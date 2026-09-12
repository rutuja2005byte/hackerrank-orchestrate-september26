"""
Phase 4 — collect messages/images and apply extracted facts.

All application rules are deterministic. Gemini only proposes facts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from forecast import optional_number, parse_date
from gemini_extract import fact_amount


def _blank_to_empty(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def select_messages(
    messages_df: pd.DataFrame,
    user_id: str,
    request_id: str,
) -> pd.DataFrame:
    """Request-linked messages, plus user-level messages with a blank request_id."""
    if messages_df is None or messages_df.empty:
        return pd.DataFrame()
    user_match = messages_df["user_id"].astype(str) == str(user_id)
    request_values = messages_df["request_id"].map(_blank_to_empty)
    linked = user_match & (request_values == str(request_id))
    user_level = user_match & (request_values == "")
    selected = messages_df[linked | user_level].copy()
    if selected.empty:
        return selected
    selected["_sent_at"] = pd.to_datetime(selected["sent_at"], errors="coerce")
    return selected.sort_values(["_sent_at", "message_id"], kind="mergesort")


def select_images(
    images_df: pd.DataFrame,
    user_id: str,
    request_id: str,
    event_ids: set[str],
    media_dir: Path,
) -> pd.DataFrame:
    """Images for this request, or for one of this user's event ids."""
    if images_df is None or images_df.empty:
        return pd.DataFrame()
    user_match = images_df["user_id"].astype(str) == str(user_id)
    request_match = images_df["request_id"].map(_blank_to_empty) == str(request_id)
    related = images_df["related_event_id"].map(_blank_to_empty)
    event_match = related.isin(event_ids)
    selected = images_df[user_match & (request_match | event_match)].copy()
    if selected.empty:
        return selected

    paths = []
    exists = []
    for image_id in selected["image_id"].astype(str):
        path = media_dir / f"{image_id}.png"
        paths.append(str(path))
        exists.append(path.exists())
    selected["image_path"] = paths
    selected["image_exists"] = exists
    return selected


def messages_as_payload(messages_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for _, row in messages_df.iterrows():
        rows.append(
            {
                "message_id": _blank_to_empty(row.get("message_id")),
                "sent_at": _blank_to_empty(row.get("sent_at")),
                "source_type": _blank_to_empty(row.get("source_type")),
                "related_event_id": _blank_to_empty(row.get("related_event_id")),
                "request_id": _blank_to_empty(row.get("request_id")),
                "message_text": _blank_to_empty(row.get("message_text")),
            }
        )
    return rows


def images_as_payload(images_df: pd.DataFrame) -> tuple[list[dict[str, Any]], list[str]]:
    items = []
    skipped = []
    for _, row in images_df.iterrows():
        image_id = _blank_to_empty(row.get("image_id"))
        if not bool(row.get("image_exists")):
            skipped.append(f"{image_id}: image file is missing, so nothing was invented.")
            continue
        items.append(
            {
                "image_id": image_id,
                "user_id": _blank_to_empty(row.get("user_id")),
                "request_id": _blank_to_empty(row.get("request_id")),
                "related_event_id": _blank_to_empty(row.get("related_event_id")),
                "path": row.get("image_path"),
            }
        )
    return items, skipped


def event_context_rows(events_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Give Gemini only identifiers, not a request to do the forecast."""
    if events_df is None or events_df.empty:
        return []
    rows = []
    for _, row in events_df.iterrows():
        amount = row.get("amount")
        rows.append(
            {
                "event_id": _blank_to_empty(row.get("event_id")),
                "event_type": _blank_to_empty(row.get("event_type")),
                "status": _blank_to_empty(row.get("status")),
                "direction": _blank_to_empty(row.get("direction")),
                "category": _blank_to_empty(row.get("category")),
                "description": _blank_to_empty(row.get("description")),
                "event_date": _blank_to_empty(row.get("event_date")),
                "settlement_date": _blank_to_empty(row.get("settlement_date")),
                "amount_is_blank": optional_number(amount) is None,
            }
        )
    return rows


def _event_index(events_df: pd.DataFrame, event_id: str) -> int | None:
    if not event_id or events_df.empty:
        return None
    matches = events_df.index[events_df["event_id"].astype(str) == event_id].tolist()
    return int(matches[0]) if matches else None


def _should_skip_fact(fact: dict[str, Any], notes: list[str]) -> bool:
    source_id = _blank_to_empty(fact.get("source_id")) or "unknown_source"
    action = _blank_to_empty(fact.get("action")) or "ignore"
    confidence = _blank_to_empty(fact.get("confidence")).lower() or "low"
    if action == "ignore" or confidence == "low":
        notes.append(f"{source_id}: ignored (action={action}, confidence={confidence}).")
        return True
    if fact.get("is_cash") is False:
        notes.append(f"{source_id}: ignored because it is not cash.")
        return True
    if fact.get("is_pending") and _blank_to_empty(fact.get("direction")) == "credit":
        notes.append(f"{source_id}: pending credit ignored until it settles.")
        return True
    if fact.get("is_pending") and action in {"confirm_income"}:
        notes.append(f"{source_id}: pending income ignored until it settles.")
        return True
    return False


def apply_extracted_facts(
    events_df: pd.DataFrame,
    facts: list[dict[str, Any]],
    home_currency: str,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Patch or append event rows from extracted facts.

    Newer facts win when they touch the same event. The LLM does not run here.
    """
    patched = events_df.copy() if events_df is not None else pd.DataFrame()
    notes: list[str] = []
    added = 0

    for fact in facts:
        if _should_skip_fact(fact, notes):
            continue
        source_id = _blank_to_empty(fact.get("source_id")) or "unknown_source"
        action = _blank_to_empty(fact.get("action"))
        event_id = _blank_to_empty(fact.get("related_event_id"))
        amount = fact_amount(fact.get("amount"))
        date_text = _blank_to_empty(fact.get("date"))
        parsed_date = parse_date(date_text) if date_text else None
        currency = _blank_to_empty(fact.get("currency")) or home_currency

        if action in {"fill_amount", "amend_amount", "amend_date", "cancel"}:
            index = _event_index(patched, event_id)
            if index is None:
                notes.append(f"{source_id}: no matching event_id={event_id}.")
                continue
            if action in {"fill_amount", "amend_amount"}:
                if amount is None:
                    notes.append(f"{source_id}: skipped {action} because amount is blank.")
                    continue
                if action == "fill_amount" and optional_number(patched.at[index, "amount"]) is not None:
                    notes.append(
                        f"{source_id}: fill_amount skipped because {event_id} already has an amount."
                    )
                    continue
                patched.at[index, "amount"] = amount
                notes.append(f"{source_id}: set {event_id} amount to {amount}.")
            if action == "amend_date":
                if parsed_date is None:
                    notes.append(f"{source_id}: skipped amend_date because the date is invalid.")
                    continue
                patched.at[index, "settlement_date"] = parsed_date.isoformat()
                notes.append(f"{source_id}: set {event_id} settlement_date to {parsed_date}.")
            if action == "cancel":
                patched.at[index, "status"] = "cancelled"
                notes.append(f"{source_id}: marked {event_id} cancelled.")
            continue

        if action == "end_income":
            new_id = f"evidence_{source_id}_end"
            if _event_index(patched, new_id) is not None:
                notes.append(f"{source_id}: income-end marker {new_id} already exists.")
                continue
            end_date = parsed_date.isoformat() if parsed_date else ""
            if not end_date:
                notes.append(f"{source_id}: skipped end_income because the date is invalid.")
                continue
            new_row = {
                "event_id": new_id,
                "user_id": patched["user_id"].iloc[0] if not patched.empty else "",
                "event_type": "income",
                "description": "Final employer payroll",
                "category": "salary",
                "direction": "credit",
                "amount": 0.0,
                "currency": currency,
                "event_date": end_date,
                "settlement_date": end_date,
                "status": "settled",
                "linked_event_id": "",
                "flexibility": "fixed",
                "minimum_allowed_amount": "",
            }
            patched = pd.concat([patched, pd.DataFrame([new_row])], ignore_index=True)
            notes.append(f"{source_id}: marked income ended on {end_date}.")
            continue

        if action in {"confirm_income", "confirm_expense"}:
            if amount is None or parsed_date is None or not fact.get("is_confirmed"):
                notes.append(
                    f"{source_id}: skipped {action} (need confirmed amount and date)."
                )
                continue
            if event_id and _event_index(patched, event_id) is not None:
                index = _event_index(patched, event_id)
                patched.at[index, "amount"] = amount
                patched.at[index, "settlement_date"] = parsed_date.isoformat()
                notes.append(f"{source_id}: updated existing {event_id} from {action}.")
                continue
            new_id = f"evidence_{source_id}"
            if _event_index(patched, new_id) is not None:
                notes.append(f"{source_id}: synthetic event {new_id} already exists.")
                continue
            direction = "credit" if action == "confirm_income" else "debit"
            event_type = "income" if action == "confirm_income" else "expense"
            new_row = {
                "event_id": new_id,
                "user_id": patched["user_id"].iloc[0] if not patched.empty else "",
                "event_type": event_type,
                "description": _blank_to_empty(fact.get("summary")) or "Confirmed evidence",
                "category": "salary" if action == "confirm_income" else "other",
                "direction": direction,
                "amount": amount,
                "currency": currency,
                "event_date": parsed_date.isoformat(),
                "settlement_date": parsed_date.isoformat(),
                "status": "scheduled",
                "linked_event_id": "",
                "flexibility": "fixed",
                "minimum_allowed_amount": "",
            }
            patched = pd.concat([patched, pd.DataFrame([new_row])], ignore_index=True)
            added += 1
            notes.append(
                f"{source_id}: added scheduled {event_type} {new_id} "
                f"{amount} on {parsed_date}."
            )
            continue

        notes.append(f"{source_id}: unknown action '{action}' was ignored.")

    if added:
        notes.append(f"Added {added} evidence-only scheduled event(s).")
    return patched, notes


def run_simple_evidence_tests() -> None:
    """Deterministic application tests. These do not call Gemini."""
    events = pd.DataFrame(
        [
            {
                "event_id": "event_blank",
                "user_id": "user_test",
                "event_type": "expense",
                "description": "Outstanding bill",
                "category": "utilities",
                "direction": "debit",
                "amount": None,
                "currency": "USD",
                "event_date": "2024-01-10",
                "settlement_date": "2024-01-10",
                "status": "pending",
                "linked_event_id": "",
                "flexibility": "fixed",
                "minimum_allowed_amount": "",
            },
            {
                "event_id": "event_keep",
                "user_id": "user_test",
                "event_type": "income",
                "description": "Payroll credit",
                "category": "salary",
                "direction": "credit",
                "amount": 1000,
                "currency": "USD",
                "event_date": "2024-01-15",
                "settlement_date": "2024-01-15",
                "status": "scheduled",
                "linked_event_id": "",
                "flexibility": "fixed",
                "minimum_allowed_amount": "",
            },
        ]
    )

    filled, notes = apply_extracted_facts(
        events,
        [
            {
                "source_id": "image_01",
                "action": "fill_amount",
                "related_event_id": "event_blank",
                "amount": 88.5,
                "currency": "USD",
                "date": "2024-01-10",
                "direction": "debit",
                "is_confirmed": True,
                "is_pending": False,
                "is_cash": True,
                "confidence": "high",
                "summary": "Bill total from image",
            }
        ],
        "USD",
    )
    assert float(filled.loc[filled.event_id == "event_blank", "amount"].iloc[0]) == 88.5
    assert any("set event_blank amount" in note for note in notes)

    cancelled, _ = apply_extracted_facts(
        events,
        [
            {
                "source_id": "message_01",
                "action": "cancel",
                "related_event_id": "event_keep",
                "amount": None,
                "currency": "USD",
                "date": "",
                "direction": "credit",
                "is_confirmed": True,
                "is_pending": False,
                "is_cash": True,
                "confidence": "high",
                "summary": "Payroll cancelled",
            }
        ],
        "USD",
    )
    assert cancelled.loc[cancelled.event_id == "event_keep", "status"].iloc[0] == "cancelled"

    pending, pending_notes = apply_extracted_facts(
        events,
        [
            {
                "source_id": "message_02",
                "action": "confirm_income",
                "related_event_id": "",
                "amount": 500,
                "currency": "USD",
                "date": "2024-02-01",
                "direction": "credit",
                "is_confirmed": False,
                "is_pending": True,
                "is_cash": True,
                "confidence": "high",
                "summary": "Pending payout",
            }
        ],
        "USD",
    )
    assert len(pending) == len(events)
    assert any("pending" in note.lower() for note in pending_notes)

    added, add_notes = apply_extracted_facts(
        events,
        [
            {
                "source_id": "message_03",
                "action": "confirm_income",
                "related_event_id": "",
                "amount": 250,
                "currency": "USD",
                "date": "2024-02-15",
                "direction": "credit",
                "is_confirmed": True,
                "is_pending": False,
                "is_cash": True,
                "confidence": "high",
                "summary": "Confirmed invoice",
            }
        ],
        "USD",
    )
    assert "evidence_message_03" in set(added["event_id"].astype(str))
    assert any("added scheduled income" in note for note in add_notes)

    from gemini_extract import parse_facts_payload

    parsed = parse_facts_payload('{"facts": [{"source_id": "x", "action": "ignore"}]}')
    assert parsed[0]["source_id"] == "x"

    print("Phase 4 simple tests: all assertions passed.")
