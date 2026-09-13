"""
Deterministic facts from messages. No request IDs, user IDs, or answers.

Gemini is optional later for images. These rules only read the message text.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

import pandas as pd
from dateutil.relativedelta import relativedelta

from forecast import optional_number, parse_date


CURRENCIES = ("IDR", "INR", "EUR", "USD", "ZAR")
AMOUNT_RE = re.compile(
    r"(?P<cur>IDR|INR|EUR|USD|ZAR)\s*(?P<amt>\d[\d.,]*)",
    flags=re.IGNORECASE,
)
DATE_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})")

PENDING_HINTS = (
    "still pending",
    "not been approved",
    "has not reached",
    "still processing",
    "still in payment processing",
    "payout is still pending",
    "weekly earnings",
    "market value",
    "no cash proceeds",
    "has not been sold",
    "displayed value",
    "belum disetujui",
    "belum masuk",
    "masih tertunda",
    "masih dalam proses",
    "nilai investasi yang ditampilkan",
)
ENDED_HINTS = (
    "employment has ended",
    "contract has ended",
    "no regular salary",
    "no off-season income",
    "household employment record has ended",
    "hubungan kerja anda telah berakhir",
    "kontrak musiman saat ini telah berakhir",
    "tidak ada pembayaran gaji rutin",
    "pendapatan yang sudah berakhir",
    "income that has ended",
)
IGNORE_INCOME_HINTS = (
    "bonus",
    "commission",
    "komisi",
    "prize",
    "hadiah",
    "refund",
    "pengembalian",
    "reimbursement",
    "penggantian",
)
RENT_INCREASE_RE = re.compile(
    r"(?:increases monthly rent by|menaikkan (?:biaya )?sewa bulanan sebesar)\s*(\d+(?:\.\d+)?)\s*%",
    flags=re.IGNORECASE,
)


def _blank(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def parse_amount_token(currency: str, raw: str) -> float | None:
    currency = currency.upper()
    text = raw.strip().rstrip(".,;")
    if not text:
        return None
    if currency in {"EUR", "USD"}:
        if "," in text and "." in text:
            text = text.replace(",", "")
        elif "," in text and "." not in text:
            text = text.replace(",", ".")
        try:
            return float(text)
        except ValueError:
            return None
    if re.search(r"[.,]\d{1,2}$", text) and text.count(".") + text.count(",") == 1:
        try:
            return float(text.replace(",", "."))
        except ValueError:
            return None
    try:
        return float(text.replace(",", "").replace(".", ""))
    except ValueError:
        return None


def first_money(text: str) -> tuple[str, float] | None:
    match = AMOUNT_RE.search(text)
    if not match:
        return None
    amount = parse_amount_token(match.group("cur"), match.group("amt"))
    if amount is None:
        return None
    return match.group("cur").upper(), amount


def all_dates(text: str) -> list[date]:
    found = []
    for token in DATE_RE.findall(text):
        parsed = parse_date(token)
        if parsed is not None:
            found.append(parsed)
    return found


def contains_any(text: str, hints: tuple[str, ...]) -> bool:
    return any(hint in text for hint in hints)


def latest_event_id(
    events_df: pd.DataFrame,
    request_date: date,
    event_type: str,
    category: str | None = None,
) -> str:
    if events_df is None or events_df.empty:
        return ""
    rows = events_df[events_df["event_type"].astype(str) == event_type].copy()
    if category:
        rows = rows[rows["category"].astype(str) == category]
    if rows.empty:
        return ""
    rows["_cash"] = pd.to_datetime(
        rows["settlement_date"].fillna(rows["event_date"]), errors="coerce"
    )
    rows = rows[rows["_cash"] < pd.Timestamp(request_date)]
    if rows.empty:
        return ""
    latest = rows.sort_values(["_cash", "event_id"]).iloc[-1]
    return _blank(latest.get("event_id"))


def latest_salary_rows(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df is None or events_df.empty:
        return pd.DataFrame()
    rows = events_df[events_df["event_type"].astype(str) == "income"].copy()
    if "category" in rows.columns:
        salary_rows = rows[rows["category"].astype(str) == "salary"]
        if not salary_rows.empty:
            rows = salary_rows
    return rows


def next_salary_date(events_df: pd.DataFrame, request_date: date) -> date:
    if events_df is None or events_df.empty:
        return request_date.replace(day=min(request_date.day, 15)) + relativedelta(months=1)
    rows = latest_salary_rows(events_df)
    if rows.empty:
        candidate = date(request_date.year, request_date.month, 15)
        return candidate if candidate >= request_date else candidate + relativedelta(months=1)
    rows["_cycle"] = pd.to_datetime(rows["event_date"], errors="coerce")
    rows["_cycle"] = rows["_cycle"].fillna(
        pd.to_datetime(rows["settlement_date"], errors="coerce")
    )
    rows = rows.dropna(subset=["_cycle"]).sort_values("_cycle")
    if rows.empty:
        candidate = date(request_date.year, request_date.month, 15)
        return candidate if candidate >= request_date else candidate + relativedelta(months=1)
    day = int(rows.iloc[-1]["_cycle"].day)
    year, month = request_date.year, request_date.month
    try:
        candidate = date(year, month, min(day, 28 if day >= 28 else day))
        if day >= 29:
            candidate = date(year, month, 28) + relativedelta(day=day)
    except ValueError:
        candidate = date(year, month, 28)
    if candidate < request_date:
        candidate = candidate + relativedelta(months=1)
    return candidate


def latest_salary_amount(events_df: pd.DataFrame, request_date: date) -> float | None:
    if events_df is None or events_df.empty:
        return None
    rows = latest_salary_rows(events_df)
    if rows.empty:
        return None
    rows["_cycle"] = pd.to_datetime(rows["event_date"], errors="coerce")
    rows["_cycle"] = rows["_cycle"].fillna(
        pd.to_datetime(rows["settlement_date"], errors="coerce")
    )
    rows = rows[rows["_cycle"] < pd.Timestamp(request_date)]
    if rows.empty:
        return None
    latest = rows.sort_values("_cycle").iloc[-1]
    return optional_number(latest.get("amount"))


def latest_rent_amount(events_df: pd.DataFrame, request_date: date) -> float | None:
    if events_df is None or events_df.empty:
        return None
    rows = events_df[events_df["category"].astype(str).isin(["rent", "housing"])].copy()
    if rows.empty:
        return None
    rows["_cash"] = pd.to_datetime(
        rows["settlement_date"].fillna(rows["event_date"]), errors="coerce"
    )
    rows = rows[rows["_cash"] < pd.Timestamp(request_date)]
    rows = rows[rows["status"].astype(str).str.lower() == "settled"]
    if rows.empty:
        return None
    latest = rows.sort_values("_cash").iloc[-1]
    return optional_number(latest.get("amount"))


def fact_template(
    source_id: str,
    action: str,
    *,
    related_event_id: str = "",
    amount: float | None = None,
    currency: str = "",
    date_value: date | None = None,
    direction: str = "unknown",
    is_confirmed: bool = True,
    is_pending: bool = False,
    summary: str = "",
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "action": action,
        "related_event_id": related_event_id,
        "amount": amount,
        "currency": currency,
        "date": date_value.isoformat() if date_value else "",
        "direction": direction,
        "is_confirmed": is_confirmed,
        "is_pending": is_pending,
        "is_cash": True,
        "confidence": "high",
        "summary": summary,
    }


def extract_message_facts(
    messages_df: pd.DataFrame,
    events_df: pd.DataFrame,
    request_date: date,
    home_currency: str,
) -> list[dict[str, Any]]:
    """Turn selected messages into apply_extracted_facts rows."""
    facts: list[dict[str, Any]] = []
    if messages_df is None or messages_df.empty:
        return facts

    for _, row in messages_df.iterrows():
        source_id = _blank(row.get("message_id"))
        text = _blank(row.get("message_text"))
        source_type = _blank(row.get("source_type")).lower()
        related = _blank(row.get("related_event_id"))
        lowered = text.lower()
        dates = all_dates(text)
        money = first_money(text)

        if contains_any(lowered, PENDING_HINTS) and not contains_any(
            lowered, ("confirmed salary", "gaji pokok yang dikonfirmasi", "gaji yang sudah dikonfirmasi")
        ):
            facts.append(
                fact_template(
                    source_id,
                    "ignore",
                    related_event_id=related,
                    is_confirmed=False,
                    is_pending=True,
                    summary="Pending or non-cash update ignored",
                )
            )
            continue

        rent_match = RENT_INCREASE_RE.search(text)
        if rent_match:
            current = latest_rent_amount(events_df, request_date)
            rent_event = latest_event_id(events_df, request_date, "expense", "rent")
            if current is None:
                rent_event = latest_event_id(events_df, request_date, "expense", "housing")
                current = latest_rent_amount(events_df, request_date)
            if current is not None and rent_event:
                factor = 1.0 + float(rent_match.group(1)) / 100.0
                facts.append(
                    fact_template(
                        source_id,
                        "amend_amount",
                        related_event_id=rent_event,
                        amount=round(current * factor, 2),
                        currency=home_currency,
                        direction="debit",
                        summary="Lease increase applied to the latest rent amount",
                    )
                )
            continue

        if contains_any(lowered, ENDED_HINTS):
            end_date = dates[0] if dates else request_date
            if end_date >= request_date:
                end_date = request_date - relativedelta(days=1)
            facts.append(
                fact_template(
                    source_id,
                    "end_income",
                    date_value=end_date,
                    direction="credit",
                    summary="Confirmed income stream has ended",
                )
            )
            remaining = money
            if remaining and (
                "remaining confirmed monthly salary" in lowered
                or "sisa gaji bulanan yang dikonfirmasi" in lowered
            ):
                currency, amount = remaining
                pay_date = dates[0] if dates else next_salary_date(events_df, request_date)
                facts.append(
                    fact_template(
                        source_id,
                        "confirm_income",
                        amount=amount,
                        currency=currency,
                        date_value=pay_date,
                        direction="credit",
                        summary="Remaining confirmed monthly salary",
                    )
                )
            continue

        if source_type == "employer" and dates and money is None:
            if any(
                phrase in lowered
                for phrase in (
                    "expected on",
                    "diperkirakan masuk pada",
                    "replaces the payroll date",
                    "menggantikan tanggal penggajian",
                )
            ):
                last_amount = latest_salary_amount(events_df, request_date)
                if last_amount is not None:
                    facts.append(
                        fact_template(
                            source_id,
                            "confirm_income",
                            amount=last_amount,
                            currency=home_currency,
                            date_value=dates[0],
                            direction="credit",
                            summary="Confirmed salary date change",
                        )
                    )
                    continue

        if source_type in {"employer", "service_provider"} and money:
            if contains_any(lowered, IGNORE_INCOME_HINTS) and not (
                "regular salary" in lowered
                or "gaji rutin" in lowered
                or "gaji pokok" in lowered
                or "confirmed base salary" in lowered
                or "confirmed salary" in lowered
            ):
                facts.append(
                    fact_template(
                        source_id,
                        "ignore",
                        is_confirmed=False,
                        is_pending=True,
                        summary="Unconfirmed bonus, commission, or refund ignored",
                    )
                )
                continue

            currency, amount = money
            if "invoice" in lowered or "faktur" in lowered:
                if dates:
                    facts.append(
                        fact_template(
                            source_id,
                            "confirm_income",
                            amount=amount,
                            currency=currency,
                            date_value=dates[0],
                            direction="credit",
                            summary="Confirmed invoice payment",
                        )
                    )
                continue
            latest_salary = latest_event_id(events_df, request_date, "income", "salary")
            ongoing = any(
                phrase in lowered
                for phrase in (
                    "increased to",
                    "naik menjadi",
                    "temporary monthly pay",
                    "gaji bulanan sementara",
                    "confirmed base salary",
                    "gaji pokok yang dikonfirmasi",
                    "confirmed salary is",
                    "monthly salary",
                    "gaji bulanan",
                    "reduced amount continues",
                    "jumlah yang lebih rendah masih berlaku",
                    "next salary is reduced",
                    "regular salary",
                    "gaji rutin",
                    "resumes on",
                    "first salary",
                    "gaji pertama",
                )
            )
            if not ongoing:
                continue
            first_salary = "first salary" in lowered or "gaji pertama" in lowered
            if latest_salary and not dates and not first_salary:
                facts.append(
                    fact_template(
                        source_id,
                        "amend_amount",
                        related_event_id=latest_salary,
                        amount=amount,
                        currency=currency,
                        direction="credit",
                        summary="Updated recurring salary amount",
                    )
                )
            else:
                pay_date = dates[0] if dates else next_salary_date(events_df, request_date)
                facts.append(
                    fact_template(
                        source_id,
                        "confirm_income",
                        related_event_id="",
                        amount=amount,
                        currency=currency,
                        date_value=pay_date,
                        direction="credit",
                        summary="Confirmed salary from message",
                    )
                )
            continue

        facts.append(
            fact_template(
                source_id,
                "ignore",
                related_event_id=related,
                summary="No confirmed cash fact extracted",
            )
        )
    return facts


def run_simple_message_tests(quiet: bool = False) -> None:
    request_date = date(2025, 8, 5)
    events = pd.DataFrame(
        [
            {
                "event_id": "event_sal",
                "user_id": "user_test",
                "event_type": "income",
                "description": "Payroll credit",
                "category": "salary",
                "direction": "credit",
                "amount": 1000,
                "currency": "IDR",
                "event_date": "2025-07-15",
                "settlement_date": "2025-07-15",
                "status": "settled",
            },
            {
                "event_id": "event_rent",
                "user_id": "user_test",
                "event_type": "expense",
                "description": "Monthly rent",
                "category": "rent",
                "direction": "debit",
                "amount": 100,
                "currency": "INR",
                "event_date": "2025-07-01",
                "settlement_date": "2025-07-01",
                "status": "settled",
            },
        ]
    )
    messages = pd.DataFrame(
        [
            {
                "message_id": "message_test_salary",
                "source_type": "employer",
                "related_event_id": "",
                "message_text": (
                    "Your monthly salary has increased to IDR 42750000. "
                    "The change applies from 2025-08-15."
                ),
            },
            {
                "message_id": "message_test_pending",
                "source_type": "employer",
                "related_event_id": "",
                "message_text": (
                    "Your quarterly bonus is still subject to the final performance review. "
                    "The final amount and payment date have not been approved yet."
                ),
            },
            {
                "message_id": "message_test_rent",
                "source_type": "service_provider",
                "related_event_id": "",
                "message_text": "The renewed lease increases monthly rent by 12%.",
            },
            {
                "message_id": "message_test_end",
                "source_type": "employer",
                "related_event_id": "",
                "message_text": (
                    "Your employment has ended. There are no regular salary payments "
                    "scheduled after the final settlement."
                ),
            },
        ]
    )
    facts = extract_message_facts(messages, events, request_date, "IDR")
    by_id = {fact["source_id"]: fact for fact in facts}
    assert by_id["message_test_salary"]["action"] == "confirm_income"
    assert by_id["message_test_salary"]["amount"] == 42750000
    assert by_id["message_test_salary"]["date"] == "2025-08-15"
    assert by_id["message_test_pending"]["action"] == "ignore"
    assert by_id["message_test_rent"]["action"] == "amend_amount"
    assert abs(float(by_id["message_test_rent"]["amount"]) - 112.0) < 0.001
    assert by_id["message_test_end"]["action"] == "end_income"
    if not quiet:
        print("Message checks passed.")
