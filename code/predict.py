"""
Phase 5 — generate one output row per request.

Predictions come only from the deterministic forecast + plan engine.
No request_id labels are hardcoded.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

from decide import Decision, format_amount, recommend_plan
from evidence import apply_extracted_facts, select_messages
from forecast import parse_date, run_forecast
from message_facts import extract_message_facts


OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]


def index_by_column(dataframe: pd.DataFrame, column: str) -> dict[str, pd.DataFrame]:
    if dataframe is None or dataframe.empty:
        return {}
    grouped = {}
    for key, group in dataframe.groupby(dataframe[column].astype(str)):
        grouped[str(key)] = group
    return grouped


def profile_for_user(profiles_df: pd.DataFrame, user_id: str) -> pd.Series:
    matches = profiles_df[profiles_df["user_id"].astype(str) == str(user_id)]
    if matches.empty:
        raise ValueError(f"No profile found for user_id={user_id}.")
    return matches.iloc[0]


def predict_one(
    request: pd.Series,
    profile: pd.Series,
    events_df: pd.DataFrame,
    payment_options_df: pd.DataFrame,
    exchange_rates_df: pd.DataFrame | None,
    messages_df: pd.DataFrame | None = None,
) -> Decision:
    """Run forecast + plan ranking for a single request."""
    request_date = parse_date(request["request_date"])
    if request_date is None:
        raise ValueError("request_date is missing or invalid.")
    selected = select_messages(
        messages_df if messages_df is not None else pd.DataFrame(),
        str(request["user_id"]),
        str(request["request_id"]),
    )
    facts = extract_message_facts(
        selected,
        events_df,
        request_date,
        str(profile.get("home_currency", "") or ""),
    )
    patched_events, _notes = apply_extracted_facts(
        events_df, facts, str(profile.get("home_currency", "") or "")
    )
    forecast = run_forecast(request, profile, patched_events, exchange_rates_df)
    return recommend_plan(
        request,
        profile,
        forecast,
        payment_options_df,
        patched_events,
        exchange_rates_df,
    )


def decision_to_row(decision: Decision) -> dict[str, str]:
    explanation = " ".join(str(decision.decision_explanation or "").split())
    return {
        "request_id": decision.request_id,
        "amount_safe_to_pay": format_amount(decision.amount_safe_to_pay),
        "affordability_status": decision.affordability_status,
        "recommended_payment_method": decision.recommended_payment_method,
        "payment_plan": decision.payment_plan or "none",
        "earliest_date_for_full_payment": decision.earliest_date_for_full_payment or "",
        "spending_changes_needed": decision.spending_changes_needed or "none",
        "decision_explanation": explanation,
    }


def generate_predictions(
    requests_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    events_df: pd.DataFrame,
    payment_options_df: pd.DataFrame,
    exchange_rates_df: pd.DataFrame | None = None,
    messages_df: pd.DataFrame | None = None,
    progress_every: int = 50,
) -> list[dict[str, str]]:
    """Preserve requests.csv order. Do not look up sample labels."""
    events_by_user = index_by_column(events_df, "user_id")
    options_by_request = index_by_column(payment_options_df, "request_id")
    rows: list[dict[str, str]] = []

    for index, request in requests_df.iterrows():
        request_id = str(request["request_id"])
        user_id = str(request["user_id"])
        profile = profile_for_user(profiles_df, user_id)
        user_events = events_by_user.get(user_id, pd.DataFrame())
        request_options = options_by_request.get(request_id, pd.DataFrame())
        decision = predict_one(
            request,
            profile,
            user_events,
            request_options,
            exchange_rates_df,
            messages_df,
        )
        rows.append(decision_to_row(decision))
        done = len(rows)
        if progress_every and done % progress_every == 0:
            print(f"Predicted {done}/{len(requests_df)} requests...")

    return rows


def write_output_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    methods: dict[str, int] = {}
    for row in rows:
        status = row["affordability_status"]
        method = row["recommended_payment_method"]
        counts[status] = counts.get(status, 0) + 1
        methods[method] = methods.get(method, 0) + 1
    return {"count": len(rows), "status": counts, "method": methods}
