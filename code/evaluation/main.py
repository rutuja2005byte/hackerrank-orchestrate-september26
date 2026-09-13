"""
Validate dataset/output.csv against the Buy or Wait? contract.

This checker does not contain hidden labels. It only validates schema,
allowed values, plan syntax, and 90-day safety of the written plan.
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = CODE_DIR.parent
DATASET_DIR = REPO_ROOT / "dataset"
OUTPUT_PATH = DATASET_DIR / "output.csv"

sys.path.insert(0, str(CODE_DIR))

from decide import (  # noqa: E402
    build_option_schedule,
    format_amount,
    is_truthy,
    overrides_from_change_text,
    user_methods,
)
from evidence import apply_extracted_facts, select_messages  # noqa: E402
from forecast import parse_date, parse_number, run_forecast, simulate_balances  # noqa: E402
from message_facts import extract_message_facts  # noqa: E402


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
STATUSES = {
    "affordable_now",
    "affordable_with_plan",
    "affordable_later",
    "not_affordable",
}
METHODS = {
    "full_payment",
    "partial_payment",
    "installments",
    "wait",
    "not_recommended",
}
PLAN_ENTRY = re.compile(r"^(\d{4}-\d{2}-\d{2}):(-?\d+(?:\.\d+)?)$")
CHANGE_ENTRY = re.compile(r"^(stop:[A-Za-z0-9_]+|reduce_to:[A-Za-z0-9_]+:\d+(?:\.\d+)?)$")


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def parse_money(value: str) -> float:
    return float(value)


def parse_plan(plan: str) -> list[tuple[date, float]]:
    if plan == "none" or plan == "":
        return []
    entries = []
    for part in plan.split("|"):
        match = PLAN_ENTRY.match(part)
        if not match:
            raise ValueError(f"bad plan entry {part!r}")
        entries.append((parse_date(match.group(1)), float(match.group(2))))
    return entries


def amounts_close(left: float, right: float, tol: float = 0.015) -> bool:
    return abs(left - right) <= tol


def check_schema(output_df: pd.DataFrame, errors: list[str]) -> None:
    if list(output_df.columns) != OUTPUT_COLUMNS:
        errors.append(f"columns must be {OUTPUT_COLUMNS}, got {list(output_df.columns)}")


def check_ids(output_df: pd.DataFrame, requests_df: pd.DataFrame, errors: list[str]) -> None:
    expected = list(requests_df["request_id"].astype(str))
    actual = list(output_df["request_id"].astype(str))
    if len(actual) != len(expected):
        errors.append(f"expected {len(expected)} rows, got {len(actual)}")
    if actual != expected:
        errors.append("request_id order does not match requests.csv")


def check_row(
    row: pd.Series,
    request: pd.Series,
    profile: pd.Series,
    options_df: pd.DataFrame,
    events_df: pd.DataFrame,
    exchange_rates_df: pd.DataFrame,
    messages_df: pd.DataFrame,
    errors: list[str],
) -> None:
    request_id = str(row["request_id"])
    prefix = f"{request_id}: "
    requested = parse_number(request["requested_amount"], "requested_amount")
    request_date = parse_date(request["request_date"])
    deadline = parse_date(request["desired_completion_date"])

    try:
        safe = parse_money(row["amount_safe_to_pay"])
    except ValueError:
        errors.append(prefix + "amount_safe_to_pay is not a number")
        return
    if safe < -1e-9 or safe - requested > 0.015:
        errors.append(prefix + "amount_safe_to_pay is outside 0..requested_amount")
    if format_amount(safe) != str(row["amount_safe_to_pay"]).strip():
        errors.append(prefix + f"amount_safe_to_pay has float noise: {row['amount_safe_to_pay']}")

    status = row["affordability_status"]
    method = row["recommended_payment_method"]
    if status not in STATUSES:
        errors.append(prefix + f"invalid affordability_status {status!r}")
    if method not in METHODS:
        errors.append(prefix + f"invalid recommended_payment_method {method!r}")

    if not row["decision_explanation"].strip():
        errors.append(prefix + "decision_explanation is empty")

    changes = row["spending_changes_needed"]
    if changes != "none":
        parts = changes.split("|")
        if not parts or len(parts) > 3 or any(not CHANGE_ENTRY.match(part) for part in parts):
            errors.append(prefix + f"invalid spending_changes_needed {changes!r}")

    earliest_text = row["earliest_date_for_full_payment"]
    earliest = parse_date(earliest_text) if earliest_text else None
    if earliest_text and earliest is None:
        errors.append(prefix + "earliest_date_for_full_payment is not YYYY-MM-DD")
    if status == "affordable_now" and earliest != request_date:
        errors.append(prefix + "affordable_now requires earliest_date_for_full_payment = request_date")

    try:
        payments = parse_plan(row["payment_plan"])
    except ValueError as error:
        errors.append(prefix + str(error))
        return

    if payments != sorted(payments, key=lambda item: item[0]):
        errors.append(prefix + "payment_plan dates are not chronological")
    if any(amount <= 0 for _date, amount in payments):
        errors.append(prefix + "payment_plan contains a non-positive amount")

    if method == "not_recommended":
        if row["payment_plan"] != "none":
            errors.append(prefix + "not_recommended must use payment_plan=none")
        if status != "not_affordable":
            errors.append(prefix + "not_recommended must be not_affordable")
        return

    if method == "full_payment":
        if len(payments) != 1:
            errors.append(prefix + "full_payment must have exactly one payment")
        elif not amounts_close(payments[0][1], requested):
            errors.append(prefix + "full_payment amount must equal requested_amount")
        if status == "affordable_now" and payments and payments[0][0] != request_date:
            errors.append(prefix + "affordable_now full_payment must be on request_date")

    if method == "wait":
        if status != "affordable_later":
            errors.append(prefix + "wait must be affordable_later")
        if len(payments) != 1:
            errors.append(prefix + "wait must have exactly one payment")
        elif earliest and payments[0][0] != earliest:
            errors.append(prefix + "wait payment date must equal earliest_date_for_full_payment")
        elif payments and not amounts_close(payments[0][1], requested):
            errors.append(prefix + "wait amount must equal requested_amount")

    if method == "partial_payment":
        if status != "affordable_with_plan":
            errors.append(prefix + "partial_payment must be affordable_with_plan")
        if not is_truthy(request.get("allows_partial_payment")):
            errors.append(prefix + "partial_payment is not allowed by the request")
        if "partial_payment" not in user_methods(profile):
            errors.append(prefix + "user does not consider partial_payment")
        if not (0 < safe < requested):
            errors.append(prefix + "partial_payment requires 0 < amount_safe_to_pay < requested")
        if earliest is None or (deadline and earliest > deadline):
            errors.append(prefix + "partial_payment requires earliest_date on or before deadline")
        if len(payments) != 2:
            errors.append(prefix + "partial_payment must have exactly two payments")
        else:
            first_date, first_amount = payments[0]
            second_date, second_amount = payments[1]
            if first_date != request_date:
                errors.append(prefix + "partial first payment must be on request_date")
            if not amounts_close(first_amount, safe):
                errors.append(prefix + "partial first payment must equal amount_safe_to_pay")
            if earliest and second_date != earliest:
                errors.append(prefix + "partial second payment must be on earliest_date")
            if not amounts_close(first_amount + second_amount, requested):
                errors.append(prefix + "partial payments must add up to requested_amount")

    if method == "installments":
        if status != "affordable_with_plan":
            errors.append(prefix + "installments must be affordable_with_plan")
        matched = False
        for _, option in options_df.iterrows():
            if str(option.get("payment_method", "")).strip() != "installments":
                continue
            schedule = build_option_schedule(option)
            if schedule is None or len(schedule) != len(payments):
                continue
            if all(
                pay_date == opt_date and amounts_close(pay_amount, opt_amount)
                for (pay_date, pay_amount), (opt_date, opt_amount) in zip(payments, schedule)
            ):
                matched = True
                break
        if not matched:
            errors.append(prefix + "installment plan does not match a supplied payment option")

    if payments:
        selected = select_messages(
            messages_df, str(request["user_id"]), str(request["request_id"])
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
        forecast = run_forecast(
            request,
            profile,
            patched_events,
            exchange_rates_df,
            spending_overrides=overrides_from_change_text(changes, patched_events),
        )
        last_payment = max(pay_date for pay_date, _amount in payments)
        lowest, _, _, _, _ = simulate_balances(
            forecast.starting_balance,
            forecast.forecast_start,
            max(forecast.forecast_end, last_payment),
            forecast.cash_events,
            extra_payments=payments,
        )
        if lowest < forecast.minimum_balance_required - 0.01:
            errors.append(
                prefix
                + "recommended plan breaks the 90-day minimum-balance safety check "
                f"(lowest={lowest:.2f}, minimum={forecast.minimum_balance_required:.2f})"
            )


def collect_validation_errors() -> tuple[int, list[str]]:
    """Return (row_count, errors). Does not print."""
    errors: list[str] = []
    output_df = load_csv(OUTPUT_PATH)
    requests_df = pd.read_csv(DATASET_DIR / "requests.csv")
    profiles_df = pd.read_csv(DATASET_DIR / "financial_profiles.csv")
    events_df = pd.read_csv(DATASET_DIR / "financial_events.csv")
    options_df = pd.read_csv(DATASET_DIR / "request_payment_options.csv")
    rates_path = DATASET_DIR / "exchange_rates.csv"
    rates_df = pd.read_csv(rates_path) if rates_path.exists() else pd.DataFrame()
    messages_path = DATASET_DIR / "messages.csv"
    messages_df = pd.read_csv(messages_path) if messages_path.exists() else pd.DataFrame()

    check_schema(output_df, errors)
    check_ids(output_df, requests_df, errors)
    if errors:
        return len(output_df), errors

    requests_by_id = {
        str(row["request_id"]): row for _, row in requests_df.iterrows()
    }
    profiles_by_id = {
        str(row["user_id"]): row for _, row in profiles_df.iterrows()
    }
    events_by_user = {
        str(user_id): group for user_id, group in events_df.groupby(events_df["user_id"].astype(str))
    }
    options_by_request = {
        str(request_id): group
        for request_id, group in options_df.groupby(options_df["request_id"].astype(str))
    }

    for _, row in output_df.iterrows():
        request = requests_by_id[str(row["request_id"])]
        profile = profiles_by_id[str(request["user_id"])]
        check_row(
            row,
            request,
            profile,
            options_by_request.get(str(row["request_id"]), pd.DataFrame()),
            events_by_user.get(str(request["user_id"]), pd.DataFrame()),
            rates_df,
            messages_df,
            errors,
        )
    return len(output_df), errors


def main() -> int:
    try:
        row_count, errors = collect_validation_errors()
    except FileNotFoundError as error:
        print(f"EVALUATION FAILED: {error}")
        return 1

    if errors:
        print(f"EVALUATION FAILED ({len(errors)} issues)")
        for error in errors[:40]:
            print(f"- {error}")
        if len(errors) > 40:
            print(f"- ... {len(errors) - 40} more")
        return 1

    print("EVALUATION PASSED")
    print(f"Checked {row_count} rows in {OUTPUT_PATH}")
    print("Validated schema, IDs, allowed values, amount ranges, dates,")
    print("totals, installment matching, partial-payment rules, and 90-day safety.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
