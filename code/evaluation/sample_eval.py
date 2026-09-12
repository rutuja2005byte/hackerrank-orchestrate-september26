"""
Score the production pipeline against dataset/sample_requests.csv.

Sample input columns are separated from expected output columns.
This file never writes answers back into the pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = CODE_DIR.parent
DATASET_DIR = REPO_ROOT / "dataset"

sys.path.insert(0, str(CODE_DIR))

from decide import format_amount  # noqa: E402
from predict import OUTPUT_COLUMNS, generate_predictions  # noqa: E402


INPUT_COLUMNS = [
    "request_id",
    "user_id",
    "request_date",
    "request_type",
    "requested_amount",
    "desired_completion_date",
    "allows_partial_payment",
    "request_text",
]
COMPARE_FIELDS = [column for column in OUTPUT_COLUMNS if column != "request_id"]


def load_split_samples(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path)
    missing_input = [column for column in INPUT_COLUMNS if column not in raw.columns]
    missing_output = [column for column in OUTPUT_COLUMNS if column not in raw.columns]
    if missing_input or missing_output:
        raise ValueError(f"sample file missing columns: {missing_input + missing_output}")
    return raw[INPUT_COLUMNS].copy(), raw[OUTPUT_COLUMNS].copy()


def normalize_expected(row: pd.Series, field: str) -> str:
    value = row[field]
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if field == "amount_safe_to_pay" and text != "":
        return format_amount(float(text))
    if field == "earliest_date_for_full_payment" and text.lower() == "nan":
        return ""
    return text


def values_match(field: str, predicted: str, expected: str) -> bool:
    if field == "amount_safe_to_pay":
        try:
            return abs(float(predicted) - float(expected)) <= 0.015
        except ValueError:
            return predicted == expected
    if field == "decision_explanation":
        return " ".join(predicted.lower().split()) == " ".join(expected.lower().split())
    return predicted == expected


def likely_cause(field: str, predicted: str, expected: str) -> str:
    if field == "amount_safe_to_pay":
        try:
            pred = float(predicted)
            exp = float(expected)
        except ValueError:
            return "safe-amount format mismatch"
        if pred > exp + 0.02:
            return "forecast is too optimistic (missing reserved expenses or variable spend)"
        if pred < exp - 0.02:
            return "forecast is too pessimistic (over-reserving or missing confirmed income)"
        return "safe-amount rounding"
    if field == "spending_changes_needed":
        if predicted == "none" and expected != "none":
            return "needed spending change was not selected from recurring flexible series"
        return "spending-change selection differs"
    if field == "earliest_date_for_full_payment":
        if predicted == "" and expected:
            return "full payment never became safe in our 90-day forecast"
        if expected == "" and predicted:
            return "we found a later full-payment date that the sample treats as never safe"
        return "payday / cash-date timing differs from the sample"
    if field == "recommended_payment_method":
        if predicted == "not_recommended" and expected in {"full_payment", "installments", "wait"}:
            return "a sample plan needs spending changes or a different cash forecast"
        if predicted == "wait" and expected == "full_payment":
            return "sample pays today after a spending change; we wait instead"
        if predicted == "full_payment" and expected == "installments":
            return "ranking preferred cheaper/earlier full payment over the sample installment"
        return "eligible-plan set or ranking differs"
    if field == "affordability_status":
        return "status follows the chosen method; usually caused by the method/safe-amount gap"
    if field == "payment_plan":
        return "plan dates/amounts follow the chosen method and forecast"
    return "field differs from the sample"


def evaluate_samples() -> int:
    samples_path = DATASET_DIR / "sample_requests.csv"
    inputs, expected = load_split_samples(samples_path)
    profiles = pd.read_csv(DATASET_DIR / "financial_profiles.csv")
    events = pd.read_csv(DATASET_DIR / "financial_events.csv")
    options = pd.read_csv(DATASET_DIR / "request_payment_options.csv")
    rates_path = DATASET_DIR / "exchange_rates.csv"
    rates = pd.read_csv(rates_path) if rates_path.exists() else pd.DataFrame()
    messages_path = DATASET_DIR / "messages.csv"
    messages = pd.read_csv(messages_path) if messages_path.exists() else pd.DataFrame()

    predicted_rows = generate_predictions(
        inputs, profiles, events, options, rates, messages, progress_every=0
    )
    predicted = pd.DataFrame(predicted_rows)

    field_hits = {field: 0 for field in COMPARE_FIELDS}
    mismatches: list[str] = []
    exact_rows = 0

    for _, expected_row in expected.iterrows():
        request_id = str(expected_row["request_id"])
        pred_matches = predicted[predicted["request_id"] == request_id]
        if pred_matches.empty:
            mismatches.append(f"{request_id}: missing prediction")
            continue
        pred_row = pred_matches.iloc[0]
        row_ok = True
        for field in COMPARE_FIELDS:
            pred_value = "" if pd.isna(pred_row[field]) else str(pred_row[field])
            exp_value = normalize_expected(expected_row, field)
            if values_match(field, pred_value, exp_value):
                field_hits[field] += 1
            else:
                row_ok = False
                mismatches.append(
                    f"{request_id} {field}: predicted={pred_value!r} "
                    f"expected={exp_value!r} | {likely_cause(field, pred_value, exp_value)}"
                )
        if row_ok:
            exact_rows += 1

    total = len(expected)
    print("=" * 80)
    print("Sample ground-truth evaluation")
    print("=" * 80)
    print(f"Samples scored: {total}")
    print(f"Exact rows (except explanation wording): {exact_rows}/{total}")
    print()
    print("Per-field accuracy")
    for field in COMPARE_FIELDS:
        hits = field_hits[field]
        print(f"  {field:32} {hits:2}/{total}  ({100 * hits / total:.1f}%)")
    print()
    if mismatches:
        print(f"Mismatches ({len(mismatches)}):")
        for line in mismatches:
            print(f"  - {line}")
    else:
        print("No field mismatches.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(evaluate_samples())
