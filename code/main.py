"""
Buy or Wait? — generate dataset/output.csv for every request.

Gemini is not used on the full-dataset run so predictions stay deterministic.
"""

from pathlib import Path
import sys
from typing import Any

from decide import run_simple_decision_tests
from evidence import run_simple_evidence_tests
from forecast import run_simple_tests
from gemini_extract import UsageRecord, write_usage_report
from message_facts import run_simple_message_tests
from predict import generate_predictions, summarize_rows, write_output_csv


REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
CSV_PATHS = {
    "requests": DATASET_DIR / "requests.csv",
    "profiles": DATASET_DIR / "financial_profiles.csv",
    "events": DATASET_DIR / "financial_events.csv",
    "exchange_rates": DATASET_DIR / "exchange_rates.csv",
    "payment_options": DATASET_DIR / "request_payment_options.csv",
    "messages": DATASET_DIR / "messages.csv",
}
OUTPUT_PATH = DATASET_DIR / "output.csv"
ROOT_OUTPUT_PATH = REPO_ROOT / "output.csv"
USAGE_REPORT_PATHS = (
    REPO_ROOT / "code" / "evaluation" / "usage_report.md",
    REPO_ROOT / "code" / "usage_report.md",
)


STATUS_LABELS = {
    "affordable_now": "Affordable now",
    "affordable_with_plan": "Affordable with a plan",
    "affordable_later": "Affordable later",
    "not_affordable": "Not affordable",
}
METHOD_LABELS = {
    "full_payment": "Pay in full",
    "partial_payment": "Pay in part",
    "installments": "Use installments",
    "wait": "Wait",
    "not_recommended": "Do not proceed",
}


def print_count_block(title: str, counts: dict[str, int], labels: dict[str, str]) -> None:
    print(title)
    for key, label in labels.items():
        if key in counts:
            print(f"  {label:<24} {counts[key]}")


def print_run_summary(rows: list[dict[str, Any]]) -> None:
    summary = summarize_rows(rows)
    print()
    print(f"Saved {summary['count']} recommendations to output.csv")
    print()
    print_count_block("Affordability", summary["status"], STATUS_LABELS)
    print()
    print_count_block("Recommendation", summary["method"], METHOD_LABELS)
    print()


def load_csv(path: Path, file_label: str, required: bool = True):
    import pandas as pd

    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required file: {path}")
        return pd.DataFrame()
    dataframe = pd.read_csv(path)
    if required and dataframe.empty:
        raise ValueError(f"{file_label} is empty: {path}")
    return dataframe


def main() -> int:
    try:
        print("Running checks...")
        run_simple_tests(quiet=True)
        run_simple_decision_tests(quiet=True)
        run_simple_evidence_tests(quiet=True)
        run_simple_message_tests(quiet=True)
        print("Checks passed.")
        print()

        requests_df = load_csv(CSV_PATHS["requests"], "requests.csv")
        profiles_df = load_csv(CSV_PATHS["profiles"], "financial_profiles.csv")
        events_df = load_csv(CSV_PATHS["events"], "financial_events.csv")
        payment_options_df = load_csv(
            CSV_PATHS["payment_options"], "request_payment_options.csv"
        )
        exchange_rates_df = load_csv(
            CSV_PATHS["exchange_rates"], "exchange_rates.csv", required=False
        )
        messages_df = load_csv(CSV_PATHS["messages"], "messages.csv", required=False)

        print(f"Writing recommendations for {len(requests_df)} requests...")
        rows = generate_predictions(
            requests_df,
            profiles_df,
            events_df,
            payment_options_df,
            exchange_rates_df,
            messages_df,
        )
        write_output_csv(OUTPUT_PATH, rows)
        write_output_csv(ROOT_OUTPUT_PATH, rows)

        usage = UsageRecord(model="none")
        usage.notes.append(
            "Full-dataset predictions are deterministic and did not call Gemini."
        )
        for usage_path in USAGE_REPORT_PATHS:
            write_usage_report(usage_path, usage, request_count=len(rows))

        print_run_summary(rows)
        return 0

    except FileNotFoundError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except KeyError as error:
        print(f"ERROR: Missing expected column {error}", file=sys.stderr)
        return 1
    except AssertionError as error:
        print(f"ERROR: test failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
