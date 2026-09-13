"""
Buy or Wait? — generate dataset/output.csv for every request.

Gemini is not used on the full-dataset run so predictions stay deterministic.
"""

from pathlib import Path
import sys

from decide import run_simple_decision_tests
from evaluation.main import collect_validation_errors
from evaluation.sample_eval import score_samples
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


STATUS_ORDER = (
    "affordable_now",
    "affordable_with_plan",
    "affordable_later",
    "not_affordable",
)
METHOD_ORDER = (
    "full_payment",
    "partial_payment",
    "installments",
    "wait",
    "not_recommended",
)
SAMPLE_FIELDS = (
    ("affordability_status", "Affordability status accuracy"),
    ("recommended_payment_method", "Payment method accuracy"),
    ("payment_plan", "Payment plan accuracy"),
    ("earliest_date_for_full_payment", "Earliest date accuracy"),
)


def banner(title: str) -> None:
    line = "=" * 70
    print(line)
    print(title)
    print(line)


def print_breakdown(title: str, counts: dict[str, int], order: tuple[str, ...], total: int) -> None:
    print(f"{title}:")
    for key in order:
        value = counts.get(key, 0)
        if value == 0:
            continue
        percent = 100.0 * value / total if total else 0.0
        print(f"  * {key:<24} : {value:4} ({percent:5.1f}%)")


def evidence_line(events_df, messages_df, images_df) -> str:
    cancelled = 0
    income_users = 0
    if events_df is not None and not events_df.empty:
        if "status" in events_df.columns:
            cancelled = int(events_df["status"].astype(str).str.lower().eq("cancelled").sum())
        if "event_type" in events_df.columns and "user_id" in events_df.columns:
            income = events_df[events_df["event_type"].astype(str) == "income"]
            income_users = int(income["user_id"].nunique())
    message_count = 0 if messages_df is None else len(messages_df)
    image_count = 0
    if images_df is not None and not images_df.empty:
        media = DATASET_DIR / "media" / "images"
        if "image_id" in images_df.columns:
            image_count = sum(
                1
                for image_id in images_df["image_id"].astype(str)
                if (media / f"{image_id}.png").exists()
            )
        else:
            image_count = len(images_df)
    return (
        f"Evidence: {message_count} messages, {image_count} image files, "
        f"{income_users} users with income records, {cancelled} cancelled events."
    )


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
        banner("Buy or Wait?")
        print("Personalised pay / wait / installment advice for each request.")
        print()

        run_simple_tests(quiet=True)
        run_simple_decision_tests(quiet=True)
        run_simple_evidence_tests(quiet=True)
        run_simple_message_tests(quiet=True)
        print("Built-in checks: 4 passed.")
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
        images_df = load_csv(DATASET_DIR / "images.csv", "images.csv", required=False)

        print(
            f"Loaded {len(requests_df)} requests, {len(profiles_df)} profiles, "
            f"{len(events_df)} events."
        )
        print(evidence_line(events_df, messages_df, images_df))
        print()

        print(f"Scoring {len(requests_df)} requests...")
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

        summary = summarize_rows(rows)
        print()
        banner("Run complete")
        print("Output file: dataset/output.csv")
        print(f"Rows written:    {summary['count']}")
        print()
        print_breakdown(
            "Affordability status", summary["status"], STATUS_ORDER, summary["count"]
        )
        print()
        print_breakdown(
            "Payment method", summary["method"], METHOD_ORDER, summary["count"]
        )

        print()
        banner("Submission contract")
        _row_count, errors = collect_validation_errors()
        if errors:
            print(f"Result: failed ({len(errors)} issues)")
            for error in errors[:8]:
                print(f"  - {error}")
            return 1
        print("Result: passed")

        print()
        banner("Public sample check")
        sample = score_samples()
        total = sample["total"]
        print(f"Sample requests: {total}")
        for field, label in SAMPLE_FIELDS:
            hits = sample["field_hits"].get(field, 0)
            print(f"{label + ':':<34} {100.0 * hits / total:5.1f}%")
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
