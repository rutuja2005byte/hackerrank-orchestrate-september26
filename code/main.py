"""
Buy or Wait? — Phase 1 + Phase 2

Phase 1 loads the first request and prints a simple surplus.
Phase 2 adds a deterministic 90-day forecast and a safer amount_safe_to_pay.
This still does not write output.csv.
"""

from pathlib import Path
import sys

import pandas as pd

from forecast import parse_number, print_forecast_summary, run_forecast, run_simple_tests


# <repo>/code/main.py -> <repo>/dataset
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"

CSV_PATHS = {
    "requests": DATASET_DIR / "requests.csv",
    "profiles": DATASET_DIR / "financial_profiles.csv",
    "events": DATASET_DIR / "financial_events.csv",
    "sample_requests": DATASET_DIR / "sample_requests.csv",
    "exchange_rates": DATASET_DIR / "exchange_rates.csv",
}


def configure_pandas_display() -> None:
    """Show full tables in the terminal so Phase 1 output is easy to read."""
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 100)


def load_csv(path: Path, file_label: str, required: bool = True) -> pd.DataFrame:
    """Read one CSV and fail clearly if a required file is missing or empty."""
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required file: {path}")
        return pd.DataFrame()

    try:
        dataframe = pd.read_csv(path)
    except Exception as error:
        raise ValueError(f"Could not read {file_label} ({path}): {error}") from error

    if required and dataframe.empty:
        raise ValueError(f"{file_label} is empty: {path}")

    return dataframe


def print_section(title: str, data) -> None:
    """Print a labeled block so request, profile, and events are easy to scan."""
    print("=" * 80)
    print(title)
    print("=" * 80)
    if isinstance(data, pd.Series):
        print(data.to_string())
    elif isinstance(data, pd.DataFrame):
        print(f"Row count: {len(data)}")
        print(data.to_string(index=False))
    else:
        print(data)
    print()


def temporary_decision(safe_amount: float, requested_amount: float) -> tuple[str, str]:
    """Very basic status rule used by Phase 1 and Phase 2."""
    if safe_amount == requested_amount:
        return "affordable_now", "full_payment"
    if 0 < safe_amount < requested_amount:
        return "affordable_with_plan", "partial_payment"
    return "not_affordable", "not_recommended"


def select_first_request_bundle(
    requests_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Phase 1 helper: first request, matching profile, and that user's events."""
    first_request = requests_df.iloc[0]
    user_id = first_request.get("user_id")
    if user_id is None or (isinstance(user_id, float) and pd.isna(user_id)):
        raise ValueError("The first request has no user_id.")
    user_id = str(user_id).strip()
    if not user_id:
        raise ValueError("The first request has an empty user_id.")

    matching_profiles = profiles_df[profiles_df["user_id"].astype(str) == user_id]
    if matching_profiles.empty:
        raise ValueError(f"No profile found for user_id={user_id}.")
    if len(matching_profiles) > 1:
        print(
            f"Warning: found {len(matching_profiles)} profiles for "
            f"{user_id}. Using the first match.\n"
        )
    profile = matching_profiles.iloc[0]
    matching_events = events_df[events_df["user_id"].astype(str) == user_id]
    return first_request, profile, matching_events


def main() -> int:
    configure_pandas_display()

    try:
        run_simple_tests()
        print()

        requests_df = load_csv(CSV_PATHS["requests"], "requests.csv")
        profiles_df = load_csv(CSV_PATHS["profiles"], "financial_profiles.csv")
        events_df = load_csv(CSV_PATHS["events"], "financial_events.csv")
        sample_requests_df = load_csv(
            CSV_PATHS["sample_requests"], "sample_requests.csv"
        )
        exchange_rates_df = load_csv(
            CSV_PATHS["exchange_rates"], "exchange_rates.csv", required=False
        )

        print_section(
            "Loaded CSV files",
            (
                f"requests.csv          : {len(requests_df)} rows\n"
                f"financial_profiles.csv: {len(profiles_df)} rows\n"
                f"financial_events.csv  : {len(events_df)} rows\n"
                f"sample_requests.csv   : {len(sample_requests_df)} rows "
                "(loaded for format check only)\n"
                f"exchange_rates.csv    : {len(exchange_rates_df)} rows "
                "(used only when an event currency differs from home_currency)"
            ),
        )

        first_request, profile, matching_events = select_first_request_bundle(
            requests_df, profiles_df, events_df
        )
        user_id = str(first_request["user_id"])

        print_section("Selected request (first row of requests.csv)", first_request)
        print_section(f"Matching profile for {user_id}", profile)
        if matching_events.empty:
            print_section(
                f"Financial events for {user_id}",
                "No matching financial events found for this user.",
            )
        else:
            event_summary = (
                matching_events.groupby(["event_type", "status"], dropna=False)
                .size()
                .reset_index(name="count")
            )
            print_section(
                f"Financial events for {user_id} (summary, {len(matching_events)} rows)",
                event_summary,
            )

        available_balance = parse_number(
            profile["current_available_balance"],
            "current_available_balance",
        )
        minimum_balance = parse_number(
            profile["minimum_balance_to_keep"],
            "minimum_balance_to_keep",
        )
        requested_amount = parse_number(
            first_request["requested_amount"],
            "requested_amount",
        )

        raw_safe_amount = available_balance - minimum_balance
        phase1_safe_amount = max(0.0, min(raw_safe_amount, requested_amount))
        phase1_status, phase1_method = temporary_decision(
            phase1_safe_amount, requested_amount
        )

        print_section(
            "Phase 1 temporary safe amount and decision",
            (
                f"current_available_balance : {available_balance}\n"
                f"minimum_balance_to_keep   : {minimum_balance}\n"
                f"requested_amount          : {requested_amount}\n"
                f"raw surplus               : {raw_safe_amount}\n"
                f"amount_safe_to_pay        : {phase1_safe_amount}\n"
                f"affordability_status      : {phase1_status}\n"
                f"recommended_payment_method: {phase1_method}\n"
                "Note: Phase 1 ignores future bills. Phase 2 includes them."
            ),
        )

        forecast = run_forecast(
            first_request, profile, matching_events, exchange_rates_df
        )
        print_forecast_summary(forecast)

        phase2_status, phase2_method = temporary_decision(
            forecast.amount_safe_to_pay, requested_amount
        )
        print_section(
            "Phase 2 temporary decision (forecast-based safe amount)",
            (
                f"amount_safe_to_pay        : {forecast.amount_safe_to_pay}\n"
                f"affordability_status      : {phase2_status}\n"
                f"recommended_payment_method: {phase2_method}"
            ),
        )
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
        print(f"ERROR: Phase 2 test failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
