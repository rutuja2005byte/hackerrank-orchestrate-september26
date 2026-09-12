"""
Phase 2 — basic 90-day cash forecast.

The dataset has no recurrence column. Recurrence is inferred only when
the same event_type + category + description repeats on a monthly cycle.

Cash rules taken from problem_statement.md / the real CSV values:
- amounts are always positive; debit subtracts, credit adds
- settlement_date is the cash date (event_date if settlement is blank)
- settled history before request_date is already inside current_available_balance
- ignore failed, cancelled, unrealized, non_cash, and pending credits
- do not invent one-off bonuses, prizes, or commissions as future income
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
from dateutil.relativedelta import relativedelta


FORECAST_DAYS = 90
IGNORE_STATUSES = {"failed", "cancelled", "unrealized"}
ONE_OFF_INCOME_HINTS = (
    "bonus",
    "arrears",
    "prize",
    "commission",
    "prorated",
    "windfall",
)


@dataclass
class CashEvent:
    """One cash movement that may hit the forecast window."""

    cash_date: date
    amount: float
    direction: str
    event_type: str
    category: str
    description: str
    source: str
    event_id: str = ""


@dataclass
class ForecastResult:
    starting_balance: float
    total_confirmed_income: float
    total_protected_expenses: float
    lowest_forecast_balance: float
    date_of_lowest_balance: date | None
    minimum_balance_required: float
    is_safe: bool
    amount_safe_to_pay: float
    requested_amount: float
    forecast_start: date
    forecast_end: date
    notes: list[str] = field(default_factory=list)
    applied_event_count: int = 0
    cash_events: list[CashEvent] = field(default_factory=list)


def parse_date(value: Any) -> date | None:
    """Read a YYYY-MM-DD value. Return None when the cell is blank."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return datetime.strptime(text[:10], "%Y-%m-%d").date()


def parse_number(value: Any, field_name: str) -> float:
    """Convert a CSV number. Raise a clear error if it is missing or invalid."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raise ValueError(f"Missing numeric value in '{field_name}'.")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric value in '{field_name}': {value!r}") from error
    if pd.isna(number):
        raise ValueError(f"Invalid numeric value in '{field_name}': {value!r}")
    return number


def optional_number(value: Any) -> float | None:
    """Return a float, or None when the amount cell is blank."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def split_categories(value: Any) -> set[str]:
    """Split a pipe-separated profile field into a set of category names."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    return {part.strip() for part in str(value).split("|") if part.strip()}


def cash_date_for_row(row: pd.Series) -> date | None:
    """Use settlement_date when present; otherwise fall back to event_date."""
    return parse_date(row.get("settlement_date")) or parse_date(row.get("event_date"))


def is_one_off_income(description: str) -> bool:
    """Bonuses, prizes, and similar labels are not treated as recurring income."""
    text = description.lower()
    return any(hint in text for hint in ONE_OFF_INCOME_HINTS)


def convert_to_home_currency(
    amount: float,
    currency: str,
    home_currency: str,
    cash_date: date,
    exchange_rates_df: pd.DataFrame | None,
    notes: list[str],
    event_id: str,
) -> float | None:
    """
    Convert a foreign-currency amount using the dated rate table.

    If currencies already match, the amount is used as-is. If no exact
    rate row exists for that date and pair, the event is skipped.
    """
    currency = str(currency or "").strip().upper()
    home_currency = str(home_currency or "").strip().upper()
    if not currency or currency == home_currency:
        return amount
    if exchange_rates_df is None or exchange_rates_df.empty:
        notes.append(
            f"{event_id}: skipped FX {currency}->{home_currency} (no rate table)."
        )
        return None

    matches = exchange_rates_df[
        (exchange_rates_df["rate_date"].astype(str) == cash_date.isoformat())
        & (exchange_rates_df["from_currency"].astype(str).str.upper() == currency)
        & (exchange_rates_df["to_currency"].astype(str).str.upper() == home_currency)
    ]
    if matches.empty:
        notes.append(
            f"{event_id}: skipped FX {currency}->{home_currency} on {cash_date} "
            "(no exact rate row)."
        )
        return None
    rate = optional_number(matches.iloc[0]["rate"])
    if rate is None:
        notes.append(f"{event_id}: skipped FX because the rate value is invalid.")
        return None
    return amount * rate


def classify_event_row(
    row: pd.Series,
    request_date: date,
    home_currency: str,
    exchange_rates_df: pd.DataFrame | None,
    notes: list[str],
) -> tuple[str, CashEvent | None]:
    """
    Decide whether a raw CSV row is history, an explicit future cash event,
    or something the forecast must ignore.
    """
    event_id = str(row.get("event_id", "") or "")
    status = str(row.get("status", "") or "").strip().lower()
    direction = str(row.get("direction", "") or "").strip().lower()
    event_type = str(row.get("event_type", "") or "").strip()
    category = str(row.get("category", "") or "").strip()
    description = str(row.get("description", "") or "").strip()

    if status in IGNORE_STATUSES:
        return "ignored", None
    if direction == "non_cash":
        return "ignored", None

    cash_date = cash_date_for_row(row)
    if cash_date is None:
        notes.append(f"{event_id}: skipped because both date fields are blank.")
        return "ignored", None

    amount = optional_number(row.get("amount"))
    if amount is None:
        notes.append(
            f"{event_id}: amount is blank, so it is not treated as zero. "
            "Image/OCR lookup is not part of Phase 2."
        )
        # Past blank amounts can still be ignored for history; future ones
        # cannot be applied until an amount is known.
        if cash_date < request_date and status == "settled":
            return "ignored", None
        return "ignored", None

    converted = convert_to_home_currency(
        amount,
        str(row.get("currency", "")),
        home_currency,
        cash_date,
        exchange_rates_df,
        notes,
        event_id,
    )
    if converted is None:
        return "ignored", None

    event = CashEvent(
        cash_date=cash_date,
        amount=converted,
        direction=direction,
        event_type=event_type,
        category=category,
        description=description,
        source=status,
        event_id=event_id,
    )

    if status == "pending" and direction == "credit":
        notes.append(f"{event_id}: pending credit ignored until it settles.")
        return "ignored", None

    if status == "settled" and cash_date < request_date:
        return "history", event

    if cash_date < request_date:
        return "ignored", None

    if status in {"pending", "scheduled", "settled"}:
        return "explicit", event

    notes.append(f"{event_id}: unknown status '{status}' was ignored.")
    return "ignored", None


def detect_monthly_series(history: list[CashEvent]) -> list[dict[str, Any]]:
    """
    Find monthly series from settled history.

    A series is kept only when the same type, category, and description
    appear at least 3 times and the median gap is about one month.
    """
    groups: dict[tuple[str, str, str], list[CashEvent]] = defaultdict(list)
    for event in history:
        groups[(event.event_type, event.category, event.description)].append(event)

    series = []
    for key, items in groups.items():
        items = sorted(items, key=lambda event: event.cash_date)
        if len(items) < 3:
            continue
        gaps = [
            (items[index].cash_date - items[index - 1].cash_date).days
            for index in range(1, len(items))
        ]
        gaps = [gap for gap in gaps if gap > 0]
        if not gaps:
            continue
        median_gap = sorted(gaps)[len(gaps) // 2]
        if not 25 <= median_gap <= 36:
            continue

        event_type, _category, description = key
        if event_type == "income" and is_one_off_income(description):
            continue

        amounts = [event.amount for event in items]
        # Conservative: smaller future income, larger future expenses.
        amount = min(amounts[-3:]) if event_type == "income" else max(amounts[-3:])
        series.append(
            {
                "key": key,
                "last_date": items[-1].cash_date,
                "amount": amount,
                "direction": items[-1].direction,
            }
        )
    return series


def project_monthly_series(
    series: list[dict[str, Any]],
    request_date: date,
    forecast_end: date,
    explicit_events: list[CashEvent],
) -> list[CashEvent]:
    """Project each monthly series forward, skipping dates already covered."""
    explicit_dates = defaultdict(list)
    for event in explicit_events:
        key = (event.event_type, event.category, event.description)
        explicit_dates[key].append(event.cash_date)

    projected: list[CashEvent] = []
    for item in series:
        current = item["last_date"]
        for _ in range(8):
            current = current + relativedelta(months=1)
            if current > forecast_end:
                break
            if current < request_date:
                continue
            clash = any(
                abs((other - current).days) <= 2
                for other in explicit_dates[item["key"]]
            )
            if clash:
                continue
            event_type, category, description = item["key"]
            projected.append(
                CashEvent(
                    cash_date=current,
                    amount=item["amount"],
                    direction=item["direction"],
                    event_type=event_type,
                    category=category,
                    description=f"{description} (projected monthly)",
                    source="projected_monthly",
                )
            )
    return projected


def simulate_balances(
    starting_balance: float,
    request_date: date,
    forecast_end: date,
    cash_events: list[CashEvent],
    payment_on_request_date: float = 0.0,
    extra_payments: list[tuple[date, float]] | None = None,
) -> tuple[float, date, float, float, int]:
    """
    Walk every date in the window.

    Safer intra-day order: planned payments, then other debits, then credits.
    extra_payments lets Phase 3 test a full payment on a later day or a
    multi-date installment / partial-payment schedule.
    """
    payments_by_day: dict[date, float] = defaultdict(float)
    if payment_on_request_date:
        payments_by_day[request_date] += payment_on_request_date
    if extra_payments:
        for pay_date, amount in extra_payments:
            if amount:
                payments_by_day[pay_date] += amount

    sim_end = forecast_end
    if payments_by_day:
        sim_end = max(sim_end, max(payments_by_day))

    by_day: dict[date, list[CashEvent]] = defaultdict(list)
    for event in cash_events:
        if request_date <= event.cash_date <= sim_end:
            by_day[event.cash_date].append(event)

    balance = starting_balance
    lowest = starting_balance
    lowest_date = request_date
    income_total = 0.0
    applied = 0

    current = request_date
    while current <= sim_end:
        today_payment = payments_by_day.get(current, 0.0)
        # Paying today uses the current available balance, not later same-day income.
        if today_payment and current == request_date:
            balance -= today_payment
            if balance < lowest:
                lowest = balance
                lowest_date = current
        day_events = by_day.get(current, [])
        debits = [event for event in day_events if event.direction == "debit"]
        credits = [event for event in day_events if event.direction == "credit"]
        for event in debits:
            balance -= event.amount
            applied += 1
            if balance < lowest:
                lowest = balance
                lowest_date = current
        for event in credits:
            balance += event.amount
            applied += 1
            if event.event_type == "income":
                income_total += event.amount
            if balance < lowest:
                lowest = balance
                lowest_date = current
        # A later planned payment can use that day's confirmed income.
        if today_payment and current != request_date:
            balance -= today_payment
            if balance < lowest:
                lowest = balance
                lowest_date = current
        current += timedelta(days=1)

    return lowest, lowest_date, income_total, balance, applied


def find_earliest_full_payment_date(
    starting_balance: float,
    minimum_balance: float,
    requested_amount: float,
    request_date: date,
    forecast_end: date,
    cash_events: list[CashEvent],
) -> date | None:
    """First date a single full payment stays at or above the minimum."""
    current = request_date
    while current <= forecast_end:
        lowest, _, _, _, _ = simulate_balances(
            starting_balance,
            request_date,
            forecast_end,
            cash_events,
            extra_payments=[(current, requested_amount)],
        )
        if lowest >= minimum_balance:
            return current
        current += timedelta(days=1)
    return None


def protected_expense_total(
    cash_events: list[CashEvent],
    request_date: date,
    forecast_end: date,
    protected_categories: set[str],
) -> float:
    """Sum forecast debits whose category the user asked to protect."""
    total = 0.0
    for event in cash_events:
        if event.direction != "debit":
            continue
        if event.category not in protected_categories:
            continue
        if request_date <= event.cash_date <= forecast_end:
            total += event.amount
    return total


def build_forecast_events(
    events_df: pd.DataFrame,
    request_date: date,
    forecast_end: date,
    home_currency: str,
    exchange_rates_df: pd.DataFrame | None,
    notes: list[str],
) -> list[CashEvent]:
    """Turn one user's rows into history-based projections plus explicit futures."""
    history: list[CashEvent] = []
    explicit: list[CashEvent] = []

    for _, row in events_df.iterrows():
        kind, event = classify_event_row(
            row, request_date, home_currency, exchange_rates_df, notes
        )
        if event is None:
            continue
        if kind == "history":
            history.append(event)
        elif kind == "explicit":
            explicit.append(event)

    series = detect_monthly_series(history)
    projected = project_monthly_series(series, request_date, forecast_end, explicit)
    notes.append(
        f"Detected {len(series)} monthly series from history and "
        f"projected {len(projected)} future occurrences."
    )
    notes.append(
        f"Applied {len(explicit)} explicit pending/scheduled/future-settled events."
    )
    return explicit + projected


def calculate_amount_safe_to_pay(
    starting_balance: float,
    minimum_balance: float,
    requested_amount: float,
    request_date: date,
    forecast_end: date,
    cash_events: list[CashEvent],
) -> tuple[float, float, date]:
    """
    Largest payment on request_date that keeps every intra-day balance
    at or above the minimum. Future expenses are part of this check.
    """
    lowest, lowest_date, _, _, _ = simulate_balances(
        starting_balance, request_date, forecast_end, cash_events, 0.0
    )
    raw_safe = lowest - minimum_balance
    safe_amount = max(0.0, min(raw_safe, requested_amount))
    return safe_amount, lowest, lowest_date


def run_forecast(
    request: pd.Series | dict[str, Any],
    profile: pd.Series | dict[str, Any],
    events_df: pd.DataFrame,
    exchange_rates_df: pd.DataFrame | None = None,
) -> ForecastResult:
    """Build the 90-day forecast for one request / profile / event set."""
    notes: list[str] = []
    request_date = parse_date(request["request_date"])
    if request_date is None:
        raise ValueError("request_date is missing or invalid.")
    forecast_end = request_date + timedelta(days=FORECAST_DAYS)

    starting_balance = parse_number(
        profile["current_available_balance"], "current_available_balance"
    )
    minimum_balance = parse_number(
        profile["minimum_balance_to_keep"], "minimum_balance_to_keep"
    )
    requested_amount = parse_number(request["requested_amount"], "requested_amount")
    home_currency = str(profile.get("home_currency", "") or "")
    protected = split_categories(profile.get("expense_categories_to_protect"))

    if events_df is None:
        events_df = pd.DataFrame()

    cash_events = build_forecast_events(
        events_df,
        request_date,
        forecast_end,
        home_currency,
        exchange_rates_df,
        notes,
    )
    safe_amount, lowest, lowest_date = calculate_amount_safe_to_pay(
        starting_balance,
        minimum_balance,
        requested_amount,
        request_date,
        forecast_end,
        cash_events,
    )
    _, _, income_total, _, applied = simulate_balances(
        starting_balance, request_date, forecast_end, cash_events, 0.0
    )
    protected_total = protected_expense_total(
        cash_events, request_date, forecast_end, protected
    )

    return ForecastResult(
        starting_balance=starting_balance,
        total_confirmed_income=income_total,
        total_protected_expenses=protected_total,
        lowest_forecast_balance=lowest,
        date_of_lowest_balance=lowest_date,
        minimum_balance_required=minimum_balance,
        is_safe=lowest >= minimum_balance,
        amount_safe_to_pay=safe_amount,
        requested_amount=requested_amount,
        forecast_start=request_date,
        forecast_end=forecast_end,
        notes=notes,
        applied_event_count=applied,
        cash_events=cash_events,
    )


def print_forecast_summary(result: ForecastResult) -> None:
    """Print the short Phase 2 summary requested for the MVP."""
    print("=" * 80)
    print("Phase 2 — 90-day forecast summary")
    print("=" * 80)
    print(f"Forecast window            : {result.forecast_start} to {result.forecast_end}")
    print(f"Starting balance           : {result.starting_balance}")
    print(f"Total confirmed income     : {result.total_confirmed_income}")
    print(f"Total protected expenses   : {result.total_protected_expenses}")
    print(f"Lowest forecast balance    : {result.lowest_forecast_balance}")
    print(f"Date of lowest balance     : {result.date_of_lowest_balance}")
    print(f"Minimum balance required   : {result.minimum_balance_required}")
    print(f"Forecast is safe           : {result.is_safe}")
    print(f"amount_safe_to_pay         : {result.amount_safe_to_pay}")
    print(f"Cash events applied        : {result.applied_event_count}")
    if result.notes:
        print("Notes:")
        for note in result.notes:
            print(f"  - {note}")
    print()


def _make_event_row(**values: Any) -> dict[str, Any]:
    """Helper used by the built-in tests to build one event dict."""
    template = {
        "event_id": "event_test",
        "user_id": "user_test",
        "event_type": "expense",
        "description": "Test expense",
        "category": "rent",
        "direction": "debit",
        "amount": 100.0,
        "currency": "USD",
        "event_date": "2024-01-01",
        "settlement_date": "2024-01-01",
        "status": "settled",
        "linked_event_id": "",
        "flexibility": "fixed",
        "minimum_allowed_amount": "",
    }
    template.update(values)
    return template


def run_simple_tests() -> None:
    """Tiny assertions for the safe-amount and minimum-balance rules."""
    request = {"request_date": "2024-01-01", "requested_amount": 900}
    profile = {
        "current_available_balance": 1000,
        "minimum_balance_to_keep": 200,
        "home_currency": "USD",
        "expense_categories_to_protect": "rent",
    }

    empty = run_forecast(request, profile, pd.DataFrame())
    assert 0 <= empty.amount_safe_to_pay <= 900
    assert empty.amount_safe_to_pay == 800
    assert empty.is_safe is True

    rent_history = pd.DataFrame(
        [
            _make_event_row(
                event_id="rent_1",
                description="Apartment rent",
                event_date="2023-10-01",
                settlement_date="2023-10-01",
                amount=100,
            ),
            _make_event_row(
                event_id="rent_2",
                description="Apartment rent",
                event_date="2023-11-01",
                settlement_date="2023-11-01",
                amount=100,
            ),
            _make_event_row(
                event_id="rent_3",
                description="Apartment rent",
                event_date="2023-12-01",
                settlement_date="2023-12-01",
                amount=100,
            ),
        ]
    )
    with_rent = run_forecast(request, profile, rent_history)
    # Last rent 2023-12-01 -> 2024-01-01, 2024-02-01, 2024-03-01 (end is 2024-03-31).
    assert with_rent.amount_safe_to_pay == 500
    assert with_rent.lowest_forecast_balance == 700
    assert with_rent.total_protected_expenses == 300

    paid_ok, _, _, _, _ = simulate_balances(
        1000,
        date(2024, 1, 1),
        date(2024, 3, 31),
        build_forecast_events(
            rent_history, date(2024, 1, 1), date(2024, 3, 31), "USD", None, []
        ),
        payment_on_request_date=with_rent.amount_safe_to_pay,
    )
    assert paid_ok >= 200

    paid_too_much, _, _, _, _ = simulate_balances(
        1000,
        date(2024, 1, 1),
        date(2024, 3, 31),
        build_forecast_events(
            rent_history, date(2024, 1, 1), date(2024, 3, 31), "USD", None, []
        ),
        payment_on_request_date=with_rent.amount_safe_to_pay + 0.01,
    )
    assert paid_too_much < 200

    cancelled = pd.DataFrame(
        [
            _make_event_row(
                event_id="cancelled_1",
                amount=400,
                event_date="2024-01-10",
                settlement_date="2024-01-10",
                status="cancelled",
            )
        ]
    )
    cancelled_result = run_forecast(request, profile, cancelled)
    assert cancelled_result.amount_safe_to_pay == 800

    pending_credit = pd.DataFrame(
        [
            _make_event_row(
                event_id="refund_1",
                event_type="refund",
                description="Pending merchant refund",
                category="shopping",
                direction="credit",
                amount=500,
                event_date="2024-01-10",
                settlement_date="2024-01-10",
                status="pending",
            )
        ]
    )
    pending_result = run_forecast(request, profile, pending_credit)
    assert pending_result.amount_safe_to_pay == 800
    assert pending_result.total_confirmed_income == 0

    print("Phase 2 simple tests: all assertions passed.")
