"""
Phase 3 — choose a payment plan from the Phase 2 forecast.

No LLM. Plans are scored with the ranking rules in problem_statement.md.
Spending changes are not applied yet; those stay none in this phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd

from forecast import (
    CashEvent,
    ForecastResult,
    build_forecast_events,
    find_earliest_full_payment_date,
    optional_number,
    parse_date,
    round_money,
    simulate_balances,
    split_categories,
)


@dataclass
class CandidatePlan:
    method: str
    payments: list[tuple[date, float]]
    total_paid: float
    start_date: date
    completes_by_deadline: bool
    payment_option_id: str = ""
    spending_changes: str = "none"

    def rank_tuple(self) -> tuple:
        """Lower is better. Matches the documented plan-ranking order."""
        option_key = 10**9
        digits = "".join(character for character in self.payment_option_id if character.isdigit())
        if digits:
            option_key = int(digits)
        return (
            0 if self.completes_by_deadline else 1,
            0 if self.spending_changes == "none" else 1,
            self.total_paid,
            self.start_date.toordinal(),
            len(self.payments),
            option_key,
        )


@dataclass
class Decision:
    request_id: str
    amount_safe_to_pay: float
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: str
    spending_changes_needed: str
    decision_explanation: str
    notes: list[str] = field(default_factory=list)


def is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def optional_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def format_amount(amount: float) -> str:
    """Avoid float noise: integers stay integers, else two decimals."""
    rounded = round_money(amount)
    if abs(rounded - round(rounded)) < 1e-9:
        return str(int(round(rounded)))
    return f"{rounded:.2f}"


def format_payment_plan(payments: list[tuple[date, float]]) -> str:
    if not payments:
        return "none"
    return "|".join(
        f"{pay_date.isoformat()}:{format_amount(amount)}" for pay_date, amount in payments
    )


def user_methods(profile: pd.Series | dict[str, Any]) -> set[str]:
    return split_categories(profile.get("payment_methods_user_will_consider"))


def months_spanned(first: date, last: date) -> int:
    return (last.year - first.year) * 12 + (last.month - first.month) + 1


def build_option_schedule(row: pd.Series) -> list[tuple[date, float]] | None:
    """Turn one payment-option row into dated payments. Do not invent amounts."""
    first = parse_date(row.get("first_payment_date"))
    count = optional_int(row.get("number_of_payments"))
    amount = optional_number(row.get("payment_amount"))
    if first is None or count is None or count < 1 or amount is None:
        return None

    method = str(row.get("payment_method", "") or "").strip()
    if method == "full_payment" or count == 1:
        return [(first, amount)]

    frequency = optional_int(row.get("payment_frequency_days"))
    if frequency is None or frequency <= 0:
        return None

    return [(first + timedelta(days=frequency * index), amount) for index in range(count)]


def plan_is_safe(
    forecast: ForecastResult,
    payments: list[tuple[date, float]],
    cash_events: list,
) -> bool:
    if not payments:
        return False
    last_payment = max(pay_date for pay_date, _amount in payments)
    lowest, _, _, _, _ = simulate_balances(
        forecast.starting_balance,
        forecast.forecast_start,
        max(forecast.forecast_end, last_payment),
        cash_events,
        extra_payments=payments,
    )
    return lowest >= forecast.minimum_balance_required


def cash_events_for_end(
    forecast: ForecastResult,
    events_df: pd.DataFrame,
    profile: pd.Series | dict[str, Any],
    exchange_rates_df: pd.DataFrame | None,
    end_date: date,
) -> list:
    """Reuse the 90-day events, or rebuild if a plan lasts longer."""
    if end_date <= forecast.forecast_end:
        return forecast.cash_events
    notes: list[str] = []
    return build_forecast_events(
        events_df,
        forecast.forecast_start,
        end_date,
        str(profile.get("home_currency", "") or ""),
        exchange_rates_df,
        notes,
    )


def collect_candidate_plans(
    request: pd.Series | dict[str, Any],
    profile: pd.Series | dict[str, Any],
    forecast: ForecastResult,
    payment_options_df: pd.DataFrame,
    events_df: pd.DataFrame,
    exchange_rates_df: pd.DataFrame | None,
    earliest: date | None,
    notes: list[str],
) -> list[CandidatePlan]:
    request_date = forecast.forecast_start
    requested = forecast.requested_amount
    deadline = parse_date(request.get("desired_completion_date"))
    methods = user_methods(profile)
    max_months = optional_int(profile.get("max_installment_months"))
    allows_partial = is_truthy(request.get("allows_partial_payment"))
    request_id = str(request.get("request_id", "") or "")

    plans: list[CandidatePlan] = []

    matching_options = payment_options_df[
        payment_options_df["request_id"].astype(str) == request_id
    ] if payment_options_df is not None and not payment_options_df.empty else pd.DataFrame()

    for _, row in matching_options.iterrows():
        method = str(row.get("payment_method", "") or "").strip()
        option_id = str(row.get("payment_option_id", "") or "")
        schedule = build_option_schedule(row)
        if schedule is None:
            notes.append(f"{option_id}: skipped because the schedule fields are incomplete.")
            continue
        if method not in methods:
            notes.append(f"{option_id}: skipped because the user will not consider {method}.")
            continue
        if method == "installments":
            if max_months is None:
                notes.append(f"{option_id}: skipped because max_installment_months is blank.")
                continue
            spanned = months_spanned(schedule[0][0], schedule[-1][0])
            if spanned > max_months:
                notes.append(
                    f"{option_id}: skipped because {spanned} months exceeds "
                    f"max_installment_months={max_months}."
                )
                continue
        last_payment = schedule[-1][0]
        cash_events = cash_events_for_end(
            forecast, events_df, profile, exchange_rates_df, last_payment
        )
        if not plan_is_safe(forecast, schedule, cash_events):
            notes.append(f"{option_id}: unsafe against the minimum-balance rule.")
            continue
        total_paid = optional_number(row.get("total_payable_amount"))
        if total_paid is None:
            total_paid = sum(amount for _date, amount in schedule)
        plans.append(
            CandidatePlan(
                method=method,
                payments=schedule,
                total_paid=total_paid,
                start_date=schedule[0][0],
                completes_by_deadline=deadline is None or last_payment <= deadline,
                payment_option_id=option_id,
            )
        )

    if (
        "partial_payment" in methods
        and allows_partial
        and 0 < forecast.amount_safe_to_pay < requested
        and earliest is not None
        and (deadline is None or earliest <= deadline)
    ):
        remainder = requested - forecast.amount_safe_to_pay
        schedule = [
            (request_date, forecast.amount_safe_to_pay),
            (earliest, remainder),
        ]
        cash_events = cash_events_for_end(
            forecast, events_df, profile, exchange_rates_df, earliest
        )
        if plan_is_safe(forecast, schedule, cash_events):
            plans.append(
                CandidatePlan(
                    method="partial_payment",
                    payments=schedule,
                    total_paid=requested,
                    start_date=request_date,
                    completes_by_deadline=True,
                )
            )
        else:
            notes.append("partial_payment: two-payment schedule was not safe.")
    elif "partial_payment" in methods and allows_partial:
        notes.append(
            "partial_payment: not eligible (need 0 < amount_safe_to_pay < requested "
            "and earliest_date on or before the deadline)."
        )

    if earliest is not None and "full_payment" in methods:
        already_has_wait_date = any(
            plan.method == "full_payment" and plan.payments[0][0] == earliest
            for plan in plans
        )
        if earliest > request_date and not already_has_wait_date:
            schedule = [(earliest, requested)]
            cash_events = cash_events_for_end(
                forecast, events_df, profile, exchange_rates_df, earliest
            )
            if plan_is_safe(forecast, schedule, cash_events):
                plans.append(
                    CandidatePlan(
                        method="wait",
                        payments=schedule,
                        total_paid=requested,
                        start_date=earliest,
                        completes_by_deadline=deadline is None or earliest <= deadline,
                    )
                )

    return plans


def status_for_method(method: str, start_date: date, request_date: date) -> str:
    if method == "full_payment" and start_date == request_date:
        return "affordable_now"
    if method in {"partial_payment", "installments"}:
        return "affordable_with_plan"
    if method == "full_payment":
        return "affordable_later"
    if method == "wait":
        return "affordable_later"
    return "not_affordable"


def build_explanation(
    decision_method: str,
    payments: list[tuple[date, float]],
    forecast: ForecastResult,
    currency: str,
) -> str:
    money = format_amount
    currency = currency or ""
    minimum = money(forecast.minimum_balance_required)
    if decision_method == "full_payment" and payments and payments[0][0] == forecast.forecast_start:
        return (
            f"Pay {currency} {money(payments[0][1])} today. "
            f"The 90-day forecast stays at or above {currency} {minimum}."
        ).strip()
    if decision_method == "wait" and payments:
        return (
            f"Wait until {payments[0][0].isoformat()}, then pay "
            f"{currency} {money(payments[0][1])} in full. "
            f"Paying earlier would put the {currency} {minimum} minimum at risk."
        ).strip()
    if decision_method == "partial_payment" and len(payments) == 2:
        return (
            f"Pay {currency} {money(payments[0][1])} today and the remaining "
            f"{currency} {money(payments[1][1])} on {payments[1][0].isoformat()}. "
            f"This completes the request and keeps the {currency} {minimum} minimum protected."
        ).strip()
    if decision_method == "installments" and payments:
        return (
            f"Use {len(payments)} installments of {currency} {money(payments[0][1])}, "
            f"starting {payments[0][0].isoformat()}. "
            f"This leaves at least {currency} {minimum} available."
        ).strip()
    return (
        f"Do not make this payment within the forecast window. "
        f"None of the eligible options keeps the {currency} {minimum} minimum protected."
    ).strip()


def recommend_plan(
    request: pd.Series | dict[str, Any],
    profile: pd.Series | dict[str, Any],
    forecast: ForecastResult,
    payment_options_df: pd.DataFrame | None = None,
    events_df: pd.DataFrame | None = None,
    exchange_rates_df: pd.DataFrame | None = None,
) -> Decision:
    """Pick the best safe eligible plan for one request."""
    notes: list[str] = []
    if payment_options_df is None:
        payment_options_df = pd.DataFrame()
    if events_df is None:
        events_df = pd.DataFrame()

    request_id = str(request.get("request_id", "") or "")
    currency = str(profile.get("home_currency", "") or "")
    earliest = find_earliest_full_payment_date(
        forecast.starting_balance,
        forecast.minimum_balance_required,
        forecast.requested_amount,
        forecast.forecast_start,
        forecast.forecast_end,
        forecast.cash_events,
    )
    earliest_text = earliest.isoformat() if earliest is not None else ""

    plans = collect_candidate_plans(
        request,
        profile,
        forecast,
        payment_options_df,
        events_df,
        exchange_rates_df,
        earliest,
        notes,
    )

    if not plans:
        return Decision(
            request_id=request_id,
            amount_safe_to_pay=round_money(forecast.amount_safe_to_pay),
            affordability_status="not_affordable",
            recommended_payment_method="not_recommended",
            payment_plan="none",
            earliest_date_for_full_payment=earliest_text,
            spending_changes_needed="none",
            decision_explanation=build_explanation("not_recommended", [], forecast, currency),
            notes=notes,
        )

    best = min(plans, key=lambda plan: plan.rank_tuple())
    status = status_for_method(best.method, best.start_date, forecast.forecast_start)
    return Decision(
        request_id=request_id,
        amount_safe_to_pay=round_money(forecast.amount_safe_to_pay),
        affordability_status=status,
        recommended_payment_method=best.method,
        payment_plan=format_payment_plan(best.payments),
        earliest_date_for_full_payment=earliest_text,
        spending_changes_needed="none",
        decision_explanation=build_explanation(best.method, best.payments, forecast, currency),
        notes=notes,
    )


def print_decision(decision: Decision) -> None:
    print("=" * 80)
    print("Phase 3 — recommended plan")
    print("=" * 80)
    print(f"request_id                     : {decision.request_id}")
    print(f"amount_safe_to_pay             : {decision.amount_safe_to_pay}")
    print(f"affordability_status           : {decision.affordability_status}")
    print(f"recommended_payment_method     : {decision.recommended_payment_method}")
    print(f"payment_plan                   : {decision.payment_plan}")
    print(f"earliest_date_for_full_payment : {decision.earliest_date_for_full_payment}")
    print(f"spending_changes_needed        : {decision.spending_changes_needed}")
    print(f"decision_explanation           : {decision.decision_explanation}")
    if decision.notes:
        print("Notes:")
        for note in decision.notes:
            print(f"  - {note}")
    print()


def _empty_forecast(safe: float, requested: float, start: date, minimum: float, balance: float) -> ForecastResult:
    from datetime import timedelta

    return ForecastResult(
        starting_balance=balance,
        total_confirmed_income=0.0,
        total_protected_expenses=0.0,
        lowest_forecast_balance=balance,
        date_of_lowest_balance=start,
        minimum_balance_required=minimum,
        is_safe=True,
        amount_safe_to_pay=safe,
        requested_amount=requested,
        forecast_start=start,
        forecast_end=start + timedelta(days=90),
        cash_events=[],
    )


def run_simple_decision_tests() -> None:
    """Assertions for eligibility, wait vs full, and installment month limits."""
    start = date(2024, 1, 1)
    request = {
        "request_id": "request_test",
        "request_date": "2024-01-01",
        "requested_amount": 400,
        "desired_completion_date": "2024-03-01",
        "allows_partial_payment": True,
    }
    profile = {
        "home_currency": "USD",
        "payment_methods_user_will_consider": "full_payment|partial_payment|installments",
        "max_installment_months": 3,
        "current_available_balance": 1000,
        "minimum_balance_to_keep": 200,
    }

    # Full amount is safe today.
    full_today = recommend_plan(
        request,
        profile,
        _empty_forecast(400, 400, start, 200, 1000),
        pd.DataFrame(
            [
                {
                    "payment_option_id": "payment_option_01",
                    "request_id": "request_test",
                    "payment_method": "full_payment",
                    "payment_amount": 400,
                    "number_of_payments": 1,
                    "first_payment_date": "2024-01-01",
                    "payment_frequency_days": "",
                    "financing_fee": 0,
                    "total_payable_amount": 400,
                }
            ]
        ),
    )
    assert full_today.affordability_status == "affordable_now"
    assert full_today.recommended_payment_method == "full_payment"
    assert full_today.payment_plan == "2024-01-01:400"
    assert full_today.earliest_date_for_full_payment == "2024-01-01"

    # Safe today, but the user will not consider full_payment.
    installment_only = recommend_plan(
        {**request, "allows_partial_payment": False},
        {**profile, "payment_methods_user_will_consider": "installments"},
        _empty_forecast(400, 400, start, 200, 1000),
        pd.DataFrame(
            [
                {
                    "payment_option_id": "payment_option_02",
                    "request_id": "request_test",
                    "payment_method": "installments",
                    "payment_amount": 200,
                    "number_of_payments": 2,
                    "first_payment_date": "2024-01-01",
                    "payment_frequency_days": 30,
                    "financing_fee": 0,
                    "total_payable_amount": 400,
                },
                {
                    "payment_option_id": "payment_option_99",
                    "request_id": "request_test",
                    "payment_method": "installments",
                    "payment_amount": 50,
                    "number_of_payments": 15,
                    "first_payment_date": "2024-01-01",
                    "payment_frequency_days": 30,
                    "financing_fee": 100,
                    "total_payable_amount": 750,
                },
            ]
        ),
    )
    assert installment_only.recommended_payment_method == "installments"
    assert installment_only.affordability_status == "affordable_with_plan"
    assert installment_only.payment_plan == "2024-01-01:200|2024-01-31:200"
    assert installment_only.earliest_date_for_full_payment == "2024-01-01"

    # Today is not fully safe, but a later confirmed salary makes full payment safe.
    wait_forecast = _empty_forecast(50, 400, start, 200, 250)
    wait_forecast.cash_events = [
        CashEvent(
            cash_date=date(2024, 2, 15),
            amount=400,
            direction="credit",
            event_type="income",
            category="salary",
            description="Payroll credit",
            source="scheduled",
        )
    ]
    wait_decision = recommend_plan(
        {**request, "allows_partial_payment": False},
        {**profile, "payment_methods_user_will_consider": "full_payment"},
        wait_forecast,
        pd.DataFrame(),
    )
    assert wait_decision.recommended_payment_method == "wait"
    assert wait_decision.affordability_status == "affordable_later"
    assert wait_decision.amount_safe_to_pay == 50
    assert wait_decision.earliest_date_for_full_payment == "2024-02-15"
    assert wait_decision.payment_plan == "2024-02-15:400"

    print("Phase 3 simple tests: all assertions passed.")
