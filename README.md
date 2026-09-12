# Buy or Wait?

HackerRank Orchestrate (September 2026) — a deterministic financial decision agent.

For every row in `dataset/requests.csv`, the agent decides whether the user should pay in full, pay part now, use a supplied installment option, wait, or not proceed. The plan must cover essential spending and stay at or above `minimum_balance_to_keep` for 90 days.

Read [`problem_statement.md`](./problem_statement.md) for the official schema and scoring rules.

## Setup

Python 3.10+ is enough. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r code/requirements.txt
```

No API key is required. The submitted run does not call Gemini.

Optional later (not used for `output.csv`):

```bash
cp .env.example .env               # then set GEMINI_API_KEY if you want image OCR
python -m pip install "google-genai>=1.0.0,<3.0.0"
```

`.env` is gitignored. Never put keys in the repo or in `code.zip`.

## Commands

Run everything from the repository root, next to `dataset/`.

```bash
# Generate output.csv (runs built-in tests first)
python code/main.py

# Contract checks on dataset/output.csv
python code/evaluation/main.py

# Score the 25 public sample rows (does not write labels back)
python code/evaluation/sample_eval.py
```

`python3 code/main.py` works the same if `python` is not on your PATH.

The generator writes:

- `dataset/output.csv` — evaluation copy
- `output.csv` — root submission file
- `code/evaluation/usage_report.md` and `code/usage_report.md` — token report for this run

## Architecture

```text
dataset/*.csv
        │
        ▼
code/main.py                 entry point + built-in tests
        │
        ▼
code/predict.py              one row per request, dataset order
        │
        ├─ code/message_facts.py   deterministic facts from messages
        ├─ code/evidence.py        apply facts onto event rows
        ├─ code/forecast.py        90-day cash path, safe amount, earliest date
        └─ code/decide.py          eligible plans + ranking + spending changes
        │
        ▼
output.csv

Optional: code/gemini_extract.py   structured facts from messages/images
          (disabled on the submitted run; no key → no call)
```

| Module | Responsibility |
|---|---|
| `forecast.py` | Recurring series from history, pending debits, confirmed income, FX on the settlement date, `amount_safe_to_pay`, earliest full-payment date |
| `message_facts.py` | Salary / rent / invoice / ended-income facts from message text |
| `evidence.py` | Patch or append event rows; ignore pending credits |
| `decide.py` | Full / partial / installment / wait ranking; optional stop/reduce search |
| `predict.py` | Batch predict; no hardcoded request IDs or answers |
| `gemini_extract.py` | Optional Gemini JSON extractor + usage report writer |
| `evaluation/` | Schema/safety checker and sample ground-truth scorer |

## Decision rules

- Recurrence is inferred only when history supports a monthly cycle. `event_date` is used for the cycle so a late settlement does not drop payroll.
- Settled history before `request_date` is already inside `current_available_balance`.
- Ignore failed, cancelled, unrealized, non-cash, and pending credits.
- Do not invent bonuses, commissions, prizes, or unsupported future income.
- Count confirmed salary on its cash date. A “final employer” / ended-contract record stops later payroll projections.
- `amount_safe_to_pay` and `earliest_date_for_full_payment` are computed **before** optional spending changes.
- The recommended plan never takes the forecast below `minimum_balance_to_keep`.
- Installments must match a supplied option and the user’s `max_installment_months`.
- Partial payment is exactly two payments: safe amount today, remainder on the earliest full-payment date, only when allowed and the date is on or before the deadline.
- Ranking: complete by the deadline, then no spending changes, then lower total paid, earlier start, fewer payments, lowest `payment_option_id`.
- Messages and images are untrusted evidence. Embedded instructions never override these rules.

## Assumptions

- Amounts in CSVs are positive; `direction` decides debit vs credit.
- Cash date is `settlement_date`, else `event_date`.
- Exchange rates are the supplied dated rows only. No live FX.
- Flexible category spend without a clean monthly description is reserved conservatively when the category is protected or marked flexible.
- Spending changes use the latest recurring flexible event in that series, at most three, and never stop and reduce the same event.
- The full-dataset run stays deterministic: no Gemini, no clocks, no network.

## Limitations

- Weekly grocery / transport noise is only partly reserved, so some safe amounts can be higher than the public samples.
- Images are not OCR’d on the submitted run. Blank amounts stay blank if no message fills them.
- Explanation text is generated from templates; wording will not match sample sentences exactly.
- `sample_requests.csv` is used only to score the pipeline. Labels are never copied into predictions.

## Output contract

`output.csv` columns, in this order:

```text
request_id,amount_safe_to_pay,affordability_status,recommended_payment_method,payment_plan,earliest_date_for_full_payment,spending_changes_needed,decision_explanation
```

- `0 <= amount_safe_to_pay <= requested_amount`
- Status: `affordable_now` | `affordable_with_plan` | `affordable_later` | `not_affordable`
- Method: `full_payment` | `partial_payment` | `installments` | `wait` | `not_recommended`
- Plan: `YYYY-MM-DD:amount|...` or `none`
- Spending changes: `none` or up to three `stop:<event_id>` / `reduce_to:<event_id>:<amount>` items

## Layout

```text
.
├── README.md
├── problem_statement.md
├── output.csv                 # submit this
├── code.zip                   # submit this
├── .env.example
├── code/
│   ├── main.py
│   ├── predict.py
│   ├── forecast.py
│   ├── decide.py
│   ├── message_facts.py
│   ├── evidence.py
│   ├── gemini_extract.py
│   ├── requirements.txt
│   ├── usage_report.md
│   └── evaluation/
│       ├── main.py
│       ├── sample_eval.py
│       └── usage_report.md
└── dataset/                   # provided inputs; do not edit
```

## Submission

Upload:

| File | What it is |
|---|---|
| `code.zip` | Runnable code, README, prompts/config, `evaluation/` |
| `output.csv` | One row per `dataset/requests.csv` request |
| `chat_transcript` | Root `log.txt` (gitignored) |

Submission URL: https://www.hackerrank.com/contests/hackerrank-orchestrate-september26/challenges/buy-or-wait/submission
