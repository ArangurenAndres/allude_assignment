# src/app.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple
import re
import os

import pandas as pd

from src.data import load_all
from src.analytics import (
    Filters,
    count_incidents,
    top_equipment,
    top_technicians,
    count_distinct_technicians,
    most_common_symptoms,
    count_mentions,
)

from src.llm import synthesize_with_llm, llm_available


# Toggle LLM via environment variable
USE_LLM = os.getenv("USE_LLM", "0") == "1"


def maybe_llm(user_question: str, deterministic_answer: str) -> str:
    """
    Wrap deterministic output with LLM synthesis if enabled.
    Otherwise return original answer.
    """
    if not deterministic_answer:
        return deterministic_answer

    # Prevent the LLM from rewriting pure numeric answers (avoids "200 equipment" style errors)
    if deterministic_answer.strip().isdigit():
        return deterministic_answer

    if USE_LLM and llm_available():
        return synthesize_with_llm(user_question, deterministic_answer)

    return deterministic_answer


# -----------------------------
# Conversation state
# -----------------------------

@dataclass
class ConversationState:
    filters: Filters = Filters()
    last_count_question_type: Optional[str] = None
    last_two_equipment_counts: Dict[str, int] = None

    def __post_init__(self):
        if self.last_two_equipment_counts is None:
            self.last_two_equipment_counts = {}


# -----------------------------
# Parsing helpers
# -----------------------------

_MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def _year_from_query(ql: str, default_year: int) -> int:
    ym = re.search(r"\b(20\d{2})\b", ql)
    return int(ym.group(1)) if ym else default_year


# NEW: half-year range parsing
def _half_year_range_from_query(q: str, default_year: int):
    """
    Detects:
      - "first half of 2024", "1st half 2024", "H1 2024"
      - "second half of 2024", "2nd half 2024", "H2 2024"

    Returns (start_ts_min, start_ts_max) or (None, None).
    """
    ql = q.lower()
    y = _year_from_query(ql, default_year)

    # H1 / first half
    if re.search(r"\b(first\s+half|1st\s+half|h1)\b", ql):
        start = pd.Timestamp(year=y, month=1, day=1)
        end = pd.Timestamp(year=y, month=7, day=1)  # exclusive
        return start, end

    # H2 / second half
    if re.search(r"\b(second\s+half|2nd\s+half|h2)\b", ql):
        start = pd.Timestamp(year=y, month=7, day=1)
        end = pd.Timestamp(year=y + 1, month=1, day=1)  # exclusive
        return start, end

    return None, None


def _month_range_from_query(q: str, default_year: int):
    ql = q.lower()
    m = re.search(
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t)?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
        ql,
    )
    if not m:
        return None, None

    mon = _MONTH_MAP[m.group(1)]
    y = _year_from_query(ql, default_year)

    start = pd.Timestamp(year=y, month=mon, day=1)
    end = start + pd.offsets.MonthBegin(1)
    return start, end


def _extract_equipment_id(q: str, known_equipment: set[str]) -> Optional[str]:
    ql = q.lower()
    for eq in known_equipment:
        if eq.lower() in ql:
            return eq
    return None


def _extract_product_line(q: str, known_product_lines: set[str]) -> Optional[str]:
    ql = q.lower()
    for pl in known_product_lines:
        if pl.lower() in ql:
            return pl
    return None


def _extract_symptom_code(q: str, known_symptoms: set[str]) -> Optional[str]:
    ql = q.lower()
    for s in known_symptoms:
        if s.lower() in ql:
            return s

    phrase_map = {
        "pressure fault": "PRESSURE_FAULT",
        "spindle timeout": "SPINDLE_TIMEOUT",
        "bearing wear": "BEARING_WEAR",
        "hydraulic leak": "HYDRAULIC_LEAK",
        "alignment drift": "ALIGNMENT_DRIFT",
        "sensor failure": "SENSOR_FAILURE",
    }

    for phrase, code in phrase_map.items():
        if phrase in ql and code in known_symptoms:
            return code

    return None


def extract_mention_keyword(q: str) -> str:
    m = re.search(r"['\"]([^'\"]+)['\"]", q)
    if m:
        return m.group(1).strip()

    ql = q.lower()
    if "leak" in ql:
        return "leak"
    if "bearing" in ql:
        return "bearing"

    m2 = re.search(r"\b(?:mention|mentions|contain|contains|containing)\b\s+(?:a|an|the)?\s*([a-z0-9_-]+)", ql)
    if m2:
        return m2.group(1).strip()

    return ""


def extract_filters(
    q: str,
    *,
    known_equipment: set[str],
    known_product_lines: set[str],
    known_symptoms: set[str],
    default_year: int,
) -> Filters:
    eq = _extract_equipment_id(q, known_equipment)
    pl = _extract_product_line(q, known_product_lines)
    sym = _extract_symptom_code(q, known_symptoms)

    # NEW: try half-year first; if not found, fall back to month parsing
    start_min, start_max = _half_year_range_from_query(q, default_year)
    if start_min is None and start_max is None:
        start_min, start_max = _month_range_from_query(q, default_year)

    return Filters(
        equipment_id=eq,
        product_line=pl,
        symptom_code=sym,
        start_ts_min=start_min,
        start_ts_max=start_max,
    )


# -----------------------------
# Intent detection
# -----------------------------

def intent(q: str) -> str:
    # Undersatnt the intent of the question by looking for keywords and pattens, simple rule based not ML based approach
    ql = q.lower()

    if ("mention" in ql or "mentions" in ql or
        "contain" in ql or "contains" in ql or "containing" in ql):
        return "count_mentions"

    if "which one had more" in ql:
        return "compare_equipment"

    if "which technician handled the most" in ql:
        return "top_technician"

    if "how many different technicians" in ql:
        return "count_distinct_technicians"

    # Distinct equipment intent
    if "distinct equipment" in ql or "different equipment" in ql:
        return "count_distinct_equipment"

    if "most common symptom" in ql:
        return "most_common_symptom"

    if "which equipment has the most incidents" in ql:
        return "top_equipment"

    if "how many" in ql or "number of" in ql:
        return "count_incidents"

    if ql.strip().startswith("what about"):
        return "what_about"

    return "unknown"


# -----------------------------
# Main loop
# -----------------------------

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    csv_path = ROOT / "data" / "maintenance_records.csv"

    print("Loading data...")
    _, work_orders = load_all(csv_path)

    known_equipment = set(work_orders["equipment_id"].astype(str).unique())
    known_product_lines = set(work_orders["product_line"].astype(str).unique())
    known_symptoms = set(work_orders["symptom_code"].astype(str).unique())
    default_year = int(pd.to_datetime(work_orders["start_ts"]).dt.year.mode().iloc[0])

    state = ConversationState()

    print("\nMaintenance Assistant (type 'exit' to quit)\n")

    while True:
        user = input("> ").strip()
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break

        f = extract_filters(
            user,
            known_equipment=known_equipment,
            known_product_lines=known_product_lines,
            known_symptoms=known_symptoms,
            default_year=default_year,
        )

        it = intent(user)

        if it == "count_mentions":
            kw = extract_mention_keyword(user)
            n = count_mentions(work_orders, keyword=kw, f=f)
            print(maybe_llm(user, str(n)))
            continue

        if it == "count_incidents":
            n = count_incidents(work_orders, f=f)
            print(maybe_llm(user, str(n)))
            continue

        if it == "count_distinct_equipment":
            # Minimal implementation without touching analytics:
            # Count distinct equipment_id in the FILTERED scope (using the same time/product/symptom/equipment filters indirectly
            # via count_incidents is not possible), so we compute with the available filter fields here.
            df = work_orders.copy()

            if f.equipment_id:
                df = df[df["equipment_id"].astype(str) == str(f.equipment_id)]
            if f.product_line:
                df = df[df["product_line"].astype(str) == str(f.product_line)]
            if f.symptom_code:
                df = df[df["symptom_code"].astype(str) == str(f.symptom_code)]
            if f.start_ts_min is not None:
                df = df[pd.to_datetime(df["start_ts"]) >= f.start_ts_min]
            if f.start_ts_max is not None:
                df = df[pd.to_datetime(df["start_ts"]) < f.start_ts_max]

            n = int(df["equipment_id"].astype(str).nunique())
            print(maybe_llm(user, str(n)))
            continue

        if it == "count_distinct_technicians":
            n = count_distinct_technicians(work_orders, f=f)
            print(maybe_llm(user, str(n)))
            continue

        if it == "most_common_symptom":
            tied = most_common_symptoms(work_orders, f=f)
            if not tied:
                print("")
            else:
                names = [s for s, _ in tied]
                n = tied[0][1]
                if len(names) == 1:
                    ans = f"{names[0]} with {n} work orders"
                else:
                    ans = f"{' and '.join(names)}, each with {n} work orders"
                print(maybe_llm(user, ans))
            continue

        if it == "top_equipment":
            df = top_equipment(work_orders, n=1, f=Filters())
            if df.empty:
                print("")
            else:
                eq = str(df.iloc[0]["equipment_id"])
                n = int(df.iloc[0]["work_order_count"])
                ans = f"{eq} with {n} work orders"
                print(maybe_llm(user, ans))
            continue

        if it == "top_technician":
            df = top_technicians(work_orders, n=1, f=Filters())
            if df.empty:
                print("")
            else:
                tech = str(df.iloc[0]["technician"])
                n = int(df.iloc[0]["work_order_count"])
                ans = f"{tech} with {n} work orders"
                print(maybe_llm(user, ans))
            continue

        print("")
