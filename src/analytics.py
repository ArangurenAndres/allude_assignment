# src/analytics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

import pandas as pd
import re



# Filters used across all analytics functions to narrow down the dataset before applying specific computations.


@dataclass(frozen=True)
class Filters:
    """
    Structured filtering object used across all analytics functions.

    Acts as the "retrieval constraint" in the RAG pipeline.
    We filter based on structured attributes.

    Attributes:
        equipment_id     -> filter by specific equipment
        product_line     -> filter by product line
        symptom_code     -> filter by symptom
        start_ts_min     -> inclusive lower bound on start timestamp
        start_ts_max     -> exclusive upper bound on start timestamp
    """
    equipment_id: Optional[str] = None
    product_line: Optional[str] = None
    symptom_code: Optional[str] = None
    start_ts_min: Optional[pd.Timestamp] = None
    start_ts_max: Optional[pd.Timestamp] = None  # end-exclusive

# Function used to apply the above filters to the work orders dataframe 

def filter_work_orders(work_orders: pd.DataFrame, f: Filters) -> pd.DataFrame:
    """
    Apply structured filters to the work_orders DataFrame.

    Structured retrieval step of the RAG system.
    It narrows down the dataset before analytics are applied.

    Returns:
        Filtered  pandas DataFrame.
    """
    df = work_orders

    if f.equipment_id:
        df = df[df["equipment_id"] == f.equipment_id]

    if f.product_line:
        df = df[df["product_line"] == f.product_line]

    if f.symptom_code:
        df = df[df["symptom_code"] == f.symptom_code]

    if f.start_ts_min is not None:
        df = df[df["start_ts"] >= f.start_ts_min]

    if f.start_ts_max is not None:
        df = df[df["start_ts"] < f.start_ts_max]

    return df



# Analytic functions 


def count_incidents(work_orders: pd.DataFrame, f: Optional[Filters] = None) -> int:
    """
    Count number of work orders within the filtered attribute.
    """
    df = filter_work_orders(work_orders, f or Filters())
    return int(len(df))


def count_distinct_equipment(work_orders: pd.DataFrame, f: Optional[Filters] = None) -> int:
    """
    Count number of distinct equipment IDs in scope.
    """
    df = filter_work_orders(work_orders, f or Filters())
    if df.empty:
        return 0
    return int(df["equipment_id"].astype(str).nunique())


def count_distinct_product_lines(work_orders: pd.DataFrame, f: Optional[Filters] = None) -> int:
    """
    Count number of distinct product lines in scope.
    """
    df = filter_work_orders(work_orders, f or Filters())
    if df.empty:
        return 0
    return int(df["product_line"].astype(str).nunique())


def top_equipment(work_orders: pd.DataFrame, n: int = 5, f: Optional[Filters] = None) -> pd.DataFrame:
    """
    Return top N equipment IDs ranked by number of work orders.
    """
    df = filter_work_orders(work_orders, f or Filters())
    if df.empty:
        return pd.DataFrame(columns=["equipment_id", "work_order_count"])

    out = (
        df.groupby("equipment_id")
        .size()
        .rename("work_order_count")
        .reset_index()
        .sort_values(["work_order_count", "equipment_id"], ascending=[False, True])
        .head(n)
        .reset_index(drop=True)
    )
    return out


def top_symptoms(work_orders: pd.DataFrame, n: int = 5, f: Optional[Filters] = None) -> pd.DataFrame:
    """
    Return top N symptom codes ranked by frequency.
    """
    df = filter_work_orders(work_orders, f or Filters())
    if df.empty:
        return pd.DataFrame(columns=["symptom_code", "work_order_count"])

    out = (
        df.groupby("symptom_code")
        .size()
        .rename("work_order_count")
        .reset_index()
        .sort_values(["work_order_count", "symptom_code"], ascending=[False, True])
        .head(n)
        .reset_index(drop=True)
    )
    return out


def incidents_over_time(work_orders: pd.DataFrame, freq: str = "ME", f: Optional[Filters] = None) -> pd.DataFrame:
    """
    Aggregate incidents over time.

    freq examples:
        "ME" -> month end
        "D"  -> daily
        "W"  -> weekly
    """
    df = filter_work_orders(work_orders, f or Filters())
    if df.empty:
        return pd.DataFrame(columns=["start_ts", "work_order_count"])

    out = (
        df.set_index("start_ts")
        .sort_index()
        .resample(freq)
        .size()
        .rename("work_order_count")
        .reset_index()
    )
    return out



# Mention functions (keyword search in text fields)


def count_mentions(work_orders: pd.DataFrame, keyword: str, f: Optional[Filters] = None) -> int:
    """
    Count work orders where keyword appears in description OR comments.
    """
    kw = str(keyword).strip().lower()
    if not kw:
        return 0

    df = filter_work_orders(work_orders, f or Filters())
    if df.empty:
        return 0

    hay = (
        df["description"].astype(str).fillna("") + "\n" +
        df["comments"].astype(str).fillna("")
    ).str.lower()

    return int(hay.str.contains(re.escape(kw), regex=True).sum())



# Technician analytics (requires "technicians" list column in work_orders) used for reasoning about technicians could be in multi-turn conversations


def top_technicians(work_orders: pd.DataFrame, n: int = 5, f: Optional[Filters] = None) -> pd.DataFrame:
    """
    Top technicians by number of DISTINCT work orders they appear in.
    Requires work_orders['technicians'] as a list of strings.
    """
    df = filter_work_orders(work_orders, f or Filters()) # if filters are provided narrow the dataset
    if df.empty:
        return pd.DataFrame(columns=["technician", "work_order_count"])# if no rows remain after filter return empty result

    if "technicians" not in df.columns:
        raise ValueError("work_orders missing expected 'technicians' list column.") # column named technicians  containing list of technician names
    # Step 3: Normalize technician represnetain 
    # Each work order may contain multiple technicians.
    # Example row:
    #   work_order_id | technicians
    #   101           | ["Alice", "Bob"]
    #
    # explode() transforms it into:
    #   work_order_id | technicians
    #   101           | "Alice"
    #   101           | "Bob"
    exploded = df[["work_order_id", "technicians"]].explode("technicians")
    #  remove rows where tehcinican value is missing
    exploded = exploded.dropna(subset=["technicians"])
    # Ensure technician names are clean strings
    exploded["technicians"] = exploded["technicians"].astype(str).str.strip()
    # Remove empty string values

    exploded = exploded[exploded["technicians"] != ""]
    # Step 4: Aggregate distinct orders per technician
    out = (
        exploded.groupby("technicians")["work_order_id"]
        .nunique() # distinct count per technician
        .rename("work_order_count") # rename result column
        .reset_index()
        .rename(columns={"technicians": "technician"})
        .sort_values(["work_order_count", "technician"], ascending=[False, True])
        .head(n)
        .reset_index(drop=True)
    )
    return out

#
def count_distinct_technicians(work_orders: pd.DataFrame, f: Optional[Filters] = None) -> int:
    """
    Count DISTINCT technicians that appear in the filtered scope.
    """
    df = filter_work_orders(work_orders, f or Filters())
    if df.empty:
        return 0

    if "technicians" not in df.columns:
        raise ValueError("work_orders missing expected 'technicians' list column.")

    exploded = df[["technicians"]].explode("technicians")
    vals = exploded["technicians"].dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    return int(vals.nunique())



# Multi-turn helper:  most common symptom

    """
    MULTI-TURN LOGIC EXPLANATION

    This function returns ALL symptom_code(s) that are tied
    for the highest frequency within the current filtered scope.

    Why this matters for multi-turn conversations:

    1) In a first turn, the user might ask:
           "What is the most common symptom?"

       If two or more symptoms have the same maximum count,
       returning only one would discard important information.

    2) By returning ALL tied symptoms, we preserve the full
       candidate set for follow-up reasoning.

       Example:
           SPINDLE_TIMEOUT -> 15
           BEARING_WEAR    -> 15

       Instead of choosing one arbitrarily, we return:
           [("BEARING_WEAR", 15), ("SPINDLE_TIMEOUT", 15)]

    3) In a second turn, the user might say:
           "Which one had more incidents in July?"
       or
           "What about BEARING_WEAR?"

       Because we preserved all tied symptoms in the first turn,
       the system can correctly resolve the follow-up query.

    4) Deterministic ordering:
       We sort alphabetically to ensure consistent output across runs.
       This guarantees reproducibility and stable conversational behavior.

    In short:
        This function prevents premature tie-breaking,
        which is critical for correct multi-turn reasoning.
    """


def most_common_symptoms(work_orders: pd.DataFrame, f: Optional[Filters] = None) -> List[Tuple[str, int]]:
    """
    Return ALL symptom_code(s) tied for highest frequency withing the filtered scope

    This supports multi-turn reasoning.

    In a conversation, the user might first ask:
    "What is the most common symptom?"

    If multiple symptoms are tied for highest frequency,
    we must return all of them to preserve full information.

    In the next turn, the user might say:
        "What about BEARING_WEAR?"
    or
    "Which one had more incidents?"
    By returning all tied symptoms (instead of just one),
    the system preserves conversational continuity and allows
    follow-up comparison or refinement questions.

    """
    df = filter_work_orders(work_orders, f or Filters())
    if df.empty:
        return []

    vc = df["symptom_code"].value_counts()
    if vc.empty:
        return []

    max_count = int(vc.iloc[0])
    tied = [(str(sym), int(cnt)) for sym, cnt in vc.items() if int(cnt) == max_count]
    tied.sort(key=lambda x: x[0])  # deterministic output
    return tied
