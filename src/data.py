from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd



# Schema definition (expected columns in the raw CSV)


REQUIRED_COLUMNS = [
    "work_order_id",  # identifier of a work order (incident); repeated across multiple event rows
    "equipment_id",
    "product_line",
    "start_date",
    "start_time",
    "end_date",
    "end_time",
    "description",
    "technician",
    "comment",
    "symptom_code",
]


# ============================================================
# Loading & Cleaning data (raw events)
# ============================================================

def load_events(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the raw maintenance log (one row per technician update / event).

    Returns a DataFrame with:
      - parsed datetime columns: start_ts, end_ts
      - normalized text columns (stripped)
      - validated required columns
    """
    csv_path = Path(csv_path)          # convert path to Path object (portable)
    df = pd.read_csv(csv_path)         # load raw rows from CSV

    _validate_columns(df)              # ensure CSV contains all required columns

    # Basic normalization (transparent and stable):
    # make sure key string fields are stripped and not NaN.
    for col in ["equipment_id", "product_line", "description", "technician", "comment", "symptom_code"]:
        df[col] = df[col].astype(str).fillna("").str.strip()

    # Ensure work_order_id is integer-like and fails fast if not parseable.
    df["work_order_id"] = pd.to_numeric(df["work_order_id"], errors="raise").astype("int64")

    # Parse timestamps from start_date/start_time and end_date/end_time
    df["start_ts"] = _combine_date_time(df["start_date"], df["start_time"], colname="start_ts")
    df["end_ts"] = _combine_date_time(df["end_date"], df["end_time"], colname="end_ts")

    return df


def _combine_date_time(date_series: pd.Series, time_series: pd.Series, colname: str) -> pd.Series:
    """
    Combine separate date + time columns into a single datetime Series.

    Uses strict parsing format:
        %Y-%m-%d %H:%M:%S

    Raises:
        ValueError if any rows fail to parse.
    """
    dt_str = date_series.astype(str).str.strip() + " " + time_series.astype(str).str.strip()
    ts = pd.to_datetime(dt_str, errors="coerce", format="%Y-%m-%d %H:%M:%S")

    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise ValueError(f"Failed parsing {colname} for {bad} rows. Check date/time formats.")

    return ts


def _validate_columns(df: pd.DataFrame) -> None:
    """
    Validate that the input DataFrame contains all required columns.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")


# ============================================================
# Work-order view (one row per incident / work_order_id)
# ============================================================

def build_work_orders(events: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the raw event log to ONE row per work_order_id.

    This is the incident-level view used by the assistant:
      - counts of incidents (distinct work_order_id)
      - top equipment, top symptoms, trends, etc.

    Strategy:
      - stable fields from the first row (deterministic after sorting)
      - start/end timestamps: min/max across updates
      - combine comments into a single blob for text retrieval
      - technicians list (unique + sorted)
    """
    _validate_columns(events)

    # Deterministic sort so "first" aggregations are stable across runs
    ev = events.sort_values(["work_order_id", "start_ts", "end_ts"], ascending=True).copy()

    # Group by work order id and compute incident-level fields
    agg = ev.groupby("work_order_id", as_index=False).agg(
        equipment_id=("equipment_id", "first"),
        product_line=("product_line", "first"),
        symptom_code=("symptom_code", "first"),
        description=("description", "first"),  # assumed stable enough to take the first
        start_ts=("start_ts", "min"),          # earliest start across updates
        end_ts=("end_ts", "max"),              # latest end across updates
        technicians=("technician", lambda s: sorted(set(x for x in s if str(x).strip()))),
        comments=("comment", _concat_comments),
        n_updates=("comment", "size"),
    )

    # Convenience fields for date-based grouping/filtering
    agg["start_date"] = agg["start_ts"].dt.date
    agg["end_date"] = agg["end_ts"].dt.date

    return agg


def _concat_comments(s: pd.Series) -> str:
    """
    Combine comment updates into a single text string.

    Rules:
      - drop empty comments
      - avoid exact consecutive duplicates
      - join with newlines to preserve readability
    """
    cleaned = []
    last = None

    for x in s.astype(str):
        t = x.strip()
        if not t:
            continue

        # Avoid exact consecutive duplicates
        if t == last:
            continue

        cleaned.append(t)
        last = t

    return "\n".join(cleaned)


## Load all function allows to laod events and build work orders in one fucnction call 

def load_all(csv_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load everything in one call.

    Returns:
        (events_df, work_orders_df)
    """
    events = load_events(csv_path)
    work_orders = build_work_orders(events)
    return events, work_orders


# Cached loader utility (object-oriented convenience)

@dataclass
class MaintenanceData:
    """
 loads and caches:
      - events (raw event rows)
      - work_orders (aggregated incident rows)

    Useful to avoid re-reading the CSV multiple times.
    """
    csv_path: Path
    _events: Optional[pd.DataFrame] = None
    _work_orders: Optional[pd.DataFrame] = None

    def load(self) -> "MaintenanceData":
        """Load events + build work_orders (cached)."""
        self._events = load_events(self.csv_path)
        self._work_orders = build_work_orders(self._events)
        return self

    @property
    def events(self) -> pd.DataFrame:
        if self._events is None:
            raise RuntimeError("Data not loaded. Call .load() first.")
        return self._events

    @property
    def work_orders(self) -> pd.DataFrame:
        if self._work_orders is None:
            raise RuntimeError("Data not loaded. Call .load() first.")
        return self._work_orders



# Script-mode debug / validation use it to debug and validate laoding process , not used for production


if __name__ == "__main__":
    # Portable path (no absolute paths)
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "data" / "maintenance_records.csv"

    data = MaintenanceData(csv_path).load()
    events = data.events
    work_orders = data.work_orders

    print("\n--- Ingestion Summary ---")
    print(f"Raw rows (events): {len(events)}")
    print(f"Distinct work orders: {len(work_orders)}")
    print(f"Distinct check (from events): {events['work_order_id'].nunique()}")
    print(f"Date range: {events['start_ts'].min()} -> {events['end_ts'].max()}")

    print("\nSample work orders (10 random, deterministic):")
    print(work_orders.sample(10, random_state=42))
