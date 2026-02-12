import sys
from pathlib import Path

import pandas as pd

# Allow importing from src/
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import REQUIRED_COLUMNS, load_all  # noqa: E402


CSV_PATH = ROOT / "data" / "maintenance_records.csv"


def test_csv_exists():
    assert CSV_PATH.exists(), f"CSV not found at {CSV_PATH}"


def test_ingestion_basic():
    events, work_orders = load_all(CSV_PATH)

    # Basic size checks
    assert len(events) > 0
    assert len(work_orders) > 0

    # Required columns exist
    for col in REQUIRED_COLUMNS:
        assert col in events.columns

    # Timestamp parsing
    assert "start_ts" in events.columns
    assert "end_ts" in events.columns
    assert pd.api.types.is_datetime64_any_dtype(events["start_ts"])
    assert pd.api.types.is_datetime64_any_dtype(events["end_ts"])

    # No parsing failures
    assert events["start_ts"].isna().sum() == 0
    assert events["end_ts"].isna().sum() == 0

    # Derived columns exist (collapsed view)
    assert "comments" in work_orders.columns
    assert "technicians" in work_orders.columns
    assert "n_updates" in work_orders.columns


def test_distinct_work_orders():
    events, work_orders = load_all(CSV_PATH)

    distinct_events = events["work_order_id"].nunique()

    # Must match collapsed table size
    assert distinct_events == len(work_orders)

    # Ensure exactly one row per work_order_id
    assert work_orders["work_order_id"].nunique() == len(work_orders)


def test_work_order_fields_not_null():
    _, work_orders = load_all(CSV_PATH)

    critical_cols = ["work_order_id", "equipment_id", "product_line", "symptom_code"]

    for col in critical_cols:
        assert work_orders[col].isna().sum() == 0

    # Not blank strings (for identifiers)
    for col in ["equipment_id", "product_line", "symptom_code"]:
        assert (work_orders[col].astype(str).str.strip() == "").sum() == 0