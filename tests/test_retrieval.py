import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import load_all  # noqa: E402
from analytics import Filters  # noqa: E402
from retrieval import search_work_orders  # noqa: E402


CSV_PATH = ROOT / "data" / "maintenance_records.csv"


def test_search_returns_results_for_common_query():
    _, work_orders = load_all(CSV_PATH)

    results = search_work_orders(work_orders, query="hydraulic leak", k=5)
    assert len(results) > 0
    assert (results["score"] > 0).all()


def test_search_with_filter_restricts_to_equipment():
    _, work_orders = load_all(CSV_PATH)

    eq = "CNC-01"
    results = search_work_orders(
        work_orders,
        query="hydraulic leak",
        k=10,
        f=Filters(equipment_id=eq),
    )

    if len(results) > 0:
        assert (results["equipment_id"] == eq).all()


def test_filtered_results_are_subset_of_unfiltered_when_not_truncated():
    _, work_orders = load_all(CSV_PATH)

    unfiltered_all = search_work_orders(work_orders, query="hydraulic leak", k=None)
    filtered_all = search_work_orders(
        work_orders,
        query="hydraulic leak",
        k=None,
        f=Filters(equipment_id="CNC-01"),
    )

    assert set(filtered_all["work_order_id"]).issubset(set(unfiltered_all["work_order_id"]))


def test_search_can_match_symptom_code_token():
    _, work_orders = load_all(CSV_PATH)

    results = search_work_orders(work_orders, query="SPINDLE_TIMEOUT", k=5)

    assert len(results) > 0
    assert (results["score"] > 0).all()
