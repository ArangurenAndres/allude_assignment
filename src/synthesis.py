# src/synthesis.py
from __future__ import annotations

import pandas as pd


def format_retrieval_answer(query: str, results: pd.DataFrame, max_items: int = 5) -> str:
    """
    Format retrieval results into a readable, grounded answer (no LLM).

    The output is deterministic and cites work_order_id for traceability.
    """
    query = str(query).strip()

    if results is None or results.empty:
        return f"No matches found for: '{query}'. Try a different term or remove filters."

    # Ensure consistent columns (defensive; retrieval should already provide these)
    cols = ["work_order_id", "equipment_id", "symptom_code", "start_ts", "score", "snippet"]
    missing = [c for c in cols if c not in results.columns]
    if missing:
        raise ValueError(f"Results missing expected columns: {missing}")

    n_total = len(results)
    n_show = min(max_items, n_total)

    lines: list[str] = []
    lines.append(f"Query: {query}")
    lines.append(f"Matches: {n_total} (showing top {n_show})")
    lines.append("")

    for i in range(n_show):
        row = results.iloc[i]
        wo = row["work_order_id"]
        eq = row["equipment_id"]
        sym = row["symptom_code"]
        ts = row["start_ts"]
        score = row["score"]
        snippet = str(row["snippet"])

        # keep snippet compact
        snippet = snippet.replace("\n", " ").strip()
        if len(snippet) > 220:
            snippet = snippet[:220].rstrip() + "..."

        lines.append(f"{i+1}. work_order_id={wo} | equipment={eq} | symptom={sym} | start={ts} | score={score}")
        lines.append(f"   snippet: {snippet}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    from pathlib import Path
    from data import load_all
    from retrieval import search_work_orders
    from analytics import Filters

    ROOT = Path(__file__).resolve().parents[1]
    csv_path = ROOT / "data" / "maintenance_records.csv"

    _, work_orders = load_all(csv_path)

    query = "hydraulic leak"
    results = search_work_orders(work_orders, query=query, k=5, f=Filters(equipment_id="CNC-01"))

    print(format_retrieval_answer(query, results))