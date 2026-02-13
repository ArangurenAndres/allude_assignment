# src/retrieval.py
from __future__ import annotations

from pathlib import Path
import re
from typing import Optional

import pandas as pd

from analytics import Filters, filter_work_orders


# ------------------------------------------------------------
# Step 1: Build searchable text (with symptom expansion)
# ------------------------------------------------------------

def build_search_text(work_orders: pd.DataFrame) -> pd.Series:
    """
    Combine structured fields + description + comments into one normalized lowercase text field.
    """
    equipment = work_orders["equipment_id"].astype(str).fillna("").str.strip()
    product = work_orders["product_line"].astype(str).fillna("").str.strip()
    symptom = work_orders["symptom_code"].astype(str).fillna("").str.strip()
    desc = work_orders["description"].astype(str).fillna("").str.strip()
    comm = work_orders["comments"].astype(str).fillna("").str.strip()

    text = (
        equipment + " " +
        product + " " +
        symptom + "\n" +
        desc + "\n" +
        comm
    ).str.lower()

    text = text.str.replace(r"\s+", " ", regex=True).str.strip()
    return text


# ------------------------------------------------------------
# Tokenization
# ------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")

def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(str(text).lower())


# ------------------------------------------------------------
# Scoring
# ------------------------------------------------------------

def _score_text(doc_text: str, query_tokens: list[str], query_phrase: str) -> int:
    score = 0

    # Token hits
    for tok in set(query_tokens):
        if tok and tok in doc_text:
            score += 2

    # Exact phrase boost
    if query_phrase and query_phrase in doc_text:
        score += 5

    return score


# ------------------------------------------------------------
# Snippet
# ------------------------------------------------------------

def _make_snippet(doc_text: str, query_tokens: list[str], window: int = 120) -> str:
    doc_text = str(doc_text)

    for tok in query_tokens:
        pos = doc_text.find(tok)
        if pos != -1:
            start = max(0, pos - window)
            end = min(len(doc_text), pos + window)
            return doc_text[start:end].strip()

    return doc_text[: 2 * window].strip()


# ------------------------------------------------------------
# Main search
# ------------------------------------------------------------

def search_work_orders(
    work_orders: pd.DataFrame,
    query: str,
    k: int | None = 5,
    f: Optional[Filters] = None,
) -> pd.DataFrame:

    if f is None:
        f = Filters()

    df = filter_work_orders(work_orders, f).copy()

    if df.empty:
        return pd.DataFrame(columns=[
            "work_order_id", "equipment_id", "symptom_code",
            "start_ts", "score", "snippet"
        ])

    query = str(query).strip().lower()
    query_tokens = _tokenize(query)

    if not query_tokens:
        raise ValueError("Query is empty after tokenization.")

    # ------------------------------------------------------------
    # Remove equipment token from scoring if filter already applied
    # (prevents double counting / overly strict threshold)
    # ------------------------------------------------------------
    if f.equipment_id:
        eq_tok = f.equipment_id.lower()
        query_tokens = [t for t in query_tokens if t != eq_tok]

    search_text = build_search_text(df)

    scores = []
    snippets = []

    for _, doc_text in search_text.items():
        s = _score_text(doc_text, query_tokens, query_phrase=query)
        scores.append(s)
        snippets.append(_make_snippet(doc_text, query_tokens))

    df["score"] = scores
    df["snippet"] = snippets

    # ------------------------------------------------------------
    # Dynamic minimum score threshold (UX improvement)
    # ------------------------------------------------------------
    unique_tokens = set(query_tokens)

    # 2+ tokens → require both tokens (score >= 4)
    # 1 token → require at least one match (score >= 2)
    min_score = 4 if len(unique_tokens) >= 2 else 2

    ranked = df[df["score"] >= min_score].sort_values(
        ["score", "start_ts"],
        ascending=[False, False]
    )

    if k is not None:
        ranked = ranked.head(k)

    return ranked[
        ["work_order_id", "equipment_id", "symptom_code",
         "start_ts", "score", "snippet"]
    ].reset_index(drop=True)


## run to debug retrieval system and see the results of the search fucntion with diff queries and fileters

if __name__ == "__main__":
    from data import load_all

    ROOT = Path(__file__).resolve().parents[1]
    csv_path = ROOT / "data" / "maintenance_records.csv"

    print("Loading data...")
    _, work_orders = load_all(csv_path)

    print("\n--- search_work_orders() demo ---")
    print(search_work_orders(work_orders, query="hydraulic leak", k=5))

    print("\n--- search_work_orders() demo (filtered to CNC-01) ---")
    print(search_work_orders(
        work_orders,
        query="hydraulic leak",
        k=5,
        f=Filters(equipment_id="CNC-01")
    ))

    print("\n--- search_work_orders() demo (SPINDLE_TIMEOUT) ---")
    print(search_work_orders(work_orders, query="SPINDLE_TIMEOUT", k=5))
