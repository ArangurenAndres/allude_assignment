# src/webapp.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, session, redirect, url_for, Response
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
from src.app import (
    ConversationState,
    extract_filters,
    extract_mention_keyword,
    intent,
    maybe_llm,
)

ROOT = Path(__file__).resolve().parents[1]

app = Flask(
    __name__,
    template_folder=str(ROOT / "templates"),
    static_folder=str(ROOT / "static"),
)

# Needed for session-based history
# Set a real value in env for production: export FLASK_SECRET_KEY="..."
app.secret_key = (
    __import__("os").getenv("FLASK_SECRET_KEY")
    or "dev-only-secret-change-me"
)

# Load data once at startup
csv_path = ROOT / "data" / "maintenance_records.csv"
_, work_orders = load_all(csv_path)

known_equipment = set(work_orders["equipment_id"].astype(str).unique())
known_product_lines = set(work_orders["product_line"].astype(str).unique())
known_symptoms = set(work_orders["symptom_code"].astype(str).unique())
default_year = int(pd.to_datetime(work_orders["start_ts"]).dt.year.mode().iloc[0])

state = ConversationState()


def answer_question(user: str) -> str:
    user = (user or "").strip()
    if not user:
        return ""

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
        return maybe_llm(user, str(n))

    if it == "count_incidents":
        n = count_incidents(work_orders, f=f)
        return maybe_llm(user, str(n))

    if it == "count_distinct_technicians":
        n = count_distinct_technicians(work_orders, f=f)
        return maybe_llm(user, str(n))

    if it == "most_common_symptom":
        tied = most_common_symptoms(work_orders, f=f)
        if not tied:
            return ""
        names = [s for s, _ in tied]
        n = tied[0][1]
        if len(names) == 1:
            ans = f"{names[0]} with {n} work orders"
        else:
            ans = f"{' and '.join(names)}, each with {n} work orders"
        return maybe_llm(user, ans)

    if it == "top_equipment":
        df = top_equipment(work_orders, n=1, f=Filters())
        if df.empty:
            return ""
        eq = str(df.iloc[0]["equipment_id"])
        n = int(df.iloc[0]["work_order_count"])
        ans = f"{eq} with {n} work orders"
        return maybe_llm(user, ans)

    if it == "top_technician":
        df = top_technicians(work_orders, n=1, f=Filters())
        if df.empty:
            return ""
        tech = str(df.iloc[0]["technician"])
        n = int(df.iloc[0]["work_order_count"])
        ans = f"{tech} with {n} work orders"
        return maybe_llm(user, ans)

    return ""


def get_history() -> list[dict]:
    hist = session.get("history")
    if not isinstance(hist, list):
        hist = []
    return hist


def push_history(q: str, a: str) -> None:
    hist = get_history()
    hist.append(
        {
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "q": q,
            "a": a,
        }
    )
    # keep it sane (last 30)
    hist = hist[-30:]
    session["history"] = hist


@app.route("/", methods=["GET", "POST"])
def home():
    answer = None
    question = ""

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            answer = answer_question(question)
            push_history(question, answer)

            # PRG pattern: avoid resubmitting on refresh
            return redirect(url_for("home"))

    history = get_history()
    return render_template("index.html", history=history)


@app.route("/clear", methods=["POST"])
def clear_history():
    session["history"] = []
    return redirect(url_for("home"))


@app.route("/export.txt", methods=["GET"])
def export_txt():
    hist = get_history()
    lines = []
    lines.append("Maintenance Assistant — Q/A History")
    lines.append("=" * 40)
    lines.append("")
    for i, item in enumerate(hist, 1):
        lines.append(f"{i}. [{item.get('ts','')}]")
        lines.append(f"Q: {item.get('q','')}")
        lines.append(f"A: {item.get('a','')}")
        lines.append("")
    content = "\n".join(lines).strip() + "\n"

    return Response(
        content,
        mimetype="text/plain",
        headers={"Content-Disposition": "attachment; filename=qa_history.txt"},
    )


if __name__ == "__main__":
    app.run(debug=True)
