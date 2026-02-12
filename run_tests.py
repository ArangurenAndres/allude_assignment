# run_tests.py
import json
from pathlib import Path
from datetime import datetime
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

ROOT = Path(__file__).resolve().parent

# --------------------------------------------------
# Load dataset once
# --------------------------------------------------

csv_path = ROOT / "data" / "maintenance_records.csv"
_, work_orders = load_all(csv_path)

known_equipment = set(work_orders["equipment_id"].astype(str).unique())
known_product_lines = set(work_orders["product_line"].astype(str).unique())
known_symptoms = set(work_orders["symptom_code"].astype(str).unique())
default_year = int(pd.to_datetime(work_orders["start_ts"]).dt.year.mode().iloc[0])

state = ConversationState()

# --------------------------------------------------
# Core logic (copied from webapp)
# --------------------------------------------------

def answer_question(user: str) -> str:
    user = user.strip()
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


# --------------------------------------------------
# Load test file
# --------------------------------------------------

with open(ROOT / "test_questions.json") as f:
    test_data = json.load(f)

results = {
    "timestamp": datetime.now().isoformat(),
    "single_turn": [],
    "multi_turn": []
}

print("\n========== SINGLE TURN TESTS ==========\n")

# ----------------------------
# Single turn
# ----------------------------

for q in test_data["single_turn"]:
    question = q["question"]
    expected = q.get("expected_answer", "")

    output = answer_question(question)

    print(f"Q: {question}")
    print(f"Expected: {expected}")
    print(f"Output:   {output}")
    print("-" * 50)

    results["single_turn"].append({
        "id": q["id"],
        "question": question,
        "expected": expected,
        "output": output
    })


# ----------------------------
# Multi turn
# ----------------------------

print("\n========== MULTI TURN TESTS ==========\n")

for convo in test_data["multi_turn"]:
    print(f"\nConversation: {convo['id']} ({convo['name']})\n")

    convo_result = {
        "id": convo["id"],
        "name": convo["name"],
        "turns": []
    }

    for turn in convo["turns"]:
        question = turn["question"]
        expected = turn.get("expected_answer", "")

        output = answer_question(question)

        print(f"Q: {question}")
        print(f"Expected: {expected}")
        print(f"Output:   {output}")
        print("-" * 50)

        convo_result["turns"].append({
            "question": question,
            "expected": expected,
            "output": output
        })

    results["multi_turn"].append(convo_result)


# --------------------------------------------------
# Save results
# --------------------------------------------------

results_dir = ROOT / "results"
results_dir.mkdir(exist_ok=True)

txt_path = results_dir / "test_results.txt"
json_path = results_dir / "test_results.json"

# TXT
with open(txt_path, "w") as f:
    for st in results["single_turn"]:
        f.write(f"Q: {st['question']}\n")
        f.write(f"Expected: {st['expected']}\n")
        f.write(f"Output: {st['output']}\n")
        f.write("-" * 50 + "\n")

    f.write("\nMULTI TURN\n\n")

    for mt in results["multi_turn"]:
        f.write(f"Conversation: {mt['id']} ({mt['name']})\n\n")
        for turn in mt["turns"]:
            f.write(f"Q: {turn['question']}\n")
            f.write(f"Expected: {turn['expected']}\n")
            f.write(f"Output: {turn['output']}\n")
            f.write("-" * 50 + "\n")

# JSON
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

print("\n Test results saved to:")
print(f"   - {txt_path}")
print(f"   - {json_path}")