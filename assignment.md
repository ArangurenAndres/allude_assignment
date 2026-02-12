# Take-Home Assignment: Manufacturing Maintenance RAG System

## Context

You're building a conversational analytics system for a **manufacturing plant's maintenance department**. Technicians log every equipment failure, repair, and preventive maintenance action into a CMMS (Computerized Maintenance Management System). The data is exported as a CSV.

Your task is to build a **RAG (Retrieval-Augmented Generation) system** that lets maintenance managers ask natural language questions about this data and get accurate, conversational answers.

---

## The Data

You're provided with `data/maintenance_records.csv` — a dataset of **556 rows** across **200 work orders** for a manufacturing facility.

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `work_order_id` | integer | Unique identifier for each maintenance incident |
| `equipment_id` | string | Machine identifier (e.g., `CNC-01`, `PRESS-03`) |
| `product_line` | string | Equipment family (e.g., `HAAS_VF_SERIES`, `SCHULER_HYDRAULIC`) |
| `start_date` | date | When the work order was opened |
| `start_time` | time | Time of day work started |
| `end_date` | date | When the work order was closed |
| `end_time` | time | Time of day work ended |
| `description` | string | Structured incident classification (hierarchical, slash-separated) |
| `technician` | string | Name of the technician who logged the entry |
| `comment` | string | Free-text technician notes about diagnosis, actions, and follow-ups |
| `symptom_code` | string | Categorical classification of the issue type |

### Key Data Characteristics

1. **Multiple rows per work order** — A single incident (work order) can have multiple rows, one per technician comment/update. When counting incidents, you should count **distinct work orders**, not rows.

2. **Structured `description` field** — Follows a hierarchical format: `TYPE / PRODUCT_LINE / SYSTEM / CATEGORY / DETAIL`. Example: `COR / HAAS_VF_SERIES / HYDRAULIC / PRESSURE / LOW PRESSURE ALARM`

3. **Categorical vs. free-text fields**:
   - `symptom_code`, `equipment_id`, `product_line`, `technician` → exact-match categorical fields
   - `description`, `comment` → free-text fields that require keyword/semantic search

4. **Temporal dimension** — Data spans January 2024 to December 2024.

5. **20 equipment** across **6 product lines**, maintained by **10 technicians**, with **15 distinct symptom codes**.

---

## What to Build

Build a system that:

### 1. Ingests the CSV data
Load and process the CSV into a format suitable for retrieval. You decide the storage approach.

### 2. Answers natural language questions
Users should be able to ask questions like:
- "How many incidents were there on CNC-01?"
- "Which equipment has the most failures?"
- "Show me all pressure-related issues"
- "What did technicians say about bearing problems?"
- "How many work orders were opened in July 2024?"

### 3. Supports multi-turn conversations
Users should be able to ask follow-up questions that reference previous context:
```
User: How many incidents were there on PRESS-01?
Bot:  PRESS-01 had 18 work orders.

User: How many different technicians worked on it?
Bot:  8 different technicians worked on PRESS-01.

User: What was the most common symptom?
Bot:  The most common symptoms on PRESS-01 were ALIGNMENT_DRIFT and
      SENSOR_FAILURE, each with 3 work orders.
```

### 4. Provides a conversational interface
Build a simple interface where users can interact with the system. This can be:
- A CLI chat loop
- A web UI
- A Jupyter notebook with interactive cells
- A REST API with example curl commands

Whatever lets you demo the system effectively.

---

## Query Types Your System Should Handle

| Type | Example | What's Tested |
|------|---------|---------------|
| **Counting** | "How many incidents on CNC-01?" | Aggregation over structured data |
| **Ranking** | "Which equipment has the most failures?" | Sorting + aggregation |
| **Filtering** | "Show me PRESSURE_FAULT incidents" | Exact categorical match |
| **Text search** | "Find issues mentioning bearing problems" | Keyword/semantic search in free-text |
| **Temporal** | "How many work orders in Q1 2024?" | Date range filtering |
| **Cross-filter** | "Pressure faults on HAAS_VF_SERIES equipment" | Multiple filter combination |
| **Detail retrieval** | "What happened with work order 100042?" | Single-record lookup |
| **Multi-turn** | "How many on CNC-01?" → "What about PRESS-01?" | Context tracking |

---

## Sample Questions

These are provided for you to test during development. Your system should handle these correctly:

1. "How many total work orders are in the dataset?"
2. "Which equipment has the most incidents?"
3. "How many incidents were there on CNC-01?"
4. "List all work orders related to SPINDLE_TIMEOUT"
5. "What did technicians report about leak issues?"
6. "How many work orders were opened in July 2024?"
7. "Which technician handled the most work orders?"
8. "Show me all incidents on the HAAS_VF_SERIES product line"

**Multi-turn sequence:**
1. "How many incidents on PRESS-01?"
2. "How many different technicians worked on it?"
3. "What was the most common symptom there?"

See `test_questions.json` for the full list with expected answers.

---

## Deliverables

Submit a **GitHub repository** containing:

1. **Working code** — Your RAG system, runnable with clear setup instructions
2. **README** — How to set up, run, and use the system
3. **Architecture notes** — Brief explanation (in README or separate doc) of:
   - Your retrieval strategy and why you chose it
   - How you handle different query types
   - How you manage multi-turn context
   - Trade-offs you considered
4. **Demo output** — Example conversations showing the system working (screenshots, logs, or saved output)

---

## Evaluation Criteria

We'll evaluate your submission on:

- **Correctness** — Does it return accurate answers? (especially counting/aggregation)
- **Architecture** — Is the retrieval strategy well-suited to the data and query types?
- **Multi-turn handling** — Can it maintain context across follow-up questions?
- **Code quality** — Is the code clean, well-organized, and easy to understand?
- **Trade-off awareness** — Do you articulate why you made the choices you did?

We care more about **thoughtful design decisions** than feature completeness. A focused system that handles core cases well is better than a sprawling one that handles edge cases poorly.

---

## Constraints

- **LLM**: Use any LLM you prefer (OpenAI, Gemini, Claude, local models, etc.)
- **Stack**: Any language/framework — Python recommended but not required
- **Time**: This is designed to take **4-5 hours**. Don't over-engineer.
- **Cost**: If using paid APIs, keep costs minimal. Free-tier models are perfectly acceptable.

---

## Tips

- Think carefully about whether pure vector search is sufficient for all query types, or if some queries need a different approach.
- The "multiple rows per work order" pattern is intentional — counting correctly is important.
- The `description` field's hierarchical structure may be useful for retrieval.
- Consider what happens when a user asks "how many" vs. "show me" — these may need different retrieval strategies.

---

Good luck! We're excited to see your approach.
