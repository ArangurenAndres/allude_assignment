# Maintenance assistant

Lightweight Retrieval - Augmented - Geenration (RAG) for querying maintenance work order data using natural language

The assistan allows users to ask questions such as 


- “How many work orders were there in July 2024?”
- “Which equipment has the most incidents?”
- “How many work orders mention a leak?”

The system parses the question, retrieves structured information from the dataset, computes deterministic analytics, and optionally uses an LLM to generate a natural-language response grounded in those results.



#  How to Run the Project
Make sure you are in the **project root directory** before running any commands.


## 1. Create a Virtual Environment (Recommended)
```bash
python -m venv .venv
source .venv/bin/activate
```

## 2. Install Dependencies



```bash
pip install -r requirements.txt
```
## Run main.py (three options to run main.py )
### Run web application (recommended)
```bash
python main.py web
```
### Run the CLI version
```bash
python main.py cli
```
Type your question directly in the terminal.
To exit type --> exit
### Run the test version

```bash
python main.py test
```
This will:

- Execute all quesstions in test_questions.json
- Print output in the terminal
- Save results in folder results / test_reuslts.txt, test_results.json

## By default the system uses deterministic Analysis for  (LLM synthesis)

This project can use a local LLM via  **Ollama** for the “Generation” part of the RAG pipeline.


```bash
brew install ollama
```
### Start the server:
```bash
ollama serve
```
#### Leave the server running and open another terminal tab to pull the model used for the project

```bash
ollama pull phi3:mini
```
### Enable the LLM layer

```bash
export USE_LLM=1
export OLLAMA_MODEL=phi3:mini

```

#### run the project
```bash
export USE_LLM=1
export OLLAMA_MODEL=phi3:mini

```




-------------------------------------------------------------------------------------------


This project implements a **Retrieval-Augmented Generation (RAG)** architecture.

## The reuslts of the tested RAG system can be found in the results folder 


This system retrieves **structured data from a tabular dataset**.


1. **Query Parsing Layer**
   - Detects user intent
   - Extracts filters (equipment, product line, symptom, time range)
   - Converts natural language into structured constraints

2. **Retrieval Layer**
   - Applies structured filters to the dataset
   - Selects relevant rows
   - Executes deterministic analytics

3. **Generation Layer (Optional)**
   - Uses an LLM to rephrase deterministic results
   - Ensures answers remain grounded in computed outputs

The deterministic analytics layer remains the source of truth.

# RAG SYSTEM

This system:
- Retrieves structured data via filters
- Computes precise analytics
- Optionally uses an LLM to generate a clean response

Instead of retrieving text embeddings, this system retrieves structured rows from a dataframe.

# Core Components

## `src/data.py` — Knowledge Base Loader

**Role in RAG:** Knowledge Base Initialization

**Responsibilities:**
- Loads `maintenance_records.csv`
- Returns structured pandas DataFrames
- Defines the data source used by retrieval

This file provides the **retrieval corpus**, equivalent to a document store in traditional RAG systems.


## `src/retrieval.py` — Structured Retrieval Layer

**Role in RAG:** Retrieval

**Responsibilities:**
- Applies structured filters (equipment, product line, symptom, time window)
- Selects relevant rows from the dataset
- Narrows down the search space

Instead of semantic similarity via embeddings, this layer performs **precise structured filtering** over tabular data.


##  `src/analytics.py` — Deterministic Reasoning Layer

**Role in RAG:** Grounded Reasoning

**Responsibilities:**
- Count distinct work orders
- Rank equipment and technicians
- Aggregate symptoms
- Count keyword mentions in text
- Compute distinct counts
- Perform time-based filtering

This layer performs deterministic computations on retrieved rows.


##  `src/app.py` — Orchestrator / Query Controller

**Role in RAG:** Query Understanding + Pipeline Orchestration

**Responsibilities:**
- Detect user intent
- Extract structured filters
- Parse time expressions (months, half-year)
- Route questions to the correct analytics function
- Optionally call the LLM wrapper

This file controls the full RAG flow:

1. Parse  
2. Retrieve  
3. Compute  
4. Generate (optional)

##  `src/synthesis.py` — Output Formatter

**Role in RAG:** Structured-to-Text Transformation

**Responsibilities:**
- Convert deterministic outputs into structured responses
- Prepare answers for presentation
- Ensure consistent formatting

This layer ensures stable, predictable outputs before optional LLM processing.

##  `src/llm.py` — Optional Generation Layer

**Responsibilities:**
- Take `(user_question, deterministic_answer)`
- Generate a concise grounded explanation
- Never compute analytics independently

The LLM is used strictly for presentation, not reasoning.

This keeps the system grounded and prevents hallucinations.


##  `src/webapp.py` — Interface Layer

**Role in RAG:** User Interaction

**Responsibilities:**
- Accept user questions
- Display answers
- Maintain session-based memory
- Export history

This layer does not perform retrieval or reasoning.  
It is purely an interface layer.


# Full End-to-End Example

Let’s walk through one example question:

> “How many work orders were there in the first half of 2024?”

## Step 1 — Data Loading
At startup the dataset is loaded:

```python
_, work_orders = load_all(csv_path)
```
## Step 2 — Intent Detection

```python
it = intent(user)
```

### Detected intent:

```python
count_incidents
```

## Step 3 — Filter Extraction

### The system parses the phrase:
first half of 2024

### It generates a time range:
start_ts_min = 2024-01-01
start_ts_max = 2024-07-01

## Step 4 A Filters object is created:

```python
Filters(
    equipment_id=None,
    product_line=None,
    symptom_code=None,
    start_ts_min=2024-01-01,
    start_ts_max=2024-07-01
)
```

## Step 5. Retrieval
#### The dataset is filtered using the extracted time range:

```python
df_filtered = work_orders[
    (start_ts >= 2024-01-01) &
    (start_ts < 2024-07-01)
]
```


## Step 6 — Deterministic Computation

```python
count_incidents(work_orders, f=f)
```

## Step 7 — Optional LLM Synthesis

```python
maybe_llm(user, "106")
```