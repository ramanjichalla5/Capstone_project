# Zepto Data & AI Platform

Three compact modules live in this repository: `data_pipeline`, `analytics`, and `support_assistant`.

## Setup

From the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

There is one consolidated requirements file. The analytics loader needs internet on its first Seaborn download; the support assistant defaults to a fully offline mock LLM.

## Run

```bash
python data_pipeline/main.py
python analytics/main.py
uvicorn support_assistant.main:app --reload
```

Then POST `{"query":"What is the delivery fee?"}` to `http://127.0.0.1:8000/ask`. To try the optional Groq-compatible LLM path, set `MOCK_LLM=0`, `GROQ_API_KEY`, and optionally `GROQ_MODEL`; the default graded path needs no credential.

## Design

The data pipeline scrapes Books to Scrape, cleans values, applies the fixed `1 GBP = 105.50 INR` conversion, and loads normalized SQLite tables. The analytics pipeline loads Titanic once, writes the CSV fallback, performs the required EDA and train-only modeling workflow, and saves a complete fitted pipeline. The support assistant embeds its eight policy texts from `main.py` into ChromaDB, routes questions through a LangGraph state graph, and exposes validated FastAPI JSON.