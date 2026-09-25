# Support Assistant

Run `uvicorn support_assistant.main:app --reload` and POST `{"query":"How much is delivery?"}` to `/ask`. The default `MOCK_LLM=1` path is deterministic and offline. A policy query routes to `retrieve_and_answer`, while an unrelated query routes to `direct_answer`.

Architecture: `POLICIES` in `main.py` is ingestion; `build_store()` chunks one document per entry, embeds with `all-MiniLM-L6-v2`, and writes the `zepto_policies` Chroma collection. `retrieve_and_answer` embeds a query and retrieves three cosine-nearest chunks, then generation uses the structured role/context/task/format/length prompt. `classify_intent` routes the LangGraph state to retrieval or direct answer. Only classification and final generation have optional LLM branches; embeddings and retrieval always run locally. In mock mode the answer is deterministic and sources/confidence are populated directly. With `MOCK_LLM=0`, an OpenAI-compatible Groq client is used and invalid JSON is retried twice.

Example mock responses:

```json
{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials...","sources":["doc_01"],"confidence":1.0}
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

Build locally with `docker build -t zepto-support support_assistant` and run with `docker run -p 7860:7860 zepto-support`.

The graph nodes are `classify_intent`, `retrieve_and_answer`, and `direct_answer`; the conditional edge is selected from the state intent.