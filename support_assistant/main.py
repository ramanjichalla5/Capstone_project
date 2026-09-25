import os, re
from pathlib import Path
from typing import TypedDict
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END

ROOT = Path(__file__).parent; DOCS = ROOT / "docs"; MODEL = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=str(ROOT / "chroma")); collection = client.get_or_create_collection("zepto_policies", metadata={"hnsw:space": "cosine"})

PROMPT = """Role: You are Zepto policy support.\nContext: {context}\nTask: Answer only the user's policy question.\nFormat: Return JSON with answer, sources, confidence.\nLength: Keep the answer under 80 words.\nConstraint: Do not answer using information not present in the provided context.\nExample: Q: What are support hours? Context: chat is available 24/7. A: support is available 24/7."""
class Answer(BaseModel):
    answer: str; sources: list[str] = []; confidence: float = Field(ge=0, le=1)
class Request(BaseModel): query: str
class State(TypedDict, total=False):
    query: str; intent: str; answer: Answer

def build_store():
    files = sorted(DOCS.glob("doc_*.txt")); ids = [p.stem for p in files]
    if collection.count() != len(files):
        texts = [p.read_text(encoding="utf-8") for p in files]; collection.upsert(ids=ids, documents=texts, embeddings=MODEL.encode(texts).tolist())
build_store()

def classify_intent(state):
    query = state["query"].lower(); words = ("delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours")
    if os.getenv("MOCK_LLM", "1") != "1":
        # Optional real mode remains compatible with any OpenAI-compatible Groq endpoint.
        from openai import OpenAI
        client_llm = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")
        result = client_llm.chat.completions.create(model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"), messages=[{"role":"user", "content": f"Classify as policy_question or general_question: {query}"}])
        intent = result.choices[0].message.content.strip()
    else: intent = "policy_question" if any(w in query for w in words) else "general_question"
    return {**state, "intent": intent}

def retrieve_and_answer(state):
    found = collection.query(query_embeddings=[MODEL.encode(state["query"]).tolist()], n_results=3); ids = found["ids"][0]; docs = found["documents"][0]
    if os.getenv("MOCK_LLM", "1") == "1": answer = f"Based on the retrieved context: {docs[0][:200]}"
    else:
        from openai import OpenAI
        llm = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1"); raw = None
        for _ in range(3):
            try:
                raw = llm.chat.completions.create(model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"), messages=[{"role":"user", "content": PROMPT.format(context="\n".join(docs)) + "\nQuestion: " + state["query"]}]).choices[0].message.content
                parsed = Answer.model_validate_json(raw); return {**state, "answer": parsed}
            except Exception: pass
        return {**state, "answer": Answer(answer="Unable to validate the LLM response.", sources=ids, confidence=0.0)}
    return {**state, "answer": Answer(answer=answer, sources=ids, confidence=1.0)}

def direct_answer(state): return {**state, "answer": Answer(answer="I can only answer questions about Zepto policies right now.", sources=[], confidence=1.0)}
def route(state): return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"
graph = StateGraph(State); graph.add_node("classify_intent", classify_intent); graph.add_node("retrieve_and_answer", retrieve_and_answer); graph.add_node("direct_answer", direct_answer); graph.set_entry_point("classify_intent"); graph.add_conditional_edges("classify_intent", route); graph.add_edge("retrieve_and_answer", END); graph.add_edge("direct_answer", END); workflow = graph.compile()
app = FastAPI(title="Zepto Support Assistant")
@app.post("/ask", response_model=Answer)
def ask(request: Request): return workflow.invoke({"query": request.query})["answer"]