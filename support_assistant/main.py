import os
from pathlib import Path
from typing import TypedDict
from fastapi import FastAPI
from pydantic import BaseModel, Field
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError:
    chromadb = None
    SentenceTransformer = None
try:
    from langgraph.graph import StateGraph, END
except ImportError:
    StateGraph = None

ROOT = Path(__file__).parent
MODEL = SentenceTransformer("all-MiniLM-L6-v2") if SentenceTransformer else None
collection = None
if chromadb and MODEL:
    client = chromadb.PersistentClient(path=str(ROOT / "chroma"))
    collection = client.get_or_create_collection("zepto_policies", metadata={"hnsw:space": "cosine"})

POLICIES = {
    "doc_01": "Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee. Priority delivery, which reserves the next available rider slot, is available at checkout for an additional INR 15. Zepto does not currently deliver to addresses outside its listed serviceable pin codes.",
    "doc_02": "Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unopened, resalable condition. Approved refunds are credited to the original payment method within 3-5 business days, or instantly to the Zepto wallet if the customer opts for wallet credit. Personal care items that have been opened are non-returnable except in the case of a manufacturing defect. Return pickup, where required, is arranged free of cost by Zepto.",
    "doc_03": "Zepto offers three account tiers: Basic (free, default tier, standard delivery fees apply), Zepto Pass (INR 49 per month, free standard delivery on all orders and 5% off select categories), and Zepto Pass+ (INR 99 per month, free priority delivery, 10% off select categories, and early access to limited-time deals 24 hours before they go live to Basic and Pass members). Membership can be cancelled at any time from account settings; cancelling stops the next billing cycle but does not refund the current membership period.",
    "doc_04": "Every Zepto order shows a live rider-tracking map from the moment it is packed until delivery, accessible from the 'Track Order' screen. Estimated delivery time updates automatically as the rider moves. If an order's status shows no movement for more than 20 minutes past its original estimated delivery time, customers should contact support directly rather than continue waiting, since this indicates a likely delivery issue.",
    "doc_05": "Orders can be cancelled free of cost any time before the order status changes to 'Packed', typically within the first 2 minutes of placing the order. Once an order has been packed, it can no longer be cancelled through the app, since the rider is dispatched immediately after packing given Zepto's quick-delivery model. If a packed order cannot be delivered due to a Zepto-side issue (for example, rider unavailability), the order is auto-cancelled and fully refunded without any cancellation fee.",
    "doc_06": "If an order arrives with damaged, spoiled, or missing items, customers must report it within 24 hours of delivery through the 'Report an Issue' button on the order page. Zepto ships a free replacement or issues a full refund for damaged, spoiled, or missing items without requiring the customer to return the original item, unless the order value exceeds INR 1000, in which case a photo of the issue must be submitted through the report form before a replacement or refund is processed.",
    "doc_07": "Zepto gift cards are available in fixed denominations of INR 100, INR 250, INR 500, and INR 1000, and are delivered by email or SMS within minutes of purchase. Gift cards are valid for 1 year from the date of issue and carry no maintenance fees. Gift card balance can be combined with one other payment method at checkout but cannot be combined with another gift card in the same transaction. Gift card balance cannot be redeemed for cash except where required by law.",
    "doc_08": "Zepto customer support is available via in-app chat 24 hours a day, 7 days a week, given the time-sensitive nature of quick commerce deliveries. Average in-app chat response time is under 2 minutes. Email support is also available for non-urgent queries and is answered within 24 hours on business days. Phone support is not offered.",
}

PROMPT = """Role: You are Zepto policy support.\nContext: {context}\nTask: Answer only the user's policy question.\nFormat: Return JSON with answer, sources, confidence.\nLength: Keep the answer under 80 words.\nConstraint: Do not answer using information not present in the provided context.\nExample: Q: What are support hours? Context: chat is available 24/7. A: support is available 24/7."""
class Answer(BaseModel):
    answer: str; sources: list[str] = []; confidence: float = Field(ge=0, le=1)
class Request(BaseModel): query: str
class State(TypedDict, total=False):
    query: str; intent: str; answer: Answer

def build_store():
    ids, texts = list(POLICIES), list(POLICIES.values())
    if collection and collection.count() != len(texts): collection.upsert(ids=ids, documents=texts, embeddings=MODEL.encode(texts).tolist())
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
    if collection:
        found = collection.query(query_embeddings=[MODEL.encode(state["query"]).tolist()], n_results=3)
        ids, docs = found["ids"][0], found["documents"][0]
    else:
        terms = set(state["query"].lower().split())
        ranked = sorted(POLICIES.items(), key=lambda item: len(terms & set(item[1].lower().split())), reverse=True)[:3]
        ids, docs = [item[0] for item in ranked], [item[1] for item in ranked]
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
if StateGraph:
    graph = StateGraph(State); graph.add_node("classify_intent", classify_intent); graph.add_node("retrieve_and_answer", retrieve_and_answer); graph.add_node("direct_answer", direct_answer); graph.set_entry_point("classify_intent"); graph.add_conditional_edges("classify_intent", route); graph.add_edge("retrieve_and_answer", END); graph.add_edge("direct_answer", END); workflow = graph.compile()
else:
    class LocalWorkflow:
        def invoke(self, state):
            state = classify_intent(state)
            return retrieve_and_answer(state) if state["intent"] == "policy_question" else direct_answer(state)
    workflow = LocalWorkflow()
app = FastAPI(title="Zepto Support Assistant")
@app.post("/ask", response_model=Answer)
def ask(request: Request): return workflow.invoke({"query": request.query})["answer"]