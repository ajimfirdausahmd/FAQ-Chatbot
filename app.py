import os, glob, time, json
from typing import List, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# Config
DATA_DIR = r"D:\Interview_Test\data"
INDEX_DIR = r"D:\Interview_Test\faiss_index"
RESULTS_FILE = r"D:\Interview_Test\eval\results.jsonl"
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

# Ingestion
def load_csv_documents(data_dir: str) -> List[Document]:
    docs: List[Document] = []
    csv_paths = glob.glob(os.path.join(data_dir, "*.csv"))
    for p in csv_paths:
        loader = CSVLoader(file_path=p, encoding="utf-8")
        rows = loader.load()
        for i, d in enumerate(rows, 1):
            d.metadata.setdefault("row", i)
            d.metadata.setdefault("source", p)  
        docs.extend(rows)
    print(f"[INGEST] Loaded {len(docs)} row-documents from {len(csv_paths)} CSV files.")
    return docs

# Vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def build_or_load_faiss(docs: List[Document], index_dir: str) -> FAISS:
    if os.path.exists(index_dir):
        print(f"[INDEX] Loading existing FAISS index from {index_dir}")
        return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    print("[INDEX] Building FAISS index...")
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(index_dir)
    print(f"[INDEX] Saved FAISS index to {index_dir}")
    return vs

# RAG prompt & helpers
template = """
You are a factual assistant for Malaysian official statistics.
Answer the question based ONLY on the given context and (optionally) chat history.
If the context does not contain the answer, ask a brief clarifying question or politely refuse.
Always include in-text bracket citations [1], [2], etc. that map to the Sources list.

Chat history:
{history}

Context:
{context}

Question: {question}

Guidelines:
- Be concise and numeric when appropriate (units, dates).
- DO NOT invent values outside the context.
- End with a "Sources:" section listing the URLs (or file paths) for each bracket index.
"""
prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
rag_chain = prompt | llm

def format_context(docs: List[Document]) -> str:
    if not docs:
        return "(no retrieved context)"
    out = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        out.append(f"[{i}] {d.page_content}\n(Source: {src})")
    return "\n\n".join(out)

def format_history(turns: List[BaseMessage]) -> str:
    return "\n".join(
        ("User: " + m.content) if isinstance(m, HumanMessage)
        else ("Assistant: " + m.content)
        for m in turns
    )

def extract_citations(retrieved_docs: List[Document]) -> List[str]:
    return [
        f"{d.metadata.get('source','unknown')}#row={d.metadata.get('row','?')}"
        for d in retrieved_docs
    ]


# State & nodes
class DialogState(TypedDict):
    turns: List[BaseMessage]
    retrieved_docs: List[Document]
    topic_flag: str
    refined_query: str
    ready_for_response: bool
    refinement_attempts: int
    question: HumanMessage
    answer: str

class TopicGrade(BaseModel):
    score: str = Field(description="Is the question about the target topics? If yes -> 'Yes'; if not -> 'No'")

def rephrase_query(state: DialogState):
    print("Entering rephrase_query")
    state["retrieved_docs"] = []
    state["topic_flag"] = ""
    state["refined_query"] = ""
    state["ready_for_response"] = False
    state["refinement_attempts"] = 0
    state["answer"] = ""

    if "turns" not in state or state["turns"] is None:
        state["turns"] = []

    if state["question"] not in state["turns"]:
        state["turns"].append(state["question"])

    if len(state["turns"]) > 1:
        chat_history = state["turns"][:-1]
        question_text = state["question"].content
        prompt_msgs = [SystemMessage(content="Rephrase the user's last question into a standalone query optimized for retrieval.")]
        prompt_msgs.extend(chat_history)
        prompt_msgs.append(HumanMessage(content=question_text))
        messages = ChatPromptTemplate.from_messages(prompt_msgs).format_messages()
        response = llm.invoke(messages)
        refined = response.content.strip()
        state["refined_query"] = refined or state["question"].content
    else:
        state["refined_query"] = state["question"].content
    return state

def classify_topic(state: DialogState):
    print("Entering classify_topic")
    sys_msg = SystemMessage(content="""
You are a STRICT Yes/No classifier.

Say **Yes** only if the question is about **annual counts of live births (kelahiran hidup)** in Malaysia, including:
- National totals; by **state**, **sex**, **ethnic group**; **sex+ethnic+state**; **district+sex**
- Simple ops on these counts (top-N, diffs, ratios, shares) or **CSV export**
- Ambiguous geo/time (e.g., “Klang Valley”, missing year) → still Yes (downstream will clarify)

Say **No** for CPI/prices, income/poverty/Gini/households/LQ, deaths/mortality/life expectancy, fertility **rates** (TFR/CBR) or any non-Malaysia topic.

Return exactly: Yes or No.
""")
    user_msg = HumanMessage(content=f"User question: {state['refined_query']}")
    structured_llm = llm.with_structured_output(TopicGrade)
    grader = ChatPromptTemplate.from_messages([sys_msg, user_msg]) | structured_llm
    result = grader.invoke({})
    state["topic_flag"] = result.score.strip()
    print(f"classify_topic: topic_flag = {state['topic_flag']}")
    return state

def topic_router(state: DialogState):
    return "fetch_docs" if state.get("topic_flag", "").lower() == "yes" else "reject_off_topic"

def fetch_docs(state: DialogState, retriever):
    print("Entering fetch_docs")
    docs = retriever.invoke(state["refined_query"])
    print(f"fetch_docs: Retrieved {len(docs)} documents")
    state["retrieved_docs"] = docs
    return state

class RelevanceGrade(BaseModel):
    score: str = Field(description="Is the document relevant to the user's question? If yes -> 'Yes'; if not -> 'No'")

def evaluate_docs(state: DialogState):
    print("Entering evaluate_docs")
    sys_msg = SystemMessage(content="Binary grade relevance: If the document helps answer the question, 'Yes', else 'No'.")
    structured_llm = llm.with_structured_output(RelevanceGrade)

    relevant: List[Document] = []
    for doc in state["retrieved_docs"]:
        user_msg = HumanMessage(content=f"Question: {state['refined_query']}\n\nDocument:\n{doc.page_content}")
        grader = ChatPromptTemplate.from_messages([sys_msg, user_msg]) | structured_llm
        res = grader.invoke({})
        if (res.score or "").strip().lower() == "yes":
            relevant.append(doc)
    state["retrieved_docs"] = relevant
    state["ready_for_response"] = len(relevant) > 0
    print(f"evaluate_docs: ready_for_response = {state['ready_for_response']}")
    return state

def decision_router(state: DialogState):
    attempts = state.get("refinement_attempts", 0)
    if state.get("ready_for_response", False):
        return "create_response"
    if attempts >= 2:
        return "fallback_response"
    return "tweak_question"

def tweak_question(state: DialogState):
    print("Entering tweak_question")
    attempts = state.get("refinement_attempts", 0)
    if attempts >= 2:
        return state
    original = state["refined_query"]
    sys_msg = SystemMessage(content="Slightly refine the question to improve retrieval. Keep meaning and constraints.")
    user_msg = HumanMessage(content=f"Original: {original}")
    messages = ChatPromptTemplate.from_messages([sys_msg, user_msg]).format_messages()
    response = llm.invoke(messages)
    refined = response.content.strip() or original
    state["refined_query"] = refined
    state["refinement_attempts"] = attempts + 1
    return state

def create_response(state: DialogState) -> DialogState:
    print("[NODE] create_response")
    history = state.get("turns", [])
    context_docs = state.get("retrieved_docs", [])
    question = state["refined_query"]
    response = rag_chain.invoke({
        "history": format_history(history),
        "context": format_context(context_docs),
        "question": question
    })
    result = (response.content or "").strip()
    state["answer"] = result
    state["turns"].append(AIMessage(content=result))
    return state

def fallback_response(state: DialogState) -> DialogState:
    print("[NODE] fallback_response")
    msg = (
        "I’m not fully confident from the current DOSM births data. "
        "Please specify the **year** (e.g., 2018–2022), **geography** (national / state / district), "
        "and any **breakdowns** (sex, ethnic group). "
        "Examples: “Selangor male births in 2021”, “Top 5 states by births in 2022”, "
        "or “Export Penang births 2015–2022 to CSV”. "
        "Note: this bot only covers **annual counts of live births** — not CPI, income, poverty, Gini, deaths, or fertility rates."
    )
    state["answer"] = msg
    state["turns"].append(AIMessage(content=msg))
    return state

def reject_off_topic(state: DialogState) -> DialogState:
    print("[NODE] reject_off_topic")
    msg = (
        "Out of scope. I only handle **annual live births counts** in Malaysia "
        "(national/state; sex/ethnic; sex+ethnic+state; district+sex). "
        "Not CPI, income/poverty/Gini, households, deaths, or fertility rates. "
        "Example: “Johor male births 2020” or “Export Penang births 2015–2022 to CSV”."
    )
    state["answer"] = msg
    state["turns"].append(AIMessage(content=msg))
    return state


# Graph
def build_graph(retriever):
    checkpointer = MemorySaver()
    workflow = StateGraph(DialogState)

    workflow.add_node("rephrase_query", rephrase_query)
    workflow.add_node("classify_topic", classify_topic)
    workflow.add_node("reject_off_topic", reject_off_topic)
    workflow.add_node("fetch_docs", lambda s: fetch_docs(s, retriever=retriever))  # bind retriever
    workflow.add_node("evaluate_docs", evaluate_docs)
    workflow.add_node("create_response", create_response)
    workflow.add_node("tweak_question", tweak_question)
    workflow.add_node("fallback_response", fallback_response)

    workflow.add_edge("rephrase_query", "classify_topic")
    workflow.add_conditional_edges("classify_topic", topic_router, {
        "fetch_docs": "fetch_docs",
        "reject_off_topic": "reject_off_topic",
    })
    workflow.add_edge("fetch_docs", "evaluate_docs")
    workflow.add_conditional_edges("evaluate_docs", decision_router, {
        "create_response": "create_response",
        "tweak_question": "tweak_question",
        "fallback_response": "fallback_response",
    })
    workflow.add_edge("tweak_question", "fetch_docs")
    workflow.add_edge("create_response", END)
    workflow.add_edge("fallback_response", END)
    workflow.add_edge("reject_off_topic", END)

    workflow.set_entry_point("rephrase_query")
    return workflow.compile(checkpointer=checkpointer)


# Eval
EVAL_QUERIES = [
    "how much live births and crude birth rate in 2000-01-01?",
    "which year is the highest live births in johor? ",
    "which district in johor have the highest live births?"
]

def run_eval_query(graph, query: str, idx: int) -> None:
    start = time.time()
    state = graph.invoke(input={"question": HumanMessage(content=query)},
                         config={"configurable": {"thread_id": f"eval-{idx}"},
                                 "recursion_limit": 40})
    latency_ms = (time.time() - start) * 1000

    response = state.get("answer", "")
    if not response:
        ai_msgs = [m for m in state.get("turns", []) if isinstance(m, AIMessage)]
        response = ai_msgs[-1].content if ai_msgs else ""

    retrieved_docs = state.get("retrieved_docs", [])
    citations = extract_citations(retrieved_docs)
    retrieval_hit = len(retrieved_docs) > 0
    hallucination = (not retrieval_hit) and (response.strip() != "")

    record = {
        "query": query,
        "response": response,
        "citations": citations,
        "latency_ms": round(latency_ms, 2),
        "retrieval_hit": retrieval_hit,
        "hallucination": hallucination
    }
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[EVAL] {idx+1:02d}/{len(EVAL_QUERIES)} saved | {latency_ms:.1f} ms | hit={retrieval_hit} | halluc={hallucination}")

# Main
def main():
    docs = load_csv_documents(DATA_DIR)
    vs = build_or_load_faiss(docs, INDEX_DIR)
    retriever = vs.as_retriever(search_kwargs={"k": 5})
    graph = build_graph(retriever)

    print("\n[RUN] Starting evaluation...\n")
    open(RESULTS_FILE, "w", encoding="utf-8").close()  # clear

    for i, q in enumerate(EVAL_QUERIES):
        run_eval_query(graph, q, i)

    print(f"\n[DONE] All results saved to {RESULTS_FILE}\n")

if __name__ == "__main__":
    main()

import statistics, json

with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    rows = [json.loads(l) for l in f]

latencies = [r["latency_ms"] for r in rows]
p50 = statistics.median(latencies)
latencies_sorted = sorted(latencies)
p95 = latencies_sorted[int(0.95*(len(latencies_sorted)-1))]
hits = sum(1 for r in rows if r["retrieval_hit"])
halluc = sum(1 for r in rows if r["hallucination"])
n = len(rows)

print("\n=== Eval Summary ===")
print(f"Total: {n}")
print(f"p50 latency: {p50:.1f} ms")
print(f"p95 latency: {p95:.1f} ms")
print(f"Retrieval hit-rate: {hits}/{n} = {hits/n:.1%}")
print(f"Hallucination rate: {halluc}/{n} = {halluc/n:.1%}")