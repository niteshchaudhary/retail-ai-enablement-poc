import streamlit as st
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import json
import datetime

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "retail-ai-enablement-poc"

QUERY_LOG_FILE = "query_log.json"

# --- Document categories ---
# This simulates what you'd do in a real enterprise:
# different source systems (Confluence, Slack, policy docs) tagged with metadata
# so the retriever can filter by source type - same problem Glean solves commercially
DOCUMENT_CATEGORIES = {
    "operations": [
        "INVENTORY MANAGEMENT", "SUPPLY CHAIN", "STORE OPERATIONS",
        "DEMAND FORECASTING", "PRODUCT CATALOG MANAGEMENT"
    ],
    "strategy": [
        "PRICING STRATEGY", "CUSTOMER PERSONALIZATION",
        "AI ENABLEMENT AT H-E-B", "COMPANY OVERVIEW"
    ]
}

def load_query_log():
    if os.path.exists(QUERY_LOG_FILE):
        with open(QUERY_LOG_FILE, "r") as f:
            return json.load(f)
    return []

def save_query_log(log):
    with open(QUERY_LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)

def log_query(question, answer, answered, category_filter):
    log = load_query_log()
    log.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "question": question,
        "answered": answered,
        "category_filter": category_filter,
        "answer_preview": answer[:120] if answered else "unanswered"
    })
    save_query_log(log)

@st.cache_resource
def build_vectorstore():
    """
    Loads domain knowledge and tags each chunk with metadata (category).
    In a real enterprise deployment, these categories would map to different
    source systems - Confluence pages, Slack channels, policy repositories.
    Metadata filtering lets you scope retrieval to relevant sources per query,
    which improves relevance and reduces noise - a key design decision.
    """
    loader = TextLoader("domain_knowledge.txt")
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(raw_docs)

    # Tag each chunk with a category based on which section it came from
    # This is the metadata layer - enables filtered retrieval downstream
    tagged_chunks = []
    for chunk in chunks:
        category = "general"
        for cat, keywords in DOCUMENT_CATEGORIES.items():
            if any(kw in chunk.page_content.upper() for kw in keywords):
                category = cat
                break
        chunk.metadata["category"] = category
        tagged_chunks.append(chunk)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(tagged_chunks, embeddings)
    return vectorstore

def build_chain(vectorstore, category_filter):
    """
    Builds a retrieval chain with optional metadata filtering.
    When category_filter is set, only chunks from that source type are retrieved.
    This mirrors how enterprise search needs to scope results -
    a question about pricing policy should retrieve from strategy docs,
    not from ops runbooks.
    """
    if category_filter and category_filter != "All":
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "filter": {"category": category_filter.lower()},
                "k": 4
            }
        )
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOllama(model="llama3.2", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
You are an internal AI assistant helping employees find answers from company knowledge.
Answer the question based only on the following context.
If the answer is not in the context, say exactly: "I don't know based on available knowledge."
Always be concise. If you reference specific data, mention where it came from.

Context:
{context}

Question: {question}
""")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# --- Streamlit UI ---

st.set_page_config(
    page_title="Employee AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Employee AI Enablement Assistant")
st.caption("Ask questions from internal company knowledge. Powered by RAG + local LLM inference.")

# Sidebar - category filter + adoption metrics
with st.sidebar:
    st.header("Knowledge Scope")
    st.caption("Filter retrieval to a specific knowledge category. In production, this maps to source systems like Confluence, Slack, or policy repos.")
    category_filter = st.radio(
        "Retrieve from:",
        ["All", "Operations", "Strategy"],
        index=0
    )

    st.divider()

    st.header("Adoption Metrics")
    st.caption("Tracks how well the assistant is serving employees - the same measurement layer you'd need at enterprise scale.")

    log = load_query_log()
    if log:
        total = len(log)
        answered = sum(1 for q in log if q["answered"])
        unanswered = total - answered
        answer_rate = round((answered / total) * 100) if total > 0 else 0

        st.metric("Total queries", total)
        st.metric("Answer rate", f"{answer_rate}%")
        st.metric("Coverage gaps", unanswered)

        if unanswered > 0:
            st.caption("Unanswered questions - these reveal knowledge gaps to fix:")
            for q in log:
                if not q["answered"]:
                    st.markdown(f"- _{q['question']}_")
    else:
        st.info("No queries yet. Ask something to start tracking.")

    st.divider()
    if st.button("Clear query log"):
        save_query_log([])
        st.rerun()

# Main chat area
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            st.caption(msg["meta"])

if prompt_input := st.chat_input("Ask a question from company knowledge..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving from knowledge base..."):
            vectorstore = build_vectorstore()
            chain = build_chain(vectorstore, category_filter)
            response = chain.invoke(prompt_input)

        answered = "i don't know" not in response.lower()
        meta = f"Scope: {category_filter} | {'Answered from knowledge base' if answered else 'Not found in knowledge base'}"

        st.markdown(response)
        st.caption(meta)

        log_query(prompt_input, response, answered, category_filter)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "meta": meta
        })