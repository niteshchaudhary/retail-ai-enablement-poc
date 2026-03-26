# Enterprise AI Enablement POC

A hands-on prototype demonstrating the core engineering patterns behind enterprise AI assistant platforms — retrieval-augmented generation (RAG), metadata-scoped knowledge retrieval, local LLM inference, and adoption measurement.

Built to explore the architectural foundations of tools like Glean, Slack AI, and Atlassian Intelligence — and the custom agent and integration layer an enterprise would build on top of them.

---

## The Problem This Solves

Enterprise AI adoption fails not at the model layer — it fails at the retrieval layer.

Getting an LLM to answer questions accurately in an enterprise context requires:
- Connecting the right internal knowledge sources
- Scoping retrieval so employees get relevant, not noisy, results
- Respecting security and access boundaries per source
- Measuring what employees are actually asking — and where the system falls short

This POC demonstrates each of those patterns end-to-end.

---

## What This Demonstrates

### 1. RAG Pipeline with Domain Grounding
- LLM answers only from approved internal knowledge — no hallucination from general training data
- "I don't know" fallback when the answer is outside the knowledge base
- Every query and response traced via LangSmith for full observability

### 2. Metadata-Scoped Retrieval
- Knowledge chunks are tagged by source category (Operations vs. Strategy)
- Retrieval can be scoped to a specific category per query
- Simulates the real enterprise problem: different source systems (Confluence, Slack, policy repos, runbooks) have different retrieval characteristics — a question about pricing strategy should not retrieve from ops runbooks
- This is the same architectural pattern enterprise search platforms use with their data connectors

### 3. Adoption Metrics Layer
- Every query is logged with timestamp, category filter, and whether it was answered
- Unanswered questions surface automatically as a "coverage gaps" list
- Answer rate tracked in real time
- This mirrors the measurement layer needed at enterprise scale: adoption without measurement is just spend. Unanswered questions tell you exactly which knowledge sources to connect next.

### 4. Conversational UI
- Streamlit-based chat interface accessible to non-technical employees
- Source scoping visible per response so employees understand where answers come from
- Designed to mirror the UX pattern of enterprise AI assistants

---

## Architecture

```
Employee Query
      |
      v
 Category Filter (metadata scope)
      |
      v
 Chroma Vector Store (semantic search)
      |
      v
 Retrieved Chunks (scoped by source type)
      |
      v
 LLM (llama3.2 via Ollama - local inference, no external API)
      |
      v
 Grounded Response + Source Attribution
      |
      v
 Query Log (answered / unanswered / category)
      |
      v
 Adoption Metrics Dashboard (sidebar)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | llama3.2 via Ollama (local inference) |
| Orchestration | LangChain |
| Vector store | Chroma |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| Observability | LangSmith |
| UI | Streamlit |
| Metadata filtering | Chroma search_kwargs filter |

---

## Key Engineering Decisions

**Why local inference (Ollama)?**
Enterprise AI deployments require data sovereignty — internal knowledge cannot be sent to external APIs. Running llama3.2 locally simulates the security posture required in enterprise environments.

**Why metadata filtering?**
In a real deployment, knowledge sources are heterogeneous — Confluence pages, Slack exports, policy documents, runbooks. Treating them as one undifferentiated corpus degrades retrieval quality. Tagging chunks by source type and filtering at query time improves relevance and mirrors how enterprise search platforms scope results.

**Why a query log?**
Most enterprise AI adoption efforts measure license counts, not actual usage. The query log tracks what employees are actually asking, whether the system answered, and where the knowledge base has gaps. Unanswered questions are the most actionable signal — they tell you exactly what to index next.

---

## Running Locally

```bash
# Prerequisites: Ollama installed and running with llama3.2 pulled
ollama pull llama3.2

# Install dependencies
pip install langchain langchain-ollama langchain-community \
    langchain-text-splitters chromadb sentence-transformers \
    langsmith streamlit

# Set LangSmith env variables (optional - for observability)
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_langsmith_key
export LANGCHAIN_PROJECT=retail-ai-enablement-poc

# Run the UI
python3 -m streamlit run app.py

# Or run the CLI version
python3 agent.py

# Run evaluation suite
python3 eval.py
```

---

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI - conversational interface, metadata filtering, adoption metrics |
| `agent.py` | CLI version - interactive RAG pipeline |
| `eval.py` | Evaluation suite - ground truth dataset, answer accuracy measurement |
| `domain_knowledge.txt` | Sample enterprise knowledge base (retail domain) |

---

## LangSmith Observability

Every query is traced end-to-end in LangSmith — retrieval steps, LLM calls, and response generation. In a production deployment this is the observability layer that enables debugging, quality monitoring, and retraining signal collection.

![LangSmith Tracing](https://github.com/user-attachments/assets/a4f78fb7-8a01-4db1-93b6-08fe5b88f044)

---

## Extending This POC

This prototype uses a single text file as the knowledge source. In a production enterprise deployment, the same architecture would connect to:
- Confluence (internal documentation)
- Slack channel exports (tribal knowledge)
- Policy and compliance repositories
- Zoom meeting transcripts
- Ticketing systems (Jira, ServiceNow)

Each source would have its own connector, chunking strategy, and metadata tag — enabling scoped retrieval across the full enterprise knowledge graph.