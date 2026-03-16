# Retail AI Enablement POC

A Retrieval Augmented Generation (RAG) pipeline built to demonstrate core AI enablement 
concepts in a retail context, using local LLM inference and LLM observability.

## What This Demonstrates
- **RAG Pipeline**: domain-grounded LLM responses using vector search
- **LLM Observability**:  every query and response traced via LangSmith
- **AI Governance**:  LLM restricted to answer only from approved domain data
- **Local LLM Inference**:  runs on Ollama (llama3.2), no external API dependency

## Tech Stack
- **LLM**: llama3.2 via Ollama (local inference)
- **Vector Database**: Chroma
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Orchestration**: LangChain
- **Observability**: LangSmith

## Architecture
User Query ->  Chroma Vector Search -> Relevant Chunks Retrieved -> 
LLM answers from context only -> Response traced in LangSmith

## How It Works
1. Retail domain knowledge is loaded and chunked into small pieces
2. Chunks are converted to vectors and stored in Chroma
3. User asks a question interactively
4. Chroma retrieves the most semantically relevant chunks
5. LLM answers based only on retrieved context
6. If answer is outside context, returns "I don't know", preventing hallucination
7. Every step is logged and traceable in LangSmith

## Running Locally
```bash
# Install dependencies
pip install langchain langchain-ollama langchain-community 
langchain-text-splitters chromadb sentence-transformers langsmith

# Set LangSmith env variables
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_langsmith_key
export LANGCHAIN_PROJECT=retail-ai-enablement-poc

# Run
python agent.py
```

## Sample Questions to Try
- "How does HEB handle demand forecasting?"
- "How is customer personalization done?"
- "What is the return policy at HEB?": demonstrates I don't know behavior

## Key AI Enablement Concepts Demonstrated
- **Grounding**: LLM answers only from domain data, not general knowledge
- **Observability**: Full trace of every LLM call visible in LangSmith
- **Fallback strategy**: Controlled "I don't know" response prevents hallucination
- **Vector search**: Semantic similarity search via Chroma, not keyword matching
```
