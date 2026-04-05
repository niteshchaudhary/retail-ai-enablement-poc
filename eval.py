from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client
import os
import uuid

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "retail-ai-enablement-poc"

# Ground truth dataset
# These are your known correct answers - this is "ground truth construction"
eval_dataset = [
    {
        "question": "How does H-E-B measure forecast accuracy?",
        "expected": "MAPE",
        "category": "demand_forecasting"
    },
    {
        "question": "What is shrink?",
        "expected": "inventory loss due to theft, damage, or expiry",
        "category": "inventory"
    },
    {
        "question": "How often are personalization models updated?",
        "expected": "nightly",
        "category": "personalization"
    },
    {
        "question": "What private label brands does H-E-B have?",
        "expected": "Central Market",
        "category": "product_catalog"
    },
    {
        "question": "What is the return policy at H-E-B?",
        "expected": "I don't know",
        "category": "out_of_scope"
    },
    {
        "question": "How does H-E-B handle supply chain disruptions?",
        "expected": "alternative sourcing",
        "category": "supply_chain"
    },
    {
        "question": "What is H-E-B's annual sales?",
        "expected": "48 billion",
        "category": "company_overview"
    },
]

# Build RAG pipeline
print("Building vector store...")
loader = TextLoader("domain_knowledge.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(documents)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

llm = ChatOllama(model="llama3.2", temperature=0)
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context.
If the answer is not in the context, say "I don't know based on the available information."

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

# Run evaluation
print("\nRunning evaluation...\n")
results = []
passed = 0
failed = 0

for item in eval_dataset:
    question = item["question"]
    expected = item["expected"]
    category = item["category"]

    answer = chain.invoke(question)
    correct = expected.lower() in answer.lower()

    status = "PASS" if correct else "FAIL"
    if correct:
        passed += 1
    else:
        failed += 1

    results.append({
        "question": question,
        "expected": expected,
        "actual": answer,
        "status": status,
        "category": category
    })

    print(f"[{status}] {category}: {question}")
    print(f"       Expected: {expected}")
    print(f"       Got: {answer[:100]}...")
    print()

# Summary
print("=" * 50)
print(f"EVALUATION SUMMARY")
print("=" * 50)
print(f"Total: {len(eval_dataset)}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print(f"Accuracy: {round(passed/len(eval_dataset)*100, 1)}%")
