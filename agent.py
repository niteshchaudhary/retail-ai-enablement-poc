from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "retail-ai-enablement-poc"

# Step 1: Retail domain document
sample_doc = """
H-E-B is a large retail grocery chain operating 420+ stores in Texas and Mexico.

Product Catalog Management:
H-E-B manages over 50,000 SKUs across fresh produce, packaged goods, and private label products.
Products are categorized by department, aisle, and shelf placement.
Pricing is updated daily based on competitor analysis and demand forecasting models.

Demand Forecasting:
H-E-B uses ML models to forecast demand at store and SKU level.
Forecasting models are retrained weekly using point-of-sale data.
Accurate forecasting reduces food waste and improves on-shelf availability.

Inventory Management:
Inventory levels are tracked in real time across all stores and distribution centers.
Automated replenishment triggers are based on reorder point models.
Out-of-stock events are flagged and escalated to store managers automatically.

Customer Personalization:
H-E-B uses purchase history and browsing behavior to personalize promotions.
Personalization models are updated nightly and served via the H-E-B app and website.
Customer segments are defined by purchase frequency, basket size, and brand affinity.

AI Enablement at H-E-B:
MLOps pipelines manage the lifecycle of demand forecasting and personalization models.
LLMOps frameworks support internal chatbots and customer facing AI assistants.
RAG systems are used to ground AI responses in H-E-B product and policy data.
LangSmith is used for LLM observability and tracing across all AI pipelines.
"""

with open("domain_knowledge.txt", "w") as f:
    f.write(sample_doc)

# Step 2: Load and chunk
loader = TextLoader("domain_knowledge.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(documents)

# Step 3: Vector store
print("Building vector store...")
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

# Step 4: RAG chain with "I don't know" instruction
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

# Step 5: Interactive input
print("\nRetail AI Assistant ready. Type 'exit' to quit.\n")
while True:
    question = input("Ask a question: ")
    if question.lower() == "exit":
        break
    answer = chain.invoke(question)
    print(f"\nA: {answer}\n")