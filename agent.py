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
# domain_knowledge.txt contains the H-E-B retail knowledge base

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