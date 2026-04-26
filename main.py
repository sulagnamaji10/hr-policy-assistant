# Document loading
from langchain_community.document_loaders import TextLoader

# Text splitting
from langchain_text_splitters import CharacterTextSplitter

# Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Vector store
from langchain_community.vectorstores import FAISS

# LLM (Ollama)
from langchain_community.llms import Ollama

# Chain
from langchain.chains import RetrievalQA

# Load HR policy document
loader = TextLoader("data/hr_policy.txt")
documents = loader.load()

# Split text into chunks
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store in vector database
db = FAISS.from_documents(docs, embeddings)

# Create retriever
retriever = db.as_retriever()

# Load Ollama model
llm = Ollama(model="llama3")

# Create RAG pipeline
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Chat loop
while True:
    query = input("Ask HR question: ")
    if query.lower() == "exit":
        break

    response = qa.run(query)
    print("\nAnswer:", response)