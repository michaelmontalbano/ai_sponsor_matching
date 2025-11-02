from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

# Path to your PDF directory
pdf_dir = "../data/papers/files"

# Load and split PDFs
docs = []
for root, _, files in os.walk(pdf_dir):
    for file in files:
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(root, file))
            docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Use a free, local embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Build FAISS vector index (local + free)
vectorstore = FAISS.from_documents(splits, embedding_model)

# Save index
vectorstore.save_local("../data/papers_index")
print("✅ RAG index built and saved locally.")
