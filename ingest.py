import os
import re
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def clean_text(text):
    # Removes non-standard binary characters that confuse Llama 3.1
    # Keeps Arabic script, numbers, and standard punctuation
    return re.sub(r'[^\w\s\u0600-\u06FF.,!?;:]', '', text)

def build_vector_store():
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created 'data' folder. Add your Saudi MOH PDFs and run again.")
        return

    # 1. Load PDFs
    loader = PyPDFDirectoryLoader("data/")
    raw_documents = loader.load()
    
    if not raw_documents:
        print("No documents found in 'data/' folder.")
        return

    # 2. Clean Text Content
    for doc in raw_documents:
        doc.page_content = clean_text(doc.page_content)

    # 3. Split into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = text_splitter.split_documents(raw_documents)

    # 4. Create Embeddings & Store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index")
    print(f"Success: Cleaned vector store created with {len(chunks)} chunks.")

if __name__ == "__main__":
    build_vector_store()