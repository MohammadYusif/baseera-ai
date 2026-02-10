import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_vector_store():
    # 1. Load PDFs
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created 'data' folder. Please add PDFs and run again.")
        return

    loader = PyPDFDirectoryLoader("data/")
    documents = loader.load()
    
    if not documents:
        print("No documents found in 'data/' folder.")
        return

    # 2. Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # 3. Embed
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 4. Save
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index")
    print(f"Success: Vector store created with {len(chunks)} chunks.")

if __name__ == "__main__":
    build_vector_store()