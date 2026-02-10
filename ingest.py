import os
import re
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker

def clean_text(text):
    """
    Strips non-standard characters while preserving Arabic/English 
    and standard punctuation to prevent model confusion.
    """
    return re.sub(r'[^\w\s\u0600-\u06FF.,!?;:]', '', text)

def build_vector_store():
    # 1. Create data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Please place your Saudi MOH PDFs in the 'data/' folder.")
        return

    # 2. Load documents
    print("Loading documents from /data...")
    loader = PyPDFDirectoryLoader("data/")
    raw_documents = loader.load()
    
    if not raw_documents:
        print("Error: No PDFs found in the 'data/' folder.")
        return

    # 3. Clean text to prevent character corruption
    for doc in raw_documents:
        doc.page_content = clean_text(doc.page_content)

    # 4. Semantic Chunking
    # Breaks text where meaning changes, rather than a fixed character count.
    print("Initializing Semantic Chunker...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # We use a percentile-based threshold to find breaks in meaning
    text_splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile"
    )
    
    print("Splitting documents semantically...")
    chunks = text_splitter.split_documents(raw_documents)

    # 5. Create and save the FAISS index
    print(f"Creating vector store with {len(chunks)} semantic chunks...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index")
    
    print("🚀 Success: Semantic vector store saved to 'faiss_index/'.")

if __name__ == "__main__":
    build_vector_store()