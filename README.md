# Baseera (بصيرة) 🌙

**An AI-Powered Support Companion for Drug Awareness in Saudi Arabia.**

Baseera is a Retrieval-Augmented Generation (RAG) system built to provide empathetic, evidence-based guidance grounded in official Saudi Ministry of Health (MOH) and NCNC protocols. By utilizing open-source LLMs, this project demonstrates high-level engineering skills in local data privacy and AI orchestration.

## 🏗️ System Architecture

The following diagram illustrates the RAG lifecycle implemented in this project:

```mermaid
graph TD
    A[Data/ PDFs] -->|PyPDFDirectoryLoader| B(ingest.py)
    B -->|RecursiveCharacterSplitter| C{Vector Store}
    C -->|HuggingFaceEmbeddings| D[FAISS Index]

    E[User Query] -->|Streamlit UI| F(app.py)
    F -->|Similarity Search| D
    D -->|Relevant Context| G(Llama 3.1 8B Instruct)
    G -->|Grounded Response| H[User]
🛠️ Tech Stack
LLM: Llama 3.1 8B Instruct (via Hugging Face Inference API)

Orchestration: LangChain (Partner Package)

Vector DB: FAISS (Local)

Embeddings: sentence-transformers/all-MiniLM-L6-v2

UI: Streamlit

🚀 Getting Started
1. Prerequisites
Ensure you have a .env file in the root directory with your Hugging Face token:

Code snippet
HUGGINGFACE_API_TOKEN=your_token_here
2. Installation
Bash
# Clone the repo
git clone [https://github.com/your-username/baseera-ai.git](https://github.com/your-username/baseera-ai.git)
cd baseera-ai

# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate  # Or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
3. Data Ingestion (The RAG Pipeline)
Before running the app, you must process the PDF documents in the data/ folder to generate the local vector store:

Bash
python ingest.py
4. Run the Application
Bash
streamlit run app.py
🧠 Features
Grounded Citations: Every response starts with "Based on [Source]..." to ensure transparency and trust.

Directory Loading: Automatically ingests any new PDFs added to the data/ folder.

System Prompt Engineering: Specialized instructions for empathetic, non-judgmental support with clear medical disclaimers.

📊 RAG Retrieval Logic
Code snippet
sequenceDiagram
    participant U as User
    participant S as Streamlit
    participant V as FAISS Index
    participant L as Llama 3.1

    U->>S: Asks about drug prevention
    S->>V: Search for relevant document chunks
    V-->>S: Return top-k matching segments
    S->>L: Context + System Prompt + Query
    L-->>S: Grounded response with citations
    S->>U: Final Answer
```
