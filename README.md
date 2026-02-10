# Baseera (بصيرة) 🌙

**An AI-Powered Support Companion for Drug Awareness in Saudi Arabia.**

Baseera is a Retrieval-Augmented Generation (RAG) system built to provide empathetic, evidence-based guidance grounded in official Saudi Ministry of Health (MOH) and NCNC protocols. By utilizing open-source LLMs, this project demonstrates strong engineering practices in **local data privacy**, **open-source AI orchestration**, and **grounded response generation**.

---

## 🏗️ System Architecture

The following diagram illustrates the upgraded RAG lifecycle using Semantic Chunking:

```mermaid
graph TD
    A[PDF Data Sources] -->|PyPDFDirectoryLoader| B[ingest.py]
    B -->|SemanticChunker| C{Vector Store}
    C -->|HuggingFaceEmbeddings| D[FAISS Index]

    E[User Query] -->|Streamlit UI| F[app.py]
    F -->|Similarity Search| D
    D -->|Relevant Context| G[Llama 3.1 8B Instruct]
    G -->|Grounded Response| H[User]
```

---

## 🛠️ Tech Stack

- **LLM:** Llama 3.1 8B Instruct (via Hugging Face Inference API)
- **Orchestration:** LangChain (Community & Partner Packages)
- **Vector Database:** FAISS (Local)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **UI:** Streamlit
- **Data Sources:** Official MOH & NCNC PDFs

---

## 🚀 Getting Started

### 1️⃣ Prerequisites

Create a `.env` file in the project root and add your Hugging Face token:

```env
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

Install the experimental package required for semantic processing:

```bash
pip install langchain-experimental
```

---

### 2️⃣ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/baseera-ai.git
cd baseera-ai

# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
# source venv/bin/activate    # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

---

### 3️⃣ Data Ingestion (Upgraded RAG Pipeline)

Before running the application, process the PDF documents inside the `data/` folder to generate the local FAISS vector store:

```bash
python ingest.py
```

What makes this version (v1.0.0) different?

- **Intelligence:** It no longer splits text every 600 characters. Instead, it uses HuggingFaceEmbeddings to determine when a topic or meaning changes, keeping medical protocols intact.
- **Precision:** By ensuring sentences aren't cut in half, the Llama 3.1 model has cleaner context to work with, significantly reducing the "ugly" character hallucinations seen in earlier versions.

---

### 4️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 🧠 Key Features

- **Semantic Document Processing:**
  Utilizes AI-driven chunking to preserve the integrity of Arabic medical guidelines, ensuring retrieval occurs at the "thought" level rather than the "character" level.

- **Grounded Citations:**
  Every response begins with _"Based on [Source]…"_ to ensure transparency and trust.

- **Localized RTL Support:**
  Dynamic UI rendering that automatically detects Arabic text and aligns it Right-to-Left (RTL) for a native user experience.

- **Automatic Directory Ingestion:**
  Any new PDF added to the `data/` folder is picked up during the next ingestion run.

- **System Prompt Engineering:**
  Specialized instructions ensure:
  - Empathetic, non-judgmental tone
  - Medical disclaimers
  - Cultural and regional sensitivity

- **Local Vector Store:**
  No external database required — all embeddings are stored locally.

---

## 📊 RAG Retrieval Logic

```mermaid
sequenceDiagram
    participant U as User
    participant S as Streamlit App
    participant V as FAISS Index
    participant L as Llama 3.1

    U->>S: Asks about drug prevention
    S->>V: Search for relevant document chunks
    V-->>S: Return top-k matching segments
    S->>L: Context + System Prompt + Query
    L-->>S: Grounded response with citations
    S-->>U: Final Answer
```

---

## 🛠️ Future Improvements & Roadmap

While Baseera is a functional RAG MVP, the following enhancements would elevate it to a production-grade system, particularly for the Saudi market.

---

### 1️⃣ Enhanced Arabic Linguistic Processing

- **Semantic Chunking for Arabic**  
  Transition from character-based splitting to semantic chunking using embeddings trained on Arabic (e.g., AraBERT, MARBERT) to prevent medical or legal concepts from being split mid-sentence.

- **Morphological Analysis**  
  Integrate Arabic NLP tools such as **CAMeL Tools** for lemmatization and diacritic normalization, improving retrieval accuracy across different word forms.

---

### 2️⃣ Saudi Dialect (Ammiyah) Understanding

- **Dialect-Aware Prompting**  
  Enhance the system prompt to better interpret Saudi dialects (Najdi, Hejazi, Southern, etc.) while maintaining responses in formal Fusha.

- **Substance Slang Recognition**  
  Improve understanding of local slang related to substances without reproducing it in responses, preserving professionalism and safety.

---

### 3️⃣ Advanced RAG Techniques

- **Hybrid Retrieval (BM25 + Vector Search)**  
  Combine keyword-based search with vector similarity to ensure high-precision retrieval for medical terms, legal references, and protocol identifiers.

- **Reranking Layer**  
  Introduce a cross-encoder reranker (e.g., **BGE-Reranker**) to re-score the top-k FAISS results and ensure the most medically and culturally relevant context is selected.

---

### 4️⃣ System Evaluation & Quality Assurance (RAGAS)

- **Automated RAG Metrics**  
  Integrate the **RAGAS** framework to continuously evaluate system quality using metrics such as:
  - **Faithfulness:** Is the response grounded in the source PDFs?
  - **Answer Relevance:** Does it directly address the user’s concern?
  - **Context Precision:** Is the retrieved context truly useful?

- **Regression Testing for Prompts**  
  Track performance changes when updating prompts, models, or retrieval strategies.

---


## ⚠️ Disclaimer

Baseera is an **educational and awareness tool**.
It does **not** replace professional medical, psychological, or legal advice.
For emergencies or medical decisions, always consult licensed professionals or official Saudi health authorities.

---

## 📌 License

This project is released for educational and research purposes.
