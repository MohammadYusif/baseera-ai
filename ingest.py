import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

SYSTEM_PROMPT = """
You are Baseera (بصيرة), a specialized AI assistant for drug awareness in Saudi Arabia. 
Your goal is to provide supportive, evidence-based guidance based ONLY on the provided context.

CRITICAL INSTRUCTION:
For every answer, you must start your response with "Based on [Source Name]..." 
explicitly citing the document provided in the context.

GUIDELINES:
1. Tone: Empathetic, non-judgmental, and professional.
2. Boundaries: Always include a disclaimer if medical help is needed (937 hotline).
3. Precision: Only answer using the context. If not found, say you don't know.
"""

# --- 1. Load the Knowledge (RAG) ---
@st.cache_resource # Keeps the index in memory for speed
def load_knowledge():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # This loads the index you created in ingest.py
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vector_store.as_retriever(search_kwargs={"k": 3})

retriever = load_knowledge()

# --- 2. Setup the Model ---
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN") # Check your .env variable name
)
chat_model = ChatHuggingFace(llm=llm)

# --- 3. UI Logic ---
st.set_page_config(page_title="Baseera | بصيرة", page_icon="🌙")
st.title("Baseera AI (Llama 3.1)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

if prompt := st.chat_input("How can Baseera help?"):
    # 1. Retrieve the relevant chunks
    docs = retriever.invoke(prompt)
    
    # 2. Format context to include source names for the LLM to see
    context_with_sources = ""
    for doc in docs:
        source_name = os.path.basename(doc.metadata.get('source', 'Unknown Source'))
        context_with_sources += f"\n---\nSOURCE: {source_name}\nCONTENT: {doc.page_content}\n"

    # 3. Build the final messages
    messages = [
        SystemMessage(content=SYSTEM_PROMPT + f"\n\nCONTEXT:\n{context_with_sources}"),
        HumanMessage(content=prompt)
    ]

    with st.chat_message("ai"):
        # 1. Generate the response content
        response = chat_model.invoke(messages)
        st.markdown(response.content)
        
        # 2. Extract unique sources from retrieved documents
        # This deduplicates the filenames so each is only listed once
        unique_sources = list(set([os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in docs]))
        
        # 3. Display the Source List in a clean expander
        with st.expander("📚 View Reference Sources"):
            for source in unique_sources:
                st.write(f"- {source}")
    
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.session_state.messages.append(response)