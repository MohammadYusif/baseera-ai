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

GUIDELINES:
1. Tone: Empathetic, non-judgmental, and professional.
2. Boundaries: You are an AI, not a doctor. Always include a disclaimer if medical help is needed.
3. Local Context: Mention Saudi resources like the 937 hotline or 'Kafa' when relevant.
4. Language: If the user speaks Arabic, respond in clear, supportive Arabic.

If the answer is not in the context, say: 'I don't have enough information on that, but you can contact 937 for professional help.'
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
    st.chat_message("human").markdown(prompt)
    
    # 1. Retrieve context
    docs = retriever.invoke(prompt)
    context = "\n".join([doc.page_content for doc in docs])
    
    # 2. Build the full prompt with System Message + Context
    messages = [
        SystemMessage(content=SYSTEM_PROMPT + f"\n\nCONTEXT:\n{context}"),
        HumanMessage(content=prompt)
    ]

    with st.chat_message("ai"):
        response = chat_model.invoke(messages)
        st.markdown(response.content)
    
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.session_state.messages.append(response)