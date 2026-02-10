import streamlit as st
import os
import re
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# --- 1. UI Configuration & RTL Logic ---
st.set_page_config(page_title="Baseera | بصيرة", page_icon="🌙")

def is_arabic(text):
    """Checks if the text contains Arabic characters to toggle RTL."""
    return bool(re.search(r'[\u0600-\u06FF]', text))

def display_message(role, content):
    """Wraps message in a div to force RTL if Arabic is detected."""
    direction = "rtl" if is_arabic(content) else "ltr"
    align = "right" if direction == "rtl" else "left"
    
    st.markdown(f"""
        <div dir="{direction}" style="text-align: {align};">
            {content}
        </div>
    """, unsafe_allow_html=True)

# --- 2. Load Knowledge (RAG) ---
@st.cache_resource
def load_knowledge():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vector_store.as_retriever(search_kwargs={"k": 3})

try:
    retriever = load_knowledge()
except Exception:
    st.error("Vector store missing. Please run 'python ingest.py' first.")
    st.stop()

# --- 3. Model Setup with Repetition & Stop Fixes ---
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1,        # Keeps responses grounded and predictable
    repetition_penalty=1.1, # Prevents loops without causing character corruption
    return_full_text=False, # Prevents repeating the user's prompt
    stop_sequences=["<|eot_id|>", "<|end_of_text|>", "User:"], # Forced stops
    huggingfacehub_api_token=hf_token
)
chat_model = ChatHuggingFace(llm=llm, token=hf_token)

# --- 4. System Prompt ---
SYSTEM_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are 'Baseera' (بصيرة), an expert AI for drug awareness in Saudi Arabia.

CORE LANGUAGE RULE:
- IF the user's message is in ENGLISH, you MUST respond ONLY in English.
- IF the user's message is in ARABIC, you MUST respond ONLY in Arabic.
- Never mix languages in a single response unless citing a specific English source name.

INSTRUCTIONS:
1. CITATION: Start with "Based on [Source Name]..." (English) or "بناءً على [Source Name]..." (Arabic).
2. TONE: Professional and empathetic.
3. CALL TO ACTION: Always mention the 937 Saudi Health hotline.

EXAMPLE:
User: "I need help with addiction."
Assistant: "Based on Treatment Protocols... I understand this is difficult. You can find help by calling 937..."
<|eot_id|>"""

# --- 5. UI Logic ---
st.title("Baseera AI (Llama 3.1)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg.type):
        display_message(msg.type, msg.content)

if prompt := st.chat_input("How can Baseera help?"):
    # User message
    with st.chat_message("human"):
        display_message("human", prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    # RAG Retrieval
    docs = retriever.invoke(prompt)
    context_str = ""
    source_map = {}
    for doc in docs:
        name = os.path.basename(doc.metadata.get('source', 'Document'))
        source_map[name] = doc.metadata.get('source')
        context_str += f"\n---\nSOURCE: {name}\nCONTENT: {doc.page_content}\n"
    
    # Build Messages
    messages = [
        SystemMessage(content=SYSTEM_PROMPT + f"\n\nCONTEXT:\n{context_str}"),
        HumanMessage(content=prompt)
    ]

    # Assistant message
    with st.chat_message("ai"):
        response = chat_model.invoke(messages)
        # Clean up any residual special tokens that might leak
        clean_content = response.content.replace("<|eot_id|>", "").strip()
        display_message("ai", clean_content)
        
        st.markdown("---")
        with st.expander("📚 Reference Sources"):
            for name, path in source_map.items():
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        st.download_button(
                            label=f"📥 {name}", 
                            data=f, 
                            file_name=name, 
                            key=f"{name}_{len(st.session_state.messages)}"
                        )

    st.session_state.messages.append(AIMessage(content=clean_content))