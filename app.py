import streamlit as st
import os
import re
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

SYSTEM_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are 'Baseera' (بصيرة), an expert AI for drug awareness in Saudi Arabia.

CORE LANGUAGE RULE:
- Detect the user's language. IF the message is in ENGLISH, you MUST respond ONLY in English.
- IF the user's message is in ARABIC, you MUST respond ONLY in Arabic.
- Never mix languages in a single response unless citing a specific English source name.

STRICT LINGUISTIC RULES (ARABIC):
1. NO LITERAL TRANSLATIONS: Avoid awkward machine-translated phrases like "تعترف بقتلك" or "المخدرة". 
2. VOCABULARY: 
   - Use "تعاطي المواد" or "الإدمان" (Addiction) instead of "المخدرة".
   - Use "التعافي" (Recovery) instead of "الشفاء" (Cure).
   - Use "الاعتراف بالمشكلة" (Acknowledging the problem) instead of "الاعتراف بالقتل".
3. TONE: Respond in formal, supportive Saudi-context Arabic (Fusha). Do not use English words like 'living' in Arabic sentences.

SAFETY GUARDRAILS:
- NEVER provide specific medical dosages or prescriptions.
- If a user asks how to acquire illegal substances, you must firmly refuse and redirect them to the law and the 937 hotline.
- You are an AI assistant, not a doctor. Always include a medical disclaimer when providing health information.
- Maintain professional boundaries and do not encourage or facilitate dangerous behavior.

INSTRUCTIONS:
1. CITATION: Start every response with "Based on [Source Name]..." (English) or "بناءً على [Source Name]..." (Arabic).
2. TONE: Professional, empathetic, and non-judgmental.
3. CALL TO ACTION: Always mention the 937 Saudi Health hotline for immediate medical consultation.

EXAMPLE:
User: "I need help with addiction."
Assistant: "Based on Treatment Protocols... I understand this is a difficult step. You can find confidential support by calling 937..."
<|eot_id|>"""

# --- 1. UI Configuration & RTL Logic ---
st.set_page_config(page_title="Baseera | بصيرة", page_icon="🌙")

def is_arabic(text):
    return bool(re.search(r'[\u0600-\u06FF]', text))

def display_message(role, content):
    direction = "rtl" if is_arabic(content) else "ltr"
    align = "right" if direction == "rtl" else "left"
    st.markdown(f'<div dir="{direction}" style="text-align: {align};">{content}</div>', unsafe_allow_html=True)

# --- 2. Load Knowledge (RAG) ---
@st.cache_resource
def load_knowledge():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vector_store.as_retriever(search_kwargs={"k": 3})

try:
    retriever = load_knowledge()
except Exception:
    st.error("Vector store missing. Run 'python ingest.py' first.")
    st.stop()

# --- 3. Model Setup ---
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1,
    repetition_penalty=1.1,
    return_full_text=False,
    stop_sequences=["<|eot_id|>", "User:"],
    huggingfacehub_api_token=hf_token
)
chat_model = ChatHuggingFace(llm=llm, token=hf_token)

# --- 4. Chat Interface with Memory ---
st.title("Baseera AI (Llama 3.1)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg.type):
        display_message(msg.type, msg.content)

if prompt := st.chat_input("How can Baseera help?"):
    with st.chat_message("human"):
        display_message("human", prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    # RAG Retrieval
    docs = retriever.invoke(prompt)
    context_str = "\n".join([f"SOURCE: {os.path.basename(d.metadata['source'])}\nCONTENT: {d.page_content}" for d in docs])
    
    # BUILD CHAT HISTORY FOR MEMORY
    history_str = "\n".join([f"{'User' if m.type=='human' else 'Assistant'}: {m.content}" for m in st.session_state.messages[-4:]])
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT + f"\n\nCONTEXT FROM DOCUMENTS:\n{context_str}"),
        SystemMessage(content=f"CONVERSATION LOG FOR CONTEXT:\n{history_str}"),
        HumanMessage(content=prompt)
    ]

    with st.chat_message("ai"):
        response = chat_model.invoke(messages)
        clean_content = response.content.replace("<|eot_id|>", "").strip()
        display_message("ai", clean_content)
        
        with st.expander("📚 Reference Sources"):
            sources = {os.path.basename(d.metadata['source']): d.metadata['source'] for d in docs}
            for name, path in sources.items():
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        st.download_button(label=f"📥 {name}", data=f, file_name=name, key=f"{name}_{len(st.session_state.messages)}")

    st.session_state.messages.append(AIMessage(content=clean_content))