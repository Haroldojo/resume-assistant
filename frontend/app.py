import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys

# ---------------------------
# Backend import fix
# ---------------------------
# Add backend folder to path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.append(backend_path)

# Now import your RAG function
from rag_groq import rag_query_refined

# ---------------------------
# Load environment
# ---------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# ---------------------------
# Load resume text
# ---------------------------
TXT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "resume.txt"))
with open(TXT_PATH, "r", encoding="utf-8") as f:
    resume_text = f.read()

# Split resume into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=40,
    length_function=len
)
chunks = text_splitter.split_text(resume_text)

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(page_title="Resume Assistant", layout="wide")

# ---------------------------
# Custom CSS for attractive Figma-style UI
# ---------------------------
st.markdown(
    """
    <style>
    .css-1d391kg {padding-top: 1rem;}
    .stButton>button {background-color:#4A90E2; color:white; border-radius:8px; padding:0.5rem 1rem;}
    .stButton>button:hover {background-color:#357ABD; color:white;}
    .stTextInput>div>div>input {border-radius:8px; padding:0.5rem;}
    .chat-box {background-color:#F5F7FA; padding:1rem; border-radius:10px; max-height:400px; overflow-y:auto;}
    .user-msg {background-color:#D1E8FF; padding:0.5rem 1rem; border-radius:10px; margin-bottom:0.5rem;}
    .bot-msg {background-color:#E2F0D9; padding:0.5rem 1rem; border-radius:10px; margin-bottom:0.5rem;}
    .header {background-color:#4A90E2; color:white; padding:1rem; border-radius:10px; text-align:center; font-size:1.5rem;}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Header
# ---------------------------
st.markdown('<div class="header">📄 Parul Verma - Resume Assistant</div>', unsafe_allow_html=True)

# ---------------------------
# Columns layout
# ---------------------------
left_col, right_col = st.columns([1,3])

# ---------------------------
# Left panel: Buttons
# ---------------------------
with left_col:
    st.markdown("### ⚡ Actions")

    # Download Resume button
    resume_pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "Parul_resume.pdf"))
    if os.path.exists(resume_pdf_path):
        with open(resume_pdf_path, "rb") as f:
            resume_bytes = f.read()
        st.download_button(
            label="📥 Download Resume",
            data=resume_bytes,
            file_name="Parul_resume.pdf",
            mime="application/pdf"
        )
    else:
        st.button("📥 Download Resume (File missing)", disabled=True)

    # Summarize Resume button (disabled for now)
    st.button("📝 Summarize Resume", disabled=True)

# ---------------------------
# Right panel: Chat
# ---------------------------
with right_col:
    st.markdown("### 💬 Ask questions about your resume")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    query = st.text_input("Type your question here:")

    if query:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": query})

        # Get answer from refined RAG
        with st.spinner("Searching your resume..."):
            answer = rag_query_refined(query, chunks, top_k=3)
        st.session_state.messages.append({"role": "bot", "content": answer})

    # Display chat in reverse order: newest on top
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(f'<div class="user-msg">**You:** {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-msg">**Assistant:** {msg["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
