import streamlit as st
from pipeline.loader import load_pdf_with_metadata,get_total_pages,get_first_page_text
from pipeline.splitter import split_documents
from pipeline.embeddings import create_vectorstore
from pipeline.retriever import get_relevant_chunks
from llm.generator import genrate_answer
import os
import shutil
import random
import base64

# ── Load external CSS from assets/style.css ─────────────
def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Convert local image to base64 for use in HTML ───────
# Streamlit cannot serve local files in raw HTML, so we encode them
def get_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ── Page Configuration ───────────────────────────────────
# Sets the browser tab title, icon, and layout
st.set_page_config(page_title="DocChat", page_icon="assets/logo.png", layout="wide")
load_css()

# Load Font Awesome for icons throughout the app
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css"/>', unsafe_allow_html=True)

# ── Main Header ──────────────────────────────────────────
# Displays the app logo and title at the top of the main area
# Logo is base64 encoded because Streamlit can't serve local files in HTML
logo_b64 = get_image_base64("assets/logo.png")
st.markdown(f"""
<div class="main-header">
    <img src="data:image/png;base64,{logo_b64}" width="90" style="border-radius:8px; margin-right:0px;"/>
    <span class="header-title">DocChat</span>
    <span class="header-sub">AI Document Assistant</span>
</div>
""", unsafe_allow_html=True)

# ── Session State Initialization ─────────────────────────
# Session state persists values across Streamlit reruns
# Without this, all values reset every time the page refreshes
if "messages" not in st.session_state:
    st.session_state["messages"] = []          # stores full chat history

if "total_pages" not in st.session_state:
    st.session_state["toal_pages"] = 0

if "first_page" not in st.session_state:
    st.session_state["first_page"] = ""

if "pdf_processed" not in st.session_state:
    st.session_state["pdf_processed"] = False  # tracks if a PDF has been indexed

if "total_questions" not in st.session_state:
    st.session_state["total_questions"] = 0    # counts questions asked this session

if "pdf_name" not in st.session_state:
    st.session_state["pdf_name"] = "None"      # stores the uploaded PDF filename

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:

    # ── Dashboard Section ────────────────────────────────────
    # Shows a quick overview of the current session at a glance
    st.markdown("""
        <p class="dashboard-title"><i class="fa-solid fa-gauge"></i> Dashboard</p>
        <p class="doc-subtitle">At a glance summary of your document session</p>
        """, unsafe_allow_html=True)

# Three stat cards: document name, questions asked, status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="stat-card">
        <div class="stat-label"><i class="fa-solid fa-file-pdf"></i> Document</div>
        <div class="stat-value" style="font-size:1em">{st.session_state["pdf_name"]}</div>
    </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""<div class="stat-card">
        <div class="stat-label"><i class="fa-solid fa-message"></i> Questions Asked</div>
        <div class="stat-value">{st.session_state["total_questions"]}</div>
    </div>""", unsafe_allow_html=True)

    with col3:
    # Status only shows Ready after PDF is fully processed, not just uploaded
        if st.session_state["pdf_processed"]:
            status = '<i class="fa-solid fa-circle-check" style="color:#4caf50"></i> Ready'
        else:
            status = '<i class="fa-solid fa-clock" style="color:#f5a623"></i> Waiting...'
        st.markdown(f"""<div class="stat-card">
        <div class="stat-label"><i class="fa-solid fa-signal"></i> Status</div>
        <div class="stat-value" style="font-size:1em">{status}</div>
    </div>""", unsafe_allow_html=True)

    st.divider()

    # PDF upload widget — accepts only PDF files
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        # Save the uploaded file temporarily to disk for processing
        st.session_state["pdf_name"] = uploaded_file.name
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Process PDF button — triggers the full RAG indexing pipeline
        if st.button("Process PDF", key="process_btn"):
            # Only re-embed if vectorstore doesn't already exist
            # This avoids hitting API rate limits unnecessarily
            if not os.path.exists("vectorstore/faiss_index/index.faiss"):
                progress = st.progress(0, text="Reading PDF...")
                pages = load_pdf_with_metadata("temp.pdf")

                progress.progress(15, text="Splitting into chunks...")
                chunks = split_documents(pages)

                progress.progress(40, text="Creating embeddings... (this may take a few minutes)")
                create_vectorstore(chunks)

                progress.progress(100, text="Done!")
                progress.empty()

            # Mark PDF as processed and reset chat history
            st.session_state["total_pages"] = get_total_pages("temp.pdf")
            st.session_state["first_page"] = get_first_page_text("temp.pdf")
            st.session_state["pdf_processed"] = True
            st.session_state["messages"] = []
            st.session_state["total_questions"] = 0
            st.success("Ready to chat!")

    st.divider()

    # Status badge — shows whether a document is loaded and ready
    if st.session_state["pdf_processed"]:
        st.markdown('<span class="badge-ready"><i class="fa-solid fa-circle-check"></i> Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-empty"><i class="fa-solid fa-circle-xmark"></i> No Document</span>', unsafe_allow_html=True)

    st.divider()
    st.markdown("<br>", unsafe_allow_html=True)

    # Action buttons — Clear resets chat only, New PDF wipes everything
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", key="clear_btn"):
            # Only clears messages, keeps the vectorstore intact
            st.session_state["messages"] = []
            st.rerun()
    with col2:
        if st.button("New PDF", key="new_pdf_btn"):
            # Deletes the vectorstore so a new PDF can be indexed
            if os.path.exists("vectorstore"):
                shutil.rmtree("vectorstore")
            st.session_state["pdf_processed"] = False
            st.session_state["messages"] = []
            st.session_state["pdf_name"] = "None"
            st.session_state["total_questions"] = 0
            st.rerun()

    st.divider()

    # About section — explains what DocChat does and how to use it
    st.markdown("""
    <p class="section-label"><i class="fa-solid fa-circle-info"></i> About DocChat</p>
    <p style="font-size:0.83em; color:#aaa; line-height:1.6;">
        DocChat is an AI-powered document assistant that lets you have a conversation with any PDF.
    </p>

    <p class="section-label" style="margin-top:12px;"><i class="fa-solid fa-list-check"></i> How to use</p>
    <p style="font-size:0.83em; color:#aaa; line-height:1.8;">
        <i class="fa-solid fa-arrow-up-from-bracket" style="color:#f5a623; width:16px;"></i> <b>Upload</b> — Drop any PDF above<br>
        <i class="fa-solid fa-bolt" style="color:#f5a623; width:16px;"></i> <b>Process</b> — Click Process PDF<br>
        <i class="fa-solid fa-comment" style="color:#f5a623; width:16px;"></i> <b>Ask</b> — Type in the chat box
    </p>

    <p class="section-label" style="margin-top:12px;"><i class="fa-solid fa-lightbulb"></i> What you can ask</p>
    <p style="font-size:0.83em; color:#aaa; line-height:1.8;">
        <i class="fa-solid fa-chevron-right" style="color:#f5a623; width:16px;"></i> Summarize a chapter<br>
        <i class="fa-solid fa-chevron-right" style="color:#f5a623; width:16px;"></i> Find specific information<br>
        <i class="fa-solid fa-chevron-right" style="color:#f5a623; width:16px;"></i> Ask follow up questions<br>
        <i class="fa-solid fa-chevron-right" style="color:#f5a623; width:16px;"></i> Compare ideas across sections
    </p>

    <p class="section-label" style="margin-top:12px;"><i class="fa-solid fa-star"></i> Tips</p>
    <p style="font-size:0.83em; color:#aaa; line-height:1.8;">
        <i class="fa-solid fa-check" style="color:#f5a623; width:16px;"></i> Be specific for better answers<br>
        <i class="fa-solid fa-check" style="color:#f5a623; width:16px;"></i> Check sources to verify<br>
        <i class="fa-solid fa-check" style="color:#f5a623; width:16px;"></i> Clear resets chat only
    </p>
    """, unsafe_allow_html=True)


# ── Chat Section ─────────────────────────────────────────
st.markdown("""
<p class="dashboard-title"><i class="fa-solid fa-comments"></i> Chat</p>
""", unsafe_allow_html=True)

# Replay all previous messages from session state so chat history is visible
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        # Show source chunks used to generate the answer
        if "sources" in message:
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.markdown(f"""<div class="source-box">
                        <i class="fa-solid fa-file-lines" style="color:#f5a623"></i>
                        <strong> Page {source['page']}</strong><br>
                        {source['chunk'][:250]}...
                    </div>""", unsafe_allow_html=True)

# ── Chat Input ───────────────────────────────────────────
# st.chat_input always stays pinned to the bottom of the page
question = st.chat_input("Ask anything about your document...")

if question:
    if not st.session_state["pdf_processed"]:
        st.warning("Please upload and process a PDF first!")
    else:
        # Display and store the user's message
        with st.chat_message("user"):
            st.write(question)
        st.session_state["messages"].append({"role": "user", "content": question})

        # Generate and display the assistant's answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Retrieve relevant chunks from vectorstore
                chunk_dicts = get_relevant_chunks(question)
                chunk_texts = [c["content"] for c in chunk_dicts]
                # Send chunks + question + chat history to LLM
                answer = genrate_answer(
    question,
    chunk_texts,
    st.session_state["messages"],
    st.session_state.get("total_pages", 0),
    st.session_state.get("first_page", "")
)
            st.write(answer)

            # Show which document chunks were used to answer
            sources = [{"page": c["page"], "chunk": c["content"]} for c in chunk_dicts]
            with st.expander("View Sources"):
                for source in sources:
                    st.markdown(f"""<div class="source-box">
                        <i class="fa-solid fa-file-lines" style="color:#f5a623"></i>
                        <strong> Page {source['page']}</strong><br>
                        {source['chunk'][:250]}...
                    </div>""", unsafe_allow_html=True)

        # Store assistant response + sources in session history
        st.session_state["messages"].append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
        st.session_state["total_questions"] += 1
        st.rerun()

# ── Pet Corner ───────────────────────────────────────────
# Animated cat GIF in the bottom right corner
# Shows a random helpful tip in a speech bubble
tips = [
    "Ask me anything!",
    "Try summarizing a chapter!",
    "Check the sources below!",
    "Upload any PDF",
]
tip = random.choice(tips)
st.markdown(f"""
<div class="pet-corner">
    <div class="pet-bubble">{tip}</div>
    <img class="pet-gif" src="https://media.tenor.com/RJbs_zZBqs0AAAAi/cute-cat.gif" alt="pet"/>
</div>
""", unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────
# Fixed footer at the bottom of every page
st.markdown("""
<div class="footer">
    <span><i class="fa-solid fa-file-lines"></i> DocChat — AI Document Assistant</span>
    <span><i class="fa-solid fa-bolt"></i> Powered by Groq & Gemini</span>
</div>
""", unsafe_allow_html=True)