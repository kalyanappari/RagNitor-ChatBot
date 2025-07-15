from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import tempfile
import validators
import json
import os
import uuid

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_core.messages import HumanMessage, AIMessage

from ui import (
    inject_custom_css, render_chat_history, user_input_box,
    new_chat_button, render_chat_sessions, create_new_chat_session,
    clean_title_text
)

# --- User Identification Setup ---
if "user_id" not in st.session_state:
    if "user_id" in st.experimental_get_query_params():
        st.session_state.user_id = st.experimental_get_query_params()["user_id"][0]
    else:
        st.session_state.user_id = str(uuid.uuid4())
        st.experimental_set_query_params(user_id=st.session_state.user_id)

USER_ID = st.session_state.user_id

# --- Config ---
st.set_page_config(page_title="🧠 RAGnitor", layout="wide")
inject_custom_css()

st.markdown("""
    <h1 style='text-align: center; font-size: 40px; font-weight: 800; margin-top: 0;'>
        <span style='color: #FF6B00;'>RAG</span><span style='color: inherit;'>nitor</span>
    </h1>
""", unsafe_allow_html=True)

SAVE_DIR = f"chat_histories/{USER_ID}"
os.makedirs(SAVE_DIR, exist_ok=True)

def load_chat_history(session_id, k=10):
    file_path = os.path.join(SAVE_DIR, f"{session_id}.json")
    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=k)
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for msg in data.get("messages", []):
                    if msg["type"] == "human":
                        memory.chat_memory.add_user_message(msg["content"])
                    elif msg["type"] == "ai":
                        memory.chat_memory.add_ai_message(msg["content"])
        except Exception as e:
            st.sidebar.error(f"Error loading chat {session_id}: {e}")
    return memory

def save_chat_history(session_id, memory):
    file_path = os.path.join(SAVE_DIR, f"{session_id}.json")
    messages_to_save = [
        {"type": msg.type, "content": msg.content} for msg in memory.chat_memory.messages
    ]
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({"messages": messages_to_save}, f, ensure_ascii=False, indent=4)
    except Exception as e:
        st.sidebar.error(f"Error saving chat {session_id}: {e}")

for key in ["chat_sessions", "active_session", "memory_store", "vectorstore", "retriever", "rag_chain", "show_tools"]:
    if key not in st.session_state:
        st.session_state[key] = {} if "store" in key or "sessions" in key else None

if not st.session_state.chat_sessions:
    for file_name in os.listdir(SAVE_DIR):
        if file_name.endswith(".json"):
            session_id = file_name.replace(".json", "")
            try:
                with open(os.path.join(SAVE_DIR, file_name), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    first_message_content = ""
                    for msg in data.get("messages", []):
                        if msg["type"] == "ai" and msg["content"].strip():
                            first_message_content = msg["content"]
                            break
                    title = clean_title_text(first_message_content) if first_message_content else f"Chat {session_id[:8]}"
                    if not title:
                        title = f"Chat {session_id[:8]}"
            except Exception:
                title = f"Chat {session_id[:8]}"
            st.session_state.chat_sessions[session_id] = title

if new_chat_button():
    if st.session_state.active_session and st.session_state.active_session in st.session_state.memory_store:
        save_chat_history(st.session_state.active_session, st.session_state.memory_store[st.session_state.active_session])
    new_id = create_new_chat_session("New Chat")
    st.session_state.active_session = new_id
    st.session_state.memory_store[new_id] = load_chat_history(new_id, k=10)
    st.session_state.vectorstore = None
    st.session_state.retriever = None
    st.session_state.rag_chain = None
    st.rerun()

selected_session = render_chat_sessions()
if selected_session and selected_session != st.session_state.active_session:
    if st.session_state.active_session and st.session_state.active_session in st.session_state.memory_store:
        save_chat_history(st.session_state.active_session, st.session_state.memory_store[st.session_state.active_session])
    st.session_state.active_session = selected_session
    st.session_state.memory_store[selected_session] = load_chat_history(selected_session, k=10)
    st.session_state.vectorstore = None
    st.session_state.retriever = None
    st.session_state.rag_chain = None
    st.rerun()

if not st.session_state.active_session and st.session_state.chat_sessions:
    st.session_state.active_session = next(iter(st.session_state.chat_sessions.keys()))
    st.session_state.memory_store[st.session_state.active_session] = load_chat_history(st.session_state.active_session, k=10)
elif not st.session_state.active_session:
    default_id = create_new_chat_session("New Chat")
    st.session_state.active_session = default_id
    st.session_state.memory_store[default_id] = load_chat_history(default_id, k=10)

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-8b-8192"
)

active_id = st.session_state.active_session
memory = st.session_state.memory_store[active_id]

render_chat_history(memory)

if memory and memory.chat_memory.messages:
    export_text = ""
    for msg in memory.chat_memory.messages:
        sender = "You" if msg.type == "human" else "Bot"
        export_text += f"{sender}: {msg.content.strip()}\n\n"

    st.download_button(
        label="📤 Export Chat",
        data=export_text.encode("utf-8", errors="ignore"),
        file_name="chat_export.txt",
        mime="text/plain"
    )

user_input, uploaded_file, url_input = user_input_box()

docs = []
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        loader = PyPDFLoader(tmp.name) if uploaded_file.type == "application/pdf" else TextLoader(tmp.name)
        docs = loader.load()
elif url_input and validators.url(url_input):
    try:
        loader = WebBaseLoader(url_input, requests_kwargs={'timeout': 15})
        raw_docs = loader.load()
        if raw_docs and raw_docs[0].page_content.strip():
            raw_docs[0].page_content = " ".join(raw_docs[0].page_content.split()[:1000])
            docs = raw_docs
            st.success("✅ Webpage loaded successfully!")
        else:
            st.warning("⚠️ No content found at the URL.")
    except Exception as e:
        st.error(f"❌ Error loading web URL: {e}")

if docs:
    chunks = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60).split_documents(docs)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
    st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 8})

    qa_prompt_template = """
You are a helpful AI assistant. Answer the question based on the provided context or your general knowledge.
If the context does not contain enough information to directly answer the question, then use your general knowledge to answer.

Context:
{context}

Question: {question}
Assistant:
"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(qa_prompt_template)

    st.session_state.rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=st.session_state.retriever,
        memory=memory,
        chain_type="stuff",
        return_source_documents=False,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

if user_input:
    try:
        with st.spinner("🧠 Bot is thinking..."):
            retriever = st.session_state.retriever
            fallback_prompt = PromptTemplate(
                input_variables=["question", "chat_history"],
                template="""
You are a helpful AI assistant. Continue the conversation based on the context below.

Previous Conversation:
{chat_history}

User: {question}
Assistant:
"""
            )
            fallback_chain = LLMChain(llm=llm, prompt=fallback_prompt, memory=memory)

            if retriever and st.session_state.rag_chain:
                response = st.session_state.rag_chain.run(user_input)
            else:
                response = fallback_chain.run({"question": user_input})

        current_id = st.session_state.active_session
        current_title = st.session_state.chat_sessions.get(current_id, "")
        if current_id and (not current_title or current_title == "New Chat"):
            title = clean_title_text(response.strip().split("\n")[0])
            if title:
                st.session_state.chat_sessions[current_id] = title

        st.session_state.user_input = ""
        st.session_state.responding = False

        save_chat_history(current_id, memory)
        st.rerun()

    except Exception as e:
        st.session_state.responding = False
        st.error(f"❌ Error: {e}")
