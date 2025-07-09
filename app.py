from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import tempfile
import validators
import json
import os

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory # Changed to Window Memory
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_core.messages import HumanMessage, AIMessage # For deserialization

from ui import (
    inject_custom_css, render_chat_history, user_input_box,
    new_chat_button, render_chat_sessions, create_new_chat_session,
    clean_title_text
)

# --- Config ---
st.set_page_config(page_title="🧠 RAGnitor", layout="wide")
inject_custom_css()

st.markdown("""
    <h1 style='text-align: center; font-size: 40px; font-weight: 800; margin-top: 0;'>
        <span style='color: #FF6B00;'>RAG</span><span style='color: inherit;'>nitor</span>
    </h1>
""", unsafe_allow_html=True)

# --- Persistence Setup ---
SAVE_DIR = "chat_histories"
os.makedirs(SAVE_DIR, exist_ok=True)

# Function to load chat history
def load_chat_history(session_id, k=10): # Added k for window memory
    file_path = os.path.join(SAVE_DIR, f"{session_id}.json")
    # Initialize with ConversationBufferWindowMemory
    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=k)
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Langchain messages need to be converted from dict to actual message objects
                for msg in data.get("messages", []):
                    if msg["type"] == "human":
                        memory.chat_memory.add_user_message(msg["content"])
                    elif msg["type"] == "ai":
                        memory.chat_memory.add_ai_message(msg["content"])
            # The window memory automatically handles truncation as messages are added
        except Exception as e:
            st.sidebar.error(f"Error loading chat {session_id}: {e}")
    return memory

# Function to save chat history (remains mostly the same, as it saves all messages for full persistence)
def save_chat_history(session_id, memory):
    file_path = os.path.join(SAVE_DIR, f"{session_id}.json")
    messages_to_save = []
    # ConversationBufferWindowMemory's .buffer property stores the actual message objects
    for msg in memory.chat_memory.messages: # Accessing all messages from the chat_memory
        messages_to_save.append({"type": msg.type, "content": msg.content})
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({"messages": messages_to_save}, f, ensure_ascii=False, indent=4)
        # st.sidebar.success(f"Saved chat: {session_id}") # Optional success message
    except Exception as e:
        st.sidebar.error(f"Error saving chat {session_id}: {e}")


# --- State Initialization ---
for key in ["chat_sessions", "active_session", "memory_store", "vectorstore", "retriever", "rag_chain", "show_tools"]:
    if key not in st.session_state:
        st.session_state[key] = {} if "store" in key or "sessions" in key else None


# --- Initial Load of Chat Sessions (on app start/refresh) ---
if not st.session_state.chat_sessions: # Only load from disk if no sessions are in state yet
    for file_name in os.listdir(SAVE_DIR):
        if file_name.endswith(".json"):
            session_id = file_name.replace(".json", "")
            # A simple way to get title from file if not stored within it:
            # You could also store a 'title' field in the JSON file when saving.
            # For now, let's derive title from the first AI message or use a placeholder.
            try:
                with open(os.path.join(SAVE_DIR, file_name), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    first_message_content = ""
                    for msg in data.get("messages", []):
                        if msg["type"] == "ai" and msg["content"].strip():
                            first_message_content = msg["content"]
                            break
                    title = clean_title_text(first_message_content) if first_message_content else f"Chat {session_id[:8]}"
                    if not title: # Fallback for completely empty or un-titleable chats
                        title = f"Chat {session_id[:8]}"
            except Exception:
                title = f"Chat {session_id[:8]}" # Fallback if file is corrupted

            st.session_state.chat_sessions[session_id] = title

# --- New Chat Handling ---
if new_chat_button():
    # Save current active session before creating a new one
    if st.session_state.active_session and st.session_state.active_session in st.session_state.memory_store:
        save_chat_history(st.session_state.active_session, st.session_state.memory_store[st.session_state.active_session])

    new_id = create_new_chat_session("New Chat")
    st.session_state.active_session = new_id
    st.session_state.memory_store[new_id] = load_chat_history(new_id, k=10) # Initialize new memory with k
    st.session_state.vectorstore = None
    st.session_state.retriever = None
    st.session_state.rag_chain = None
    st.rerun()

# --- Session Switching ---
selected_session = render_chat_sessions() # This function in ui.py now also handles deleting files
if selected_session and selected_session != st.session_state.active_session:
    # Save the current active session's memory before switching
    if st.session_state.active_session and st.session_state.active_session in st.session_state.memory_store:
        save_chat_history(st.session_state.active_session, st.session_state.memory_store[st.session_state.active_session])

    st.session_state.active_session = selected_session
    st.session_state.memory_store[selected_session] = load_chat_history(selected_session, k=10) # Load the memory for the newly selected session with k
    st.session_state.vectorstore = None # Reset RAG components for the new session, as they are document-specific
    st.session_state.retriever = None
    st.session_state.rag_chain = None
    st.rerun()

# --- Default Session Fallback ---
if not st.session_state.active_session and st.session_state.chat_sessions:
    # If no active session, but existing sessions were loaded, pick the first one
    st.session_state.active_session = next(iter(st.session_state.chat_sessions.keys()))
    st.session_state.memory_store[st.session_state.active_session] = load_chat_history(st.session_state.active_session, k=10) # Load memory with k
elif not st.session_state.active_session: # Still no active session, create a new one
    default_id = create_new_chat_session("New Chat")
    st.session_state.active_session = default_id
    st.session_state.memory_store[default_id] = load_chat_history(default_id, k=10) # Initialize new memory with k

# --- Load LLM ---
llm = Ollama(model="llama3")

# --- Get Current Memory ---
active_id = st.session_state.active_session
memory = st.session_state.memory_store[active_id]

# --- Show Chat History ---
render_chat_history(memory)

# Export Button
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

# --- Input Box UI ---
user_input, uploaded_file, url_input, append_to_memory = user_input_box()

# --- Ingest Uploaded Document or URL ---
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
            # Limit to first 1000 words to avoid long context
            raw_docs[0].page_content = " ".join(raw_docs[0].page_content.split()[:1000])
            docs = raw_docs
            st.success("✅ Webpage loaded successfully!")
        else:
            st.warning("⚠️ No content found at the URL.")

    except Exception as e:
        st.error(f"❌ Error loading web URL: {e}")

# --- Build Vector Store ---
if docs:
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    if not append_to_memory or not st.session_state.vectorstore:
        st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
    else:
        st.session_state.vectorstore.add_documents(chunks)

    st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 10})

    # --- NEW: Define a custom prompt for RetrievalQA to guide the LLM ---
    # This prompt tells the LLM to use the context if available and relevant, otherwise use general knowledge.
    qa_prompt_template = """
You are a helpful AI assistant. When answering questions:
If context is provided, use it to directly answer the question.
If the context does not provide enough information to fully answer, treat the question as a general user query and respond using your general knowledge or reasoning.
Ensure the final answer is clear, relevant, and concise.

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
        chain_type="stuff", # 'stuff' is common for RAG where context is "stuffed" into the prompt
        return_source_documents=False, # Set to True if you want to see the retrieved sources
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT} # Pass the custom prompt here
    )

# --- Bot Response ---
if user_input:
    try:
        with st.spinner("🧠 Bot is thinking..."):
            retriever = st.session_state.retriever
            
            # This fallback prompt is still useful if no retriever is active at all
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
                # When a retriever is active and rag_chain is built with the custom prompt,
                # we just use the rag_chain. The prompt itself handles the decision
                # of using context vs. general knowledge.
                response = st.session_state.rag_chain.run(user_input)
            else:
                # If no retriever or rag_chain is set up (e.g., initial state), use fallback
                response = fallback_chain.run({"question": user_input})

        current_id = st.session_state.active_session
        current_title = st.session_state.chat_sessions.get(current_id, "")
        if current_id and (not current_title or current_title == "New Chat"):
            title = clean_title_text(response.strip().split("\n")[0])
            if title:
                st.session_state.chat_sessions[current_id] = title

        st.session_state.user_input = ""
        st.session_state.responding = False

        # --- SAVE MEMORY AFTER EACH TURN ---
        save_chat_history(current_id, memory)

        st.rerun()

    except Exception as e:
        st.session_state.responding = False
        st.error(f"❌ Error: {e}")