# ui.py
import streamlit as st
import uuid
import os # New Import for file deletion

# 👉 Inject custom CSS styles for chat layout
def inject_custom_css():
    st.markdown("""
        <style>
            .chat-container {
                max-width: 800px;
                margin: auto;
                padding: 20px;
                font-family: 'Segoe UI', sans-serif;
            }
            .chat-bubble {
                padding: 12px 18px;
                margin-bottom: 10px;
                border-radius: 10px;
                max-width: 80%;
                white-space: pre-wrap;
                color: #000000;
            }
            .user {
                background-color: #e1f5fe;
                align-self: flex-end;
                margin-left: auto;
            }
            .bot {
                background-color: #f1f8e9;
                align-self: flex-start;
                margin-right: auto;
            }
            .small-tools {
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 10px;
            }
            .custom-input-row {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 10px;
                margin-top: 20px;
            }
            input[type="text"] {
                height: 48px;
                font-size: 16px;
                padding: 0 10px;
            }
        </style>
    """, unsafe_allow_html=True)

# 👉 Show chat history in bubbles
def render_chat_history(memory):
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    if memory and memory.chat_memory.messages:
        for msg in memory.chat_memory.messages:
            role_class = "user" if msg.type == "human" else "bot"
            speaker = "🧑 You" if msg.type == "human" else "🤖 Bot"
            
            # ✅ Clean message content to remove extra spaces
            msg_content = msg.content.strip().replace("\n\n", "\n").replace("\n", "<br>")

            st.markdown(
                f'<div class="chat-bubble {role_class}"><strong>{speaker}:</strong><br>{msg_content}</div>',
                unsafe_allow_html=True
            )

    st.markdown('</div>', unsafe_allow_html=True)


def file_url_menu(show_menu):
    uploaded_file, url_input = None, ""
    if show_menu:
        with st.container():
            st.markdown("<div class='small-tools'>", unsafe_allow_html=True)
            col1, col2 = st.columns([1, 1])
            with col1:
                uploaded_file = st.file_uploader("Upload File", type=["pdf", "txt"])
            with col2:
                url_input = st.text_input("Web URL", placeholder="Paste web URL here")
    return uploaded_file, url_input

def user_input_box():
    if "show_menu" not in st.session_state:
        st.session_state.show_menu = False
    if "responding" not in st.session_state:
        st.session_state.responding = False
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "user_input_field" not in st.session_state:
        st.session_state.user_input_field = ""

    uploaded_file, url_input = None, ""
    append_to_memory = True

    st.markdown("""
        <style>
            .custom-input-row {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 10px;
                margin-top: 20px;
            }
            input[type="text"] {
                height: 48px;
                font-size: 16px;
                padding: 0 10px;
            }
            button[kind="primary"] {
                height: 48px !important;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-input-row">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([0.08, 0.8, 0.12])

    with col1:
        if st.button("⬇", key="dropdown", use_container_width=True, disabled=st.session_state.responding):
            st.session_state.show_menu = not st.session_state.show_menu

    with col2:
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "",
                placeholder="Type your message here...",
                key="user_input_field",
                label_visibility="collapsed",
                disabled=st.session_state.responding
            )
            submitted = st.form_submit_button("Send", use_container_width=True, disabled=st.session_state.responding)

    st.markdown('</div>', unsafe_allow_html=True)

    # Show menu if toggled
    if st.session_state.show_menu:
        with st.expander("📎 Upload or Link"):
            uploaded_file, url_input = file_url_menu(True)
            
            if uploaded_file or url_input:
                st.session_state.show_menu = False


    # Set the input and trigger response
    if submitted and user_input.strip():
        st.session_state.user_input = user_input.strip()
        st.session_state.responding = True
        st.rerun()

    return (
    st.session_state.user_input if st.session_state.responding else None,
    uploaded_file,
    url_input
)



# 👉 New Chat Button
def new_chat_button():
    return st.sidebar.button("🆕 New Chat")

# 👉 Show list of old sessions in sidebar
def render_chat_sessions():
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}

    st.sidebar.markdown("### 📂 Chat Sessions")
    sessions_to_delete = []

    # Define SAVE_DIR here as well for consistency with file deletion
    SAVE_DIR = "chat_histories" 

    for session_id, title in st.session_state.chat_sessions.items():
        col1, col2 = st.sidebar.columns([0.8, 0.2])
        with col1:
            # Add a disabled state to the button if it's the active session
            disabled_state = (session_id == st.session_state.active_session)
            if st.button(title, key=session_id, disabled=disabled_state):
                return session_id
        with col2:
            if st.button("🗑", key=f"delete_{session_id}"):
                sessions_to_delete.append(session_id)

    for session_id in sessions_to_delete:
        if session_id == st.session_state.active_session:
            st.session_state.active_session = None # Clear active if deleted

        # Remove from session_state
        st.session_state.chat_sessions.pop(session_id, None)
        st.session_state.memory_store.pop(session_id, None)

        # Delete the corresponding file
        file_path = os.path.join(SAVE_DIR, f"{session_id}.json")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                st.sidebar.info(f"Deleted chat file: {session_id}")
            except Exception as e:
                st.sidebar.error(f"Error deleting file {file_path}: {e}")

    return None


# 👉 Utility to create new session with title based on first bot reply
def create_new_chat_session(bot_reply: str = ""):
    new_id = str(uuid.uuid4())
    # Clean title is generated in app.py after bot reply, this is just initial
    clean_title = bot_reply.strip().split("\n")[0][:40] or f"Chat {len(st.session_state.chat_sessions) + 1}"
    st.session_state.chat_sessions[new_id] = clean_title
    return new_id

def clean_title_text(text):
    import re
    text = re.sub(r'[*_`>#-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()[:40]
