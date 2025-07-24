# RagNitor-ChatBot
# RAGnitor – Retrieval-Augmented Chatbot with LLM Memory and Web Intelligence

## Demo
Try it here: http://3.86.227.167:8501/

## Overview
RAGnitor is a GenAI chatbot that combines LLMs, document understanding, and web scraping with persistent multi-user memory.

## Features
- Supports PDFs, text files, and website links
- Natural and contextual conversation
- Multi-user session-based memory
- Dual chat mode: General LLM and RAG
- Lightweight: Runs on EC2 free-tier (1 GB RAM)

## Tech Stack
- **LLM**: Groq’s llama3-8b-8192
- **LangChain**: Agent logic and memory
- **FAISS + Nomic Embeddings**: Vector search
- **Streamlit**: UI with session management
- **BeautifulSoup**: Web content extraction
- **Docker**: Containerized deployment
- **AWS EC2**: Cloud hosting

## Setup Instructions
### 1. Clone the Repository
git clone https://github.com/kalyanappari/RagNitor-ChatBot.git
cd RagNitor-ChatBot 

### 2. Install all the requirements.
pip install -r requirements.txt

### 3. Run Locally
streamlit run app/main.py

## 4.Deployment (Optional)
docker build -t ragnitor .
docker run -p 8501:8501 ragnitor

### File Upload Tips
Upload first, then query for best RAG output.

Ask specific queries like:

"Summarize this document"

"What are the key points?"

No upload? It'll work in LLM-only mode.

### Roadmap
HuggingFace model switch

OAuth login and named chat memory

Persistent vector DB support

Image and multimodal integration


