"""Streamlit frontend for Krisp Meeting Assistant Chat."""

import streamlit as st
import requests
import json
from datetime import datetime
import asyncio
import websockets
from typing import Dict, Any, List, Optional
import uuid
import time


# Configuration
BACKEND_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/api/ws/chat"

# Page config
st.set_page_config(
    page_title="Krisp Meeting Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main container */
    .stApp {
        background-color: #f7f8fa;
    }
    
    /* Chat messages */
    .user-message {
        background-color: #007AFF;
        color: white;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        max-width: 70%;
        float: right;
        clear: both;
    }
    
    .assistant-message {
        background-color: #E8E8ED;
        color: #1C1C1E;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        max-width: 70%;
        float: left;
        clear: both;
    }
    
    /* Sources */
    .source-card {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
        font-size: 0.9em;
    }
    
    /* Sidebar */
    .sidebar-header {
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 10px;
        color: #1C1C1E;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 5px;
    }
    
    .status-online {
        background-color: #34C759;
    }
    
    .status-offline {
        background-color: #FF3B30;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    
    if "backend_status" not in st.session_state:
        st.session_state.backend_status = check_backend_status()
    
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    
    if "current_sources" not in st.session_state:
        st.session_state.current_sources = []


def check_backend_status() -> bool:
    """Check if backend is available."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def send_message(message: str) -> Optional[Dict[str, Any]]:
    """Send message to backend and get response."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/chat",
            json={
                "message": message,
                "conversation_id": st.session_state.conversation_id
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"Failed to send message: {str(e)}")
        return None


def search_knowledge_base(query: str, search_type: str = "hybrid") -> List[Dict[str, Any]]:
    """Search the knowledge base."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/search",
            json={
                "query": query,
                "search_type": search_type,
                "top_k": 10
            },
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Search error: {response.status_code}")
            return []
            
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        return []


def display_chat_message(role: str, content: str, sources: Optional[List[Dict]] = None):
    """Display a chat message with optional sources."""
    if role == "user":
        st.markdown(f'<div class="user-message">{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-message">{content}</div>', unsafe_allow_html=True)
        
        # Display sources if available
        if sources:
            with st.expander("📎 Sources", expanded=False):
                for source in sources:
                    st.markdown(f"""
                    <div class="source-card">
                        <strong>Meeting:</strong> {source.get('meeting_id', 'Unknown')}<br>
                        <strong>Speaker:</strong> {source.get('speaker', 'N/A')}<br>
                        <strong>Preview:</strong> {source.get('content_preview', '')}
                    </div>
                    """, unsafe_allow_html=True)


def main():
    """Main application function."""
    init_session_state()
    
    # Header
    st.title("🎙️ Krisp Meeting Assistant")
    st.markdown("AI-powered chat interface for your meeting transcriptions")
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">⚙️ Settings</div>', unsafe_allow_html=True)
        
        # Backend status
        status_class = "status-online" if st.session_state.backend_status else "status-offline"
        status_text = "Connected" if st.session_state.backend_status else "Disconnected"
        
        st.markdown(
            f'<span class="status-indicator {status_class}"></span>{status_text}',
            unsafe_allow_html=True
        )
        
        if st.button("🔄 Refresh Connection"):
            st.session_state.backend_status = check_backend_status()
            st.rerun()
        
        st.divider()
        
        # Search section
        st.markdown('<div class="sidebar-header">🔍 Search Knowledge Base</div>', unsafe_allow_html=True)
        
        search_query = st.text_input("Search query", key="search_input")
        search_type = st.selectbox(
            "Search type",
            ["hybrid", "vector", "keyword"],
            index=0
        )
        
        if st.button("Search", type="secondary"):
            if search_query:
                with st.spinner("Searching..."):
                    st.session_state.search_results = search_knowledge_base(search_query, search_type)
        
        # Display search results
        if st.session_state.search_results:
            st.markdown("**Search Results:**")
            for i, result in enumerate(st.session_state.search_results[:5]):
                chunk = result.get("chunk", {})
                score = result.get("score", 0)
                
                with st.expander(f"Result {i+1} (Score: {score:.2f})"):
                    st.write(f"**Meeting:** {chunk.get('meeting_id', 'Unknown')}")
                    st.write(f"**Content:** {chunk.get('content', '')[:200]}...")
        
        st.divider()
        
        # Data ingestion section
        st.markdown('<div class="sidebar-header">📤 Upload Data</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload transcription",
            type=["txt", "json"],
            help="Upload meeting transcription files"
        )
        
        if uploaded_file is not None:
            if st.button("Process Upload"):
                with st.spinner("Processing..."):
                    # Read file content
                    content = uploaded_file.read().decode("utf-8")
                    
                    # Send to backend
                    response = requests.post(
                        f"{BACKEND_URL}/api/ingest",
                        json={
                            "meeting_id": uploaded_file.name.split(".")[0],
                            "transcription_text": content,
                            "metadata": {
                                "filename": uploaded_file.name,
                                "uploaded_at": datetime.utcnow().isoformat()
                            }
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ Processed {result['chunks_created']} chunks")
                    else:
                        st.error("Failed to process file")
        
        st.divider()
        
        # Statistics
        if st.button("📊 Show Statistics"):
            try:
                response = requests.get(f"{BACKEND_URL}/api/stats")
                if response.status_code == 200:
                    stats = response.json()
                    st.markdown("**System Statistics:**")
                    st.json(stats)
            except:
                st.error("Failed to fetch statistics")
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.session_state.conversation_id = str(uuid.uuid4())
            st.session_state.current_sources = []
            st.rerun()
    
    # Main chat interface
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            display_chat_message(
                message["role"],
                message["content"],
                message.get("sources")
            )
    
    # Chat input
    if prompt := st.chat_input("Ask about your meetings...", disabled=not st.session_state.backend_status):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message immediately
        with chat_container:
            display_chat_message("user", prompt)
        
        # Get response
        with st.spinner("Thinking..."):
            response_data = send_message(prompt)
            
            if response_data:
                # Add assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_data["response"],
                    "sources": response_data.get("sources", [])
                })
                
                # Store current sources
                st.session_state.current_sources = response_data.get("sources", [])
                
                # Display assistant message
                with chat_container:
                    display_chat_message(
                        "assistant",
                        response_data["response"],
                        response_data.get("sources")
                    )
        
        # Rerun to update the UI
        st.rerun()
    
    # Footer with helpful prompts
    if len(st.session_state.messages) == 0:
        st.markdown("### 💡 Try asking:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            - What were the main decisions from recent meetings?
            - Show me all action items assigned to John
            - Summarize the product roadmap discussions
            """)
        
        with col2:
            st.markdown("""
            - What did we discuss about the Q4 goals?
            - Find all mentions of the new feature launch
            - Who committed to delivering the API updates?
            """)


if __name__ == "__main__":
    main()
