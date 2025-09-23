import streamlit as st
import json
import os
from langchain_core.messages import HumanMessage, AIMessage, messages_to_dict, messages_from_dict
from agent import create_agent
from vector_store import get_vector_store

# Page config
st.set_page_config(
    page_title="Scientific Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for classy UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global theme */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Chat messages styling */
    .stChatMessage {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Sidebar styling */
    .stSidebar > div {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stSidebar .stMarkdown {
        color: #e2e8f0;
    }
    
    /* Main header */
    .main-header {
        text-align: center;
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: #ffffff;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .main-header h1, .main-header h2 {
        color: #ffffff;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        margin: 0;
        font-weight: 600;
    }
    
    /* Feature boxes */
    .feature-box {
        background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        color: #e2e8f0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Status indicators */
    .status-success {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: #ffffff;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 500;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(72, 187, 120, 0.3);
    }
    
    .status-warning {
        background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
        color: #ffffff;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 500;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(237, 137, 54, 0.3);
    }
    
    .status-info {
        background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
        color: #ffffff;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 500;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(66, 153, 225, 0.3);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Chat input */
    .stChatInput > div > div > div > div {
        background: rgba(45, 55, 72, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: #e2e8f0;
        backdrop-filter: blur(10px);
    }
    
    .stChatInput input {
        color: #e2e8f0 !important;
        background: transparent !important;
    }
    
    .stChatInput input::placeholder {
        color: #a0aec0 !important;
    }
    
    /* Feature list */
    .feature-list {
        list-style: none;
        padding: 0;
    }
    
    .feature-item {
        color: #cbd5e0;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        transition: color 0.3s ease;
    }
    
    .feature-item:hover {
        color: #667eea;
    }
    
    /* Section headers */
    .section-header {
        color: #e2e8f0;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: #a0aec0;
        padding: 2rem 1rem;
        background: rgba(26, 32, 44, 0.5);
        border-radius: 12px;
        margin-top: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .footer em {
        color: #cbd5e0;
        font-weight: 500;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        margin: 2rem 0;
    }
    
    /* Spinner customization */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Success/Error messages */
    .element-container .stSuccess {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: #ffffff;
        border: none;
        border-radius: 8px;
    }
    
    .element-container .stError {
        background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
        color: #ffffff;
        border: none;
        border-radius: 8px;
    }
    
    .element-container .stWarning {
        background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
        color: #ffffff;
        border: none;
        border-radius: 8px;
    }
    
    .element-container .stInfo {
        background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
        color: #ffffff;
        border: none;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'agent_initialized' not in st.session_state:
        st.session_state.agent_initialized = False
    if 'agent' not in st.session_state:
        st.session_state.agent = None

def save_chat_history():
    """Save chat history to session state and file"""
    if st.session_state.messages:
        try:
            dict_messages = messages_to_dict(st.session_state.messages)
            # Store in session state for persistence
            st.session_state.chat_history = dict_messages
        except Exception as e:
            st.error(f"Error saving chat history: {e}")

def load_chat_history():
    """Load chat history from session state"""
    if 'chat_history' in st.session_state and st.session_state.chat_history:
        try:
            st.session_state.messages = messages_from_dict(st.session_state.chat_history)
            return True
        except Exception as e:
            st.error(f"Error loading chat history: {e}")
            return False
    return False

def clear_chat():
    """Clear chat history"""
    st.session_state.messages = []
    if 'chat_history' in st.session_state:
        del st.session_state.chat_history
    st.success("Chat cleared!")

def main():
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="main-header"><h2>🔬 Research Agent</h2></div>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-box">
        Ask questions about scientific papers on graph theory, sparsity, 
        and the pebble game, or any general knowledge questions!
        </div>
        """, unsafe_allow_html=True)
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
                clear_chat()
                st.rerun()
        
        with col2:
            if st.button("📁 Load History", type="secondary", use_container_width=True):
                if load_chat_history():
                    st.success("History loaded!")
                    st.rerun()
                else:
                    st.warning("No history found!")
        
        st.markdown("---")
        
        # Features section
        st.markdown('<div class="section-header">✨ Features</div>', unsafe_allow_html=True)
        features = [
            "🔬 Scientific paper Q&A",
            "🌐 Web search capabilities", 
            "💾 Chat history persistence",
            "⚡ Real-time responses",
            "🎯 Context-aware answers"
        ]
        
        for feature in features:
            st.markdown(f'<div class="feature-item">{feature}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Status section
        st.markdown('<div class="section-header">📊 Status</div>', unsafe_allow_html=True)
        if st.session_state.agent_initialized:
            st.markdown('<div class="status-success">✅ Agent Ready</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="status-info">💬 Messages: {len(st.session_state.messages)}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-warning">⏳ Initializing...</div>', unsafe_allow_html=True)

    # Main interface
    st.markdown('<div class="main-header"><h1>🔬 Scientific Research Agent</h1></div>', 
                unsafe_allow_html=True)

    # Initialize agent if not done
    if not st.session_state.agent_initialized:
        with st.spinner("🚀 Initializing agent and loading scientific papers..."):
            try:
                vector_store = get_vector_store()
                st.session_state.agent = create_agent(vector_store)
                st.session_state.agent_initialized = True
                st.success("✅ Agent initialized successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to initialize agent: {str(e)}")
                st.stop()

    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            if isinstance(message, HumanMessage):
                with st.chat_message("user", avatar="👤"):
                    st.markdown(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant", avatar="🔬"):
                    st.markdown(message.content)

    # Chat input
    if prompt := st.chat_input("Ask me anything about scientific papers or general knowledge...", 
                              disabled=not st.session_state.agent_initialized):
        
        # Add user message
        st.session_state.messages.append(HumanMessage(content=prompt))
        
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant", avatar="🔬"):
            with st.spinner("🤔 Thinking..."):
                try:
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    # Stream the response
                    for event in st.session_state.agent.stream({"messages": st.session_state.messages}):
                        for value in event.values():
                            if (isinstance(value.get("messages", [{}])[-1], AIMessage) and 
                                value["messages"][-1].content):
                                chunk = value["messages"][-1].content
                                full_response += chunk
                                # Update with typing indicator
                                response_placeholder.markdown(full_response + "▌")
                    
                    # Final response without cursor
                    response_placeholder.markdown(full_response)
                    
                    # Add to chat history
                    if full_response:
                        st.session_state.messages.append(AIMessage(content=full_response))
                        save_chat_history()
                    
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")
                    # Remove the failed user message
                    st.session_state.messages.pop()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p><small>Built using Streamlit & LangChain</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()