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

# Custom CSS for better UI
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stSidebar {
        background-color: #f0f2f6;
    }
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .feature-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
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
        st.markdown("### ✨ Features")
        features = [
            "🔬 Scientific paper Q&A",
            "🌐 Web search capabilities", 
            "💾 Chat history persistence",
            "⚡ Real-time responses",
            "🎯 Context-aware answers"
        ]
        
        for feature in features:
            st.markdown(f"• {feature}")
        
        st.markdown("---")
        
        # Status section
        st.markdown("### 📊 Status")
        if st.session_state.agent_initialized:
            st.success("✅ Agent Ready")
            st.info(f"💬 Messages: {len(st.session_state.messages)}")
        else:
            st.warning("⏳ Initializing...")

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
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><em>🚀 Powered by NVIDIA API, Google Search, and FAISS Vector Store</em></p>
        <p><small>Built using Streamlit & LangChain</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()