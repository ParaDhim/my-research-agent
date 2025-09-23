import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from agent import create_agent_graph
from vector_store import get_vector_store
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="🔬 RAG Research Agent", 
    layout="wide",
    page_icon="🔬",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
    /* Main container styling */
    .main > div {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin-bottom: 0;
    }
    
    /* Chat container styling */
    .chat-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        border: 1px solid #e0e0e0;
    }
    
    /* Message styling */
    .stChatMessage {
        margin-bottom: 1rem !important;
    }
    
    /* User message styling */
    div[data-testid="chatAvatarIcon-human"] {
        background-color: #667eea !important;
    }
    
    /* AI message styling */
    div[data-testid="chatAvatarIcon-ai"] {
        background-color: #764ba2 !important;
    }
    
    /* Input styling */
    .stChatInput {
        border-radius: 25px !important;
        border: 2px solid #667eea !important;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Processing indicator */
    .processing-indicator {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f2ff, #e8ebff, #f0f2ff);
        background-size: 200% 100%;
        animation: shimmer 2s infinite;
        border-radius: 10px;
        margin: 1rem 0;
        color: #667eea;
        font-weight: 500;
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    /* Status indicators */
    .status-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 1rem;
        background: #f8f9ff;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    .status-text {
        color: #667eea;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    /* Welcome message */
    .welcome-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .welcome-title {
        color: #333;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .welcome-text {
        color: #666;
        font-size: 1.1rem;
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }
    
    .feature-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        justify-content: center;
    }
    
    .feature-tag {
        background: white;
        color: #667eea;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div class="header-container">
    <div class="header-title">🔬 RAG Research Agent</div>
    <div class="header-subtitle">Advanced AI-powered research assistant for scientific papers</div>
</div>
""", unsafe_allow_html=True)

# --- AGENT AND VECTOR STORE INITIALIZATION ---
@st.cache_resource
def setup_agent():
    """Load secrets, initialize vector store, and create the agent graph."""
    try:
        # Load API keys from Streamlit secrets
        nvidia_api_key = st.secrets["NVIDIA_API_KEY"]
        google_api_key = st.secrets["GOOGLE_API_KEY"]
        google_cse_id = st.secrets["GOOGLE_CSE_ID"]
        
        vector_store = get_vector_store()
        
        # Create the agent graph
        app = create_agent_graph(vector_store, nvidia_api_key, google_api_key, google_cse_id)
        return app
    except (KeyError, FileNotFoundError) as e:
        st.error(f"🚫 Could not load API keys from secrets. Please ensure secrets.toml is configured correctly. Error: {e}")
        st.stop()

# Initialize agent
with st.spinner("🔧 Initializing AI Agent..."):
    app = setup_agent()

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# --- STATUS DISPLAY ---
if st.session_state.is_processing:
    st.markdown("""
    <div class="status-container">
        <span class="status-text">🤖 AI is thinking...</span>
        <div class="typing-indicator"></div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="status-container">
        <span class="status-text">✅ Ready to help you with your research</span>
        <span style="color: #28a745;">●</span>
    </div>
    """, unsafe_allow_html=True)

# --- WELCOME MESSAGE ---
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-title">👋 Welcome to Your Research Assistant</div>
        <div class="welcome-text">
            I'm here to help you explore scientific papers and research on various topics. 
            Ask me anything about complex research areas, and I'll provide detailed, well-sourced answers.
        </div>
        <div class="feature-tags">
            <span class="feature-tag">📊 Data Analysis</span>
            <span class="feature-tag">📚 Literature Review</span>
            <span class="feature-tag">🔍 Research Insights</span>
            <span class="feature-tag">📈 Graph Theory</span>
            <span class="feature-tag">🧠 AI-Powered</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- CHAT HISTORY DISPLAY ---
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message.type):
        # Clean up any remaining HTML tags in the content
        clean_content = message.content.replace('<br>', '\n').replace('<br/>', '\n').replace('<br />', '\n')
        st.markdown(clean_content)

st.markdown('</div>', unsafe_allow_html=True)

# --- CHAT INPUT ---
# Disable input when processing
chat_input_key = f"chat_input_{len(st.session_state.messages)}"
prompt = st.chat_input(
    "💭 What would you like to research today?" if not st.session_state.is_processing else "🤖 AI is processing your request...",
    disabled=st.session_state.is_processing,
    key=chat_input_key
)

# --- MESSAGE PROCESSING ---
if prompt and not st.session_state.is_processing:
    # Set processing state
    st.session_state.is_processing = True
    
    # Add user message to session state and display it
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    with st.chat_message("human"):
        st.markdown(prompt)
    
    # Prepare the input for the agent
    inputs = {"messages": st.session_state.messages}
    
    # Display AI response with typing indicator
    with st.chat_message("ai"):
        response_placeholder = st.empty()
        
        # Show processing indicator
        response_placeholder.markdown("""
        <div class="processing-indicator">
            <div class="typing-indicator" style="margin-right: 10px;"></div>
            Analyzing your question and searching through research papers...
        </div>
        """, unsafe_allow_html=True)
        
        full_response = ""
        
        try:
            # Stream the agent's response
            for event in app.stream(inputs):
                for value in event.values():
                    if isinstance(value["messages"][-1], AIMessage):
                        # Clean up any HTML tags in the response
                        new_content = value["messages"][-1].content
                        new_content = new_content.replace('<br>', '\n').replace('<br/>', '\n').replace('<br />', '\n')
                        full_response = new_content
                        
                        # Update with current response and cursor
                        response_placeholder.markdown(full_response + " ▌")
            
            # Final response without cursor
            response_placeholder.markdown(full_response)
            
            # Add the final AI response to the session state
            st.session_state.messages.append(AIMessage(content=full_response))
            
        except Exception as e:
            error_message = f"🚫 Sorry, I encountered an error while processing your request: {str(e)}"
            response_placeholder.markdown(error_message)
            st.session_state.messages.append(AIMessage(content=error_message))
        
        finally:
            # Reset processing state
            st.session_state.is_processing = False
            # Force a rerun to update the UI
            st.rerun()

# --- SIDEBAR WITH ADDITIONAL INFO ---
with st.sidebar:
    st.markdown("### 📋 Quick Help")
    st.markdown("""
    **💡 Tips for better results:**
    - Be specific in your questions
    - Ask about research methodologies
    - Request paper summaries
    - Inquire about data analysis techniques
    
    **🔍 Example questions:**
    - "Explain (k,ℓ)-sparse graphs"
    - "What is the pebble game algorithm?"
    - "Compare different graph theory approaches"
    """)
    
    st.markdown("### 📊 Session Stats")
    st.metric("Messages", len(st.session_state.messages))
    st.metric("Status", "🟢 Active" if not st.session_state.is_processing else "🟡 Processing")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.is_processing = False
        st.rerun()

# --- FOOTER ---
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666; font-size: 0.9rem;">
    Powered by advanced AI • Built with Streamlit • Research made simple
</div>
""", unsafe_allow_html=True)