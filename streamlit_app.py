import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from agent import create_agent_graph
from vector_store import get_vector_store

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="🔬 RAG Research Agent", layout="wide")
st.title("🔬 RAG Research Agent on Scientific Papers")
st.markdown("Ask me anything about (k,ℓ)-sparse graphs, the pebble game, or other related topics!")

# --- AGENT AND VECTOR STORE INITIALIZATION ---
# This part is cached to avoid reloading on every interaction.
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
        st.error(f"Could not load API keys from secrets. Please ensure secrets.toml is configured correctly in Hugging Face. Error: {e}")
        st.stop()

app = setup_agent()

# --- CHAT HISTORY MANAGEMENT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# --- CHAT INTERFACE ---
if prompt := st.chat_input("What is your question?"):
    # Add user message to session state and display it
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("human"):
        st.markdown(prompt)
    
    # Prepare the input for the agent
    inputs = {"messages": st.session_state.messages}
    
    # Display an empty container for the agent's response
    with st.chat_message("ai"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Stream the agent's response
        for event in app.stream(inputs):
            for value in event.values():
                if isinstance(value["messages"][-1], AIMessage):
                    full_response += value["messages"][-1].content
                    response_placeholder.markdown(full_response + "▌")
        
        # Update the placeholder with the final response
        response_placeholder.markdown(full_response)

    # Add the final AI response to the session state
    st.session_state.messages.append(AIMessage(content=full_response))