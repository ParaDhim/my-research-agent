import re
import operator
import streamlit as st
from typing import TypedDict, Annotated, List
from openai import OpenAI
from langchain.tools import tool
from langchain_google_community.search import GoogleSearchAPIWrapper
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Load API keys from Streamlit secrets
@st.cache_data
def get_api_keys():
    """Get API keys from Streamlit secrets"""
    try:
        nvidia_api_key = st.secrets["NVIDIA_API_KEY"]
        google_api_key = st.secrets["GOOGLE_API_KEY"] 
        google_cse_id = st.secrets["GOOGLE_CSE_ID"]
        return nvidia_api_key, google_api_key, google_cse_id
    except KeyError as e:
        st.error(f"Missing API key in secrets: {e}")
        st.error("Please add your API keys to Streamlit secrets:")
        st.code("""
        # In .streamlit/secrets.toml:
        NVIDIA_API_KEY = "your_nvidia_api_key"
        GOOGLE_API_KEY = "your_google_api_key" 
        GOOGLE_CSE_ID = "your_google_cse_id"
        """)
        st.stop()

# Initialize clients
nvidia_api_key, google_api_key, google_cse_id = get_api_keys()

@st.cache_resource
def get_openai_client():
    """Initialize and cache the OpenAI client"""
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=nvidia_api_key
    )

@st.cache_resource
def get_search_wrapper():
    """Initialize and cache the Google search wrapper"""
    return GoogleSearchAPIWrapper(
        google_api_key=google_api_key, 
        google_cse_id=google_cse_id
    )

# Global clients
client = get_openai_client()
search_wrapper = get_search_wrapper()

# Global variable to store vector store
vector_store = None

@tool
def paper_qa_tool(query: str) -> str:
    """
    Answers specific, detailed questions about scientific papers on graph theory,
    sparsity, and the pebble game. Use this for questions that reference specific
    paper details or concepts.
    """
    try:
        retriever = vector_store.as_retriever(search_kwargs={'k': 3})
        context_docs = retriever.get_relevant_documents(query)
        
        # Clean out gibberish patterns
        gibberish_pattern = re.compile(r'/DAN <[A-Fa-f0-9]+>')
        cleaned_docs = [doc for doc in context_docs 
                       if not gibberish_pattern.search(doc.page_content)]
        
        if not cleaned_docs:
            return "No relevant information found in the scientific papers for this query."
        
        context_text = "\n\n".join([doc.page_content for doc in cleaned_docs])
        return f"Based on the scientific papers:\n\n{context_text}"
        
    except Exception as e:
        return f"Error searching scientific papers: {str(e)}"

@tool
def web_search_tool(query: str) -> str:
    """
    Provides up-to-date answers from the web for general knowledge, definitions,
    or topics not covered in the local scientific papers. Also provides source links.
    """
    try:
        results = search_wrapper.results(query, num_results=3)
        if not results:
            return "No web search results found for this query."
            
        formatted_results = []
        for res in results:
            formatted_results.append(
                f"**{res['title']}**\n"
                f"Source: {res['link']}\n"
                f"{res['snippet']}\n"
            )
        
        return "Web search results:\n\n" + "\n---\n".join(formatted_results)
        
    except Exception as e:
        return f"Error performing web search: {str(e)}"

# Define the AgentState
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

def call_model(state):
    """Call the NVIDIA model"""
    try:
        # Convert messages to OpenAI format
        openai_messages = []
        for msg in state['messages']:
            if isinstance(msg, HumanMessage):
                openai_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                openai_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                # Add tool results as user messages for context
                openai_messages.append({"role": "user", "content": f"Tool result: {msg.content}"})
        
        # Call NVIDIA API
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=openai_messages,
            temperature=0.2,
            top_p=1,
            max_tokens=4096
        )
        
        response_text = completion.choices[0].message.content
        return {"messages": [AIMessage(content=response_text)]}
        
    except Exception as e:
        error_msg = f"Error calling model: {str(e)}"
        return {"messages": [AIMessage(content=error_msg)]}

def should_continue(state):
    """Determine whether to continue or end"""
    last_message = state['messages'][-1]
    
    # For now, we'll implement a simple rule-based approach
    # Check if the query seems to need tools
    if isinstance(last_message, HumanMessage):
        content = last_message.content.lower()
        
        # Check for scientific paper related queries
        paper_keywords = ['paper', 'research', 'study', 'graph theory', 'sparsity', 'pebble game', 'algorithm']
        web_keywords = ['current', 'latest', 'news', 'what is', 'define', 'today']
        
        if any(keyword in content for keyword in paper_keywords):
            # Use paper tool
            return "use_paper_tool"
        elif any(keyword in content for keyword in web_keywords):
            # Use web search
            return "use_web_tool" 
    
    return "end"

def call_paper_tool(state):
    """Call the paper QA tool"""
    query = state['messages'][-1].content
    result = paper_qa_tool.invoke(query)
    return {"messages": [ToolMessage(content=result, tool_call_id="paper_search")]}

def call_web_tool(state):
    """Call the web search tool"""
    query = state['messages'][-1].content
    result = web_search_tool.invoke(query)
    return {"messages": [ToolMessage(content=result, tool_call_id="web_search")]}

@st.cache_resource
def create_agent(_vs):
    """Create and return the agent with the given vector store."""
    global vector_store
    vector_store = _vs
    
    # Build the workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("paper_tool", call_paper_tool)
    workflow.add_node("web_tool", call_web_tool)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "use_paper_tool": "paper_tool",
            "use_web_tool": "web_tool", 
            "end": END
        }
    )
    
    # Add edges back to agent after tool use
    workflow.add_edge("paper_tool", "agent")
    workflow.add_edge("web_tool", "agent")
    
    # Compile the workflow
    app = workflow.compile()
    return app