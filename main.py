import re
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import streamlit as st
import sys

# --- Mock Streamlit ---
# Intercept Streamlit calls so importing `agent` and `vector_store` don't crash or hang
def mock_cache(func=None, **kwargs):
    if func is None:
        return lambda f: f
    return func

st.cache_resource = mock_cache
st.cache_data = mock_cache

class MockProgress:
    def progress(self, *args, **kwargs): pass
    def empty(self, *args, **kwargs): pass

st.progress = lambda *args, **kwargs: MockProgress()
st.info = lambda *args, **kwargs: None
st.success = lambda *args, **kwargs: None
st.warning = lambda *args, **kwargs: None
st.error = lambda *args, **kwargs: None
st.stop = lambda *args, **kwargs: sys.exit(1)

# Now it is safe to import local modules
from agent import create_agent
from vector_store import get_vector_store
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

agent_app = None
vector_store = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_app, vector_store
    try:
        print("Starting up: initializing vector store...", flush=True)
        vector_store = get_vector_store()
        print("Vector store up. Initializing agent...", flush=True)
        agent_app = create_agent(vector_store)
        print("Agent ready.", flush=True)
    except Exception as e:
        print(f"Error during startup: {e}", flush=True)
    yield
    print("Shutting down...", flush=True)
    agent_app = None

app = FastAPI(title="Multi-Tool Research Agent API", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The research or general query.")

class QueryResponse(BaseModel):
    answer: str = Field(description="The response from the agent.")
    sources: List[str] = Field(description="Extracted source links or document references.")

def extract_sources_from_messages(messages) -> List[str]:
    """Helper to extract sources (URLs or paper metadata) from tool messages."""
    sources = set()
    url_pattern = re.compile(r'(https?://\S+)')
    
    for msg in messages:
        if isinstance(msg, ToolMessage):
            # If web search, extract URLs
            if "web_search" in msg.name or msg.tool_call_id == "web_search":
                # Web search is formatted like: Source: <url>
                urls = url_pattern.findall(msg.content)
                for u in urls:
                    sources.add(u)
            elif "paper_search" in msg.name or msg.tool_call_id == "paper_search":
                # For papers, we might extract references. Currently the prompt just needs sources.
                # If there's a specific pattern, we could grab it. Defaulting to general search below.
                pass
                
            # Generic grab just in case
            urls = url_pattern.findall(msg.content)
            for u in urls:
                sources.add(u)
                
    return list(sources)

@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    if not agent_app:
        raise HTTPException(status_code=503, detail="Agent is not initialized yet.")
        
    try:
        # We start a fresh conversation state for a single query.
        initial_state = {"messages": [HumanMessage(content=request.query)]}
        result = agent_app.invoke(initial_state)
        
        # Last message should be the AI answer
        messages = result.get("messages", [])
        if not messages:
            return QueryResponse(answer="No response generated.", sources=[])
            
        final_answer = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
        
        # Extract sources from preceding ToolMessages
        sources = extract_sources_from_messages(messages)
        
        return QueryResponse(
            answer=final_answer,
            sources=sources
        )
    except Exception as e:
        print(f"Exception during query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Check if the API and underlying Agent are ready."""
    if agent_app is None:
        return {"status": "starting_up", "vector_store": "unknown"}
    return {"status": "ok", "agent_initialized": True}
