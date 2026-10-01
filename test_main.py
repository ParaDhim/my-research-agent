import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import main AFTER the imports above
import main
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

@pytest.fixture
def mock_agent():
    with patch("main.get_vector_store") as mock_gvs, \
         patch("main.create_agent") as mock_ca:
        
        mock_agent_instance = MagicMock()
        mock_ca.return_value = mock_agent_instance
        
        # TestClient uses context manager to trigger lifespan events
        with TestClient(main.app) as client:
            yield client, mock_agent_instance

def test_health_check_ok(mock_agent):
    client, _ = mock_agent
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "agent_initialized": True}

def test_health_check_uninitialized():
    # If we don't use TestClient context manager, lifespan hasn't run
    client = TestClient(main.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "starting_up", "vector_store": "unknown"}

def test_query_no_sources(mock_agent):
    client, mock_agent_instance = mock_agent
    
    # Mocking standard conversational response
    mock_agent_instance.invoke.return_value = {
        "messages": [
            HumanMessage(content="Hello"),
            AIMessage(content="Hello! How can I help you today?")
        ]
    }
    
    response = client.post("/query", json={"query": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Hello! How can I help you today?"
    assert data["sources"] == []

def test_query_with_sources(mock_agent):
    client, mock_agent_instance = mock_agent
    
    mock_agent_instance.invoke.return_value = {
        "messages": [
            HumanMessage(content="What is the latest AI news?"),
            ToolMessage(content="**AI Breakthroughs**\nSource: https://example.com/ai-news\nHere is a snippet.", tool_call_id="web_search", name="web_search_tool"),
            AIMessage(content="According to recent news, there are many AI breakthroughs.")
        ]
    }
    
    response = client.post("/query", json={"query": "What is the latest AI news?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "According to recent news, there are many AI breakthroughs."
    assert "https://example.com/ai-news" in data["sources"]

def test_query_validation_error(mock_agent):
    client, _ = mock_agent
    # Query too short or missing
    response = client.post("/query", json={"query": ""})
    assert response.status_code == 422
    
def test_query_exception_handling(mock_agent):
    client, mock_agent_instance = mock_agent
    mock_agent_instance.invoke.side_effect = Exception("Model is down")
    
    response = client.post("/query", json={"query": "Crash the model"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Model is down"
