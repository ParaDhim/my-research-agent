import re
from typing import TypedDict, Annotated, List
import operator

from langchain.tools import tool
from langchain_google_community.search import GoogleSearchAPIWrapper
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# This class defines the structure of the agent's state
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

def create_agent_graph(vector_store, nvidia_api_key, google_api_key, google_cse_id):
    """Creates and compiles the LangGraph agent."""

    # 1. Define Tools
    @tool
    def paper_qa_tool(query: str) -> str:
        """
        Answers specific, detailed questions about scientific papers on graph theory,
        sparsity, and the pebble game. Use this for questions that reference specific
        paper details or concepts.
        """
        print("--- Calling Paper Q&A Tool ---")
        retriever = vector_store.as_retriever(search_kwargs={'k': 3})
        context_docs = retriever.get_relevant_documents(query)
        
        # Simple cleaning to remove potential gibberish from parsed PDFs
        gibberish_pattern = re.compile(r'/DAN <[A-Fa-f0-9]+>')
        cleaned_docs = [doc for doc in context_docs if not gibberish_pattern.search(doc.page_content)]
        
        if not cleaned_docs:
            return "No relevant information found in the documents after cleaning."
        
        context_text = "\n\n".join([doc.page_content for doc in cleaned_docs])
        return context_text

    search_wrapper = GoogleSearchAPIWrapper(google_api_key=google_api_key, google_cse_id=google_cse_id)

    @tool
    def web_search_tool(query: str) -> str:
        """
        Provides up-to-date answers from the web for general knowledge, definitions,
        or topics not covered in the local scientific papers. Also provides source links.
        """
        print("--- Calling Web Search Tool ---")
        results = search_wrapper.results(query, num_results=3)
        return "\n".join([f"Title: {res['title']}\nLink: {res['link']}\nSnippet: {res['snippet']}\n" for res in results])

    tools = [paper_qa_tool, web_search_tool]
    tool_node = ToolNode(tools)

    # 2. Define the Model
    # We use ChatOpenAI pointed at the NVIDIA endpoint
    model = ChatOpenAI(
        model="openai/gpt-oss-120b",
        openai_api_key=nvidia_api_key,
        openai_api_base="https://integrate.api.nvidia.com/v1/ ",
        temperature=0.2
    ).bind_tools(tools)

    # 3. Define Graph Nodes
    def call_model(state):
        """The primary node that calls the LLM."""
        print("--- AGENT: Thinking... ---")
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state):
        """Router: decides whether to call a tool or end the conversation."""
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "continue"
        return "end"

    # 4. Build and Compile the Graph
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", tool_node)
    
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"continue": "action", "end": END},
    )
    workflow.add_edge("action", "agent")

    return workflow.compile()