import re
from typing import TypedDict, Annotated, List
import operator
import json
from openai import OpenAI
from langchain.tools import tool
from langchain_google_community.search import GoogleSearchAPIWrapper
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage
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

    # 2. Setup OpenAI client for NVIDIA API
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=nvidia_api_key
    )

    # 3. Define Graph Nodes
    def call_model(state):
        """The primary node that calls the LLM."""
        print("--- AGENT: Thinking... ---")
        
        # Convert LangChain messages to OpenAI format
        openai_messages = []
        
        # Add system message for tool usage
        system_prompt = """You are a research assistant with access to scientific papers and web search tools.

Available tools:
1. paper_qa_tool - for questions about scientific papers on graph theory, sparsity, and pebble game
2. web_search_tool - for general knowledge and web search

When you need to use a tool, respond with a JSON object in this exact format:
{"tool_calls": [{"name": "tool_name", "arguments": {"query": "your query here"}}]}

Otherwise, provide a direct answer to the user's question."""
        
        openai_messages.append({"role": "system", "content": system_prompt})
        
        # Convert state messages to OpenAI format
        for msg in state["messages"]:
            if msg.type == "human":
                openai_messages.append({"role": "user", "content": msg.content})
            elif msg.type == "ai":
                openai_messages.append({"role": "assistant", "content": msg.content})
            elif msg.type == "tool":
                openai_messages.append({"role": "user", "content": f"Tool result: {msg.content}"})
        
        # Call NVIDIA API
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=openai_messages,
            temperature=0.2,
            top_p=1,
            max_tokens=4096
        )
        
        response_content = completion.choices[0].message.content
        print(f"Model response: {response_content}")
        
        # Check if response contains tool calls
        tool_calls = []
        try:
            # Try to parse as JSON for tool calls
            if response_content.strip().startswith('{"tool_calls"'):
                parsed_response = json.loads(response_content)
                if "tool_calls" in parsed_response:
                    for i, tool_call in enumerate(parsed_response["tool_calls"]):
                        tool_calls.append({
                            "id": f"call_{i}",
                            "name": tool_call["name"],
                            "args": tool_call["arguments"]
                        })
                    # Set content to empty string when there are tool calls
                    response_content = ""
        except json.JSONDecodeError:
            # Not a tool call, treat as regular response
            pass
        
        # Create AIMessage with or without tool calls
        response_message = AIMessage(content=response_content, tool_calls=tool_calls)
        return {"messages": [response_message]}

    def should_continue(state):
        """Router: decides whether to call a tool or end the conversation."""
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
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