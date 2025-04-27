import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# Tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# LLM with bound tool
def get_llm_with_tools(tools):
    model = AzureChatOpenAI(
        azure_deployment="gpt-4.1",  # or your deployment
        api_version="2024-12-01-preview",
        model = "gpt-4.1",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2)
    return model.bind_tools(tools)

llm_with_tools = get_llm_with_tools([multiply])

# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", END)

# Compile graph
graph = builder.compile()