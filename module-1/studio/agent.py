import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage

from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode

def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide]

# Load environment variables from .env
load_dotenv()

from langchain_openai import ChatOpenAI, AzureChatOpenAI

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

# Define LLM with bound tools
llm_with_tools = get_llm_with_tools(tools)

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with writing performing arithmetic on a set of inputs.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

# Compile graph
graph = builder.compile()
