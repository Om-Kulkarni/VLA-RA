"""
LangGraph Workflow Orchestration

This module builds the StateGraph and orchestrates the transition between the various
Agent Nodes (Scout, Librarian, Analyst, Critic).
"""
from langgraph.graph import StateGraph, START, END
from chains.state import GraphState
from agents.scout import scout_node

def create_workflow():
    """
    Initializes and compiles the main LangGraph workflow.
    
    Returns:
        Compiled StateGraph: The executable workflow.
    """
    # Initialize the graph with our state definition
    workflow = StateGraph(GraphState)
    
    # Add nodes to the graph
    workflow.add_node("scout", scout_node)
    
    # Define the edges (routing)
    workflow.add_edge(START, "scout")
    
    # Temporarily route scout to END since it's the only node so far
    workflow.add_edge("scout", END)
    
    # Compile the graph into a runnable sequence
    return workflow.compile()
