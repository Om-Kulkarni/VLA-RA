"""
LangGraph Workflow Orchestration

This module builds the StateGraph and orchestrates the transition between the various
Agent Nodes (Scout, Librarian, Analyst, Critic).
"""
from langgraph.graph import StateGraph, START, END
from chains.state import GraphState
from agents.scout import scout_node
from agents.librarian import librarian_node

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
    workflow.add_node("librarian", librarian_node)
    
    # Define the edges (routing)
    workflow.add_edge(START, "scout")
    workflow.add_edge("scout", "librarian")
    
    # Temporarily route librarian to END since subsequent nodes aren't added yet
    workflow.add_edge("librarian", END)
    
    # Compile the graph into a runnable sequence
    return workflow.compile()
