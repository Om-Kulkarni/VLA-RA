"""
LangGraph Workflow Orchestration

This module builds the StateGraph and orchestrates the transition between the various
Agent Nodes (Scout, Librarian, Analyst, Critic).
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from chains.state import GraphState
from agents.scout import scout_node
from agents.librarian import librarian_node
from agents.critic import critic_node
from agents.analyst import analyst_node


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
    workflow.add_node("critic", critic_node)
    workflow.add_node("analyst", analyst_node)

    # Define the edges (routing)
    workflow.add_edge(START, "scout")
    workflow.add_edge("scout", "librarian")

    # Route librarian to critic, resolving earlier placeholder
    workflow.add_edge("librarian", "critic")
    workflow.add_edge("critic", "analyst")
    workflow.add_edge("analyst", END)

    # Compile the graph into a runnable sequence with an interrupt
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory, interrupt_before=["analyst"])


def create_express_workflow():
    """
    Builds a stripped-down workflow for express mode (--arxiv-id flag).
    Skips Scout and Librarian entirely. Papers are pre-loaded into state by main.py.

    Graph: START → critic → analyst → END
    Interrupts before analyst (same HITL gate as the full workflow).

    Returns:
        Compiled StateGraph: The express workflow.
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("critic", critic_node)
    workflow.add_node("analyst", analyst_node)

    workflow.add_edge(START, "critic")
    workflow.add_edge("critic", "analyst")
    workflow.add_edge("analyst", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory, interrupt_before=["analyst"])

