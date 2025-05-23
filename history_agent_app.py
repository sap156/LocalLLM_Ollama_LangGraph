# Historical Events AI Agent
# An agentic application that searches for historical events using Tavily API
# and processes them with local LLMs via Ollama and LangGraph

import os
import json
from datetime import datetime
from typing import Dict, TypedDict, List, Annotated, Literal
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL") # Get from .env file
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # Get from .env file

# State definition
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    current_date: str
    search_results: str
    final_response: str

# Initialize components
def initialize_components():
    """Initialize Ollama LLM and Tavily search tool"""
    try:
        llm = Ollama(model=OLLAMA_MODEL, temperature=0.7)
        
        if not TAVILY_API_KEY:
            st.error("Please set TAVILY_API_KEY environment variable")
            return None, None
            
        search_tool = TavilySearchResults(
            api_key=TAVILY_API_KEY,
            max_results=5,
            search_depth="advanced"
        )
        
        return llm, search_tool
    except Exception as e:
        st.error(f"Error initializing components: {e}")
        return None, None

# Agent nodes
# These functions define the individual operations (nodes) in the workflow.

# 1. Search Agent
# This agent searches for historical events that happened on today's date using the Tavily API.
def search_historical_events(state: AgentState, search_tool) -> AgentState:
    """Search for historical events that happened on today's date"""
    # Get the current date in "Month Day" format
    current_date = datetime.now().strftime("%B %d")
    
    # Construct the search query
    search_query = f"historical events that happened on {current_date} in history famous important"
    
    try:
        # Invoke the search tool with the query
        search_results = search_tool.invoke(search_query)
        
        # Format the search results for display
        formatted_results = "\n\n".join([
            f"**Source {i+1}:** {result.get('title', 'No title')}\n"
            f"**URL:** {result.get('url', 'No URL')}\n"
            f"**Content:** {result.get('content', 'No content')}"
            for i, result in enumerate(search_results)
        ])
        
        # Update the state with the search results and a success message
        return {
            **state,
            "current_date": current_date,
            "search_results": formatted_results,
            "messages": state["messages"] + [
                AIMessage(content=f"Found historical events for {current_date}")
            ]
        }
    except Exception as e:
        # Handle errors and update the state with an error message
        error_msg = f"Error searching for historical events: {e}"
        return {
            **state,
            "current_date": current_date,
            "search_results": error_msg,
            "messages": state["messages"] + [AIMessage(content=error_msg)]
        }

# 2. Analyze Agent
# This agent analyzes the search results to extract key insights using the LLM.
def analyze_events(state: AgentState, llm) -> AgentState:
    """Use LLM to analyze the historical events"""
    # Create a system prompt to guide the LLM's analysis
    system_prompt = f"""You are a knowledgeable historian. Today is {state['current_date']}.

Your task is to analyze the search results about historical events that happened on this date and extract key insights.

Search Results:
{state['search_results']}"""

    try:
        # Create the full prompt for the LLM
        full_prompt = f"{system_prompt}\n\nPlease provide a detailed analysis of these historical events."
        
        # Invoke the LLM with the prompt
        response = llm.invoke(full_prompt)
        
        # Update the state with the analysis and a success message
        return {
            **state,
            "analysis": response,
            "messages": state["messages"] + [
                AIMessage(content="Analysis complete!")
            ]
        }
    except Exception as e:
        # Handle errors and update the state with an error message
        error_msg = f"Error analyzing events: {e}"
        return {
            **state,
            "analysis": error_msg,
            "messages": state["messages"] + [AIMessage(content=error_msg)]
        }

# 3. Structure Response Agent
# This agent structures the analyzed historical events into a well-organized and engaging response.
def structure_response(state: AgentState, llm) -> AgentState:
    """Use LLM to structure the analyzed historical events into a response"""
    # Create a system prompt to guide the LLM's response structuring
    system_prompt = f"""You are a historian and storyteller. Today is {state['current_date']}.

Your task is to structure the following analysis into a well-structured, engaging response:

Analysis:
{state['analysis']}"""

    try:
        # Create the full prompt for the LLM
        full_prompt = f"{system_prompt}\n\nPlease structure the analysis as follows:\n1. **Introduction**\n2. **Major Historical Events**\n3. **Notable Births/Deaths**\n4. **Fun Facts**\n5. **Reflection**"
        
        # Invoke the LLM with the prompt
        response = llm.invoke(full_prompt)
        
        # Update the state with the final response and a success message
        return {
            **state,
            "final_response": response,
            "messages": state["messages"] + [
                AIMessage(content="Response structuring complete!")
            ]
        }
    except Exception as e:
        # Handle errors and update the state with an error message
        error_msg = f"Error structuring response: {e}"
        return {
            **state,
            "final_response": error_msg,
            "messages": state["messages"] + [AIMessage(content=error_msg)]
        }

def should_continue(state: AgentState) -> Literal["analyze", "end"]:
    """Determine if we should continue to analysis or end"""
    if "search_results" in state and state["search_results"]:
        return "analyze"
    return "end"

# Build the graph
# This function creates the LangGraph workflow, which defines the sequence of operations for the AI agent.
def create_agent_graph():
    """Create the LangGraph workflow"""
    # Initialize the LLM (Ollama) and the search tool (Tavily API)
    llm, search_tool = initialize_components()
    
    # If initialization fails, return None to indicate an error
    if not llm or not search_tool:
        return None
    
    # Create the state graph, which manages the flow of data and operations
    workflow = StateGraph(AgentState)
    
    # Add nodes to the graph
    # Each node represents a specific operation in the workflow
    workflow.add_node("search", lambda state: search_historical_events(state, search_tool))
    workflow.add_node("analyze", lambda state: analyze_events(state, llm))
    workflow.add_node("structure", lambda state: structure_response(state, llm))
    
    # Define the edges between nodes
    # This specifies the order in which the nodes are executed
    workflow.add_edge("search", "analyze")
    workflow.add_edge("analyze", "structure")
    workflow.add_edge("structure", END)  # END signifies the end of the workflow
    
    # Set the entry point of the workflow
    # The workflow starts with the "search" node
    workflow.set_entry_point("search")
    
    # Compile the graph to finalize its structure
    return workflow.compile()

# Streamlit UI
# This function defines the user interface for the application using Streamlit.
def main():
    # Set the page configuration for the Streamlit app
    st.set_page_config(
        page_title="Historical Events AI Agent",  # Title of the web page
        page_icon="📚",  # Icon displayed in the browser tab
        layout="wide"  # Use a wide layout for the app
    )
    
    # Display the main title of the application
    st.title("📚 Historical Events AI Agent")
    st.markdown("*Discover what happened on this day in history using AI*")
    
    # Sidebar with additional information and requirements
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This AI agent:
        - 🔍 Searches for historical events using Tavily API
        - 🤖 Processes results with local LLM (Ollama)
        - 📖 Provides structured, engaging summaries
        - 🏠 Runs completely on your local machine
        """)
        
        st.header("Requirements")
        st.markdown("""
        - Ollama installed with a model (e.g., mistral)
        - Tavily API key
        - Python dependencies installed
        """)
        
        # Display the current date in the sidebar
        current_date = datetime.now().strftime("%B %d, %Y")
        st.info(f"Today's Date: {current_date}")
    
    # Check if the Tavily API key is set
    if not TAVILY_API_KEY:
        # Display an error message if the API key is missing
        st.error("⚠️ Please set your TAVILY_API_KEY environment variable")
        st.code("export TAVILY_API_KEY=your_api_key_here")
        return
    
    # Main interface with two columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Button to trigger the discovery of historical events
        if st.button("🔍 Discover Today's Historical Events", type="primary", use_container_width=True):
            # Create and run the agent workflow
            agent = create_agent_graph()
            
            # If the agent fails to initialize, display an error message
            if agent is None:
                st.error("Failed to initialize the agent. Please check your setup.")
                return
            
            # Initialize the state for the workflow
            initial_state = {
                "messages": [HumanMessage(content="Find historical events for today")],
                "current_date": "",
                "search_results": "",
                "final_response": ""
            }
            
            # Run the workflow with a progress spinner
            with st.spinner("🔍 Searching for historical events..."):
                try:
                    # Execute the workflow and get the result
                    result = agent.invoke(initial_state)
                    
                    # Display the results in the second column
                    with col2:
                        if result.get("final_response"):
                            st.markdown("## 📖 Historical Events Analysis")
                            st.markdown(result["final_response"])
                        else:
                            st.error("No results generated. Please try again.")
                            
                        # Show raw search results in an expandable section
                        if result.get("search_results"):
                            with st.expander("🔍 Raw Search Results"):
                                st.text(result["search_results"])
                                
                except Exception as e:
                    # Display an error message if the workflow fails
                    st.error(f"Error running the agent: {e}")
    
    # Footer with additional information
    st.markdown("---")
    st.markdown("Built with ❤️ using Ollama, LangGraph, Tavily, and Streamlit")

if __name__ == "__main__":
    main()