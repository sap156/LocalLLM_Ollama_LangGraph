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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")  # Default to 'mistral' if not set
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # Get from .env file or environment

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
def search_historical_events(state: AgentState, search_tool) -> AgentState:
    """Search for historical events that happened on today's date"""
    current_date = datetime.now().strftime("%B %d")
    
    search_query = f"historical events that happened on {current_date} in history famous important"
    
    try:
        search_results = search_tool.invoke(search_query)
        
        # Format search results
        formatted_results = "\n\n".join([
            f"**Source {i+1}:** {result.get('title', 'No title')}\n"
            f"**URL:** {result.get('url', 'No URL')}\n"
            f"**Content:** {result.get('content', 'No content')}"
            for i, result in enumerate(search_results)
        ])
        
        return {
            **state,
            "current_date": current_date,
            "search_results": formatted_results,
            "messages": state["messages"] + [
                AIMessage(content=f"Found historical events for {current_date}")
            ]
        }
    except Exception as e:
        error_msg = f"Error searching for historical events: {e}"
        return {
            **state,
            "current_date": current_date,
            "search_results": error_msg,
            "messages": state["messages"] + [AIMessage(content=error_msg)]
        }

def analyze_and_structure_events(state: AgentState, llm) -> AgentState:
    """Use LLM to analyze and structure the historical events"""
    
    system_prompt = f"""You are a knowledgeable historian and storyteller. Today is {state['current_date']}.

Your task is to analyze the search results about historical events that happened on this date and create a well-structured, engaging response.

Please structure your response as follows:
1. **Introduction**: A brief, engaging introduction about the significance of this date in history
2. **Major Historical Events**: List 3-5 of the most important/interesting events with:
   - Year and brief description
   - Why this event was significant
   - Any interesting details or connections
3. **Notable Births/Deaths**: If any famous people were born or died on this date
4. **Fun Facts**: Any interesting or lesser-known facts about events on this date
5. **Reflection**: A brief concluding thought about what we can learn from these historical events

Make it engaging and educational, suitable for a general audience. Use clear headings and bullet points where appropriate.

Search Results:
{state['search_results']}"""

    try:
        # Create the prompt for the LLM
        full_prompt = f"{system_prompt}\n\nPlease provide a comprehensive analysis of these historical events."
        
        response = llm.invoke(full_prompt)
        
        return {
            **state,
            "final_response": response,
            "messages": state["messages"] + [
                AIMessage(content="Analysis complete!")
            ]
        }
    except Exception as e:
        error_msg = f"Error analyzing events: {e}"
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
def create_agent_graph():
    """Create the LangGraph workflow"""
    llm, search_tool = initialize_components()
    
    if not llm or not search_tool:
        return None
    
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes with partial functions to pass tools
    workflow.add_node("search", lambda state: search_historical_events(state, search_tool))
    workflow.add_node("analyze", lambda state: analyze_and_structure_events(state, llm))
    
    # Define the workflow edges
    workflow.add_edge("search", "analyze")
    workflow.add_edge("analyze", END)
    
    # Set entry point
    workflow.set_entry_point("search")
    
    # Compile the graph
    return workflow.compile()

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Historical Events AI Agent",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Historical Events AI Agent")
    st.markdown("*Discover what happened on this day in history using AI*")
    
    # Sidebar with information
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
        
        current_date = datetime.now().strftime("%B %d, %Y")
        st.info(f"Today's Date: {current_date}")
    
    # Check if Tavily API key is set
    if not TAVILY_API_KEY:
        st.error("⚠️ Please set your TAVILY_API_KEY environment variable")
        st.code("export TAVILY_API_KEY=your_api_key_here")
        return
    
    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("🔍 Discover Today's Historical Events", type="primary", use_container_width=True):
            # Create and run the agent
            agent = create_agent_graph()
            
            if agent is None:
                st.error("Failed to initialize the agent. Please check your setup.")
                return
            
            # Initialize state
            initial_state = {
                "messages": [HumanMessage(content="Find historical events for today")],
                "current_date": "",
                "search_results": "",
                "final_response": ""
            }
            
            # Run the workflow with progress tracking
            with st.spinner("🔍 Searching for historical events..."):
                try:
                    result = agent.invoke(initial_state)
                    
                    # Display results
                    with col2:
                        if result.get("final_response"):
                            st.markdown("## 📖 Historical Events Analysis")
                            st.markdown(result["final_response"])
                        else:
                            st.error("No results generated. Please try again.")
                            
                        # Show search results in an expander
                        if result.get("search_results"):
                            with st.expander("🔍 Raw Search Results"):
                                st.text(result["search_results"])
                                
                except Exception as e:
                    st.error(f"Error running the agent: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Ollama, LangGraph, Tavily, and Streamlit")

if __name__ == "__main__":
    main()