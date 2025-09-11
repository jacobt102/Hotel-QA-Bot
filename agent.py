import pandas as pd
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.tools import tool
from typing import Optional, Annotated, TypedDict

#Langgraph
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt.tool_node import ToolNode

load_dotenv()

#State
class State(TypedDict):
    messages: Annotated[list,add_messages]



#Load csv into df and only keep necessary columns
@st.cache_resource
def get_csv():
    """Load hotel data from CSV file"""
    return pd.read_csv("hotels.csv").drop(["location_base","staff_base","value_for_money_base"],axis=1)
st.set_page_config(page_title="Hotel QA Agent", layout="centered")

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = InMemoryChatMessageHistory()

@st.cache_resource
def get_model():
    """Initialize the language model"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Please set the OPENAI_API_KEY environment variable.")
        st.stop()
    return ChatOpenAI(model="gpt-4o-mini",temperature=0.6,openai_api_key=api_key)


@tool
def query_hotels(city: Optional[str]=None, country: Optional[str]=None, star_rating: Optional[int]=None, cleanliness: Optional[int]=None, 
                 comfort: Optional[int]=None, facilities: Optional[int]=None, sort_by: Optional[str]=None, num_results: Optional[int]=10) -> pd.DataFrame:
    """Queries hotels based on provided criteria. 
    city, country (case-insensitive string filters),
    minimum thresholds for star rating, cleanliness, comfort, and facilities (using parameters star_rating, cleanliness, comfort, facilities),
    sorting by a selected column (e.g., star rating, cleanliness, comfort, facilities using ),
    limiting the number of results returned
    Returns: a DataFrame of hotels matching the criteria.
    """
    #Bind num_results between 1 and 10
    num_results = min(max(num_results,1),10)
    csv = get_csv()
    if city:
        csv = csv[csv['city'].str.lower() == city.lower()]
    if country:
        csv = csv[csv['country'].str.lower() == country.lower()]
    if star_rating:
        csv = csv[csv['star_rating'] >= star_rating]
    if cleanliness:
        csv = csv[csv['cleanliness_base'] >= cleanliness]
    if comfort:
        csv = csv[csv['comfort_base'] >= comfort]
    if facilities:
        csv = csv[csv['facilities_base'] >= facilities]
    if sort_by:
        csv = csv.sort_values(by=sort_by, ascending=False)
    if num_results:
        csv = csv.head(num_results)
    return csv

@st.cache_resource
def get_graph():
    llm = get_model()
    
    tools = [query_hotels] if query_hotels else []
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    
    def chatbot(state: State):
        result = llm_with_tools.invoke(state["messages"])
        return {"messages", [result]}
    
    
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot",chatbot)
    
    if tools:
        tool_node = ToolNode(tools=tools)
        graph_builder.add_node("tools",tool_node)
        graph_builder.add_conditional_edges("chatbot", tools_condition)
        graph_builder.add_edge("tools", "chatbot")

    graph_builder.add_edge(START, "chatbot")
    graph = graph_builder.compile()
    return graph

def invoke_graph(all_messages):
    """
    all_messages: list of langchain_core.messages BaseMessage
    Returns the final assistant message text.
    """
    graph = get_graph()
    # Run the graph with the accumulated messages
    state = graph.invoke({"messages": all_messages})
    # Find the last assistant message
    last_ai = None
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage):
            last_ai = m
            break
    return last_ai.content if last_ai else ""



    
if __name__ == "__main__":
    system_message = SystemMessage(content="You are a helpful assistant that helps users find hotels based on their preferences. " \
    "Use the provided tool to query hotel data as needed. ")
    prior_messages = [system_message]
    prior_messages.append(HumanMessage(content="I want to find a hotel in Paris, France with at least 4 stars"))
    text = invoke_graph(prior_messages)  
    print(text)