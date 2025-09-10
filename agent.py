import pandas as pd
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

#Langgraph
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition

load_dotenv()

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



def query_hotels(city: str=None, country: str=None, star_rating: int=None, cleanliness: int=None, 
                 comfort: int=None, facilities: int=None, sort_by: str=None, num_results: int=None) -> pd.DataFrame:
    """Queries hotels based on provided criteria. 
    city, country (case-insensitive string filters),
    minimum thresholds for star rating, cleanliness, comfort, and facilities (using parameters star_rating, cleanliness, comfort, facilities),
    sorting by a selected column (e.g., star rating, cleanliness, comfort, facilities using ),
    limiting the number of results returned
    Returns: a DataFrame of hotels matching the criteria.
    """

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





def main():
    st.title("Hotel QA Agent")
    st.markdown("---")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Please set the OPENAI_API_KEY environment variable.")
        st.stop()

if __name__ == "__main__":
    main()   