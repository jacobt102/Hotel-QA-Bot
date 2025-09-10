import pandas as pd
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory



load_dotenv()

#Load csv into df and only keep necessary columns
hotels = pd.read_csv("hotels.csv")
hotels = hotels.drop(["location_base","staff_base","value_for_money_base"],axis=1)
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

def main():
    st.title("Hotel QA Agent")
    st.markdown("---")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Please set the OPENAI_API_KEY environment variable.")
        st.stop()

        