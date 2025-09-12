import pandas as pd
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.tools import tool
from typing import Optional, Annotated, TypedDict

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
def load_csv():
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
                 comfort: Optional[int]=None, facilities: Optional[int]=None, sort_by: Optional[str]=None, num_results: Optional[int]=10) -> str:
    """Queries hotels based on provided criteria. 
    city, country (case-insensitive string filters),
    minimum thresholds for star rating, cleanliness, comfort, and facilities (using parameters star_rating, cleanliness, comfort, facilities),
    sorting by a selected column (e.g., star rating, cleanliness, comfort, facilities using ),
    limiting the number of results returned
    Returns: a string matching the dataframe of hotels matching the criteria.
    """
    COLUMN_MAP = {
        "comfort": "comfort_base",
        "cleanliness": "cleanliness_base",
        "facilities": "facilities_base",
        "star_rating": "star_rating"
    }
    #Bind num_results between 1 and 10
    num_results = min(max(num_results,1),10)
    csv = load_csv()
    
    # Apply filters
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
    
    # Apply sorting if necessary and converting from dict
    if sort_by:
        sort_col = COLUMN_MAP.get(sort_by, sort_by)  
        if sort_col in csv.columns:
            csv = csv.sort_values(by=sort_col, ascending=False)
    
    # Limit results
    csv = csv.head(num_results)
    
    return csv.to_string(index=False)

@st.cache_resource
def get_graph():
    """
    Construct and return the LangGraph graph for the chatbot. The graph includes the following nodes:
    - chatbot: the main chatbot node that takes the user messages and returns the assistant response
    - tools: the node containing the query_hotels tool
    The graph also includes the following edges:
    - tools -> chatbot: conditional edge that invokes the tools node with the user messages
    - START -> chatbot: edge that starts the graph execution from the chatbot node
    The graph is compiled and cached using Streamlit's @st.cache_resource decorator.
    """
    llm = get_model()
    
    tools = [query_hotels]
    llm_with_tools = llm.bind_tools(tools)
    
    def chatbot(state: State):
        result = llm_with_tools.invoke(state["messages"])
        return {"messages": [result]}  
    
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
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
    # Run the graph w/ messages
    state = graph.invoke({"messages": all_messages})
    # Find the last assistant message
    last_ai = None
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage):
            last_ai = m
            break
    return last_ai.content if last_ai else "Response generation failed."

def main():
    initialize_session_state()
    
    global _hotel_data
    _hotel_data = load_csv()

    st.title("Hotel AI Assistant")
    st.markdown("---")

    graph = get_graph()
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not found in .env file")
        st.info(
            "To fix this:\n"
            "1. Create a `.env` file in the project root\n"
            "2. Add your OpenAI API key: `OPENAI_API_KEY=your_api_key_here`\n"
            "3. Get your API key from: https://platform.openai.com/api-keys"
        )
        st.stop()

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button(" Clear Chat"):
            st.session_state.messages = []
            st.session_state.chat_history = InMemoryChatMessageHistory()
            st.rerun()

    with col2:
        st.markdown("Start chatting below! You can ask about hotels and I can help you:")
        st.markdown("* Filter by city, country, star rating, cleanliness, comfort, and facilities")
        st.markdown("* Sort by star rating, cleanliness, comfort, or facilities")
        st.markdown("* Limit the number of results")

    st.markdown("---")

    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            elif message["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(message["content"])

    user_response = st.chat_input("Ask me about hotels!")
    
    if user_response:
        user_msg = HumanMessage(content=user_response)
        
        st.session_state.messages.append({"role": "user", "content": user_response})
        st.session_state.chat_history.add_message(user_msg)

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_response)

        with st.chat_message("assistant"):
            with st.spinner(" Thinking..."):
                try:
                    system_message = SystemMessage(content="""You are a helpful hotel assistant. Use the query_hotels tool to search for hotels based on user requests. 
                    
                    When a user asks for hotels:
                    1. Use the query_hotels tool with appropriate parameters
                    2. Present the results in a friendly way
                    3. If no results, suggest alternative searches
                    4. If a city or country is not provided, don't filter by it and print overall results for the query.
                    5. List out all hotel details when outputting results.
                    6. Do not give out any hotel information that is not in the dataset.
                    Always use the tool for hotel queries. When outputting results, give all available details for each hotel.""")

                    # Compose full message list for the graph
                    all_msgs = [system_message] + st.session_state.chat_history.messages

                    ai_text = invoke_graph(all_msgs)
                        
                    
                    if ai_text:
                        st.write(ai_text)
                        st.session_state.messages.append({"role": "assistant", "content": ai_text})
                        ai_msg = AIMessage(content=ai_text)
                        st.session_state.chat_history.add_message(ai_msg)
                    else:
                        error_msg = "No response generated"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

                except Exception as e:
                    error_msg = f" Error: {str(e)}"
                    st.error(error_msg)
                    st.info("Please check your API keys, internet connection, and ensure hotels.csv exists.")
                    ai_msg = AIMessage(content=f"I encountered an error: {str(e)}")
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.session_state.chat_history.add_message(ai_msg)

if __name__ == "__main__":
    main()