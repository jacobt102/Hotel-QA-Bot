# Hotel AI Assistant

A conversational AI assistant built with Streamlit and LangGraph that helps users query and find hotels from a CSV dataset. The application uses **OpenAI's GPT-4o-mini** model to understand natural language queries and automatically calls appropriate tools to search and filter hotel data.

## Features

- **Natural Language Queries**: Ask questions about hotels in plain English
- **Advanced Filtering**: Filter hotels by city, country, star rating, cleanliness, comfort, and facilities
- **Intelligent Sorting**: Sort results by star rating, cleanliness, comfort, or facilities
- **Interactive Chat Interface**: Streamlit-based chat UI with conversation history
- **Tool-Augmented AI**: Uses LangGraph framework for structured AI tool calling
- **Error Handling**: Robust error handling and user feedback

## Requirements

### System Requirements
- Python 3.8 or higher
- Internet connection for OpenAI API access

### Dependencies
#### Also available in `requirements.txt` file
```
streamlit
pandas
python-dotenv
langchain-openai
langchain-core
langchain
langgraph
```

### Data Requirements
- `hotels.csv` file must be present in the project root directory
- Dataset used is linked [here](https://www.kaggle.com/datasets/alperenmyung/international-hotel-booking-analytics?select=hotels.csv)
- The CSV must contain the following columns:
  - `city`: Hotel city location
  - `country`: Hotel country location
  - `star_rating`: Hotel star rating (numeric)
  - `cleanliness_base`: Cleanliness score (numeric)
  - `comfort_base`: Comfort score (numeric)
  - `facilities_base`: Facilities score (numeric)
- The following columns will be automatically removed if present:
  - `location_base`
  - `staff_base`
  - `value_for_money_base`

### API Key Requirements
- Valid OpenAI API key with access to GPT-4o-mini model
- API key must be configured in environment variables

## Installation

1. **Clone or download the project files**
   ```bash
   git clone <repository-url>
   cd hotel-ai-assistant
   ```

2. **Install required Python packages**
   ```bash
   pip install streamlit pandas python-dotenv langchain-openai langchain-core langchain langgraph
   ```

3. **Create environment file**
   Create a `.env` file in the project root directory:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Prepare hotel data**
   Ensure `hotels.csv` is in the project root directory with the required columns.

## Usage

### Starting the Application
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Using the Chat Interface

1. **Basic Hotel Search**
   - "Show me hotels in Paris"
   - "Find hotels in France"
   - "Give me the best hotels"

2. **Filtered Searches**
   - "Show me 5-star hotels in New York"
   - "Find clean hotels with good facilities in Tokyo"
   - "Hotels in London with comfort rating above 8"

3. **Sorted Results**
   - "Show me hotels sorted by star rating"
   - "Find hotels sorted by cleanliness"
   - "Best facilities hotels in Italy"

4. **Limited Results**
   - "Show me top 5 hotels in Spain"
   - "Give me 3 luxury hotels"

### Query Parameters

The AI assistant can interpret and use the following search parameters:

- **city**: Filter by specific city (case-insensitive)
- **country**: Filter by specific country (case-insensitive)
- **star_rating**: Minimum star rating threshold (1-5)
- **cleanliness**: Minimum cleanliness score threshold
- **comfort**: Minimum comfort score threshold
- **facilities**: Minimum facilities score threshold
- **sort_by**: Sort results by 'star_rating', 'cleanliness', 'comfort', or 'facilities'
- **num_results**: Limit number of results (1-10, default: 10)

## Application Architecture

### Core Components

1. **Streamlit Frontend**
   - Chat interface for user interaction
   - Session state management for conversation history
   - Real-time message display

2. **LangGraph Workflow**
   - State management for conversation flow
   - Tool calling orchestration
   - Message routing between AI and tools

3. **Hotel Query Tool**
   - CSV data filtering and sorting
   - Parameter validation and constraints
   - Formatted result output

4. **OpenAI Integration**
   - GPT-4o-mini model for natural language understanding
   - Tool binding for structured function calls
   - Response generation

### Data Flow

1. User enters query in chat interface
2. Message added to conversation history
3. LangGraph processes message with system prompt
4. AI model determines if tool calling is needed
5. Hotel query tool filters CSV data based on parameters
6. Results returned to AI model
7. AI model formats and presents results to user
8. Response displayed in chat interface

### Error Handling

- **API Key Validation**: Checks for OpenAI API key on startup
- **File Validation**: Ensures hotels.csv exists and is readable
- **Session Context**: Handles Streamlit session context issues
- **Tool Execution**: Catches and reports tool execution errors
- **Network Issues**: Graceful handling of API connectivity problems

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Required OpenAI API key for GPT-4o-mini access

### System Settings
- **Model**: GPT-4o-mini (configurable in `get_model()` function)
- **Temperature**: 0.6 (controls AI response creativity)
- **Max Results**: 10 hotels per query (configurable)
- **Cache**: Streamlit resource caching for CSV data and model initialization

### UI Configuration
- **Page Title**: "Hotel QA Agent"
- **Layout**: Centered layout
- **Chat Input**: Bottom-positioned chat input field
- **Clear Chat**: Button to reset conversation history

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not found" Error**
   - Ensure `.env` file exists in project root
   - Verify API key is correctly formatted in `.env`
   - Check API key has proper permissions

2. **"NoSessionContext" Error**
   - Restart the application
   - Check if hotels.csv is accessible
   - Verify all dependencies are installed

3. **No Hotel Results**
   - Check if hotels.csv contains data
   - Verify column names match requirements
   - Try broader search criteria

4. **Import Errors**
   - Install all required packages: `pip install -r requirements.txt`
   - Check Python version compatibility
   - Verify virtual environment activation


### Performance Optimization

- CSV data is cached using `@st.cache_resource`
- Model initialization is cached
- Global data storage for tool access
- Limited result sets to prevent overload

## File Structure

```
project-root/
├── app.py                 # Main application file
├── hotels.csv            # Hotel dataset (required)
├── .env                  # Environment variables
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## API Usage Costs

- Uses OpenAI GPT-4o-mini model
- Costs depend on token usage per conversation
- Estimated cost: $0.0001-0.001 per query
- Monitor usage through OpenAI dashboard

## Support and Maintenance

### Regular Maintenance
- Update OpenAI API key if expired
- Refresh hotels.csv with new data as needed
- Update dependencies for security patches

### Extending Functionality
- Add new search parameters in `query_hotels` function
- Modify system prompt for different AI behavior
- Add new data sources or formats/ exapnd from csv
- Implement user authentication if needed

For technical support or feature requests, please refer to the project documentation or contact the development team.
