# ReAct Agent Demo

A minimal hands-on project demonstrating a ReAct (Reasoning and Acting) agent using Gemini via LangChain.

## Project Structure

```
├── app.py          # Streamlit UI for user interaction
├── agent.py        # ReAct agent implementation with LangChain
├── tools.py        # Tool definitions and in-memory data tables
├── requirements.txt # Python dependencies
└── README.md       # This file
```

## Features

- **ReAct Pattern**: Implements reasoning loops (Thought → Action → Observation → Final Answer)
- **Gemini Integration**: Uses Google's Gemini LLM via LangChain
- **Structured Output**: Pydantic schemas for data validation
- **Tool Calling**: Agent can query orders and shipments
- **Interactive UI**: Streamlit-based web interface
- **Reasoning Trace**: Full visibility into agent's decision process

## Setup

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Get Google API Key:**

   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key for use in the app

3. **Run the application:**

   ```bash
   streamlit run app.py
   ```

4. **Open your browser:**
   - Navigate to `http://localhost:8501`
   - Enter your Google API key in the sidebar
   - Ask questions about orders and shipments

## Sample Queries

- "Where is my order #1234?"
- "What's the status of order #1235?"
- "Tell me about shipment #1001"
- "Is order #1236 delivered?"
- "Show me all orders for customer 1001"
- "What orders did customer 1001 place in September 2025?"

## How It Works

1. **User Input**: User enters a question in the Streamlit interface
2. **Agent Reasoning**: The ReAct agent analyzes the question and decides what action to take
3. **Tool Execution**: Agent calls appropriate tools (get_order, get_shipment, etc.)
4. **Observation**: Agent processes the tool results
5. **Iteration**: Agent continues reasoning until it has enough information
6. **Final Answer**: Agent provides a complete, grounded answer

## Data

The demo includes sample data for:

- **Orders**: Order details, customer info, status, amounts
- **Shipments**: Tracking info, carrier, delivery status, location

## Architecture

- **tools.py**: Defines Pydantic schemas and tool functions
- **agent.py**: Implements the ReAct agent with LangChain integration
- **app.py**: Provides the Streamlit web interface

## Requirements

- Python 3.8+
- Google API key for Gemini
- Internet connection for API calls
