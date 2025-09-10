"""
agent.py - ReAct agent implementation using LangChain and Gemini

This file contains:
- ReAct agent class that implements reasoning loops
- Integration with Gemini LLM via LangChain
- Tool calling and structured output handling
- Reasoning trace tracking for debugging
"""

import json
import os
import re
from typing import Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from tools import (
    get_order,
    get_shipment,
    get_shipment_by_order_id,
    get_order_by_customer_id,
)
from prompt import SYSTEM_PROMPT


class ReActAgent:
    """
    ReAct (Reasoning and Acting) Agent implementation

    This agent follows the ReAct pattern:
    1. Thought: Reason about what to do next
    2. Action: Choose and execute an action
    3. Observation: Observe the result
    4. Repeat until final answer
    """

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """
        Initialize the ReAct agent

        Args:
            api_key: Google API key for Gemini
            model_name: Name of the Gemini model to use
        """
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, google_api_key=api_key, temperature=0.1
        )

        self.tools = [
            get_order,
            get_shipment,
            get_shipment_by_order_id,
            get_order_by_customer_id,
        ]
        self.tools_by_name = {tool.name: tool for tool in self.tools}

        self.reasoning_trace = []

    def _extract_action_and_input(
        self, text: str
    ) -> tuple[Optional[str], Optional[Dict]]:
        """
        Extract action name and input from agent response

        Args:
            text: The agent's response text

        Returns:
            Tuple of (action_name, action_input) or (None, None)
        """
        # Look for action pattern: Action: action_name
        action_match = re.search(r"Action:\s*(\w+)", text)
        if not action_match:
            return None, None

        action_name = action_match.group(1)

        # Look for input pattern: Action Input: {...}
        input_match = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)
        if not input_match:
            return action_name, {}

        try:
            action_input = json.loads(input_match.group(1))
            return action_name, action_input
        except json.JSONDecodeError:
            return action_name, {}

    def _execute_action(self, action_name: str, action_input: Dict) -> str:
        """
        Execute the specified action with given input

        Args:
            action_name: Name of the action to execute
            action_input: Input parameters for the action

        Returns:
            String result of the action execution
        """

        if action_name not in self.tools_by_name:
            return f"Error: Unknown action '{action_name}'"

        try:
            result = self.tools_by_name[action_name].invoke(action_input)
            return str(result)
        except Exception as e:
            return f"Error executing action '{action_name}': {str(e)}"

    def _create_system_prompt(self) -> str:
        """Create the system prompt for the ReAct agent"""
        return SYSTEM_PROMPT

    def run(
        self,
        query: str,
        max_iterations: int = 10,
        chat_history: list = None,
    ) -> Dict[str, Any]:
        """
        Run the ReAct agent on a given query

        Args:
            query: The user's question
            max_iterations: Maximum number of reasoning iterations
            chat_history: Previous conversation history

        Returns:
            Dictionary containing the final answer and reasoning trace
        """
        self.reasoning_trace = []

        # Create conversation history with system prompt
        messages = [SystemMessage(content=self._create_system_prompt())]

        # Add chat history if provided
        if chat_history:
            for msg in chat_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

        # Add current query
        messages.append(HumanMessage(content=query))

        for iteration in range(max_iterations):
            try:
                # Get response from LLM
                response = self.llm.invoke(messages)
                response_text = response.content

                # Add to reasoning trace
                self.reasoning_trace.append(
                    {"iteration": iteration + 1, "response": response_text}
                )

                # Check for final answer
                if "Final Answer:" in response_text:
                    final_answer = response_text.split("Final Answer:")[-1].strip()
                    return {
                        "final_answer": final_answer,
                        "reasoning_trace": self.reasoning_trace,
                        "success": True,
                    }

                # Extract action and input
                action_name, action_input = self._extract_action_and_input(
                    response_text
                )

                if action_name is None:
                    # No action found, treat as final answer
                    return {
                        "final_answer": response_text,
                        "reasoning_trace": self.reasoning_trace,
                        "success": True,
                    }

                # Execute action
                observation = self._execute_action(action_name, action_input)

                # Add observation to conversation
                messages.append(AIMessage(content=response_text))
                messages.append(HumanMessage(content=f"Observation: {observation}"))

                # Add to reasoning trace
                self.reasoning_trace.append(
                    {
                        "iteration": iteration + 1,
                        "action": action_name,
                        "action_input": action_input,
                        "observation": observation,
                    }
                )

            except Exception as e:
                error_msg = f"Error in iteration {iteration + 1}: {str(e)}"
                self.reasoning_trace.append(
                    {"iteration": iteration + 1, "error": error_msg}
                )
                return {
                    "final_answer": f"I encountered an error: {error_msg}",
                    "reasoning_trace": self.reasoning_trace,
                    "success": False,
                }

        # Max iterations reached
        return {
            "final_answer": "I reached the maximum number of iterations without finding a complete answer.",
            "reasoning_trace": self.reasoning_trace,
            "success": False,
        }


def create_agent() -> ReActAgent:
    """
    Create and return a ReAct agent instance

    Args:
        api_key: Google API key for Gemini

    Returns:
        Configured ReActAgent instance
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    return ReActAgent(api_key)
