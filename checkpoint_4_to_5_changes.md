# Changes from Checkpoint 4 to Checkpoint 5

## Current State

We have a bot that can access external data through tools and can track user orders and provide details about their order shipments. However, it uses a simple tool-calling approach where the LLM directly calls tools and gets responses in a single iteration. The bot lacks proper reasoning capabilities and structured conversation flow.

## Failing Queries

- Complex multi-step queries that require reasoning
- Queries that need multiple tool calls in sequence
- Queries requiring the agent to think through the problem step by step

## Problems with the Current Version

The bot uses a basic tool-calling approach where:

- It directly calls tools without proper reasoning
- It lacks structured conversation flow
- It doesn't follow the ReAct (Reasoning + Acting) pattern
- It has limited ability to handle complex multi-step queries
- The conversation history is not properly maintained

## What is the solution?

We need to implement a proper ReAct (Reasoning + Acting) pattern where the agent:

1. **Reasons** about what to do next
2. **Acts** by calling appropriate tools
3. **Observes** the results and continues reasoning

This requires:

- Structured response format with "Thought:", "Action:", and "Final Answer:" sections
- Proper conversation history management
- Multi-iteration reasoning loops
- Better error handling and iteration limits

## Code to Modify

### 1. Remove ToolMessage import and tool-calling approach

**File: agent.py**

- Remove `ToolMessage` from the import statement on line 18
- Change from:

```python
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage, AIMessage
```

- To:

```python
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
```

### 2. Update system prompt to use external template

**File: agent.py**

- Modify the `_create_system_prompt()` function to use the external prompt template:

```python
def _create_system_prompt(self) -> str:
    """Create system prompt with available tools"""
    tool_descs = "\n".join([f"- {t.name}: {t.description}" for t in self.tools])
    tool_names = ", ".join([t.name for t in self.tools])
    current_date = datetime.now().strftime("%Y-%m-%d")
    return SYSTEM_PROMPT_TEMPLATE.format(
        tools=tool_descs, tool_names=tool_names, current_date=current_date
    )
```

### 3. Add action extraction and execution methods

**File: agent.py**

- Add `_extract_action_and_input()` method to parse structured responses:

```python
def _extract_action_and_input(
    self, text: str
) -> Tuple[Optional[str], Optional[Dict]]:
    """
    More robust extractor:
    - Finds the Action name (alphanumeric + underscores).
    - Finds 'Action Input:' and extracts the first balanced JSON object following it.
    - Returns (action_name, action_input_dict) or (None, None) if no action found.
    """
    action_match = re.search(r"Action:\s*([A-Za-z0-9_]+)", text)
    if not action_match:
        return None, None
    action_name = action_match.group(1)
    ai_idx = text.find("Action Input:")
    if ai_idx == -1:
        return action_name, {}
    brace_start = text.find("{", ai_idx)
    if brace_start == -1:
        return action_name, {}
    depth = 0
    end_idx = None
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end_idx = i
                break
    if end_idx is None:
        # Could not find balanced JSON — fall back
        return action_name, {}
    json_text = text[brace_start : end_idx + 1]
    if "'" in json_text and '"' not in json_text:
        return action_name, {}
    try:
        action_input = json.loads(json_text)
    except json.JSONDecodeError:
        return action_name, {}
    return action_name, action_input
```

- Add `_execute_action()` method to execute tools:

```python
def _execute_action(self, action_name: str, action_input: Dict) -> str:
    """
    Execute the tool
    """
    if action_name not in self.tools_by_name:
        return f"Error: Unknown action '{action_name}'"

    tool = self.tools_by_name[action_name]
    try:
        result = tool.invoke(action_input)
        # Convert dict/list results to JSON for consistent string output
        if isinstance(result, (dict, list)):
            response = result["data"]
            return response
        return str(result)
    except Exception as e:
        return f"Error executing action '{action_name}': {str(e)}"
```

### 4. Completely rewrite the run() method

**File: agent.py**

- Replace the entire `run()` method with a ReAct implementation:

```python
def run(self, query: str, max_iterations: int = 10) -> Dict[str, Any]:
    """
    Run the agent on a given query
    """
    self.reasoning_trace = []

    # Create conversation history with system prompt
    messages = [SystemMessage(content=self._create_system_prompt())]

    # Add current query
    messages.append(HumanMessage(content=query))

    # add chat history to the messages list to provide context to the llm

    for iteration in range(max_iterations):
        try:
            # Get response from LLM
            response = self.llm.invoke(messages)
            logger.info(f"Response: {response}")
            response_text = response.content

            # Add to reasoning trace
            self.reasoning_trace.append(
                {"iteration": iteration + 1, "response": response_text}
            )

            # check for thought
            if "Thought:" in response_text:
                agent_thought = response_text.split("Thought:")[-1].strip()
                agent_thought = agent_thought.split("Action:")[0].strip()
                agent_thought = agent_thought.split("Final Answer:")[0].strip()
                messages.append(AIMessage(content=agent_thought))

            # Check for final answer
            if "Final Answer:" in response_text:
                final_answer = response_text.split("Final Answer:")[-1].strip()
                return {
                    "final_answer": final_answer,
                    "reasoning_trace": self.reasoning_trace,
                    "num_iterations": iteration + 1,
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
                    "num_iterations": iteration + 1,
                    "success": True,
                }

            # Execute action
            observation = self._execute_action(action_name, action_input)

            # Add observation to conversation
            messages.append(AIMessage(content=response_text))
            messages.append(
                AIMessage(name=action_name, content=f"Observation: {observation}")
            )

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
                "num_iterations": iteration + 1,
                "success": False,
            }

    # Max iterations reached
    return {
        "final_answer": "I reached the maximum number of iterations without finding a complete answer.",
        "reasoning_trace": self.reasoning_trace,
        "success": False,
    }
```

### 5. Create external prompt template

**File: prompt.py**

- Create a new file with structured ReAct prompt template:

```python
SYSTEM_PROMPT_TEMPLATE = """You are a helpful and professional customer service assistant for an e-commerce company.
You assist customers by answering questions about their orders, shipments, and related issues.
If needed, you can call tools to fetch accurate information before replying.

current date: {current_date}

You have access to the following tools (name and input schema):
{tools}

Strict response format rules (follow exactly):

1) You must ALWAYS respond in ONE of these two forms — either an ACTION or a FINAL ANSWER.
   - Never mix both in the same step.
   - Never output an empty response.

FORMAT 1 — to call a tool (ACTION):
Thought: <short reasoning about what to do next>
Action: <tool_name> (always choose from allowed tools: {tool_names})
Action Input: <JSON object with parameters for the tool, EXACTLY one JSON object; must use double quotes and no extra text>
(Do NOT include any explanatory text between or inside the JSON. Only the JSON object.)

FORMAT 2 — to finish (FINAL ANSWER):
Thought: <final reasoning>
Final Answer: <a clear, polite, helpful reply to the customer in natural language>

2) JSON rules for Action Input:
- Must be a valid JSON object (e.g. {{"order_id": "12345"}}).
- Use double quotes for keys and string values.
- Do NOT include comments or trailing commas.
- If no parameters are required, use an empty object `{{}}`.
- Keys must match the tool's parameter names exactly as shown above.

3) Examples (copy style exactly):

Example ACTION:
Thought: I should fetch the order details to confirm shipping status.
Action: get_order
Action Input: {{"order_id": "ORD-12345"}}

Example FINAL ANSWER:
Thought: I now know the final answer.
Final Answer: Your order ORD-12345 was shipped on 2025-09-09 and is expected to arrive in 2 days.

4) Important continuation rules:
- After receiving an Observation, you MUST respond again.
- If the Observation provides enough information to answer the customer's question, then always produce a FINAL ANSWER.
- If the Observation is not sufficient, then produce another ACTION.
- Never produce an empty response.
- You must always continue until you output a FINAL ANSWER.

You may reference conversation history if available. Always be concise and return only one of the two allowed formats and never return an empty response.
"""
```

## Testing

The following queries will now work with proper reasoning:

- "List all orders for customer 1001" - Agent will reason about needing to call get_order_by_customer_id, execute it, and provide a final answer
- "Where is my order #1234?" - Agent will reason about calling get_order first, then potentially get_shipment_by_order_id, and provide a comprehensive answer
- "What's the status of my recent orders?" - Agent will reason through multiple steps to gather information and provide a complete response

## Key Improvements

1. **Structured Reasoning**: The agent now follows a proper Thought → Action → Observation → Final Answer pattern
2. **Multi-iteration Support**: Can handle complex queries requiring multiple tool calls
3. **Better Error Handling**: Proper iteration limits and error recovery
4. **Conversation History**: Maintains context across multiple reasoning steps
5. **Flexible Response Format**: Can handle both tool calls and direct answers
6. **Robust Parsing**: Better extraction of actions and parameters from LLM responses
