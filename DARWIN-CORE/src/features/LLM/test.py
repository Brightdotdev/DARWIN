import json
import logging
import sys
from typing import Any, Dict, List, Callable, Union

from src.lib.KeyManager import load_keys_from_keychain


# Assume LLMClient and LLMResponse are already imported from your codebase
# from your_module import LLMClient, LLMResponse

# ---------------------------------------------------------------------------
# 1. Define the tools we want the agent to use
# ---------------------------------------------------------------------------

def calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression and return the result."""
    try:
        # WARNING: eval can be dangerous – restrict allowed operations for production
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def get_weather(city: str) -> str:
    """Return a fake weather report for the given city."""
    # In production you'd call a real API
    weather_data = {
        "london": "rainy, 12°C",
        "paris": "cloudy, 15°C",
        "berlin": "sunny, 20°C",
    }
    return weather_data.get(city.lower(), f"No data for {city}")

# Map tool names to their implementations and JSON schemas (for the LLM prompt)
TOOLS: Dict[str, Dict[str, Any]] = {
    "calculator": {
        "description": "Evaluate a mathematical expression. Input: expression (string).",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "The math expression, e.g., '2+2'"}
            },
            "required": ["expression"]
        },
        "function": calculator,
    },
    "get_weather": {
        "description": "Get the current weather for a city. Input: city (string).",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g., 'London'"}
            },
            "required": ["city"]
        },
        "function": get_weather,
    }
}

# ---------------------------------------------------------------------------
# 2. JSON schema that the LLM must follow in its response
# ---------------------------------------------------------------------------
AGENT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {
            "type": "string",
            "description": "Your reasoning about what to do next"
        },
        "tool_call": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": list(TOOLS.keys())},
                "arguments": {"type": "object"}  # flexible; you could make it stricter per tool
            },
            "required": ["name", "arguments"]
        },
        "final_answer": {
            "type": "string",
            "description": "The final answer to the user; set only when ready"
        }
    },
    "required": ["thought"],
    "oneOf": [
        {"required": ["tool_call"]},
        {"required": ["final_answer"]}
    ],
    "additionalProperties": False
}

# ---------------------------------------------------------------------------
# 3. The agentic loop
# ---------------------------------------------------------------------------

import logging
from typing import Dict, List

import json
import logging
from typing import Dict, List


def run_agent(
    client,               # LLMClient instance
    user_query: str,
    max_iterations: int = 5,
    tier: str = "smart",
    debug: bool = True,
) -> str:
    tool_descriptions = "\n".join(
        f"- {name}: {info['description']}"
        for name, info in TOOLS.items()
    )
    system_prompt = (
        "You are a helpful assistant with access to the following tools:\n"
        f"{tool_descriptions}\n\n"
        "You must respond ONLY with a JSON object that contains a 'thought' field.\n"
        "If you need to use a tool, include a 'tool_call' object with 'name' and 'arguments'.\n"
        "Arguments MUST be a JSON object with named parameters, NOT an array.\n"
        'Example: {"tool_call": {"name": "calculator", "arguments": {"expression": "2+2"}}}\n'
        "When you have ALL the information needed to answer the user's question, "
        "include a 'final_answer' field with the complete answer.\n"
        "IMPORTANT: Once you have all tool results, you MUST provide a final_answer immediately.\n"
        "Do NOT include both 'tool_call' and 'final_answer' in the same response."
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_query},
    ]

    # Track what we've learned
    tool_results_summary = []

    for step in range(max_iterations):
        if debug:
            print(f"\n{'='*60}")
            print(f"STEP {step+1}/{max_iterations}")
            print(f"{'='*60}")

        response = client._call(
            messages,
            tier=tier,
            response_format={"type": "json_object"},
        )

        if response.error:
            return f"LLM error: {response.error}"

        if debug:
            print(f"Raw response:\n{response.text}")

        try:
            parsed = json.loads(response.text)
        except json.JSONDecodeError as e:
            error_msg = f"Your last response was not valid JSON. Error: {e}. Please follow the required format exactly."
            messages.append({"role": "user", "content": error_msg})
            if debug:
                print(f"JSON parse error: {e}")
            continue

        thought = parsed.get("thought", "")
        if debug:
            print(f"Thought: {thought}")

        # Case 1: final answer - WE FOUND IT!
        if "final_answer" in parsed and parsed["final_answer"] is not None:
            if debug:
                print(f"FINAL ANSWER: {parsed['final_answer']}")
            return parsed["final_answer"]

        # Case 2: tool call
        if "tool_call" in parsed and parsed["tool_call"] is not None:
            tool_req = parsed["tool_call"]
            tool_name = tool_req.get("name")
            tool_args = tool_req.get("arguments", {})

            if debug:
                print(f"Tool call: {tool_name}")
                print(f"Arguments: {json.dumps(tool_args, indent=2)}")

            # Validate arguments format
            if not isinstance(tool_args, dict):
                error_msg = (
                    f"Error: arguments must be a JSON object with named parameters, not a {type(tool_args).__name__}. "
                    f'Example: {{"city": "Paris"}} not ["Paris"]'
                )
                messages.append({"role": "user", "content": error_msg})
                continue

            if tool_name not in TOOLS:
                error_msg = f"Error: unknown tool '{tool_name}'. Available: {list(TOOLS.keys())}"
                messages.append({"role": "user", "content": error_msg})
                continue

            # Execute the tool
            try:
                result = TOOLS[tool_name]["function"](**tool_args)
                if debug:
                    print(f"Tool result: {result}")
            except Exception as e:
                result = f"Error: {e}"
                if debug:
                    print(f"EXECUTION ERROR: {e}")

            # Track what we learned
            tool_results_summary.append(f"- {tool_name}: {result}")
            
            # CRITICAL FIX: Tell the model what it now knows and push it toward a final answer
            results_so_far = "\n".join(tool_results_summary)
            feedback = (
                f"Tool '{tool_name}' returned: {result}\n\n"
                f"Results gathered so far:\n{results_so_far}\n\n"
                f"If you now have ALL the information needed to answer the user's original question "
                f"('{user_query}'), respond with a 'final_answer' field. "
                f"Do NOT make more tool calls if you already have everything you need."
            )
            messages.append({"role": "user", "content": feedback})
            continue

        # Case 3: Neither - this is the problem case from Step 3
        # The model has tool results but didn't provide final_answer or tool_call
        if tool_results_summary:
            # Remind the model what it knows and demand a final answer
            results_so_far = "\n".join(tool_results_summary)
            error_msg = (
                f"Your response must contain either 'tool_call' or 'final_answer'. "
                f"You have already gathered these results:\n{results_so_far}\n\n"
                f"If you have enough information to answer the user's question "
                f"('{user_query}'), provide a 'final_answer' NOW. "
                f"If you still need more information, specify a 'tool_call'."
            )
        else:
            error_msg = (
                "Your response must contain either 'tool_call' or 'final_answer'. "
                "Please include one of them."
            )
        
        messages.append({"role": "user", "content": error_msg})
        if debug:
            print(f"FORMAT ERROR: {error_msg}")

    return "Agent did not produce a final answer within the allowed iterations."

# ---------------------------------------------------------------------------
# 4. Example simulation using your client
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Minimal client setup for demonstration ---
    # In your real code you would already have a configured LLMClient.
    # Here we create a dummy one (assuming an OpenAI key is set in env).
    from src.features.LLM.LLMManager import LLMManager

    
    load_keys_from_keychain(log=True)
    initialized = LLMManager.initialize()
    if not initialized:
        print("\n⚠️  No LLM configuration found.")
        print("   Run: uv run main.py --wizard\n")
        sys.exit(1)

    client = LLMManager.get_client()

    # --- Run a query that requires tools ---
    question = "What is the weather in Paris and what is 123 * 456?"
    print(f"User: {question}")
    final = run_agent(client, question)
    print(f"Agent: {final}")