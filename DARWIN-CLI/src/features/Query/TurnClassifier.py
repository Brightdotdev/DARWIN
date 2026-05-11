# src/features/Query/TurnClassifier.py
"""
Classify the user's message into a TurnType using a small LLM call.
"""
from __future__ import annotations

from src.features.LLM.LLMTypes import LLMResponse
from src.features.Query.TurnTypes import TurnType

TURN_CLASSIFIER_SYSTEM_PROMPT = (
    "You are a conversation analyst. Given the user's latest message, the assistant's last response (if any), "
    "and an active follow-up question (if any), determine the type of turn.\n\n"
    "Turn types:\n"
    "- new_query: a completely new question or topic, not referencing the last response\n"
    "- follow_up_answer: the user is answering a follow-up question that was just asked\n"
    "- elaboration: asking for more details about a specific part of the last response (e.g., 'tell me more about point 2')\n"
    "- correction: correcting a mistake in the last response (e.g., 'that date is wrong')\n"
    "- re_render: asking to display the same information differently (e.g., 'make it a table', 'show as timeline')\n"
    "- chitchat: casual conversation, not a research request (e.g., 'thanks', 'hello')\n\n"
    "Return a JSON object: {\"turn_type\": \"<type>\", \"confidence\": 0.0-1.0}"
)


async def classify_turn(
    client,
    user_message: str,
    last_assistant_response: str = "",
    active_follow_up: str = "",
) -> TurnType:
    """Return the turn type determined by the small LLM."""
    # Build a concise prompt
    parts = [f"User message: {user_message}"]
    if last_assistant_response:
        parts.append(f"Assistant last said: {last_assistant_response[:200]}")
    if active_follow_up:
        parts.append(f"Active follow-up question: {active_follow_up[:200]}")
    prompt = "\n".join(parts)

    try:
        resp: LLMResponse = await client.acall_small(
            [
                {"role": "system", "content": TURN_CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            json_schema={
                "type": "object",
                "properties": {
                    "turn_type": {
                        "type": "string",
                        "enum": [
                            "new_query",
                            "follow_up_answer",
                            "elaboration",
                            "correction",
                            "re_render",
                            "chitchat",
                        ],
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["turn_type", "confidence"],
            },
        )
        if resp.error or not resp.json:
            return "new_query"
        return resp.json["turn_type"]
    except Exception:
        return "new_query"