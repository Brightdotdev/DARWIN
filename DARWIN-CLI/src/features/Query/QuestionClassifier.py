# src/features/Query/QuestionClassifier.py
"""
Classify the user's query into a QuestionType using rules + optional LLM.
Fixed: system prompt moved into messages list.
"""
from __future__ import annotations

import re
from typing import Optional
from src.features.LLM.LLMTypes import LLMResponse
from src.features.Query.QueryTypes import QuestionType

# ── Simple rule‑based patterns ─────────────────────────────────────────────
RULE_PATTERNS: list[tuple[str, QuestionType]] = [
    (r"\b(vs\.?|versus|compared?\s+(to|with)|differences?\s+between)\b", "comparison"),
    (r"\bhow\s+(to|do|can|would|should)\b", "instruction"),
    (r"^\s*(why|what\s+(is|are|was|were)|who|when|where)\b", "fact"),
    (r"\b(explain|elaborate|describe|tell\s+me)\b", "reasoning"),
    (r"\b(do\s+you\s+think|your\s+opinion|what\s+do\s+you\s+believe)\b", "opinion"),
    (r"\b(what\s+if|suppose|hypothet|imagine)\b", "hypothetical"),
    (r"\b((which|what)\s+one|should\s+I\s+(choose|pick))\b", "clarification"),
    (r"\b(multiple\s+part|several\s+questions|firstly|secondly)\b", "multi_part"),
    (r"^\s*(is\s+there|are\s+there|does|do|can|will|should|has|have)\b", "yes_no"),
]

CLASSIFIER_SYSTEM_PROMPT = (
    "You are a query classifier. Given a user question, determine the best category "
    "from the following list:\n\n"
    "fact, instruction, reasoning, comparison, opinion, hypothetical, clarification, "
    "multi_part, yes_no, statement\n\n"
    "Return a JSON object:\n"
    '{\n  "question_type": "<chosen category>",\n}'
)


def classify_by_rules(query: str) -> tuple[Optional[QuestionType], float]:
    q = query.lower().strip()
    for pattern, qtype in RULE_PATTERNS:
        if re.search(pattern, q):
            return qtype, 0.7
    return None, 0.0


async def classify_by_llm(client, query: str, conversation_summary: str = "") -> tuple[QuestionType, float]:
    prompt = f"Conversation so far: {conversation_summary}\nUser: {query}" if conversation_summary else query

    messages = [
        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    try:
        resp: LLMResponse = await client.acall_small(
            messages,
            temperature=0.0,
            json_schema={          # <-- using a JSON schema here is optional, but fine
                "type": "object",
                "properties": {
                    "question_type": {
                        "type": "string",
                        "enum": [
                            "fact", "instruction", "reasoning", "comparison",
                            "opinion", "hypothetical", "clarification",
                            "multi_part", "yes_no", "statement"
                        ]
                    },
                },
                "required": ["question_type"]
            },
        )
        if resp.error or not resp.json:
            return "statement", 0.0
        return resp.json["question_type"], resp.json.get("confidence", 0.5)
    except Exception:
        return "statement", 0.0


async def classify_query(
    client,
    query: str,
    conversation_summary: str = "",
    rule_threshold: float = 0.8
) -> QuestionType:
    rule_type, rule_conf = classify_by_rules(query)
    if rule_type is not None and rule_conf >= rule_threshold:
        return rule_type
    llm_type, _ = await classify_by_llm(client, query, conversation_summary)
    return llm_type