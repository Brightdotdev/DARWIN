# src/features/Query/QueryRewriter.py
"""
Rewrite the user query for optimal retrieval.
Fixed: no longer passes 'system' as a top‑level kwarg.
"""
from src.features.LLM.LLMTypes import LLMResponse

REWRITE_SYSTEM_PROMPT = (
    "You are a search query rewriter. Given a user question, "
    "produce a concise, keyword‑rich query suitable for hybrid search (vector + keyword).\n"
    "Rules:\n"
    "- Resolve pronouns and vague references using the conversation summary if provided.\n"
    "- Add relevant synonyms or technical terms.\n"
    "- Keep the original intent.\n"
    "- Output ONLY the rewritten query string, no extra text."
)


async def rewrite_query(
    client,
    original_query: str,
    conversation_summary: str = "",
) -> str:
    """Return a rewritten query, or the original if the LLM fails."""
    prompt = original_query
    if conversation_summary:
        prompt = f"Conversation summary: {conversation_summary}\nQuery: {original_query}"

    # Build messages list that includes the system prompt as a message
    messages = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    try:
        resp: LLMResponse = await client.acall_small(
            messages,
            temperature=0.0,
        )
        if resp.error or not resp.text.strip():
            return original_query
        return resp.text.strip()
    except Exception:
        return original_query