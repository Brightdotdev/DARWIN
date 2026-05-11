# src/features/Query/ArtifactSelector.py
"""
Choose the best ResponsePresentation for the current query + context.
Fixed: system prompt moved into messages list.
"""
from __future__ import annotations

from typing import List
from src.features.LLM.LLMTypes import LLMResponse
from src.features.Query.QueryTypes import QuestionType, ResponsePresentation

# ── Static fallback map ────────────────────────────────────────────────────
FALLBACK_MAP: dict[QuestionType, ResponsePresentation] = {
    "fact":          "summary_card",
    "instruction":   "stepper",
    "reasoning":     "flowchart",
    "comparison":    "comparison_table",
    "opinion":       "pros_cons",
    "hypothetical":  "parameter_sandbox",
    "multi_part":    "accordion",
    "yes_no":        "summary_card",
    "statement":     "summary_card",
    "clarification": "summary_card",
}

ARTIFACT_SELECTOR_PROMPT = (
    "You are an artifact planner. Based on the user's question type "
    "and a preview of the retrieved context, choose the best visual artifact "
    "type from the list below.\n\n"
    "Available artifact types:\n"
    "{available_artifacts}\n\n"
    "Respond with a JSON object:\n"
    '{{"selected": "<artifact type>", "rationale": "why you chose it (one sentence)"}}'
)


async def select_artifact(
    client,
    question_type: QuestionType,
    context_previews: List[str],
    allow_llm_override: bool = True,
) -> ResponsePresentation:
    """Return the artifact type. Falls back to static map if LLM fails or override disabled."""
    default = FALLBACK_MAP.get(question_type, "summary_card")

    if not allow_llm_override or not context_previews:
        return default

    previews = "\n\n".join(c[:300] for c in context_previews[:3])
    available = ", ".join(FALLBACK_MAP.values())
    prompt = ARTIFACT_SELECTOR_PROMPT.format(available_artifacts=available)

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Question type: {question_type}\nContext previews:\n{previews}"},
    ]

    try:
        resp: LLMResponse = await client.acall_small(
            messages,
            temperature=0.0,
        )
        if resp.error or not resp.json:
            return default
        selected = resp.json.get("selected")
        if selected in FALLBACK_MAP.values():
            return selected
        return default
    except Exception:
        return default