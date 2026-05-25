"""
src/core/Response/LLMSchemaGenerator.py
---------------------------------------
Dynamic schema generation per block type.
The LLM reads SearchContext and populates domain-specific JSON.
"""

import json
import re
from typing import Dict, List, Callable, Any

from src.features.Query.Response.ResponseTypes import SearchContext

class LLMSchemaGenerator:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_block(self, block_type: str, search_context: SearchContext, query: str) -> Dict[str, Any]:
        caller = self._select_caller(block_type)
        prompt = self._build_prompt(block_type, search_context, query)
        raw = caller(prompt)
        print("******************************************************************")
        print("********************************************")
        print(raw)
        print("********************************************")
        print("******************************************************************")
       
        return "noooo"

    def generate_fallback(self, block_type: str, block_data: Dict[str, Any],
                          search_context: SearchContext, query: str) -> str:
        if block_type == "markdown":
            return block_data.get("content", "")
        caller = self.llm_client.call_small
        prompt = self._build_fallback_prompt(block_type, block_data, search_context, query)
        return caller(prompt)

    # ------------------------------------------------------------------
    # LLM method selection by complexity
    # ------------------------------------------------------------------

    def _select_caller(self, block_type: str) -> Callable[[str], str]:
        if block_type == "markdown":
            return self.llm_client.call_small
        if block_type in ("summary_card", "list", "pros_cons"):
            return self.llm_client.call_normal
        return self.llm_client.call_smart

    # ------------------------------------------------------------------
    # Prompt builders (one per block family)
    # ------------------------------------------------------------------

    def _build_prompt(self, block_type: str, ctx: SearchContext, query: str) -> str:
        passages_text = self._format_passages(ctx.passages)
        concepts_text = ", ".join(c.concept for c in ctx.concepts) or "None"
        sources_text = ", ".join(ctx.files) or "Unknown"

        base = f"""Query: {query}
Sources: {sources_text}
Key Concepts: {concepts_text}
Passages:
{passages_text}

"""

        builders = {
            "markdown": base + """Write a clear, concise markdown response that answers the query using only the provided passages.
Return ONLY a JSON object: {"content": "your markdown here"}""",

            "summary_card": base + """Extract the most important facts as a summary card.
Return JSON with:
- "title": string
- "fields": list of {"label": string, "value": string}
- "summary": one-paragraph overview""",

            "comparison_table": base + """Compare key entities across sources DYNAMICALLY.
Analyze the passages and determine what dimensions are worth comparing.
Do NOT use generic columns — derive them from the data.
Return JSON with:
- "title": string
- "columns": list of column names (first is the entity/concept)
- "rows": list of row objects
- "insights": list of strings highlighting key differences""",

            "timeline": base + """Extract chronological events and order them.
Return JSON with:
- "title": string
- "events": list of {"period": string, "event": string, "source": string, "significance": string, "order": int}""",

            "flowchart": base + """Extract a process or decision flow from the passages.
Return JSON with:
- "title": string
- "nodes": list of {"id": string, "label": string, "type": string (start|process|decision|end)}
- "edges": list of {"source": string, "target": string, "label": string}""",

            "list": base + """Create a ranked or ordered list from the passages.
Return JSON with:
- "title": string
- "items": list of {"rank": int, "label": string, "description": string, "source": string}""",

            "entity_map": base + """Extract entities and their relationships.
Return JSON with:
- "title": string
- "nodes": list of {"id": string, "label": string, "type": string}
- "edges": list of {"source": string, "target": string, "label": string, "strength": float (0-1)}""",

            "pros_cons": base + """Evaluate the subject as pros and cons.
Return JSON with:
- "title": string
- "subject": string
- "pros": list of strings
- "cons": list of strings
- "verdict": one-sentence summary""",

            "hierarchy": base + """Extract hierarchical or genealogical relationships.
Return JSON with:
- "title": string
- "nodes": list of {"id": string, "name": string, "level": int, "source": string}
- "edges": list of {"parent": string, "child": string, "relationship_type": string}
- "root": top-level node id""",
        }

        return builders.get(block_type, builders["markdown"])

    def _build_fallback_prompt(self, block_type: str, data: Dict[str, Any],
                               ctx: SearchContext, query: str) -> str:
        data_json = json.dumps(data, indent=2)
        return f"""The user wants a plain-text explanation instead of a visualization.

Query: {query}
Visualization type: {block_type}
Data: {data_json}

Write a concise, well-structured plain-text summary that conveys the same information.
Use bullet points or short paragraphs. Do not mention that this is a "fallback".
Just explain the content clearly."""

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def _format_passages(self, passages: List[Any]) -> str:
        if not passages:
            return "No passages retrieved."
        chunks = []
        for i, p in enumerate(passages[:15], 1):
            text = getattr(p, 'passage_text', str(p))
            source = getattr(p, 'source', 'unknown')
            page = getattr(p, 'page_start', '?')
            chunks.append(
                f"[{i}] {source} (p.{page}):\n{text[:800]}{'...' if len(text) > 800 else ''}"
            )
        return "\n\n".join(chunks)

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        raw = raw.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            for pattern in [r'\{{.*\}}', r'\[.*\]']:
                m = re.search(pattern, raw, re.DOTALL)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except Exception:
                        pass
            return {"raw_response": raw, "parse_error": True}