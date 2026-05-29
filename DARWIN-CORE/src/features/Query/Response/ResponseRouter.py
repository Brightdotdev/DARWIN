"""
src/core/Response/ResponseRouter.py
-----------------------------------
Adaptive response router.

Simple queries  → single block, deterministic detection, fast path.
Complex queries → classifier (small LLM) picks a script,
                  assembler builds multi-block response.

Fixes:
- Hierarchy requires 3+ structural parent-child pairs or explicit request.
- Comparison is prioritized when requires_comparison=True or multi-file.
- Temporal detection requires actual temporal language in passages.
"""

import re
import json
from typing import List, Dict, Optional, Any, Literal
from dataclasses import dataclass, field

from src.features.Query.Response.ResponseTypes import (
    ResolvedQuery, SearchContext, ResponsePlan, ResponseBlock,
    WikiConcept, ResolvedPassage,
)
from src.features.Query.Response.LLMSchemaGenerator import LLMSchemaGenerator

try:
    from src.core.Logger.DarwinLogger import darwin, LogType
except Exception:
    # Fallback if logger not available
    class _FakeDarwin:
        def log(self, msg, *args): pass
    darwin = _FakeDarwin()
    class LogType:
        INFO = "info"
        ERROR = "error"


class ResponseRouter:
    def __init__(self, llm_client=None, mode: Literal["simple", "adaptive"] = "adaptive"):
        self.llm_client = llm_client
        self.mode = mode
        self.schema_generator = LLMSchemaGenerator(llm_client) if llm_client else None

    def route(self, resolved_query: ResolvedQuery, search_context: SearchContext) -> ResponsePlan:
        if self.mode == "simple" or not self.llm_client:
            return self._simple_route(resolved_query, search_context)

        if self._is_complex(resolved_query, search_context):
            return self._adaptive_route(resolved_query, search_context)

        return self._simple_route(resolved_query, search_context)

    # ═══════════════════════════════════════════════════════════════════
    # Complexity Heuristic (deterministic, zero LLM calls)
    # ═══════════════════════════════════════════════════════════════════

    def _is_complex(self, rq: ResolvedQuery, ctx: SearchContext) -> bool:
        if rq.requires_comparison:
            return True
        if len(ctx.files) > 1:
            return True
        if len(ctx.passages) > 15:
            return True
        if len(ctx.concepts) > 8:
            return True
        if rq.decomposed_queries and len(rq.decomposed_queries) >= 2:
            return True

        complex_words = [
            "compare", "contrast", "versus", "vs", "difference between",
            "relationship between", "how to", "step by step", "explain the process",
            "walk me through", "what are the steps", "pros and cons",
            "advantages and disadvantages",
        ]
        q_lower = rq.original_query.lower()
        if any(w in q_lower for w in complex_words):
            return True

        return False

    # ═══════════════════════════════════════════════════════════════════
    # Simple Route (single block)
    # ═══════════════════════════════════════════════════════════════════

    def _simple_route(self, rq: ResolvedQuery, ctx: SearchContext) -> ResponsePlan:
        block_type = self._auto_detect_single_block(rq, ctx)

        if self.schema_generator:
            data = self.schema_generator.generate_block(block_type, ctx, rq.original_query)
            fallback = None
            if block_type != "markdown":
                fallback = self.schema_generator.generate_fallback(block_type, data, ctx, rq.original_query)
        else:
            data = {"content": "Static fallback: no LLM client configured."}
            fallback = None

        block = ResponseBlock(
            block_type=block_type,
            title=data.get("title", rq.original_query),
            data=data,
            schema=self._get_schema_stub(block_type),
            markdown_breakdown=None if block_type == "markdown" else f"{block_type} visualization from retrieved data.",
            markdown_fallback=fallback,
        )

        return ResponsePlan(
            script_key=None,
            blocks=[block],
            is_complex=False,
            metadata={"route": "simple", "block_type": block_type},
        )

    # ═══════════════════════════════════════════════════════════════════
    # Adaptive Route (multi-block)
    # ═══════════════════════════════════════════════════════════════════

    def _adaptive_route(self, rq: ResolvedQuery, ctx: SearchContext) -> ResponsePlan:
        script = self._classify_script(rq, ctx)
        blocks = self._assemble_blocks(script, rq, ctx)

        return ResponsePlan(
            script_key=script.get("script_key", "custom"),
            blocks=blocks,
            is_complex=True,
            metadata={
                "route": "adaptive",
                "requested_blocks": script.get("blocks", []),
                "complexity_reasons": script.get("reasons", []),
            },
        )

    def _classify_script(self, rq: ResolvedQuery, ctx: SearchContext) -> Dict[str, Any]:
        if not self.llm_client:
            return {"script_key": "simple_markdown", "blocks": ["markdown"], "reasons": ["no_llm"]}

        passages_summary = self._summarize_passages(ctx)
        prompt = f"""You are a response planner. Given a query and retrieved data, decide the best sequence of presentation blocks.

Available block types:
markdown, summary_card, comparison_table, timeline, flowchart, list, entity_map, pros_cons

Rules:
1. ALWAYS start with "markdown" (intro) and end with "markdown" (synthesis/outro).
2. Maximum 4 blocks total.
3. Use "comparison_table" only if multiple sources/files and comparison intent.
4. Use "timeline" only if temporal/chronological data is present.
5. Use "flowchart" only if process/steps are present.
6. Use "entity_map" only if many interrelated concepts/entities.
7. Use "pros_cons" only if evaluating a subject with clear tradeoffs.
8. Prefer fewer blocks unless the query is genuinely complex.

Query: {rq.original_query}
Files: {ctx.files}
Passage count: {len(ctx.passages)}
Concept count: {len(ctx.concepts)}
Requires comparison: {rq.requires_comparison}
Decomposed queries: {len(rq.decomposed_queries)}

Passage summary:
{passages_summary}

Return ONLY a JSON object:
{{
  "script_key": "descriptive_name",
  "blocks": ["markdown", "..."],
  "reasons": ["why this script was chosen"]
}}
"""

        raw = self.llm_client.call_small(prompt)
        try:
            parsed = json.loads(
                raw.strip().split("```json")[-1].split("```")[0].strip()
                if "```" in raw else raw.strip()
            )
            blocks = parsed.get("blocks", ["markdown"])

            # Enforce constraints
            if blocks[0] != "markdown":
                blocks.insert(0, "markdown")
            if blocks[-1] != "markdown":
                blocks.append("markdown")
            if len(blocks) > 4:
                blocks = blocks[:4]
                if blocks[-1] != "markdown":
                    blocks[-1] = "markdown"

            parsed["blocks"] = blocks
            return parsed
        except Exception as e:
            darwin.log(f"[ResponseRouter] Script classification failed: {e}", LogType.ERROR)
            return {"script_key": "fallback", "blocks": ["markdown"], "reasons": ["classification_error"]}

    def _assemble_blocks(self, script: Dict[str, Any], rq: ResolvedQuery, ctx: SearchContext) -> List[ResponseBlock]:
        blocks = []
        block_types = script.get("blocks", ["markdown"])

        for idx, block_type in enumerate(block_types):
            # Slice context per block position / type
            if block_type == "markdown" and idx == 0:
                sliced = self._slice_context(ctx, max_passages=5)
            elif block_type == "markdown" and idx == len(block_types) - 1:
                sliced = self._slice_context(ctx, max_passages=5)
            elif block_type == "comparison_table":
                sliced = self._slice_context(ctx, max_passages=12)
            elif block_type in ("timeline", "flowchart"):
                sliced = self._slice_context(ctx, max_passages=10)
            else:
                sliced = ctx

            data = self.schema_generator.generate_block(block_type, sliced, rq.original_query)
            fallback = None
            if block_type != "markdown":
                fallback = self.schema_generator.generate_fallback(block_type, data, sliced, rq.original_query)

            block = ResponseBlock(
                block_type=block_type,
                title=data.get("title", block_type.replace("_", " ").title()),
                data=data,
                schema=self._get_schema_stub(block_type),
                markdown_breakdown=None if block_type == "markdown" else self._generate_breakdown(block_type, data),
                markdown_fallback=fallback,
                metadata={"position": idx, "context_passages": len(sliced.passages)},
            )
            blocks.append(block)

        return blocks

    def _slice_context(self, ctx: SearchContext, max_passages: int) -> SearchContext:
        return SearchContext(
            query=ctx.query,
            intent=ctx.intent,
            files=ctx.files,
            concepts=ctx.concepts,
            passages=ctx.passages[:max_passages],
            related_topics=ctx.related_topics,
            merged_prompt=ctx.merged_prompt,
        )

    def _generate_breakdown(self, block_type: str, data: Dict[str, Any]) -> str:
        if block_type == "comparison_table":
            cols = data.get("columns", [])
            insights = data.get("insights", [])
            breakdown = f"This table compares entities across {len(cols)-1} dimension(s)."
            if insights:
                breakdown += f" Key insight: {insights[0]}"
            return breakdown
        if block_type == "timeline":
            events = data.get("events", [])
            return f"Timeline showing {len(events)} chronological events."
        if block_type == "flowchart":
            nodes = data.get("nodes", [])
            return f"Process flow with {len(nodes)} steps."
        if block_type == "entity_map":
            nodes = data.get("nodes", [])
            edges = data.get("edges", [])
            return f"Concept map with {len(nodes)} entities and {len(edges)} relationships."
        if block_type == "pros_cons":
            pros = len(data.get("pros", []))
            cons = len(data.get("cons", []))
            return f"Evaluation showing {pros} advantage(s) and {cons} disadvantage(s)."
        if block_type == "list":
            items = data.get("items", [])
            return f"Ranked list of {len(items)} items."
        if block_type == "summary_card":
            fields = data.get("fields", [])
            return f"Summary card with {len(fields)} key facts."
        if block_type == "hierarchy":
            nodes = data.get("nodes", [])
            return f"Hierarchy with {len(nodes)} nodes."
        return f"{block_type} visualization generated from retrieved data."

    def _summarize_passages(self, ctx: SearchContext) -> str:
        lines = []
        for p in ctx.passages[:8]:
            text = getattr(p, 'passage_text', str(p))[:300]
            source = getattr(p, 'source', '?')
            lines.append(f"- {source}: {text}...")
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════
    # Single-block Auto-Detection (fixed logic)
    # ═══════════════════════════════════════════════════════════════════

    def _auto_detect_single_block(self, rq: ResolvedQuery, ctx: SearchContext) -> str:
        # 1. Explicit user request
        if rq.explicit_response_type:
            return rq.explicit_response_type

        # 2. Comparison: prioritize when query demands it
        if self._is_comparison_query(rq, ctx):
            return "comparison_table"

        # 3. Temporal signals
        if self._has_temporal_signals(ctx):
            return "timeline"

        # 4. Hierarchy: strict structural evidence
        if self._has_hierarchy_structure(rq, ctx):
            return "hierarchy"

        # 5. Graph signals
        if self._has_graph_signals(rq):
            return "entity_map"

        # Default
        return "markdown"

    def _is_comparison_query(self, rq: ResolvedQuery, ctx: SearchContext) -> bool:
        comp_words = ["compare", "contrast", "versus", "vs", "difference", "similarities", "between"]
        if any(w in rq.original_query.lower() for w in comp_words):
            return True
        if rq.requires_comparison:
            return True
        if rq.decomposed_queries and len(rq.decomposed_queries) >= 2:
            return True
        if len(ctx.files) >= 2 and any(w in rq.original_query.lower() for w in ["and", "or", "both"]):
            return True
        return False

    def _has_temporal_signals(self, ctx: SearchContext) -> bool:
        temporal = {
            "day", "days", "year", "years", "generation", "generations",
            "time", "times", "age", "ages", "before", "after", "when",
            "then", "until", "during", "period", "era", "epoch", "chronology",
            "first", "second", "third", "finally", "sequence", "chronological",
        }
        hits = 0
        for p in ctx.passages:
            words = set(re.findall(r'\b\w+\b', p.passage_text.lower()))
            hits += len(words & temporal)
        return hits >= 3 and len(ctx.passages) >= 3

    def _has_hierarchy_structure(self, rq: ResolvedQuery, ctx: SearchContext) -> bool:
        hierarchy_keywords = [
            "genealogy", "descendants", "generations", "family tree",
            "lineage", "ancestry", "pedigree", "hierarchy", "tree of",
        ]
        explicit = any(t in rq.original_query.lower() for t in hierarchy_keywords)
        if explicit and len(ctx.passages) >= 2:
            return True

        pairs = set()
        for p in ctx.passages:
            for m in re.finditer(r"(\w+)\s+(?:son|daughter)\s+of\s+(\w+)", p.passage_text, re.IGNORECASE):
                pairs.add((m.group(1).lower(), m.group(2).lower()))
            for m in re.finditer(r"(\w+)\s+begat\s+(\w+)", p.passage_text, re.IGNORECASE):
                pairs.add((m.group(2).lower(), m.group(1).lower()))
            for m in re.finditer(r"(\w+)\s+(?:father|mother)\s+of\s+(\w+)", p.passage_text, re.IGNORECASE):
                pairs.add((m.group(2).lower(), m.group(1).lower()))

        return len(pairs) >= 3

    def _has_graph_signals(self, rq: ResolvedQuery) -> bool:
        graph_words = ["graph", "network", "map", "connections", "relationships between", "linked", "concept map"]
        return any(w in rq.original_query.lower() for w in graph_words)

    def _get_schema_stub(self, block_type: str) -> Dict[str, Any]:
        stubs = {
            "markdown": {"type": "object", "properties": {"content": {"type": "string"}}},
            "summary_card": {"type": "object", "properties": {"title": {"type": "string"}, "fields": {"type": "array"}, "summary": {"type": "string"}}},
            "comparison_table": {"type": "object", "properties": {"title": {"type": "string"}, "columns": {"type": "array"}, "rows": {"type": "array"}, "insights": {"type": "array"}}},
            "timeline": {"type": "object", "properties": {"title": {"type": "string"}, "events": {"type": "array"}}},
            "flowchart": {"type": "object", "properties": {"title": {"type": "string"}, "nodes": {"type": "array"}, "edges": {"type": "array"}}},
            "list": {"type": "object", "properties": {"title": {"type": "string"}, "items": {"type": "array"}}},
            "entity_map": {"type": "object", "properties": {"title": {"type": "string"}, "nodes": {"type": "array"}, "edges": {"type": "array"}}},
            "pros_cons": {"type": "object", "properties": {"title": {"type": "string"}, "subject": {"type": "string"}, "pros": {"type": "array"}, "cons": {"type": "array"}, "verdict": {"type": "string"}}},
            "hierarchy": {"type": "object", "properties": {"title": {"type": "string"}, "nodes": {"type": "array"}, "edges": {"type": "array"}, "root": {"type": "string"}}},
        }
        return stubs.get(block_type, {"type": "object"})