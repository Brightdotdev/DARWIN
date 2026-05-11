# src/features/Query/QueryPipeline.py  (updated with handle_message)
"""
Dynamic conversational query pipeline.
handle_message classifies the turn and dispatches to the appropriate handler.
"""
from __future__ import annotations

import json
import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import asdict

from src.features.LLM.LLMClient import LLMClient
from src.features.Query.QueryTypes import (
    RawQuery, QueryContext, URLContent, QuestionType, RetrievalPolicy,
    ResponsePresentation, LLMResponse,TurnType
)
from src.features.Query.QueryPreprocessor import preprocess_raw_query
from src.features.Query.QuestionClassifier import classify_query
from src.features.Query.QueryRewriter import rewrite_query
from src.features.Query.RetrievalPolicy import get_retrieval_policy
from src.features.Query.ArtifactSelector import select_artifact
from src.features.Query.ArtifactGenerator import generate_artifact
from src.lib.UrlFetcher import fetch_url
from src.features.Storage.DarwinStore import DarwinStore
from src.features.Ingestion.Chunking.ContextResolver import get_full_context
from src.core.Logger.DarwinLogger import darwin, LogType
from src.features.Retrival.main import handle_retrieval, RetrievalArgs,RetrievalMethod,RetrievalResult
from src.features.Query.TurnClassifier import classify_turn
from src.features.Query.ContextWindowManager import (
    trim_contexts,
    trim_history,
    estimate_tokens,
)

DEFAULT_MAX_CONTEXT_TOKENS = 3000
DEFAULT_MAX_HISTORY_TOKENS = 500

RETRIEVAL_METHOD_MAP = {
    "hybrid":   RetrievalMethod.HYBRID,
    "semantic": RetrievalMethod.SEMANTIC,
    "keyword":  RetrievalMethod.KEYWORD,
}
def _build_followup_for_indexing(non_searchable_files: List[str]) -> Dict[str, Any]:
    """Return a FollowUpBlock asking the user to index files."""
    return {
        "type": "follow_up",
        "data": {
            "question": "Some files are not indexed yet.",
            "options": [
                {"id": "index", "label": "Index now", "value": "index"}
            ],
            "reason": "file_not_searchable",
            "files": non_searchable_files,
        }
    }

async def retrieve(
    query: str,
    policy: RetrievalPolicy,
    file_filter: List[str],
) -> List[RetrievalResult]:
    darwin.log(f"Retrieving with method={policy.retrieval}, top_k={policy.top_k}", LogType.INFO)
    method = RETRIEVAL_METHOD_MAP.get(policy.retrieval, RetrievalMethod.HYBRID)
    args = RetrievalArgs(
        query=query,
        method=method,
        files=file_filter if file_filter else [],
        top_k=policy.top_k,
    )
    results = await asyncio.to_thread(handle_retrieval, args)
    darwin.log(f"Retrieved {len(results)} results.", LogType.INFO)
    return results


def _resolve_contexts(
    store: DarwinStore,
    results: List[RetrievalResult],
    max_tokens: int,
) -> List[str]:
    darwin.log(f"Resolving contexts for {len(results)} results...", LogType.INFO)
    contexts: List[str] = []
    for r in results:
        try:
            full = get_full_context(r, store, r.source)
            contexts.append(full.text)
        except Exception:
            contexts.append(getattr(r, "text", ""))
    # Trim token budget
    trimmed = trim_contexts(contexts, max_tokens)
    darwin.log(f"Contexts: {len(contexts)} resolved, {len(trimmed)} kept after token trimming.", LogType.INFO)
    return trimmed


async def _full_pipeline(
    client: LLMClient,
    store: DarwinStore,
    raw: RawQuery,
    *,
    conversation_summary: str = "",
    allow_search: bool = True,
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    enable_llm_override: bool = True,
    ctx: Optional[QueryContext] = None,
) -> Dict[str, Any]:
    """Original 8-step pipeline as a reusable subroutine."""
    if ctx is None:
        darwin.log("▶ Step 1: Preprocessing query...", LogType.INFO)
        ctx = preprocess_raw_query(raw)

    if ctx.non_searchable_files:
        darwin.log(f"Files not indexed: {ctx.non_searchable_files}", LogType.WARNING)
        return _build_followup_for_indexing(ctx.non_searchable_files)

    # Fetch external URLs
    fetched_contents: List[URLContent] = []
    if allow_search and ctx.extracted_urls:
        darwin.log(f"Fetching {len(ctx.extracted_urls)} URLs...", LogType.INFO)
        fetch_tasks = [fetch_url(u.url, allow_search=True) for u in ctx.extracted_urls]
        results = await asyncio.gather(*fetch_tasks)
        for uc in results:
            if uc:
                fetched_contents.append(uc)
    ctx.fetched_url_contents = fetched_contents

    darwin.log("▶ Step 2: Classifying question...", LogType.INFO)
    qtype: QuestionType = await classify_query(client, ctx.query, conversation_summary)
    darwin.log(f"Question type: {qtype}", LogType.INFO)

    darwin.log("▶ Step 2.5: Rewriting query for retrieval...", LogType.INFO)
    rewritten = await rewrite_query(client, ctx.query, conversation_summary)
    darwin.log(f"Rewritten query: {rewritten[:100]}...", LogType.INFO)

    darwin.log("▶ Step 3: Retrieval policy lookup...", LogType.INFO)
    policy: RetrievalPolicy = get_retrieval_policy(qtype)
    # Force RAG if user pinned files
    if ctx.searchable_files and not policy.requires_rag:
        policy = RetrievalPolicy(
            retrieval=policy.retrieval,
            top_k=policy.top_k,
            requires_rag=True,
            searchable_topics=policy.searchable_topics,
            requires_tools=policy.requires_tools,
        )
    darwin.log(f"Policy: {policy}", LogType.INFO)

    all_retrieved: List[RetrievalResult] = []
    if policy.requires_rag:
        darwin.log("▶ Step 4: Retrieval...", LogType.INFO)
        batch = await retrieve(
            query=rewritten,
            policy=policy,
            file_filter=ctx.searchable_files if ctx.searchable_files else [],
        )
        all_retrieved.extend(batch)

        # Dedup by chunk identity: (source, page, chunk_index) with fallback
        seen = set()
        deduped = []
        for r in all_retrieved:
            chunk_index = getattr(r, "chunk_index", None)
            if chunk_index is None:
                import hashlib
                chunk_id = hashlib.md5(getattr(r, "text", "").encode()).hexdigest()[:12]
            else:
                chunk_id = (getattr(r, "source", ""), getattr(r, "page", None), chunk_index)
            if chunk_id not in seen:
                seen.add(chunk_id)
                deduped.append(r)
        all_retrieved = sorted(deduped, key=lambda x: x.score, reverse=True)[:policy.top_k]
        darwin.log(f"Total unique results after dedup: {len(all_retrieved)}", LogType.INFO)
    else:
        darwin.log("RAG not required, skipping retrieval.", LogType.INFO)

    darwin.log("▶ Step 5: Context resolution & trimming...", LogType.INFO)
    contexts = _resolve_contexts(store, all_retrieved, max_context_tokens)

    # Context quality gate
    MIN_CONTEXT_TOKENS = 50
    total_context_tokens = sum(estimate_tokens(c) for c in contexts)
    has_blobs = bool(ctx.blobs)
    has_urls = bool(ctx.fetched_url_contents)

    if total_context_tokens < MIN_CONTEXT_TOKENS and not has_blobs and not has_urls:
        if ctx.searchable_files and not all_retrieved:
            return {
                "type": "follow_up",
                "data": {
                    "question": "I searched the pinned files but couldn't find enough relevant content. Could you be more specific or rephrase your question?",
                    "options": [
                        {"id": "rephrase", "label": "Rephrase my question", "value": "rephrase"},
                        {"id": "broader", "label": "Search more broadly", "value": "broader"},
                    ],
                    "reason": "missing_context",
                }
            }
        # else fall through with empty context -> the artifact generator will use general knowledge

    darwin.log("▶ Step 6: Artifact type selection...", LogType.INFO)
    artifact_type: ResponsePresentation = await select_artifact(
        client, qtype, contexts, allow_llm_override=enable_llm_override,
    )
    darwin.log(f"Selected artifact: {artifact_type}", LogType.INFO)

    darwin.log("▶ Step 7: Artifact generation...", LogType.INFO)
    resp: LLMResponse = await generate_artifact(
        client,
        original_query=ctx.query,
        artifact_type=artifact_type,
        resolved_contexts=contexts,
        blobs=ctx.blobs,
        url_contents=ctx.fetched_url_contents,
        conversation_summary=conversation_summary,
    )
    if resp.error:
        darwin.log(f"Artifact generation error: {resp.error}", LogType.ERROR)
        return {"type": "error", "message": resp.error}

    darwin.log("Artifact generated successfully.", LogType.SUCCESS)
    return {
        "type": "artifact",
        "artifact_type": artifact_type,
        "data": resp.json,
        "resolved_contexts": contexts,  # for potential reuse
    }


async def handle_message(
    client: LLMClient,
    store: DarwinStore,
    raw: RawQuery,
    *,
    conversation_db: Optional[Any] = None,
    conversation_id: Optional[str] = None,
    allow_search: bool = True,
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    max_history_tokens: int = DEFAULT_MAX_HISTORY_TOKENS,
    enable_llm_override: bool = True,
) -> Dict[str, Any]:
    """
    Dynamic conversational handler.
    Classifies the turn and dispatches to the appropriate flow.
    """
    # ── load conversation state ──────────────────────────────────
    last_turn = None
    conversation_summary = ""
    if conversation_db and conversation_id:
        last_turn = conversation_db.get_last_turn(conversation_id)
        if last_turn:
            # Build summary from last few turns
            turns = conversation_db.get_last_turns(conversation_id, limit=3)
            parts = []
            for t in turns:
                q = t.get("user_query_json", "")
                r = t.get("response_text", "")[:200]
                if q:
                    parts.append(f"User: {q}")
                if r:
                    parts.append(f"Assistant: {r}")
            conversation_summary = "\n".join(parts)
            conversation_summary = trim_history(conversation_summary, max_history_tokens)

    # ── Turn classification ─────────────────────────────────────
    last_artifact_type = last_turn.get("artifact_type") if last_turn else None
    last_response_text = last_turn.get("response_text", "") if last_turn else ""
    active_follow_up_json = last_turn.get("active_follow_up_block_json") if last_turn else ""
    active_follow_up = ""
    if active_follow_up_json:
        try:
            active_follow_up = json.loads(active_follow_up_json).get("question", "")
        except Exception:
            pass

    turn_type: TurnType = await classify_turn(
        client,
        raw.query,
        last_assistant_response=last_response_text[:300],
        active_follow_up=active_follow_up,
    )
    darwin.log(f"Turn classified as: {turn_type}", LogType.INFO)

    # ── Branching logic ─────────────────────────────────────────
    if turn_type == "chitchat":
        # Plain text response, no artifact
        try:
            resp: LLMResponse = await client.acall_small(
                [{"role": "user", "content": raw.query}],
                temperature=0.7,
            )
            if resp.error:
                return {"type": "error", "message": resp.error}
            return {
                "type": "plain_text",
                "text": resp.text.strip(),
            }
        except Exception as e:
            return {"type": "error", "message": str(e)}

    elif turn_type == "re_render":
        if not last_turn or not last_turn.get("resolved_contexts_json") or not last_artifact_type:
            # Fallback to full pipeline
            return await _full_pipeline(client, store, raw,
                                        conversation_summary=conversation_summary,
                                        allow_search=allow_search,
                                        max_context_tokens=max_context_tokens,
                                        enable_llm_override=enable_llm_override)
        # Re‑use contexts, just re‑select artifact and generate
        contexts = json.loads(last_turn["resolved_contexts_json"])
        # Use artifact selector again (with the same contexts)
        artifact_type = await select_artifact(
            client, last_turn.get("classification", "statement"), contexts,
            allow_llm_override=enable_llm_override,
        )
        resp = await generate_artifact(
            client,
            original_query=raw.query,
            artifact_type=artifact_type,
            resolved_contexts=contexts,
            conversation_summary=conversation_summary,
        )
        if resp.error:
            return {"type": "error", "message": resp.error}
        return {
            "type": "artifact",
            "artifact_type": artifact_type,
            "data": resp.json,
            "resolved_contexts": contexts,
        }

    elif turn_type == "elaboration":
        if not last_turn or not last_turn.get("resolved_contexts_json"):
            # Fallback
            return await _full_pipeline(client, store, raw,
                                        conversation_summary=conversation_summary,
                                        allow_search=allow_search,
                                        max_context_tokens=max_context_tokens,
                                        enable_llm_override=enable_llm_override)
        contexts = json.loads(last_turn["resolved_contexts_json"])
        # Still need to retrieve more specific info? For simplicity, just regenerate with a new query on same contexts
        artifact_type = last_artifact_type or "summary_card"
        resp = await generate_artifact(
            client,
            original_query=f"(Elaboration request) {raw.query}",
            artifact_type=artifact_type,
            resolved_contexts=contexts,
            conversation_summary=conversation_summary,
        )
        if resp.error:
            return {"type": "error", "message": resp.error}
        return {
            "type": "artifact",
            "artifact_type": artifact_type,
            "data": resp.json,
            "resolved_contexts": contexts,
        }

    elif turn_type == "correction":
        if not last_turn or not last_turn.get("artifact_json"):
            # Fallback
            return await _full_pipeline(client, store, raw,
                                        conversation_summary=conversation_summary,
                                        allow_search=allow_search,
                                        max_context_tokens=max_context_tokens,
                                        enable_llm_override=enable_llm_override)
        try:
            current_artifact = json.loads(last_turn["artifact_json"])
            # Patch the artifact using the correction (simple: ask LLM to merge)
            patch_prompt = f"The user says: {raw.query}\nCurrent artifact: {json.dumps(current_artifact)}\nPlease apply the correction and output the corrected artifact JSON."
            resp: LLMResponse = await client.acall_small(
                [{"role": "user", "content": patch_prompt}],
                temperature=0.1,
                json_schema={"type": "object", "properties": {}, "additionalProperties": True},  # flexible
            )
            if resp.error:
                return {"type": "error", "message": resp.error}
            return {
                "type": "artifact",
                "artifact_type": last_artifact_type or "summary_card",
                "data": resp.json,
                "resolved_contexts": json.loads(last_turn.get("resolved_contexts_json", "[]")),
                "corrected": True,
            }
        except Exception as e:
            return {"type": "error", "message": str(e)}

    elif turn_type == "follow_up_answer":
        if not active_follow_up_json:
            # No active follow-up, treat as new_query
            return await _full_pipeline(client, store, raw,
                                        conversation_summary=conversation_summary,
                                        allow_search=allow_search,
                                        max_context_tokens=max_context_tokens,
                                        enable_llm_override=enable_llm_override)
        # We can use the user's answer to refine the query and re-run the pipeline
        # For now, just prepend the follow-up answer to the original query and run full pipeline
        raw.query = f"(Answer to previous question: {active_follow_up}) {raw.query}"
        return await _full_pipeline(client, store, raw,
                                    conversation_summary=conversation_summary,
                                    allow_search=allow_search,
                                    max_context_tokens=max_context_tokens,
                                    enable_llm_override=enable_llm_override)

    else:  # new_query (default)
        return await _full_pipeline(client, store, raw,
                                    conversation_summary=conversation_summary,
                                    allow_search=allow_search,
                                    max_context_tokens=max_context_tokens,
                                    enable_llm_override=enable_llm_override)