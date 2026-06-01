"""
src/features/Query/SearchOrchestrator.py
----------------------------------------
Unified search entry point. Receives resolved queries from upstream
intent routing / query expansion layers.

Multi-file support: searches each file independently, merges results.
Flips between Option A (concepts-first) and Option B (passages-first)
based on intent.
"""

from typing import List, Dict, Optional, Literal,Any
from dataclasses import dataclass, field

from src.core.Storage.DarwinStore import DarwinStore
from src.features.Knowledge.Wiki.WikiUtils import DocumentWiki, WikiPage
from src.features.Knowledge.WikiOrchestrator import WikiOrchestrator, WikiConcept,WikiContext
from src.features.Retrival.RetrievalOrchestrator import RetrievalOrchestrator, RetrievalContext, ResolvedPassage
from src.core.Logger.DarwinLogger import darwin, LogType

# ═══════════════════════════════════════════════════════════════════════════
# Canonical types
# ═══════════════════════════════════════════════════════════════════════════
from src.features.Query.Response.ResponseTypes import ResolvedQuery


# ═══════════════════════════════════════════════════════════════════════════
# SearchContext (search-specific, stays here)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SearchContext:
    """Unified search output."""
    query: str
    intent: str
    files: List[str]
    concepts: List[WikiConcept]
    passages: List[ResolvedPassage]
    related_topics: List[str]
    merged_prompt: str
    audit_summary: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Prompt Builders (Multi-file aware)
# ═══════════════════════════════════════════════════════════════════════════

def _build_broad_prompt(
    query: str,
    concepts: List[WikiConcept],
    passages: List[ResolvedPassage],
    files: List[str],
) -> str:
    """
    Option A: Concepts first, passages nested.
    For overview / explanation questions.
    Grouped by source file when multi-file.
    """
    if not concepts:
        return f"Question: {query}\n\nNo relevant concepts found in documents."

    blocks = []

    # Group concepts by source file for clarity
    concepts_by_file: Dict[str, List[WikiConcept]] = {}
    for c in concepts:
        concepts_by_file.setdefault(c.source_file, []).append(c)

    for file_idx, (filename, file_concepts) in enumerate(concepts_by_file.items(), 1):
        blocks.append(f"### Source {file_idx}: {filename}\n")

        for i, concept in enumerate(file_concepts):
            # Find passages that belong to this concept
            concept_passages = [
                p for p in passages
                if p.source == filename and (
                    p.primary_concept == concept.raw_concept
                    or concept.raw_concept in p.related_concepts
                )
            ]

            block = f"""#### Concept {i+1}: {concept.concept}
**Pages:** {concept.get_page_range()} | **Score:** {concept.score:.3f} | **Sources:** {concept.source_count}
**Keywords:** {', '.join(concept.keywords[:5])}
"""
            if concept.audit and concept.audit.was_penalized:
                block += f"**Penalty:** {concept.audit.structural_type} factor={concept.audit.penalty_factor} ({concept.audit.raw_score:.3f} → {concept.audit.final_score:.3f})\n"

            if concept_passages:
                block += "\n**Evidence:**\n"
                for j, p in enumerate(concept_passages[:3], 1):
                    block += f"\n> [{j}] Page {p.page_start}-{p.page_end} | Score {p.avg_score:.3f}\n"
                    preview = p.passage_text[:800]
                    if len(p.passage_text) > 800:
                        preview += "..."
                    block += f"{preview}\n"
            else:
                block += "\n*(No direct passage evidence retrieved for this concept)*\n"

            blocks.append(block)

    # Orphaned passages (passages without concept tags)
    orphaned = [p for p in passages if not p.primary_concept]
    if orphaned:
        blocks.append("### Additional Evidence\n")
        for i, p in enumerate(orphaned[:3], 1):
            blocks.append(
                f"> [{i}] {p.source} | Page {p.page_start}-{p.page_end} | Score {p.avg_score:.3f}\n"
                f"{p.passage_text[:600]}{'...' if len(p.passage_text) > 600 else ''}\n"
            )

    joined_blocks = chr(10).join(blocks)

    return f"""Answer based on the following document concepts and evidence.

Question: {query}

{joined_blocks}

Answer:"""


def _build_dense_prompt(
    query: str,
    passages: List[ResolvedPassage],
    concepts: List[WikiConcept],
    files: List[str],
) -> str:
    """
    Option B: Passages first, concept headers injected.
    For find / quote / exact evidence questions.
    """
    if not passages:
        return f"Question: {query}\n\nNo relevant passages found."

    blocks = []
    current_concept = None
    current_file = None

    # Sort passages by source file, then by score
    sorted_passages = sorted(passages, key=lambda p: (p.source, -p.avg_score))

    for i, p in enumerate(sorted_passages):
        # File header when source changes
        if p.source != current_file:
            current_file = p.source
            blocks.append(f"\n### Source: {p.source}\n")

        # Inject concept header when concept changes
        if p.primary_concept and p.primary_concept != current_concept:
            current_concept = p.primary_concept
            concept_obj = next(
                (c for c in concepts if c.raw_concept == current_concept and c.source_file == p.source),
                None,
            )
            if concept_obj:
                blocks.append(
                    f"\n#### Concept: {current_concept} "
                    f"(Pages {concept_obj.get_page_range()})\n"
                )
            else:
                blocks.append(f"\n#### Concept: {current_concept}\n")

        score_parts = [f"avg={p.avg_score:.3f}"]
        if p.semantic_score:
            score_parts.append(f"sem={p.semantic_score:.3f}")
        if p.keyword_score:
            score_parts.append(f"kw={p.keyword_score:.3f}")

        gap_info = ""
        if len(p.chunk_indices) > len(p.original_results):
            gap_count = len(p.chunk_indices) - len(p.original_results)
            gap_info = f" (includes {gap_count} gap-filled chunks)"

        penalty_info = ""
        if p.overall_penalty_factor < 1.0:
            penalty_info = f" [PENALTY: {p.dominant_structural_type} factor={p.overall_penalty_factor}]"

        blocks.append(
            f"[Passage {i+1}: Page {p.page_start}-{p.page_end} | "
            f"{' | '.join(score_parts)}{gap_info}{penalty_info}]\n"
            f"{p.passage_text}\n"
        )

    passage_text = "\n---\n".join(blocks)

    return f"""Answer based on the following passages.

Question: {query}

Passages:
{passage_text}

Answer:"""


def _build_combined_prompt(
    query: str,
    concepts: List[WikiConcept],
    passages: List[ResolvedPassage],
    files: List[str],
) -> str:
    """
    Option A+B hybrid: When intent is 'both', use dense layout but
    prepend a concept overview section. Grouped by file.
    """
    if not passages and not concepts:
        return f"Question: {query}\n\nNo relevant information found."

    # Concept overview by file (compact)
    overview = ""
    if concepts:
        overview_lines = ["## Document Concepts Overview"]
        concepts_by_file: Dict[str, List[WikiConcept]] = {}
        for c in concepts:
            concepts_by_file.setdefault(c.source_file, []).append(c)

        for filename, file_concepts in concepts_by_file.items():
            overview_lines.append(f"\n### {filename}")
            for c in file_concepts[:5]:
                penalty_note = ""
                if c.audit and c.audit.was_penalized:
                    penalty_note = f" [penalized: {c.audit.structural_type}]"
                overview_lines.append(
                    f"- **{c.concept}** ({c.get_page_range()}) — "
                    f"score: {c.score:.3f}, keywords: {', '.join(c.keywords[:3])}{penalty_note}"
                )
        overview = "\n".join(overview_lines) + "\n"

    # Passages (dense layout, grouped by file)
    passage_blocks = []
    current_file = None
    for p in passages[:12]:
        if p.source != current_file:
            current_file = p.source
            passage_blocks.append(f"\n### Source: {p.source}\n")

        concept_tag = f" | Concept: {p.primary_concept}" if p.primary_concept else ""
        penalty_tag = ""
        if p.overall_penalty_factor < 1.0:
            penalty_tag = f" | Penalty: {p.dominant_structural_type}={p.overall_penalty_factor}"
        passage_blocks.append(
            f"[{len(passage_blocks)}] Page {p.page_start}-{p.page_end}{concept_tag}{penalty_tag} | "
            f"Score: {p.avg_score:.3f}\n{p.passage_text[:1000]}"
            f"{'...' if len(p.passage_text) > 1000 else ''}\n"
        )

    evidence_text = "\n---\n".join(passage_blocks)

    return f"""Answer based on the following document overview and evidence.

Question: {query}

{overview}

## Evidence
{evidence_text}

Answer:"""


# ═══════════════════════════════════════════════════════════════════════════
# SearchOrchestrator
# ═══════════════════════════════════════════════════════════════════════════

class SearchOrchestrator:
    """
    Unified search entry point. Receives resolved queries from upstream layers.

    Multi-file: searches each file independently, merges results.

    Intent routing:
        broad  → Wiki-first per file, then hydrate (Option A)
        dense  → Retrieval-first per file, then lookup concepts (Option B)
        both   → Parallel dispatch per file, combined layout
    """

    def __init__(self, store: DarwinStore):
        self.store = store
        self.wiki_orch = WikiOrchestrator(store)
        self.retrieval_orch = RetrievalOrchestrator(store)

    def _load_wiki_safe(self, filename: str) -> Optional[DocumentWiki]:
        """Load wiki for a file, return None if missing or empty."""
        pages_dicts = self.store.load_wiki(filename)
        if not pages_dicts:
            # None or empty list → no wiki
            return None
        # Ensure it's a list
        if pages_dicts is None:
            pages_dicts = []
        try:
            wiki = DocumentWiki(
                filename=filename,
                pages=[WikiPage.from_dict(p) for p in pages_dicts],
            )
            return wiki if wiki.pages else None
        except Exception as e:
            darwin.log(f"Failed to load wiki for {filename}: {e}", LogType.WARNING)
            return None

    def search(
        self,
        resolved_query: ResolvedQuery,
        files: List[str],
        top_k: int = 5,
        max_chars: int = 6000,
        filter_structural: bool = True,
        merge_sources: bool = True,
        fill_gaps: bool = True,
    ) -> SearchContext:
        query = resolved_query.original_query
        intent = resolved_query.intent
        embedding = resolved_query.query_embedding

        # Normalize files to list
        if isinstance(files, str):
            files = [files]

        darwin.log(
            f"[SearchOrchestrator] intent={intent} query='{query}' files={files}",
            LogType.INFO,
        )

        all_concepts: List[WikiConcept] = []
        all_passages: List[ResolvedPassage] = []
        all_related: List[str] = []
        all_retrieval_contexts: List[RetrievalContext] = []

        # ── Search each file independently ──
        for filename in files:
            # Safely load wiki (may be None)
            wiki = self._load_wiki_safe(filename)

            # For "broad" and "both", we need wiki – if missing, fall back to dense-like behavior
            if intent in ("broad", "both") and wiki is None:
                darwin.log(
                    f"[SearchOrchestrator] No wiki for {filename}, falling back to dense mode for this file",
                    LogType.WARNING,
                )
                # Fallback: treat as dense (retrieval only, no concept grouping)
                ret_ctx = self.retrieval_orch.retrieve_and_resolve(
                    query=query,
                    files=[filename],
                    method="hybrid",
                    top_k=top_k,
                    max_chars=max_chars // len(files),
                    filter_structural=filter_structural,
                    merge_sources=merge_sources,
                    fill_gaps=fill_gaps,
                    wiki=None,  # no wiki to enrich
                )
                all_passages.extend(ret_ctx.passages)
                all_retrieval_contexts.append(ret_ctx)
                # No concepts or related topics for this file
                continue

            if intent == "broad":
                # Wiki first, then hydrate top concepts with retrieval
                wiki_ctx = self.wiki_orch.search(
                    query=query,
                    filename=filename,
                    query_embedding=embedding,
                    top_k=top_k,
                    merge_sources=merge_sources,
                    filter_structural=filter_structural,
                )

                # Retrieve evidence for top concepts only
                if wiki_ctx.concepts:
                    concept_queries = []
                    for c in wiki_ctx.concepts[:3]:
                        concept_queries.extend(c.keywords[:3])
                    enhanced_query = query + " " + " ".join(concept_queries)
                    darwin.log(f"Enhanced query {enhanced_query}")

                    # RetrievalOrchestrator now handles 'wiki' param internally
                    ret_ctx = self.retrieval_orch.retrieve_and_resolve(
                        query=enhanced_query,
                        files=[filename],
                        method="hybrid",
                        top_k=top_k * 2,
                        max_chars=max_chars // len(files),
                        filter_structural=filter_structural,
                        merge_sources=merge_sources,
                        fill_gaps=fill_gaps,
                        wiki=wiki,  # pass wiki (may be None, but we already ensured it exists)
                    )
                else:
                    ret_ctx = RetrievalContext(
                        query=query, method="hybrid", passages=[],
                        total_chars=0, penalized_count=0,
                        fallback_used=False, raw_results=[],
                        audit_summary={},
                    )

                all_concepts.extend(wiki_ctx.concepts)
                all_passages.extend(ret_ctx.passages)
                all_related.extend(wiki_ctx.related_topics)
                all_retrieval_contexts.append(ret_ctx)

            elif intent == "dense":
                # Retrieval first, then lookup concept headers
                ret_ctx = self.retrieval_orch.retrieve_and_resolve(
                    query=query,
                    files=[filename],
                    method="hybrid",
                    top_k=top_k,
                    max_chars=max_chars // len(files),
                    filter_structural=filter_structural,
                    merge_sources=merge_sources,
                    fill_gaps=fill_gaps,
                    wiki=wiki,  # wiki may be None – retrieval will work without it
                )

                # Fetch concepts for context (don't drive the layout)
                wiki_ctx = self.wiki_orch.search(
                    query=query,
                    filename=filename,
                    query_embedding=embedding,
                    top_k=top_k,
                    merge_sources=merge_sources,
                    filter_structural=filter_structural,
                ) if wiki else WikiContext(
                    query=query, concepts=[], related_topics=[],
                    total_concepts=0, penalized_count=0,
                )

                all_concepts.extend(wiki_ctx.concepts)
                all_passages.extend(ret_ctx.passages)
                all_related.extend(wiki_ctx.related_topics)
                all_retrieval_contexts.append(ret_ctx)

            else:  # both
                wiki_ctx = self.wiki_orch.search(
                    query=query,
                    filename=filename,
                    query_embedding=embedding,
                    top_k=top_k,
                    merge_sources=merge_sources,
                    filter_structural=filter_structural,
                ) if wiki else WikiContext(
                    query=query, concepts=[], related_topics=[],
                    total_concepts=0, penalized_count=0,
                )

                ret_ctx = self.retrieval_orch.retrieve_and_resolve(
                    query=query,
                    files=[filename],
                    method="hybrid",
                    top_k=top_k,
                    max_chars=max_chars // len(files),
                    filter_structural=filter_structural,
                    merge_sources=merge_sources,
                    fill_gaps=fill_gaps,
                    wiki=wiki,
                )

                all_concepts.extend(wiki_ctx.concepts)
                all_passages.extend(ret_ctx.passages)
                all_related.extend(wiki_ctx.related_topics)
                all_retrieval_contexts.append(ret_ctx)

        # Deduplicate related topics
        seen_related = set()
        unique_related = []
        for r in all_related:
            if r.lower() not in seen_related:
                seen_related.add(r.lower())
                unique_related.append(r)

        # Sort concepts and passages by score
        all_concepts.sort(key=lambda c: c.score, reverse=True)
        all_passages.sort(key=lambda p: p.avg_score, reverse=True)

        # Build audit summary
        audit_summary = {
            "wiki": {
                "total_concepts": len(all_concepts),
                "penalized_concepts": sum(
                    1 for c in all_concepts if c.audit and c.audit.was_penalized
                ),
            },
            "retrieval": {},
            "passages_with_penalties": sum(
                1 for p in all_passages if p.overall_penalty_factor < 1.0
            ),
        }
        # Merge retrieval audit summaries
        for ctx in all_retrieval_contexts:
            if ctx and ctx.audit_summary:
                for k, v in ctx.audit_summary.items():
                    if k in audit_summary["retrieval"] and isinstance(v, int):
                        audit_summary["retrieval"][k] = audit_summary["retrieval"].get(k, 0) + v
                    else:
                        audit_summary["retrieval"][k] = v

        # Build prompt based on intent
        if intent == "broad":
            prompt = _build_broad_prompt(query, all_concepts, all_passages, files)
        elif intent == "dense":
            prompt = _build_dense_prompt(query, all_passages, all_concepts, files)
        else:
            prompt = _build_combined_prompt(query, all_concepts, all_passages, files)

        return SearchContext(
            query=query,
            intent=intent,
            files=files,
            concepts=all_concepts,
            passages=all_passages,
            related_topics=unique_related[:10],
            merged_prompt=prompt,
            audit_summary=audit_summary,
        )

    def search_for_llm(
        self,
        resolved_query: ResolvedQuery,
        files: List[str],
        **kwargs,
    ) -> str:
        """Convenience: returns just the prompt string."""
        ctx = self.search(resolved_query, files, **kwargs)
        return ctx.merged_prompt