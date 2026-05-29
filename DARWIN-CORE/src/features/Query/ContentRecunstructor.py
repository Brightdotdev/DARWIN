from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.features.Query.lib import PreprocessedQuery,ReconstructedContext,ResolvedPlaceholder


class ContextReconstructor:
    """
    Takes a PreprocessedQuery and resolved placeholder content,
    rebuilds the full context for the extraction LLM.
    """

    def __init__(self, max_tokens: int = 2000, tokens_per_char: float = 0.25):
        """
        Args:
            max_tokens: maximum token budget for the final context.
            tokens_per_char: rough estimate of tokens per character (varies by model).
        """
        self.max_tokens = max_tokens
        self.tokens_per_char = tokens_per_char

    def reconstruct(
        self,
        query: PreprocessedQuery,
        rag_results: List[str],
        resolved_links: Dict[str, str],      # placeholder → full content
        resolved_blobs: Dict[str, str],       # placeholder → full content
    ) -> ReconstructedContext:
        """
        Replace placeholders in cleaned_text with their resolved content,
        inject RAG results, and trim to token budget.

        Args:
            query: the original PreprocessedQuery.
            rag_results: list of text chunks from vector search.
            resolved_links: mapping from "[link::domain]" to fetched page content.
            resolved_blobs: mapping from "[blob::id]" to full blob content.

        Returns:
            ReconstructedContext with the final text ready for LLM extraction.
        """
        truncation_log = []
        resolved = {}

        # 1. Replace link placeholders
        text = query.cleaned_text
        for link in query.extracted_links:
            full = resolved_links.get(link.placeholder, link.raw_url)
            summary = None
            was_truncated = False
            token_count = self._estimate_tokens(full)

            if token_count > self.max_tokens * 0.3:  # Don't let one link dominate
                summary = self._summarize(full, max_tokens=200)
                replacement = f"[Link: {link.domain}] {summary}"
                was_truncated = True
                truncation_log.append(f"Link {link.domain} summarized ({token_count} → ~200 tokens)")
            else:
                replacement = f"[Link: {link.domain}] {full}"

            text = text.replace(link.placeholder, replacement)
            resolved[link.placeholder] = ResolvedPlaceholder(
                placeholder=link.placeholder,
                original=link.raw_url,
                full_content=full,
                summary=summary,
                was_truncated=was_truncated,
                token_count=token_count,
            )

        # 2. Replace blob placeholders
        for blob in query.extracted_blobs:
            full = resolved_blobs.get(blob.placeholder, blob.content_preview)
            summary = None
            was_truncated = False
            token_count = self._estimate_tokens(full)

            if token_count > self.max_tokens * 0.2:
                summary = self._summarize(full, max_tokens=150)
                replacement = f"[Blob] {summary}"
                was_truncated = True
                truncation_log.append(f"Blob {blob.blob_id} summarized ({token_count} → ~150 tokens)")
            else:
                replacement = f"[Blob] {full}"

            text = text.replace(blob.placeholder, replacement)
            resolved[blob.placeholder] = ResolvedPlaceholder(
                placeholder=blob.placeholder,
                original=blob.content_preview,
                full_content=full,
                summary=summary,
                was_truncated=was_truncated,
                token_count=token_count,
            )

        # 3. Inject RAG results
        if rag_results:
            rag_section = "\n\n[Additional Context from Knowledge Base:]\n"
            rag_budget = self.max_tokens * 0.3  # 30% for RAG
            rag_tokens = 0
            for i, chunk in enumerate(rag_results):
                chunk_tokens = self._estimate_tokens(chunk)
                if rag_tokens + chunk_tokens > rag_budget:
                    truncation_log.append(f"RAG results truncated after {i+1}/{len(rag_results)} chunks")
                    break
                rag_section += f"\n[{i+1}] {chunk}\n"
                rag_tokens += chunk_tokens
            text += rag_section

        # 4. Trim to max_tokens
        total_tokens = self._estimate_tokens(text)
        was_truncated = False
        if total_tokens > self.max_tokens:
            text = self._trim_to_budget(text, self.max_tokens)
            was_truncated = True
            truncation_log.append(f"Final context trimmed from {total_tokens} to {self.max_tokens} tokens")

        return ReconstructedContext(
            final_text=text,
            total_tokens=self._estimate_tokens(text),
            was_truncated=was_truncated,
            truncation_log=truncation_log,
            resolved_placeholders=resolved,
        )

    def _estimate_tokens(self, text: str) -> int:
        """Rough token count estimation."""
        return int(len(text) * self.tokens_per_char)

    def _summarize(self, text: str, max_tokens: int = 200) -> str:
        """
        Simple truncation‑based summarization.
        Replace with an actual LLM summarizer if needed.
        """
        max_chars = int(max_tokens / self.tokens_per_char)
        if len(text) <= max_chars:
            return text
        # Take first and last portions
        half = max_chars // 2
        return text[:half] + "\n...[truncated]...\n" + text[-half:]

    def _trim_to_budget(self, text: str, max_tokens: int) -> str:
        """Trim text to fit within token budget, preserving structure."""
        max_chars = int(max_tokens / self.tokens_per_char)
        if len(text) <= max_chars:
            return text
        # Prioritize: keep query + linked content, trim RAG section last
        sections = text.split("\n\n[Additional Context")
        if len(sections) > 1:
            main, rag = sections[0], "[Additional Context" + sections[1]
            main_budget = int(max_chars * 0.7)
            rag_budget = max_chars - main_budget
            if len(main) > main_budget:
                main = main[:main_budget] + "\n...[trimmed]..."
            if len(rag) > rag_budget:
                rag = rag[:rag_budget] + "\n...[trimmed]..."
            return main + "\n\n" + rag
        return text[:max_chars] + "\n...[trimmed]..."

    # ------------------------------------------------------------------
    # Optional: LLM‑powered summarization
    # ------------------------------------------------------------------

    def summarize_with_llm(self, text: str, max_tokens: int = 200) -> str:
        """
        Use a fast LLM to summarize text. Call this if you want better
        summaries than the simple truncation above.

        This is a hook — implement with your own call_fast_model().
        """
        # Example:
        # prompt = f"Summarize this in under {max_tokens} tokens:\n\n{text}"
        # return call_fast_model(prompt)
        return self._summarize(text, max_tokens)  # Fallback