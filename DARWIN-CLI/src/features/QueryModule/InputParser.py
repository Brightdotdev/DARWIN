"""
Darwin Query Module — Input Parser
Detects blobs, extracts URLs, separates clean query from attachments.
"""

import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from src.features.QueryModule.types import BlobChunk, BlobHandle, BlobType, ParsedInput, URLRef


class InputParser:
    """
    Parses raw user input into structured components.
    Handles blob detection, URL extraction, and query cleaning.
    """

    # Heuristic thresholds
    CODE_BLOB_THRESHOLD = 300
    DATA_BLOB_THRESHOLD = 150
    DOCUMENT_BLOB_THRESHOLD = 500
    LONG_QUESTION_THRESHOLD = 300

    # Code detection keywords
    CODE_KEYWORDS = ["def ", "class ", "import ", "function", "const ", "let ", "var ",
                     "return", "if ", "for ", "while ", "try:", "except", "=>", "{}", "[]"]

    # Data structure markers
    DATA_MARKERS = ["[", "{", "|", "csv", "json", "xml", "yaml", "tsv", ": ", ";"]

    def __init__(self, llm_manager=None):
        self.llm_manager = llm_manager  # Optional cheap LLM for blob summarization

    def parse(self, raw_input: str, pinned_files: Optional[List[str]] = None) -> ParsedInput:
        """Main entry point: parse raw input into structured components."""
        pinned_files = pinned_files or []
        
        # Step 1: Extract URLs
        urls, text_without_urls = self._extract_urls(raw_input)
        
        # Step 2: Detect blob type
        blob_type = self._detect_blob_type(text_without_urls)
        
        # Step 3: Separate query from blob
        clean_query, blob_content = self._separate_query_and_blob(text_without_urls, blob_type)
        
        # Step 4: Process blob if present
        blob_handle = None
        if blob_type != BlobType.NONE and blob_content:
            blob_handle = self._process_blob(blob_content, blob_type)
        
        # Step 5: Clean up query
        clean_query = self._clean_query(clean_query)
        
        return ParsedInput(
            clean_query=clean_query,
            blob=blob_handle,
            extracted_urls=urls,
            pinned_files=pinned_files,
            raw_input=raw_input,
        )

    def _extract_urls(self, text: str) -> Tuple[List[URLRef], str]:
        """Extract URLs from text and return cleaned text."""
        url_pattern = re.compile(
            r"https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?",
            re.IGNORECASE,
        )
        urls = []
        found_urls = url_pattern.findall(text)
        
        for url in found_urls:
            domain_match = re.search(r"https?://([^/]+)", url)
            domain = domain_match.group(1) if domain_match else None
            urls.append(URLRef(url=url, domain=domain))
        
        text_without_urls = url_pattern.sub("", text).strip()
        # Clean up extra whitespace from URL removal
        text_without_urls = re.sub(r"\s+", " ", text_without_urls)
        
        return urls, text_without_urls

    def _detect_blob_type(self, text: str) -> BlobType:
        """Detect if text contains a blob and what type."""
        length = len(text)
        
        if length < 300:
            # Short text — check for code patterns anyway
            code_score = sum(1 for kw in self.CODE_KEYWORDS if kw in text)
            if code_score >= 3:
                return BlobType.CODE
            return BlobType.NONE
        
        # Check for code blob
        code_score = sum(1 for kw in self.CODE_KEYWORDS if kw in text)
        if length > self.CODE_BLOB_THRESHOLD and code_score >= 3:
            return BlobType.CODE
        
        # Check for data blob
        data_score = sum(1 for marker in self.DATA_MARKERS if marker in text)
        if length > self.DATA_BLOB_THRESHOLD and data_score >= 3:
            return BlobType.DATA
        
        # Check for document blob
        if length > self.DOCUMENT_BLOB_THRESHOLD:
            # Has paragraph structure?
            paragraphs = [p for p in text.split("\n\n") if len(p.strip()) > 50]
            if len(paragraphs) >= 2:
                return BlobType.DOCUMENT
        
        # Long natural language question
        if length > self.LONG_QUESTION_THRESHOLD:
            return BlobType.LONG_QUESTION
        
        return BlobType.NONE

    def _separate_query_and_blob(self, text: str, blob_type: BlobType) -> Tuple[str, str]:
        """Separate the actual question from blob content."""
        if blob_type == BlobType.NONE:
            return text, ""
        
        # Strategy: look for a short question at the beginning or end
        lines = text.split("\n")
        
        # Check first few lines for a short question
        for i in range(min(3, len(lines))):
            line = lines[i].strip()
            if len(line) < 200 and ("?" in line or any(kw in line.lower() for kw in ["how", "what", "why", "who", "when", "where", "can you", "explain"])):
                clean_query = line
                blob_content = "\n".join(lines[:i] + lines[i+1:]).strip()
                if not blob_content:
                    blob_content = "\n".join(lines).strip()
                    clean_query = self._extract_question_from_blob(blob_content)
                return clean_query, blob_content
        
        # Check last few lines
        for i in range(max(0, len(lines)-3), len(lines)):
            line = lines[i].strip()
            if len(line) < 200 and "?" in line:
                clean_query = line
                blob_content = "\n".join(lines[:i]).strip()
                return clean_query, blob_content
        
        # No clear question found — extract from blob
        blob_content = text
        clean_query = self._extract_question_from_blob(text)
        return clean_query, blob_content

    def _extract_question_from_blob(self, text: str) -> str:
        """Try to infer a question from blob content."""
        sentences = re.split(r"[.!?]\s+", text)
        for sent in sentences[:5]:
            sent = sent.strip()
            if len(sent) > 20 and len(sent) < 200:
                # Look for interrogative or imperative patterns
                lowered = sent.lower()
                if any(kw in lowered for kw in ["how", "what", "why", "who", "when", "where", "explain", "describe", "analyze"]):
                    return sent + "?" if not sent.endswith("?") else sent
        
        # Fallback: use first sentence truncated
        first = sentences[0].strip() if sentences else text[:100]
        return (first[:150] + "...") if len(first) > 150 else first

    def _process_blob(self, content: str, blob_type: BlobType) -> BlobHandle:
        """Chunk blob, store, and summarize."""
        blob_id = f"blob_{uuid.uuid4().hex[:12]}"
        
        if blob_type == BlobType.CODE:
            chunks = self._chunk_code(content)
        elif blob_type == BlobType.DATA:
            chunks = self._chunk_data(content)
        elif blob_type == BlobType.DOCUMENT:
            chunks = self._chunk_document(content)
        else:
            chunks = self._chunk_text(content)
        
        summary = self._summarize_blob(content, blob_type)
        
        return BlobHandle(
            blob_id=blob_id,
            blob_type=blob_type,
            summary=summary,
            chunk_count=len(chunks),
            chunks=chunks,
            raw_content=content[:5000],  # Store truncated raw
        )

    def _chunk_code(self, content: str) -> List[BlobChunk]:
        """Chunk code by functions/classes."""
        import re
        # Split by function/class definitions
        pattern = re.compile(r"(?=^\s*(?:def\s+|class\s+))", re.MULTILINE)
        parts = pattern.split(content)
        
        chunks = []
        for i, part in enumerate(parts):
            if part.strip():
                chunks.append(BlobChunk(
                    index=i,
                    content=part.strip(),
                    token_count=self._estimate_tokens(part),
                    metadata={"type": "code_block"},
                ))
        return chunks if chunks else [BlobChunk(index=0, content=content, token_count=self._estimate_tokens(content))]

    def _chunk_data(self, content: str) -> List[BlobChunk]:
        """Chunk structured data by records/rows."""
        lines = content.split("\n")
        chunks = []
        buffer = []
        buffer_tokens = 0
        chunk_idx = 0
        
        for line in lines:
            line_tokens = self._estimate_tokens(line)
            if buffer_tokens + line_tokens > 500 and buffer:
                chunks.append(BlobChunk(
                    index=chunk_idx,
                    content="\n".join(buffer),
                    token_count=buffer_tokens,
                    metadata={"type": "data_records"},
                ))
                chunk_idx += 1
                buffer = [line]
                buffer_tokens = line_tokens
            else:
                buffer.append(line)
                buffer_tokens += line_tokens
        
        if buffer:
            chunks.append(BlobChunk(
                index=chunk_idx,
                content="\n".join(buffer),
                token_count=buffer_tokens,
                metadata={"type": "data_records"},
            ))
        return chunks if chunks else [BlobChunk(index=0, content=content, token_count=self._estimate_tokens(content))]

    def _chunk_document(self, content: str) -> List[BlobChunk]:
        """Chunk documents by paragraphs."""
        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 20]
        chunks = []
        buffer = []
        buffer_tokens = 0
        chunk_idx = 0
        
        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)
            if buffer_tokens + para_tokens > 600 and buffer:
                chunks.append(BlobChunk(
                    index=chunk_idx,
                    content="\n\n".join(buffer),
                    token_count=buffer_tokens,
                    metadata={"type": "paragraphs"},
                ))
                chunk_idx += 1
                buffer = [para]
                buffer_tokens = para_tokens
            else:
                buffer.append(para)
                buffer_tokens += para_tokens
        
        if buffer:
            chunks.append(BlobChunk(
                index=chunk_idx,
                content="\n\n".join(buffer),
                token_count=buffer_tokens,
                metadata={"type": "paragraphs"},
            ))
        return chunks if chunks else [BlobChunk(index=0, content=content, token_count=self._estimate_tokens(content))]

    def _chunk_text(self, content: str) -> List[BlobChunk]:
        """Generic text chunking by sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", content)
        chunks = []
        buffer = []
        buffer_tokens = 0
        chunk_idx = 0
        
        for sent in sentences:
            sent_tokens = self._estimate_tokens(sent)
            if buffer_tokens + sent_tokens > 500 and buffer:
                chunks.append(BlobChunk(
                    index=chunk_idx,
                    content=" ".join(buffer),
                    token_count=buffer_tokens,
                    metadata={"type": "sentences"},
                ))
                chunk_idx += 1
                buffer = [sent]
                buffer_tokens = sent_tokens
            else:
                buffer.append(sent)
                buffer_tokens += sent_tokens
        
        if buffer:
            chunks.append(BlobChunk(
                index=chunk_idx,
                content=" ".join(buffer),
                token_count=buffer_tokens,
                metadata={"type": "sentences"},
            ))
        return chunks if chunks else [BlobChunk(index=0, content=content, token_count=self._estimate_tokens(content))]

    def _summarize_blob(self, content: str, blob_type: BlobType) -> str:
        """Generate 1-2 sentence summary. Uses cheap LLM if available, else heuristic."""
        if self.llm_manager:
            try:
                prompt = f"Summarize this {blob_type.value} in 1-2 sentences:\n\n{content[:2000]}"
                # Assume llm_manager has a cheap/quick method
                if hasattr(self.llm_manager, "quick_generate"):
                    return self.llm_manager.quick_generate(prompt, max_tokens=80)
            except Exception:
                pass
        
        # Heuristic fallback
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        if not lines:
            return "Empty blob."
        
        first = lines[0][:100]
        if blob_type == BlobType.CODE:
            funcs = len([l for l in lines if l.startswith("def ") or l.startswith("class ")])
            return f"Code blob with {funcs} functions/classes. Starts with: {first}..."
        elif blob_type == BlobType.DATA:
            rows = len(lines)
            return f"Data blob with ~{rows} rows/records. Preview: {first}..."
        elif blob_type == BlobType.DOCUMENT:
            paras = len([l for l in lines if len(l) > 50])
            return f"Document blob with ~{paras} paragraphs. Topic: {first}..."
        else:
            return f"Text blob ({len(content)} chars). Preview: {first}..."

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation. Uses tiktoken if available, else heuristic."""
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            # Fallback: ~4 chars per token for English
            return len(text) // 4

    def _clean_query(self, query: str) -> str:
        """Final query cleanup."""
        query = query.strip()
        # Remove extra whitespace
        query = re.sub(r"\s+", " ", query)
        # Remove leading/trailing punctuation except ?
        query = query.strip(".!,:;-")
        return query