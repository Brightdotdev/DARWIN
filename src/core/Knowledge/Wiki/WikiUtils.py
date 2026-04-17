"""
src/core/Wiki/WikiUtils.py
--------------------------
Utilities for wiki generation: argument handling, templates,
and output formatting.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
from pathlib import Path


# ============================================================================
# Constants
# ============================================================================

DEFAULT_MODE = "index"           # index or summary
DEFAULT_MAX_CONCEPTS = 50        # Max concepts to include
DEFAULT_CHUNKS_PER_CONCEPT = 10  # Max chunks per concept for context

WIKI_MODES = ["index", "summary"]


# ============================================================================
# Args
# ============================================================================

@dataclass
class WikiGenerationArgs:
    target: str
    file: Optional[str] = None
    project: Optional[str] = None
    mode: str = DEFAULT_MODE
    max_concepts: int = DEFAULT_MAX_CONCEPTS
    chunks_per_concept: int = DEFAULT_CHUNKS_PER_CONCEPT
    llm_model: Optional[str] = None      # For summary mode
    output_dir: Optional[str] = None     # Where to save .md files
    inspect: bool = False
    rerun: bool = False


# ============================================================================
# Templates
# ============================================================================

INDEX_TEMPLATE = """# {title} — Concept Index

*Generated from {filename}*
*{concept_count} concepts extracted*

---

{concept_sections}
"""

CONCEPT_SECTION_TEMPLATE = """## {concept}

*Appears in {chunk_count} chunks | Related: {related_concepts}*

{chunk_references}
"""

CHUNK_REFERENCE_TEMPLATE = """- **Page {page}** — {heading}
  > {snippet}
"""

SUMMARY_PROMPT_TEMPLATE = """You are generating a Wikipedia-style article about "{concept}" 
based on the document "{filename}".

Use ONLY the context provided below. Every claim must be followed by a citation [Page X].
Write in encyclopedic, neutral tone. Include:
- A clear definition or description
- Key details and context
- Connections to other concepts mentioned

Context from the document:
{context}

Wikipedia-style summary of "{concept}":
"""

INDEX_PROMPT_TEMPLATE = """Generate a master index for the document "{filename}".

Below are the top concepts extracted from the document with their frequencies.
Create a hierarchical index that groups related concepts together.

Concepts:
{concepts_list}

Master Index:
"""


# ============================================================================
# Output formatting
# ============================================================================

def format_wiki_output(pages: List, mode: str) -> str:
    """Format wiki pages for display or file output."""
    output = []
    
    for page in pages:
        output.append(f"# {page.title}\n")
        output.append(page.content)
        output.append("\n---\n")
    
    return "\n".join(output)


def extract_snippet(text: str, concept: str, max_length: int = 200) -> str:
    """Extract a snippet around the concept mention."""
    import re
    
    # Find the concept in text (case insensitive)
    pattern = re.compile(re.escape(concept), re.IGNORECASE)
    match = pattern.search(text)
    
    if not match:
        return text[:max_length] + "..." if len(text) > max_length else text
    
    # Get context around the match
    start = max(0, match.start() - 100)
    end = min(len(text), match.end() + 100)
    
    snippet = text[start:end]
    
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    
    return snippet.strip()


def save_wiki_files(
    pages: List,
    output_dir: Path,
    filename: str,
) -> List[Path]:
    """Save wiki pages as individual markdown files."""
    import re
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    # Save index
    index_path = output_dir / f"{filename}_index.md"
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(pages[0].content if pages else "")
    saved_paths.append(index_path)
    
    # Save individual concept pages
    for page in pages[1:]:
        # Sanitize filename
        safe_title = re.sub(r'[<>:"/\\|?*]', '_', page.title)
        safe_title = safe_title.replace(' ', '_')
        
        page_path = output_dir / f"{safe_title}.md"
        with open(page_path, 'w', encoding='utf-8') as f:
            f.write(page.content)
        saved_paths.append(page_path)
    
    return saved_paths