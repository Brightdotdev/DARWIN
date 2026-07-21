from dataclasses import dataclass, field
from enum import Enum
import re
from typing import List
from typing import Dict, Any, Optional



# ── Block-type sentinels (must match ChunkCleaners + extractor) ───────────────
# These are present in the *raw* extractor output (before cleaning).
# The chunker reads them to make split decisions, then they are cleaned away
# by prepare_embedding_text before chunks reach the embedding model.
TABLE_OPEN_TOKEN  = "[TABLE]"          # fitz extractor display text
TABLE_CLOSE_TOKEN = "[/TABLE]"
TABLE_EMB_TOKEN   = "[Table]"          # fitz extractor embedding text
CAPTION_TOKEN     = "[Caption:"        # fitz extractor embedding text
PAGE_HEADER_RE    = re.compile(r"\[Page\s+[^\]]+\]")

# Legacy separator kept for backwards compat with pymupdf4llm output
LEGACY_PAGE_SEP_RE = re.compile(r"^---\s*$", re.MULTILINE)




# ─────────────────────────────────────────────────────────────────────────────
# Overlap direction
# ─────────────────────────────────────────────────────────────────────────────

class OverlapDirection(str, Enum):
    """
    Describes which side(s) of a chunk carry overlap sentences.

    NONE   — no overlap; chunk is self-contained
    LEFT   — chunk *opens* with sentences repeated from the tail of the
             previous chunk  (this chunk "looks left")
    RIGHT  — chunk *closes* with sentences repeated at the head of the
             next chunk  (this chunk "donates right")
    BOTH   — interior window: received from the left AND donates to the right

    Use str-Enum so the value serialises cleanly to JSON as a plain string
    ("none", "left", "right", "both") without needing a custom encoder.
    """
    NONE  = "none"
    LEFT  = "left"
    RIGHT = "right"
    BOTH  = "both"



# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

class ChunkLink:
    def __init__(self, url: str, text: str):
        self.url = url
        self.text = text
    
    def to_dict(self):
        return {"url": self.url, "text": self.text}
    
    def __json__(self):  # For json.dumps
        return self.to_dict()




@dataclass
class Chunk:
    text: str
    clean: str
    index: int
    source: str
    token_count: int
    heading: str
    page: int
    page_end: int
    overlap_direction: OverlapDirection = OverlapDirection.NONE
    links: List[ChunkLink] = field(default_factory=list)
    structural_type: Optional[str] = "body"  # CHANGED: was None — now always present
    
    def __post_init__(self):
        """Ensure overlap_direction is always an Enum."""
        if isinstance(self.overlap_direction, str):
            try:
                self.overlap_direction = OverlapDirection(self.overlap_direction)
            except ValueError:
                self.overlap_direction = OverlapDirection.NONE
    
    @property
    def is_overlap(self) -> bool:
        """Legacy property for backward compatibility."""
        return self.overlap_direction != OverlapDirection.NONE
    

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Chunk into a storage-safe dictionary.

        Rules:
        - No Enums (convert to string)
        - No dataclass nesting (flatten ChunkLink)
        - JSON / Parquet safe
        - Stable schema for embeddings + metadata pipelines
        """

        return {
            "text": self.text,
            "clean": self.clean,
            "index": self.index,
            "source": self.source,
            "token_count": self.token_count,
            "heading": self.heading,
            "page": self.page,
            "page_end": self.page_end,

            # Enum → string (critical for Parquet/JSON compatibility)
            "overlap_direction": (
                self.overlap_direction.value
                if hasattr(self.overlap_direction, "value")
                else str(self.overlap_direction)
            ),

            # Flatten links (avoid nested dataclass in storage layer)
            "links": [
                {
                    "url": link.url,
                    "text": link.text
                }
                for link in self.links
            ],

            # Structural type for penalty system
            "structural_type": self.structural_type,
        }
    
    @classmethod 
    def from_dict(cls, data: Dict[str, Any], default_source: str = "") -> "Chunk":
        """Reconstruct a Chunk from a dictionary (e.g., from storage)."""
        # Handle links
        links = []
        if "links" in data and data["links"]:
            for link_data in data["links"]:
                if isinstance(link_data, dict):
                    links.append(ChunkLink(url=link_data["url"], text=link_data["text"]))
                elif isinstance(link_data, ChunkLink):
                    links.append(link_data)
        
        # Handle overlap_direction
        overlap_direction = data.get("overlap_direction", OverlapDirection.NONE)
        if isinstance(overlap_direction, str):
            try:
                overlap_direction = OverlapDirection(overlap_direction)
            except ValueError:
                overlap_direction = OverlapDirection.NONE
        
        return cls(
            text=data.get("text", ""),
            clean=data.get("clean", ""),
            index=data["index"],
            source=data.get("source", default_source),
            token_count=data.get("token_count", 0),
            heading=data.get("heading", ""),
            page=data.get("page", 0),
            page_end=data.get("page_end", 0),
            overlap_direction=overlap_direction,
            links=links,
            structural_type=data.get("structural_type", "body"),
        )
    
    
    def __json__(self):  # For json.dumps
        return self.to_dict()


@dataclass
class ChunkingArgs:
    files:     list[str] = field(default_factory=list)
    rerun:     bool = False

# ─────────────────────────────────────────────────────────────────────────────
# Page tracking
# ─────────────────────────────────────────────────────────────────────────────

def count_page_seps(text: str) -> int:
    """
    Count page boundaries in a text fragment.

    Supports both formats:
    - New extractor:  [Page N]  lines
    - Legacy pymupdf4llm / old extractor:  ---  lines
    """
    new_style    = len(PAGE_HEADER_RE.findall(text))
    legacy_style = len(LEGACY_PAGE_SEP_RE.findall(text))
    return new_style + legacy_style