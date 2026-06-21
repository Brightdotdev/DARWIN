"""
Darwin Wiki / Knowledge Module — Core Types & Constants (v3.0)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any

# ── Constants ──────────────────────────────────────────────────────────────

VALID_ENTITY_TYPES = {
    "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
    "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY",
    "QUANTITY", "ORDINAL", "CARDINAL",
}

# Entity types that are almost never useful for a knowledge index.
# Entirely suppressed during extraction.
REJECTED_ENTITY_TYPES = {"CARDINAL", "ORDINAL"}

# Words that should never become entity canonical names.
ENTITY_STOPWORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
    "day", "days", "year", "years", "month", "months",
    "thou", "thee", "thy", "thine", "ye", "hath", "doth", "shalt",
    "the", "that", "this", "those", "these", "there", "their", "them",
    "all", "some", "many", "more", "most", "much",
    "mine", "his", "her", "our", "your", "its",
    "also", "unto", "upon", "into", "from", "with", "shall", "will",
}

# ── Data Classes ─────────────────────────────────────────────────────────

@dataclass
class Mention:
    text: str
    chunk_id: int
    span: List[int]  # [start, end] character offsets
    doc_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "span": self.span,
            "doc_id": self.doc_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Mention":
        return cls(
            text=d["text"],
            chunk_id=d["chunk_id"],
            span=d["span"],
            doc_id=d["doc_id"],
        )


@dataclass
class EntityRecord:
    canonical_name: str
    type: str
    mentions: List[Mention] = field(default_factory=list)
    chunk_ids: List[int] = field(default_factory=list)
    doc_ids: List[str] = field(default_factory=list)
    frequency: int = 0
    spread: int = 0
    salience: float = 0.0

    @property
    def entity_id(self) -> str:
        return f"{self.canonical_name}|{self.type.lower()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_name": self.canonical_name,
            "type": self.type,
            "mentions": [m.to_dict() for m in self.mentions],
            "chunk_ids": self.chunk_ids,
            "doc_ids": self.doc_ids,
            "frequency": self.frequency,
            "spread": self.spread,
            "salience": round(self.salience, 4),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EntityRecord":
        return cls(
            canonical_name=d["canonical_name"],
            type=d["type"],
            mentions=[Mention.from_dict(m) for m in d.get("mentions", [])],
            chunk_ids=d.get("chunk_ids", []),
            doc_ids=d.get("doc_ids", []),
            frequency=d.get("frequency", 0),
            spread=d.get("spread", 0),
            salience=d.get("salience", 0.0),
        )