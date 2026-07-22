"""
Darwin NLP Utilities — Core Types & Constants (v5.1)
CHANGES (v5.1):
- EntityRecord: base_name (merging key), display_name (label), quality_signal (persisted)
- entity_id is now type‑agnostic (base_name only)
- to_dict/from_dict updated for new fields
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Set, Optional

# ── Constants ──────────────────────────────────────────────────────────────

VALID_ENTITY_TYPES = {
    "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
    "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY",
    "QUANTITY", "ORDINAL", "CARDINAL",
}


ENTITY_BLACKLIST = {
    # Pronouns / determiners
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs",
    "this", "that", "these", "those", "there", "here", "where", "when", "what", "which",
    "who", "whom", "whose", "why", "how", "all", "some", "any", "every", "each", "either",
    "neither", "none", "both", "few", "many", "much", "more", "most", "other", "another",
    "such", "only", "own", "same", "so", "than", "too", "very", "just", "also",
    # Temporal deictics
    "day", "days", "night", "nights", "year", "years", "month", "months", "week", "weeks",
    "hour", "hours", "minute", "minutes", "second", "seconds", "time", "times", "period",
    "age", "ages", "era", "epoch", "date", "dates", "century", "centuries", "millennium",
    "today", "tomorrow", "yesterday", "now", "then", "soon", "later", "before", "after",
    "once", "twice", "again", "already", "yet", "still", "ever", "never", "always",
    "often", "sometimes", "usually", "finally", "recently", "currently", "formerly",
    # Generic relational nouns (spaCy loves these as PERSON/ORG)
    "man", "men", "woman", "women", "person", "people", "folk", "human", "humans",
    "son", "sons", "daughter", "daughters", "child", "children", "kid", "kids",
    "father", "fathers", "mother", "mothers", "parent", "parents",
    "brother", "brothers", "sister", "sisters", "family", "families",
    "king", "kings", "queen", "queens", "lord", "lords", "god", "gods",
    "priest", "priests", "prophet", "prophets", "angel", "angels", "devil", "devils",
    "place", "places", "land", "lands", "country", "countries", "nation", "nations",
    "city", "cities", "town", "towns", "world", "worlds", "earth", "heaven", "heavens",
    "hell", "hells", "house", "houses", "home", "homes", "room", "rooms",
    "way", "ways", "part", "parts", "kind", "kinds", "sort", "sorts", "type", "types",
    "thing", "things", "matter", "matters", "case", "cases", "point", "points",
    "end", "ends", "beginning", "beginnings", "middle", "middles", "side", "sides",
    # Modal / frequent verbs mislabeled as entities
    "shall", "will", "would", "should", "may", "might", "can", "could",
    "must", "ought", "need", "dare", "used", "hath", "doth", "didst", "doeth",
    "said", "say", "says", "spoke", "speak", "speaks", "told", "tell", "tells",
    "came", "come", "comes", "went", "go", "goes", "made", "make", "makes",
    "took", "take", "takes", "gave", "give", "gives", "saw", "see", "sees",
    "knew", "know", "knows", "thought", "think", "thinks", "found", "find", "finds",
    "brought", "bring", "brings", "put", "puts", "set", "sets", "let", "lets",
    "left", "leave", "leaves", "held", "hold", "holds", "kept", "keep", "keeps",
    "called", "call", "calls", "named", "name", "names", "written", "write", "writes",
    # Function words / connectors
    "unto", "upon", "into", "from", "with", "within", "without", "about", "above",
    "across", "after", "against", "along", "among", "around", "at", "before", "behind",
    "below", "beneath", "beside", "between", "beyond", "by", "down", "during", "except",
    "for", "in", "inside", "near", "of", "off", "on", "onto", "out", "outside", "over",
    "past", "since", "through", "throughout", "till", "to", "toward", "under", "until",
    "up", "upon", "via", "versus", "vs", "worth", "per", "pro", "re", "versus",
    "and", "or", "but", "nor", "yet", "so", "because", "although", "though", "while",
    "whereas", "if", "unless", "whether", "either", "neither", "both", "not", "no",
    "be", "being", "been", "am", "is", "are", "was", "were", "have", "has", "had",
    "do", "does", "did", "done", "get", "got", "gotten", "become", "became",
    # Numbers / ordinals (even if spaCy labels them)
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
    "eleven", "twelve", "twenty", "thirty", "hundred", "thousand", "million",
    # Fragments / artifacts
    "et", "al", "etc", "eg", "ie", "viz", "vs", "cf", "ibid", "op", "cit",
    "chapter", "chapters", "verse", "verses", "page", "pages", "line", "lines",
    "paragraph", "paragraphs", "section", "sections", "volume", "volumes",
    "note", "notes", "footnote", "footnotes", "appendix", "appendices",
    "figure", "figures", "table", "tables", "diagram", "diagrams",
    "see", "seen", "saw", "look", "looked", "looks", "appear", "appears", "appeared",
    "seem", "seems", "seemed", "become", "became", "remains", "remained",
}


# ── Data Classes ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Mention:
    text: str
    chunk_id: int
    span: Tuple[int, int]
    doc_id: str
    source: str = "ner"          # "ner" | "noun_chunk" | "both"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "span": list(self.span),
            "doc_id": self.doc_id,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Mention":
        span = d["span"]
        if isinstance(span, list):
            span = tuple(span)
        return cls(
            text=d["text"],
            chunk_id=d["chunk_id"],
            span=span,
            doc_id=d["doc_id"],
            source=d.get("source", "ner"),
        )


@dataclass
class EntityRecord:
    """
    Entity with lazy-derived fields, variant tracking, and multi-source mentions.
    v5.1: base_name (merging key), display_name (label), quality_signal persisted.
    """
    base_name: str                # stripped of determiners, lowercase – used for merging
    display_name: str             # best readable name (e.g., "Book of Enoch")
    type: str
    mentions: List[Mention] = field(default_factory=list)
    salience: float = 0.0
    type_confidence: Dict[str, int] = field(default_factory=dict)
    variant_names: Set[str] = field(default_factory=set)
    quality_signal: float = 0.0   # persisted signal score

    @property
    def entity_id(self) -> str:
        # Type‑agnostic: now merged across types
        return self.base_name.lower().replace(" ", "_")

    @property
    def canonical_name(self) -> str:
        # For backward compatibility, return display_name
        return self.display_name

    @property
    def chunk_ids(self) -> List[int]:
        return sorted({m.chunk_id for m in self.mentions})

    @property
    def doc_ids(self) -> List[str]:
        return sorted({m.doc_id for m in self.mentions})

    @property
    def frequency(self) -> int:
        return len(self.mentions)

    @property
    def spread(self) -> int:
        return len(self.doc_ids)

    @property
    def ner_mentions(self) -> int:
        return sum(1 for m in self.mentions if m.source in ("ner", "both"))

    @property
    def noun_chunk_mentions(self) -> int:
        return sum(1 for m in self.mentions if m.source in ("noun_chunk", "both"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_name": self.base_name,
            "display_name": self.display_name,
            "type": self.type,
            "mentions": [m.to_dict() for m in self.mentions],
            "salience": round(self.salience, 4),
            "type_confidence": self.type_confidence,
            "variant_names": sorted(self.variant_names),
            "quality_signal": round(self.quality_signal, 4),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EntityRecord":
        return cls(
            base_name=d.get("base_name", d.get("canonical_name", "").lower()),
            display_name=d.get("display_name", d.get("canonical_name", "")),
            type=d["type"],
            mentions=[Mention.from_dict(m) for m in d.get("mentions", [])],
            salience=d.get("salience", 0.0),
            type_confidence=d.get("type_confidence", {}),
            variant_names=set(d.get("variant_names", [])),
            quality_signal=d.get("quality_signal", 0.0),
        )