"""
Darwin Query Module — Content Validator
Checks if retrieved context supports suggested representation.
Handles downgrade chains and edge cases.
"""

import re
from typing import Any, Dict, List, Optional

from src.features.QueryModule.types import (
    DOWNGRADE_CHAIN,
    REPRESENTATION_REQUIREMENTS,
    ResponsePresentation,
    RetrievalContext,
    ValidationResult,
)


class ContentValidator:
    """
    Validates whether retrieved content can support a given representation.
    Implements downgrade chains for PARTIAL/FAIL states.
    """

    def __init__(self):
        pass

    def validate(
        self,
        context: RetrievalContext,
        suggested: ResponsePresentation,
        explicit_lock: Optional[ResponsePresentation] = None,
    ) -> ValidationResult:
        """
        Validate content against representation requirements.
        
        If explicit_format_lock is set, skip validation and return PASS.
        """
        # Explicit format lock bypasses validation
        if explicit_lock:
            return ValidationResult(
                representation=explicit_lock,
                status="PASS",
                reason="Explicit format lock from user request",
            )
        
        requirements = REPRESENTATION_REQUIREMENTS.get(suggested, {})
        
        # Count passages
        passage_count = len(context.passages)
        min_passages = requirements.get("min_passages", 0)
        
        if passage_count < min_passages:
            # Try downgrade chain
            return self._downgrade(context, suggested, f"Only {passage_count} passages, need {min_passages}")
        
        # Check entity count
        min_entities = requirements.get("min_entities", 0)
        if min_entities > 0:
            entity_count = self._count_entities(context)
            if entity_count < min_entities:
                return self._downgrade(context, suggested, f"Only {entity_count} entities, need {min_entities}")
        
        # Check dates
        if requirements.get("requires_dates"):
            date_count = self._count_dates(context)
            if date_count < 1:
                return self._downgrade(context, suggested, "No dates found for timeline")
        
        # Check relationships
        if requirements.get("requires_relationships"):
            rel_count = self._count_relationships(context)
            if rel_count < 2:
                return self._downgrade(context, suggested, "Insufficient relationships for graph")
        
        # Check steps
        if requirements.get("requires_steps"):
            step_count = self._count_steps(context)
            if step_count < 2:
                return self._downgrade(context, suggested, "Insufficient steps found")
        
        # Check variables
        if requirements.get("requires_variables"):
            var_count = self._count_variables(context)
            if var_count < 1:
                return self._downgrade(context, suggested, "No variables/parameters found")
        
        # Check branches
        if requirements.get("requires_branches"):
            branch_count = self._count_branches(context)
            if branch_count < 2:
                return self._downgrade(context, suggested, "Insufficient branches for tree")
        
        # Check structure
        if requirements.get("requires_structure"):
            has_structure = self._has_structure(context)
            if not has_structure:
                return self._downgrade(context, suggested, "No structured data for table")
        
        return ValidationResult(
            representation=suggested,
            status="PASS",
            reason="All requirements met",
        )

    def _downgrade(
        self,
        context: RetrievalContext,
        original: ResponsePresentation,
        reason: str,
    ) -> ValidationResult:
        """Walk downgrade chain to find best fit."""
        chain = DOWNGRADE_CHAIN.get(original, [ResponsePresentation.MARKDOWN])
        
        for candidate in chain:
            req = REPRESENTATION_REQUIREMENTS.get(candidate, {})
            min_p = req.get("min_passages", 0)
            
            if len(context.passages) >= min_p:
                # Additional checks
                if req.get("min_entities") and self._count_entities(context) < req["min_entities"]:
                    continue
                if req.get("requires_dates") and self._count_dates(context) < 1:
                    continue
                
                return ValidationResult(
                    representation=candidate,
                    status="PARTIAL",
                    reason=f"Downgraded from {original.value}: {reason}",
                    downgrade_chain=[original] + chain,
                )
        
        # Ultimate fallback
        return ValidationResult(
            representation=ResponsePresentation.MARKDOWN,
            status="FAIL",
            reason=f"Failed to meet requirements for {original.value}: {reason}",
            downgrade_chain=[original] + chain,
        )

    def _count_entities(self, context: RetrievalContext) -> int:
        """Count unique capitalized phrases as entities."""
        all_text = " ".join([p.content for p in context.passages])
        found = re.findall(r"\b[A-Z][a-zA-Z\s]{2,20}\b", all_text)
        seen = set()
        for f in found:
            f = f.strip()
            if f and f not in seen:
                seen.add(f)
        return len(seen)

    def _count_dates(self, context: RetrievalContext) -> int:
        """Count date patterns in passages."""
        date_patterns = [
            r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b",
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
            r"\b\d{4}\b",
        ]
        count = 0
        for p in context.passages:
            for pat in date_patterns:
                count += len(re.findall(pat, p.content, re.I))
        return count

    def _count_relationships(self, context: RetrievalContext) -> int:
        """Count relationship-indicating phrases."""
        rel_words = ["related to", "connected", "links", "depends on", "influences",
                     "causes", "leads to", "results in", "parent of", "child of",
                     "part of", "contains", "uses", "implements", "extends"]
        count = 0
        for p in context.passages:
            lowered = p.content.lower()
            for rw in rel_words:
                count += lowered.count(rw)
        return count

    def _count_steps(self, context: RetrievalContext) -> int:
        """Count step-indicating patterns."""
        step_markers = ["step", "first", "second", "third", "next", "then", "after",
                        "finally", "1.", "2.", "3.", "(1)", "(2)", "(3)",
                        "begin by", "start with", "followed by"]
        count = 0
        for p in context.passages:
            lowered = p.content.lower()
            for sm in step_markers:
                count += lowered.count(sm)
        return count

    def _count_variables(self, context: RetrievalContext) -> int:
        """Count variable/parameter patterns."""
        var_patterns = [r"\b\w+\s*=\s*\d+", r"\b\w+\s*:\s*\d+",
                        r"\b(?:parameter|variable|setting|config|option)\b"]
        count = 0
        for p in context.passages:
            for pat in var_patterns:
                count += len(re.findall(pat, p.content, re.I))
        return count

    def _count_branches(self, context: RetrievalContext) -> int:
        """Count branch/condition patterns."""
        branch_markers = ["if ", "else", "otherwise", "unless", "when ", "depending",
                          "case", "scenario", "option", "choice"]
        count = 0
        for p in context.passages:
            lowered = p.content.lower()
            for bm in branch_markers:
                count += lowered.count(bm)
        return count

    def _has_structure(self, context: RetrievalContext) -> bool:
        """Check if content has tabular or structured data patterns."""
        for p in context.passages:
            if "|" in p.content or "\t" in p.content or re.search(r"\b\w+:\s*\w+", p.content):
                return True
        return False