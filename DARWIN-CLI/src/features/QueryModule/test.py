"""
Darwin Query Module — Standalone Test Runner
Run with:

    uv run test.py
"""

from unittest.mock import MagicMock
import traceback
import sys

# Add output path
sys.path.insert(0, "/mnt/agents/output/src/core/Agent/QueryModule")

from src.features.QueryModule.types import (
    BlobHandle,
    BlobType,
    ClassificationResult,
    ConfidenceLevel,
    DeferredChoice,
    DeferredQuestion,
    RetrievalContext,
    RetrievalMode,
    RetrievalPolicy,
    GenerationResult,
    QuestionType,
    ResponsePresentation,
)

from src.features.QueryModule.InputParser import InputParser
from src.features.QueryModule.Classifier import Classifier
from src.features.QueryModule.Retriever import Retriever
from src.features.QueryModule.ContentValidator import ContentValidator
from src.features.QueryModule.GenerationOrchestrator import GenerationOrchestrator
from src.features.QueryModule.ResponseBuilder import ResponseBuilder
from src.features.QueryModule.FollowUpSkill import FollowUpSkill
from src.features.QueryModule.SkillRouter import SkillRouter


# =========================================================
# INPUT PARSER TESTS
# =========================================================

def test_short_query_no_blob():
    parser = InputParser()

    result = parser.parse("Who are the watchers?")

    assert result.blob is None
    assert result.clean_query == "Who are the watchers?"
    assert result.extracted_urls == []


def test_url_extraction():
    parser = InputParser()

    result = parser.parse(
        "Check this out: https://example.com/article and tell me about it"
    )

    assert len(result.extracted_urls) == 1
    assert result.extracted_urls[0].url == "https://example.com/article"
    assert result.extracted_urls[0].domain == "example.com"


def test_fact_question():
    c = Classifier()

    result = c.classify("What is the capital of France?")

    assert QuestionType.FACT in result.question_types
    assert result.confidence >= 0.85


def test_no_rag_skip_retrieval():
    r = Retriever()

    policy = RetrievalPolicy(
        retrieval=RetrievalMode.HYBRID,
        top_k=5,
        requires_rag=False,
    )

    ctx = r.retrieve("What is 2+2?", policy)

    assert ctx.is_empty()
    assert ctx.total_found == 0


def test_response_builder():
    builder = ResponseBuilder()

    gen_result = GenerationResult(
        primary_output='{"title":"Test"}',
        fallback_text="fallback",
        representation=ResponsePresentation.SUMMARY_CARD,
        citations=[],
        sources=[],
        confidence=ConfidenceLevel.HIGH,
        used_retrieval=True,
    )

    package = builder.build(gen_result)

    assert package.mode == "direct"


# =========================================================
# SIMPLE TEST RUNNER
# =========================================================

def run_tests():
    test_functions = [
        obj for name, obj in globals().items()
        if callable(obj) and name.startswith("test_")
    ]

    passed = 0
    failed = 0

    print(f"Running {len(test_functions)} tests...\n")

    for test in test_functions:
        try:
            test()
            print(f"[PASS] {test.__name__}")
            passed += 1

        except AssertionError as e:
            print(f"[FAIL] {test.__name__}")
            print(f"       AssertionError: {e}")
            failed += 1

        except Exception:
            print(f"[ERROR] {test.__name__}")
            traceback.print_exc()
            failed += 1

    print("\n=================================================")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=================================================")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)