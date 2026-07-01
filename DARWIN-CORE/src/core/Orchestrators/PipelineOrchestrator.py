"""
Darwin PipelineOrchestrator — end-to-end pipeline execution.
Typed parameters, per-stage session logging, stage purge, and knowledge wiring.
"""

import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Storage.DarwinStore import DarwinStore

# ── Stage handlers ─────────────────────────────────────────────────────────────
from src.pipleline.Ingestion.Parsing.lib import ParsingArgs
from src.pipleline.Ingestion.Chunking.lib import ChunkingArgs
from src.pipleline.Indexing.lib import IndexingArgs
from src.pipleline.Ingestion.main import handle_ingestion
from src.pipleline.Indexing.main import handle_indexing

# ── Knowledge ────────────────────────────────────────────────────────────────
from src.features.Knowledge.WikiOrchestrator import WikiOrchestrator, create_wiki_orchestrator


# ═══════════════════════════════════════════════════════════════════════════════
# Enums & Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

class PipelineStage(str, Enum):
    """IntelliSense-friendly stage identifiers. Comparable to plain strings."""
    INGESTION = "ingestion"
    INDEXING = "indexing"
    KNOWLEDGE = "knowledge"


@dataclass
class KnowledgeArgs:
    """Parameters forwarded to WikiOrchestrator."""
    files: List[str] = field(default_factory=list)
    rerun: bool = False
    knowledge_stages: List[str] = field(default_factory=lambda: ["wiki", "graph", "html"])
    similarity_threshold: float = 0.82
    window_size: int = 5
    min_mentions: int = 1
    min_entropy: float = 0.3
    max_entity_len: int = 60
    pmi_threshold: Optional[float] = None
    max_nodes_html: int = 150
    html_title: str = "Knowledge Graph"
    skip_labels: Optional[Set[str]] = None


@dataclass
class PipelineParameters:
    """
    Strongly-typed bag of stage arguments.
    Only populate the fields for the stages you intend to run.
    """
    parsing: Optional[ParsingArgs] = None
    chunking: Optional[ChunkingArgs] = None
    indexing: Optional[IndexingArgs] = None
    knowledge: Optional[KnowledgeArgs] = None


@dataclass
class PipelineResult:
    success: bool = False
    parameters_validated: bool = False
    stages_completed: List[PipelineStage] = field(default_factory=list)
    stages_failed: Dict[str, str] = field(default_factory=dict)
    logs: Dict[str, List[Any]] = field(default_factory=dict)
    summary: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class PipelineOrchestrator:
    """
    Stages: ingestion → indexing → knowledge
    Per-stage session logging via DarwinLogger. Knowledge stage wired in.
    """

    STAGE_ORDER: List[PipelineStage] = [
        PipelineStage.INGESTION,
        PipelineStage.INDEXING,
        PipelineStage.KNOWLEDGE,
    ]

    def __init__(
        self,
        store: DarwinStore,
        wiki_orchestrator: Optional[WikiOrchestrator] = None,
    ):
        self.store = store
        self.wiki = wiki_orchestrator or create_wiki_orchestrator(store)
        self._handlers = {
            PipelineStage.INGESTION: self._run_ingestion,
            PipelineStage.INDEXING: self._run_indexing,
            PipelineStage.KNOWLEDGE: self._run_knowledge,
        }

    # ── Coercion & Validation ─────────────────────────────────────────────────

    def _coerce_stages(
        self,
        stages: Union[List[PipelineStage], List[str]],
    ) -> List[PipelineStage]:
        """Accept enums or strings, normalize to PipelineStage."""
        result: List[PipelineStage] = []
        for s in stages:
            if isinstance(s, PipelineStage):
                result.append(s)
            elif isinstance(s, str):
                try:
                    result.append(PipelineStage(s))
                except ValueError:
                    valid = [m.value for m in PipelineStage]
                    raise ValueError(f"Unknown stage '{s}'. Valid: {valid}")
            else:
                raise TypeError(f"Stage must be PipelineStage or str, got {type(s).__name__}")
        return result

    def _validate(
        self,
        params: PipelineParameters,
        stages: List[PipelineStage],
    ) -> Optional[str]:
        """Pre-flight check before any compute begins."""
        for stage in stages:
            if stage == PipelineStage.INGESTION:
                if params.parsing is None:
                    return "Ingestion stage requires PipelineParameters.parsing"
                if params.chunking is None:
                    return "Ingestion stage requires PipelineParameters.chunking"
            elif stage == PipelineStage.INDEXING:
                if params.indexing is None:
                    return "Indexing stage requires PipelineParameters.indexing"
            elif stage == PipelineStage.KNOWLEDGE:
                if params.knowledge is None:
                    return "Knowledge stage requires PipelineParameters.knowledge"
                if not params.knowledge.files:
                    return "Knowledge stage requires at least one file in KnowledgeArgs.files"
        return None

    def _log_levels(self, mode: str) -> Set[LogType]:
        """Quiet = warnings, errors, and successes only. Normal = everything."""
        if mode == "quiet":
            return {LogType.WARNING, LogType.ERROR, LogType.SUCCESS}
        return {LogType.INFO, LogType.SUCCESS, LogType.WARNING, LogType.ERROR}

    # ── Main entrypoint ─────────────────────────────────────────────────────────

    def build_pipeline(
        self,
        params: PipelineParameters,
        stages: Union[List[PipelineStage], List[str]],
        *,
        strict: bool = True,
        log_mode: str = "quiet",
    ) -> PipelineResult:
        """
        Execute pipeline stages with typed parameters, per-stage session logging,
        and structured log capture.

        Args:
            params: Strongly-typed stage arguments.
            stages: List of PipelineStage enums (or strings, coerced automatically).
            strict: If True, halt on first stage failure.
            log_mode: "quiet" (default) returns WARNING/ERROR/SUCCESS only.
                      "normal" returns everything including INFO.

        Returns:
            PipelineResult with per-stage logs under result.logs[stage.value].
            Orchestrator-level logs live under result.logs["orchestrator"].
        """
        if log_mode not in ("quiet", "normal"):
            raise ValueError(f"log_mode must be 'quiet' or 'normal', got {log_mode}")

        result = PipelineResult(success=True, parameters_validated=False)
        stages = self._coerce_stages(stages)
        levels = self._log_levels(log_mode)

        # ── Orchestrator session wraps everything ──
        with darwin.session("pipeline_orchestrator"):
            # 1. Pre-validate
            validation_error = self._validate(params, stages)
            if validation_error:
                darwin.error(f"Validation failed: {validation_error}")
                result.success = False
                result.summary = f"Validation failed: {validation_error}"
                result.logs["orchestrator"] = darwin.get_logs(
                    session="pipeline_orchestrator", levels=levels
                )
                return result

            result.parameters_validated = True
            darwin.success(
                f"Parameters validated. Stages: {[s.value for s in stages]}"
            )

            # 2. Canonical order
            ordered = [s for s in self.STAGE_ORDER if s in stages]
            ordered += [s for s in stages if s not in self.STAGE_ORDER]

            # 3. Execute per-stage with isolated sessions
            for stage in ordered:
                session_name = f"pipeline_{stage.value}"
                with darwin.session(session_name):
                    darwin.info(f"▶ Stage starting: {stage.value}")
                    try:
                        self._handlers[stage](params)
                        darwin.success(f"✓ Stage complete: {stage.value}")
                        result.stages_completed.append(stage)
                    except Exception as e:
                        err_msg = f"{e}"
                        darwin.error(
                            f"✗ Stage failed: {stage.value}",
                            payload={"error": err_msg},
                        )
                        darwin.error(traceback.format_exc())
                        result.stages_failed[stage.value] = err_msg
                        result.success = False
                        if strict:
                            darwin.warning("Strict mode: halting pipeline")
                            break

                # Capture this stage's logs (failed or not)
                result.logs[stage.value] = darwin.get_logs(
                    session=session_name, levels=levels
                )

            # 4. Finalize
            result.summary = self._build_summary(result)
            if result.success:
                darwin.success(f"Pipeline complete: {result.summary}")
            else:
                darwin.error(f"Pipeline finished with failures: {result.summary}")

            # Capture orchestrator-level logs
            result.logs["orchestrator"] = darwin.get_logs(
                session="pipeline_orchestrator", levels=levels
            )

        return result

    # ── Purge / Artifact removal ──────────────────────────────────────────────

    def purge(
        self,
        stages: Union[List[PipelineStage], List[str]],
        filenames: List[str],
    ) -> None:
        """
        Scrub artifacts for specific files in specific stages.
        Logs every deletion attempt and outcome. Returns nothing.
        """
        if not filenames:
            darwin.warning("Purge called with no filenames — nothing to do")
            return

        stages = self._coerce_stages(stages)
        darwin.info(f"Purging {len(stages)} stage(s) for {len(filenames)} file(s)")

        for stage in stages:
            for filename in filenames:
                try:
                    if stage == PipelineStage.INGESTION:
                        self.store.delete_markdown(filename)
                        self.store.delete_records(filename)
                    elif stage == PipelineStage.INDEXING:
                        self.store.delete_embeddings(filename)
                        self.store.delete_vector(filename)
                        self.store.delete_bm25(filename)
                        self.store.delete_stopwords(filename)
                    elif stage == PipelineStage.KNOWLEDGE:
                        self.wiki.delete_knowledge(filename)
                    darwin.success(f"[purge] {stage.value}/{filename} removed")
                except Exception as e:
                    darwin.error(f"[purge] {stage.value}/{filename} failed: {e}")

        darwin.success("Purge complete")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_summary(self, result: PipelineResult) -> str:
        parts = [
            f"Pipeline {'OK' if result.success else 'FAILED'}",
            f"Completed: {len(result.stages_completed)}",
            f"Failed: {len(result.stages_failed)}",
        ]
        if result.stages_failed:
            parts.append(f"Errors: {list(result.stages_failed.keys())}")
        return " | ".join(parts)

    # ── Stage runners ───────────────────────────────────────────────────────────

    def _run_ingestion(self, params: PipelineParameters) -> None:
        if not params.parsing or not params.chunking:
            raise ValueError("ParsingArgs and ChunkingArgs required")
        handle_ingestion(params.parsing, params.chunking, self.store)

    def _run_indexing(self, params: PipelineParameters) -> None:
        if not params.indexing:
            raise ValueError("IndexingArgs required")
        handle_indexing(params.indexing, self.store)

    def _run_knowledge(self, params: PipelineParameters) -> None:
        if not params.knowledge:
            raise ValueError("KnowledgeArgs required")
        args = params.knowledge
        if not args.files:
            darwin.warning("Knowledge stage: no files specified — skipping")
            return
        self.wiki.handle_knowledge_batch(
            filenames=args.files,
            rerun=args.rerun,
            stages=args.knowledge_stages,
            similarity_threshold=args.similarity_threshold,
            window_size=args.window_size,
            min_mentions=args.min_mentions,
            min_entropy=args.min_entropy,
            max_entity_len=args.max_entity_len,
            pmi_threshold=args.pmi_threshold,
            max_nodes_html=args.max_nodes_html,
            html_title=args.html_title,
            skip_labels=args.skip_labels,
        )