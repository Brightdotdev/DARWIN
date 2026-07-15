"""
Darwin PipelineOrchestrator — end-to-end pipeline execution.
Typed parameters, per-stage session logging, stage purge.
"""

import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Storage.DarwinStore import DarwinStore

# ── Stage handlers ─────────────────────────────────────────────────────────────
from src.core.Pipeline.Ingestion.Parsing.lib import ParsingArgs
from src.core.Pipeline.Indexing.lib import IndexingArgs
from src.core.Pipeline.Ingestion.main import handle_ingestion
from src.core.Pipeline.Indexing.main import handle_indexing

from src.core.Orchestrators.lib import PipelineParameters as PipelineInput  # public API


# ═══════════════════════════════════════════════════════════════════════════════
# Enums & Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

class PipelineStage(str, Enum):
    """IntelliSense-friendly stage identifiers. Comparable to plain strings."""
    INGESTION = "ingestion"
    INDEXING = "indexing"




@dataclass
class StageParameters:
    parsing: Optional[ParsingArgs] = None
    indexing: Optional[IndexingArgs] = None


@dataclass
class PipelineResult:
    success: bool = False
    parameters_validated: bool = False
    stages_completed: List[PipelineStage] = field(default_factory=list)
    stages_failed: Dict[str, str] = field(default_factory=dict)
    logs: Dict[str, List[str]] = field(default_factory=dict)
    summary: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class PipelineOrchestrator:
    """
    Stages: ingestion → indexing
    """

    STAGE_ORDER: List[PipelineStage] = [
        PipelineStage.INGESTION,
        PipelineStage.INDEXING,
    ]

    def __init__(
        self,
        store: DarwinStore,
    ):
        self.store = store
        
        if not self.store:
            darwin.error("A valid DarwinStore instance is required.")
            self.store = None
        store.repair_registry()
        self._handlers = {
            PipelineStage.INGESTION: self._run_ingestion,
            PipelineStage.INDEXING: self._run_indexing,
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
                    darwin.error(f"Unknown stage '{s}'. Valid: {valid}")
                    return []
            else:
                darwin.error(f"Stage must be PipelineStage or str, got {type(s).__name__}")
                return []
        return result

    def _validate(
        self,
        params: StageParameters,
        stages: List[PipelineStage],
    ) -> Optional[str]:
        """Pre-flight check before any compute begins."""
        for stage in stages:
            if stage == PipelineStage.INGESTION:
                if params.parsing is None:
                    return "Ingestion stage requires StageParameters.parsing"
               
            elif stage == PipelineStage.INDEXING:
                if params.indexing is None:
                    return "Indexing stage requires StageParameters.indexing"
            
        return None

    def _log_levels(self) -> Set[LogType]:
        """Always filter to warnings, errors, and successes only."""
        return {LogType.WARNING, LogType.ERROR, LogType.SUCCESS}

    @staticmethod
    def _format_logs(events: list) -> list[str]:
        """Convert LogEvents to readable [session][LEVEL] message strings."""
        return [f"[{e.session}][{e.log_type.name}] {e.message}" for e in events]

    # ── Main entrypoint ─────────────────────────────────────────────────────────

    def build_pipeline(
        self,
        params: PipelineInput,
    ) -> PipelineResult:
        """
        Execute pipeline stages with typed parameters, per-stage session logging,
        and structured log capture.

        Args:
            params: PipelineParameters from Orchestrators.lib — the single
                    source of truth for defaults. Use .build() internally.

        Returns:
            PipelineResult with per-stage logs under result.logs[stage.value].
            Orchestrator-level logs live under result.logs["orchestrator"].
        """
        internal, raw_stages = params.build()
        stages = self._coerce_stages(raw_stages)
        strict = params.strict

        result = PipelineResult(success=True, parameters_validated=False)
        levels = self._log_levels()

        # ── Orchestrator session wraps everything ──
        with darwin.session("PIPELINE"):
            # 1. Pre-validate
            validation_error = self._validate(internal, stages)
            if validation_error:
                darwin.error(f"Pipeline parameter validation failed: {validation_error}")
                result.success = False
                result.summary = f"Validation failed: {validation_error}"
                result.logs["orchestrator"] = self._format_logs(
                    darwin.get_logs(session="PIPELINE", levels=levels)
                )
                return result

            result.parameters_validated = True
            darwin.success(
                f"Parameters validated for stages: {[s.value for s in stages]}"
            )

            # 2. Canonical order
            ordered = [s for s in self.STAGE_ORDER if s in stages]
            ordered += [s for s in stages if s not in self.STAGE_ORDER]

            # 3. Execute per-stage with isolated sessions
            for stage in ordered:
                session_name = f"{stage.value.capitalize()}"
                with darwin.session(session_name):
                    darwin.info(f"Starting stage '{stage.value}'")
                    try:
                        self._handlers[stage](internal)
                        darwin.success(f"Stage '{stage.value}' completed successfully")
                        result.stages_completed.append(stage)
                    except Exception as e:
                        err_msg = f"{e}"
                        darwin.error(
                            f"Stage '{stage.value}' failed: {err_msg}",
                            payload={"error": err_msg},
                        )
                        darwin.error(traceback.format_exc())
                        result.stages_failed[stage.value] = err_msg
                        result.success = False
                        if strict:
                            darwin.warning("Strict mode enabled, halting pipeline after failure")
                            break

                # Capture this stage's logs (failed or not)
                result.logs[stage.value] = self._format_logs(
                    darwin.get_logs(session=session_name, levels=levels)
                )

            # 4. Finalize
            result.summary = self._build_summary(result)
            if result.success:
                darwin.success(f"Pipeline completed: {result.summary}")
            else:
                darwin.error(f"Pipeline completed with failures: {result.summary}")

            # Capture orchestrator-level logs
            result.logs["orchestrator"] = self._format_logs(
                darwin.get_logs(session="PIPELINE", levels=levels)
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
            darwin.warning("Purge called with no filenames, nothing to do")
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
                        
                    darwin.success(f"Removed '{filename}' from stage '{stage.value}'")
                except Exception as e:
                    darwin.error(f"Failed to remove '{filename}' from stage '{stage.value}': {e}")

        darwin.success("Purge completed successfully")

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

    def _run_ingestion(self, params: StageParameters) -> None:
        if not params.parsing:
            darwin.error("ParsingArgs  required")
            return
        handle_ingestion(params.parsing, self.store)

    def _run_indexing(self, params: StageParameters) -> None:
        
        if not params.indexing:
            darwin.error("IndexingArgs required")
            return
        handle_indexing(params.indexing, self.store)

