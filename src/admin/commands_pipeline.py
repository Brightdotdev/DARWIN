"""
Pipeline execution and maintenance command handlers for darwin-admin.

Commands: run, ingest, index, purge
"""

from admin.lib import (
    ns_to,
    PipelineRunArgs,
    PipelineIngestArgs,
    PipelineIndexArgs,
    PipelinePurgeArgs,
)
from admin.renderer import _r


def _run_pipeline_stages(
    *,
    target: str | None,
    files: str | None,
    stages: list[str],
    ocr: bool,
    recursive: bool,
    precise: bool,
    threshold: float | None,
    idf_smoothing: float | None,
    only_kw: bool,
    only_semantic: bool,
    rerun: bool,
    store_ref: str | None,
    index_mode: str = "auto",
    _get_pipeline,
) -> bool:
    """Shared pipeline execution. Returns True on success."""
    pipe = _get_pipeline(store_ref)
    from core.Orchestrators.lib import PipelineParameters as PipelineInput

    kwargs = {"stages": stages}
    if target is not None:
        kwargs["target"] = target
    if store_ref is not None:
        kwargs["store"] = store_ref
    if files is not None:
        kwargs["files"] = files
    if ocr:
        kwargs["ocr"] = True
    if recursive:
        kwargs["recursive"] = True
    if precise:
        kwargs["precise"] = True
    if rerun:
        kwargs["rerun"] = True
    if threshold is not None:
        kwargs["threshold"] = threshold
    if idf_smoothing is not None:
        kwargs["idf_smoothing"] = idf_smoothing
    if only_kw:
        kwargs["only_kw"] = True
    if only_semantic:
        kwargs["only_semantic"] = True
    if index_mode and index_mode != "auto":
        kwargs["index_mode"] = index_mode
    params = PipelineInput(**kwargs)
    _r.info(f"Running pipeline stages: {', '.join(stages)}")
    result = pipe.build_pipeline(params)
    if result.success:
        _r.ok(f"{', '.join(s.value for s in result.stages_completed)} complete")
    if result.stages_failed:
        for stage, err in result.stages_failed.items():
            _r.err(f"{stage}: {err}")
    _r._out()
    _r._out(f"  {result.summary}")
    return result.success


def cmd_pipeline_run(args: PipelineRunArgs, _get_pipeline) -> None:
    _run_pipeline_stages(
        target=args.target,
        files=args.files,
        stages=["ingestion", "indexing"],
        ocr=args.ocr,
        recursive=args.recursive,
        precise=args.precise,
        threshold=args.threshold,
        idf_smoothing=args.idf_smoothing,
        only_kw=args.only_kw,
        only_semantic=args.only_semantic,
        rerun=args.rerun,
        store_ref=args.store,
        index_mode=args.index_mode,
        _get_pipeline=_get_pipeline,
    )


def cmd_pipeline_ingest(args: PipelineIngestArgs, _get_pipeline) -> None:
    _run_pipeline_stages(
        target=args.target,
        files=args.files,
        stages=["ingestion"],
        ocr=args.ocr,
        recursive=args.recursive,
        precise=args.precise,
        threshold=None,
        idf_smoothing=None,
        only_kw=False,
        only_semantic=False,
        rerun=args.rerun,
        store_ref=args.store,
        _get_pipeline=_get_pipeline,
    )


def cmd_pipeline_index(args: PipelineIndexArgs, _get_pipeline) -> None:
    _run_pipeline_stages(
        target="",
        files=args.files,
        stages=["indexing"],
        ocr=False,
        recursive=False,
        precise=False,
        threshold=args.threshold,
        idf_smoothing=args.idf_smoothing,
        only_kw=args.only_kw,
        only_semantic=args.only_semantic,
        rerun=args.rerun,
        store_ref=args.store,
        _get_pipeline=_get_pipeline,
    )


def cmd_pipeline_purge(args: PipelinePurgeArgs, _get_pipeline) -> None:
    pipe = _get_pipeline(args.store)
    pipe.purge(args.stages, args.filenames)
    _r.ok(f"Purged {', '.join(args.stages)} for {len(args.filenames)} file(s)")


def add_args(sub, _get_pipeline) -> None:
    """Register pipeline subcommands."""
    p = sub.add_parser("pipeline", help="Pipeline execution and maintenance")
    sp = p.add_subparsers(dest="pipeline_action", required=True)

    r = sp.add_parser("run", help="Full ingest + index pipeline")
    r.add_argument("target", nargs="?", default=None, help="Directory or URL to process")
    r.add_argument("--store", help="Store ID, name, or path (default: auto)")
    r.add_argument("--files", "-f", nargs="+", help="Specific file(s) to process")
    r.add_argument("--ocr", action="store_true", help="Enable OCR for scanned docs")
    r.add_argument("--recursive", action="store_true", help="Recursive directory scan")
    r.add_argument("--precise", action="store_true", help="Precision parsing mode")
    r.add_argument("--threshold", type=float, default=None, help="Indexing threshold")
    r.add_argument("--idf-smoothing", type=float, default=None, help="BM25 IDF smoothing")
    r.add_argument("--only-kw", action="store_true", help="Keyword-only indexing")
    r.add_argument("--only-semantic", action="store_true", help="Semantic-only indexing")
    r.add_argument("--rerun", action="store_true", help="Re-process even if done")
    r.add_argument(
        "--index-mode", choices=["auto", "text", "structured"], default="auto",
        help="Index mode: auto (detect), text (paragraph chunks), structured (record-level)",
    )
    r.set_defaults(func=lambda a: cmd_pipeline_run(ns_to(PipelineRunArgs, a), _get_pipeline))

    i = sp.add_parser("ingest", help="Ingestion only (parse + chunk)")
    i.add_argument("target", nargs="?", default=None, help="Directory or URL to process")
    i.add_argument("--store", help="Store ID, name, or path (default: auto)")
    i.add_argument("--files", "-f", nargs="+", help="Specific file(s) to process")
    i.add_argument("--ocr", action="store_true")
    i.add_argument("--recursive", action="store_true")
    i.add_argument("--precise", action="store_true")
    i.add_argument("--rerun", action="store_true")
    i.set_defaults(func=lambda a: cmd_pipeline_ingest(ns_to(PipelineIngestArgs, a), _get_pipeline))

    ix = sp.add_parser("index", help="Indexing only (BM25 + embeddings)")
    ix.add_argument("--files", "-f", nargs="+", help="Specific file(s) to index")
    ix.add_argument("--store", help="Store ID, name, or path (default: auto)")
    ix.add_argument("--threshold", type=float, default=None)
    ix.add_argument("--idf-smoothing", type=float, default=None)
    ix.add_argument("--only-kw", action="store_true")
    ix.add_argument("--only-semantic", action="store_true")
    ix.add_argument("--rerun", action="store_true")
    ix.set_defaults(func=lambda a: cmd_pipeline_index(ns_to(PipelineIndexArgs, a), _get_pipeline))

    pu = sp.add_parser("purge", help="Purge pipeline artifacts for files")
    pu.add_argument("filenames", nargs="+", help="Files to purge")
    pu.add_argument("--store", help="Store ID, name, or path (default: auto)")
    pu.add_argument(
        "--stages", nargs="+", default=["ingestion", "indexing"],
        choices=["ingestion", "indexing"], help="Stages to purge",
    )
    pu.set_defaults(func=lambda a: cmd_pipeline_purge(ns_to(PipelinePurgeArgs, a), _get_pipeline))
