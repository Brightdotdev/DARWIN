"""
Pipeline execution and maintenance command handlers for darwin-admin.

Commands: run, ingest, index, purge
"""

import sys

from src.admin.renderer import _r


def _run_pipeline_stages(
    target,
    files,
    stages,
    ocr,
    recursive,
    precise,
    threshold,
    idf_smoothing,
    only_kw,
    only_semantic,
    rerun,
    store_ref,
    _get_pipeline,
):
    pipe = _get_pipeline(store_ref)
    from src.core.Orchestrators.lib import PipelineParameters as PipelineInput

    params = PipelineInput(
        target=target or "",
        store=store_ref,
        stages=stages,
        files=[files] if files else None,
        ocr=ocr,
        recursive=recursive,
        precise=precise,
        rerun=rerun,
        threshold=threshold if threshold is not None else 0.5,
        idf_smoothing=idf_smoothing if idf_smoothing is not None else 1.5,
        only_kw=only_kw,
        only_semantic=only_semantic,
    )
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


def cmd_pipeline_run(args, _get_pipeline):
    success = _run_pipeline_stages(
        args.target, args.files, ["ingestion", "indexing"],
        args.ocr, args.recursive, args.precise,
        args.threshold, args.idf_smoothing,
        args.only_kw, args.only_semantic, args.rerun,
        store_ref=getattr(args, "store", None),
        _get_pipeline=_get_pipeline,
    )
    if not success:
        sys.exit(1)


def cmd_pipeline_ingest(args, _get_pipeline):
    success = _run_pipeline_stages(
        args.target, args.files, ["ingestion"],
        args.ocr, args.recursive, args.precise,
        None, None, False, False, args.rerun,
        store_ref=getattr(args, "store", None),
        _get_pipeline=_get_pipeline,
    )
    if not success:
        sys.exit(1)


def cmd_pipeline_index(args, _get_pipeline):
    success = _run_pipeline_stages(
        None, args.files, ["indexing"],
        False, False, False,
        args.threshold, args.idf_smoothing,
        args.only_kw, args.only_semantic, args.rerun,
        store_ref=getattr(args, "store", None),
        _get_pipeline=_get_pipeline,
    )
    if not success:
        sys.exit(1)


def cmd_pipeline_purge(args, _get_pipeline):
    try:
        pipe = _get_pipeline(getattr(args, "store", None))
        pipe.purge(args.stages, args.filenames)
        _r.ok(f"Purged {', '.join(args.stages)} for {len(args.filenames)} file(s)")
    except Exception as e:
        _r.err(f"{type(e).__name__}: {e}")
        sys.exit(1)


def add_args(sub, _get_pipeline):
    """Register pipeline subcommands."""
    p = sub.add_parser("pipeline", help="Pipeline execution and maintenance")
    sp = p.add_subparsers(dest="pipeline_action", required=True)

    r = sp.add_parser("run", help="Full ingest + index pipeline")
    r.add_argument("target", help="Path or static URL to ingest")
    r.add_argument("--store", help="Store ID, name, or path (default: auto)")
    r.add_argument("--files", "-f", help="Specific files (comma-separated)")
    r.add_argument("--ocr", action="store_true", help="Enable OCR for scanned docs")
    r.add_argument("--recursive", action="store_true", help="Recursive directory scan")
    r.add_argument("--precise", action="store_true", help="Precision parsing mode")
    r.add_argument(
        "--threshold", type=float, default=0.5, help="Indexing threshold (default: 0.5)"
    )
    r.add_argument(
        "--idf-smoothing", type=float, default=1.5,
        help="BM25 IDF smoothing (default: 1.5)",
    )
    r.add_argument("--only-kw", action="store_true", help="Keyword-only indexing")
    r.add_argument("--only-semantic", action="store_true", help="Semantic-only indexing")
    r.add_argument("--rerun", action="store_true", help="Re-process even if done")
    r.set_defaults(func=lambda a: cmd_pipeline_run(a, _get_pipeline))

    i = sp.add_parser("ingest", help="Ingestion only (parse + chunk)")
    i.add_argument("target", help="Path or static URL to ingest")
    i.add_argument("--store", help="Store ID, name, or path (default: auto)")
    i.add_argument("--files", "-f", help="Specific files")
    i.add_argument("--ocr", action="store_true")
    i.add_argument("--recursive", action="store_true")
    i.add_argument("--precise", action="store_true")
    i.add_argument("--rerun", action="store_true")
    i.set_defaults(func=lambda a: cmd_pipeline_ingest(a, _get_pipeline))

    ix = sp.add_parser("index", help="Indexing only (BM25 + embeddings)")
    ix.add_argument("--files", "-f", help="Specific files to index")
    ix.add_argument("--store", help="Store ID, name, or path (default: auto)")
    ix.add_argument("--threshold", type=float, default=0.5)
    ix.add_argument("--idf-smoothing", type=float, default=1.5)
    ix.add_argument("--only-kw", action="store_true")
    ix.add_argument("--only-semantic", action="store_true")
    ix.add_argument("--rerun", action="store_true")
    ix.set_defaults(func=lambda a: cmd_pipeline_index(a, _get_pipeline))

    pu = sp.add_parser("purge", help="Purge pipeline artifacts for files")
    pu.add_argument("filenames", nargs="+", help="Files to purge")
    pu.add_argument("--store", help="Store ID, name, or path (default: auto)")
    pu.add_argument(
        "--stages", nargs="+", default=["ingestion", "indexing"],
        choices=["ingestion", "indexing"], help="Stages to purge",
    )
    pu.set_defaults(func=lambda a: cmd_pipeline_purge(a, _get_pipeline))
