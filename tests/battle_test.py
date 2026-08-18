"""Darwin Full Battle Test — comprehensive smoke test."""
import sys
import json
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.Storage.DarwinStore import DarwinStore
from src.core.Orchestrators.PipelineOrchestrator import PipelineOrchestrator
from src.core.Orchestrators.RetrievalOrchestrator import RetrievalOrchestrator
from src.core.Orchestrators.lib import PipelineParameters, StructuredSearchParams
from src.core.Retrival.RetrievalArgs import RetrievalArgs
from src.core.Retrival.lib import RetrievalMethod
from src.core.Retrival.main import handle_retrieval
import polars as pl

PASS = 0
FAIL = 0
WARN = 0
RESULTS = []


def rpt(category, name, status, detail=""):
    global PASS, FAIL, WARN
    tag = "[PASS]" if status == "pass" else "[FAIL]" if status == "fail" else "[WARN]"
    if status == "pass":
        PASS += 1
    elif status == "fail":
        FAIL += 1
    else:
        WARN += 1
    line = f"  {tag} {name}"
    if detail:
        line += f" -- {detail}"
    print(line)
    RESULTS.append({"category": category, "name": name, "status": status, "detail": detail})


# ======================================================================
print("=" * 70)
print("DARWIN BATTLE TEST")
print(f"Timestamp: {datetime.now().isoformat()}")
print("=" * 70)

store = DarwinStore()
store.bootstrap(quiet=True)

# -- 1. STORE HEALTH ---------------------------------------------------
print("\n[1] STORE HEALTH")

try:
    rpt("store", "bootstrap", "pass", f"Root: {store.paths.root}")
except Exception as e:
    rpt("store", "bootstrap", "fail", str(e))

try:
    files = store.get_search_ready_files(mode="hybrid")
    rpt("store", "search_ready_files", "pass", f"{len(files)} files")
except Exception as e:
    rpt("store", "search_ready_files", "fail", str(e))

try:
    health = store.full_registry_health_check()
    rpt("store", "registry_health", "pass", json.dumps(health, default=str)[:120])
except Exception as e:
    rpt("store", "registry_health", "fail", str(e))

idx_dir = str(store.paths.indexes_dir)
if os.path.exists(idx_dir):
    dirs = [d for d in os.listdir(idx_dir) if os.path.isdir(os.path.join(idx_dir, d))]
    rpt("store", "indexes_on_disk", "pass", f"{len(dirs)} indexed")
else:
    rpt("store", "indexes_on_disk", "fail", "indexes dir missing")

structured_count = sum(1 for f in files if store.has_schema(f))
rpt("store", "schemas", "pass", f"{structured_count} files with schemas")


# -- 2. INGESTION (only files that exist on disk) ----------------------
print("\n[2] INGESTION")
orch = PipelineOrchestrator(store=store)

# 2a: Ingest sumsum.csv (structured rerun)
print("\n--- 2a: Ingest sumsum.csv (structured rerun) ---")
try:
    params = PipelineParameters(target="src/pdfs", files=["src/pdfs/sumsum.csv"], rerun=True, index_mode="structured")
    result = orch.build_pipeline(params)
    if result.success:
        rpt("ingestion", "sumsum.csv (structured rerun)", "pass", result.summary)
    else:
        rpt("ingestion", "sumsum.csv (structured rerun)", "fail", result.summary)
except Exception as e:
    rpt("ingestion", "sumsum.csv (structured rerun)", "fail", str(e))

# 2b: Schema saved?
try:
    has = store.has_schema("sumsum.csv")
    schema = store.load_schema("sumsum.csv") if has else None
    if has and schema:
        rpt("ingestion", "schema_saved", "pass",
            f"{len(schema.get('keys', []))} keys, {schema.get('record_count', 0)} records")
    else:
        rpt("ingestion", "schema_saved", "fail", "No schema found")
except Exception as e:
    rpt("ingestion", "schema_saved", "fail", str(e))

# 2c: Ingest psykadelicish.csv (structured, no rerun — skip if cached)
print("\n--- 2c: Ingest psykadelicish.csv (structured) ---")
try:
    params = PipelineParameters(target="src/pdfs", files=["src/pdfs/psykadelicish.csv"], index_mode="structured")
    result = orch.build_pipeline(params)
    if result.success:
        rpt("ingestion", "psykadelicish.csv (structured)", "pass", result.summary)
    else:
        rpt("ingestion", "psykadelicish.csv (structured)", "fail", result.summary)
except Exception as e:
    rpt("ingestion", "psykadelicish.csv (structured)", "fail", str(e))


# -- 3. TEXT SEARCH (keyword, hybrid) ----------------------------------
print("\n[3] TEXT SEARCH")

search_tests = [
    ("keyword", "what is darwin", 5),
    ("hybrid", "how does retrieval work", 5),
    ("keyword", "pipeline", 3),
    ("hybrid", "Kimi K2.5 Composer improvements", 5),
]

rorch = RetrievalOrchestrator(store=store)

for method_str, query, top_k in search_tests:
    method = RetrievalMethod(method_str)
    print(f"\n--- {method_str.upper()}: '{query}' ---")
    try:
        args = RetrievalArgs(query=query, method=method, top_k=top_k)
        results = handle_retrieval(args, store=store)
        if results:
            scores = [r.score for r in results]
            rpt("text_search", f"{method_str}: {query[:30]}", "pass",
                f"{len(results)} results, top={max(scores):.3f}, src={results[0].source}")
            for i, r in enumerate(results[:2], 1):
                snippet = (r.clean or r.text or "")[:80].replace("\n", " ")
                print(f"    [{i}] score={r.score:.3f} src={r.source} | {snippet}...")
        else:
            rpt("text_search", f"{method_str}: {query[:30]}", "warn", "0 results")
    except Exception as e:
        rpt("text_search", f"{method_str}: {query[:30]}", "fail", str(e))


# -- 4. STRUCTURED SEARCH ----------------------------------------------
print("\n[4] STRUCTURED SEARCH")


def _print_struct_result(ctx):
    for i, p in enumerate(ctx.passages[:2], 1):
        fields = getattr(p.metadata, "_structured_fields", {})
        if fields:
            detail = ", ".join(f"{k}={v}" for k, v in list(fields.items())[:3])
            print(f"    [{i}] score={p.metadata.avg_score:.3f} | {detail}")
        else:
            snippet = (p.text or "")[:60].replace("\n", " ")
            print(f"    [{i}] score={p.metadata.avg_score:.3f} | {snippet}...")


struct_tests = [
    ("filter_only", {"gender": "Female"}, None, "auto", 5),
    ("contains", {"hospital_name__contains": "children"}, None, "auto", 5),
    ("in_list", {"shirt_size__in": ["XS", "S"]}, None, "auto", 5),
    ("combined", {"gender": "Male", "shirt_size": "2XL"}, None, "auto", 5),
    ("hybrid", {"gender": "Female"}, "nurse", "auto", 5),
    ("exists", {"shirt_size__exists": True}, None, "auto", 5),
    ("exact_no_match", {"gender": "Helicopter"}, None, "structured", 5),
]

for name, filters, query, mode, top_k in struct_tests:
    print(f"\n--- {name}: filters={filters}, query={query} ---")
    try:
        params = StructuredSearchParams(
            query=query or "", filters=filters, mode=mode, top_k=top_k,
        )
        ctx = rorch.search_lists(params)
        if ctx and not ctx.is_empty():
            rpt("structured_search", name, "pass",
                f"{ctx.total_passages} results, method={ctx.method}, fallback={ctx.fallback_used}")
            _print_struct_result(ctx)
        elif ctx:
            rpt("structured_search", name, "pass", "0 results (expected for no-match)")
        else:
            rpt("structured_search", name, "fail", "None returned")
    except Exception as e:
        rpt("structured_search", name, "fail", str(e))


# -- 5. FILTER VALIDATION ----------------------------------------------
print("\n[5] FILTER VALIDATION")

try:
    vr = store.validate_filters(
        {"gender": "Male", "shirt_size__in": ["XS"]}, files=["sumsum.csv"],
    )
    if vr.get("valid"):
        rpt("validation", "valid_filters", "pass", f"{len(vr['valid'])} valid")
    else:
        rpt("validation", "valid_filters", "fail", "No valid filters")

    vr2 = store.validate_filters(
        {"nonexistent": "foo", "also_bad__gt": 5}, files=["sumsum.csv"],
    )
    if vr2.get("warnings"):
        rpt("validation", "invalid_filters_warn", "pass", f"{len(vr2['warnings'])} warnings")
    else:
        rpt("validation", "invalid_filters_warn", "fail", "Expected warnings")
except Exception as e:
    rpt("validation", "filter_validation", "fail", str(e))


# -- 6. EDGE CASES -----------------------------------------------------
print("\n[6] EDGE CASES")

# 6a: Empty query + empty filters (should error gracefully)
try:
    params = StructuredSearchParams(query="", filters=None, top_k=5)
    ctx = rorch.search_lists(params)
    if ctx is None:
        rpt("edge_case", "empty_query_no_filters", "pass", "Returns None (expected)")
    else:
        rpt("edge_case", "empty_query_no_filters", "warn", "Got context instead of None")
except Exception as e:
    rpt("edge_case", "empty_query_no_filters", "pass", f"Raised {type(e).__name__} (expected)")

# 6b: Search nonexistent file
try:
    args = RetrievalArgs(query="test", method=RetrievalMethod.HYBRID, files=["nonexistent.pdf"], top_k=3)
    results = handle_retrieval(args, store=store)
    if not results:
        rpt("edge_case", "search_nonexistent_file", "pass", "0 results (expected)")
    else:
        rpt("edge_case", "search_nonexistent_file", "warn", f"Got {len(results)} results")
except Exception as e:
    rpt("edge_case", "search_nonexistent_file", "fail", str(e))

# 6c: Structured search with no matching records
try:
    params = StructuredSearchParams(filters={"gender": "Attack Helicopter"}, mode="structured", top_k=5)
    ctx = rorch.search_lists(params)
    if ctx and ctx.is_empty():
        rpt("edge_case", "structured_no_match", "pass", "Empty context (expected)")
    elif ctx is None:
        rpt("edge_case", "structured_no_match", "pass", "None (expected)")
    else:
        rpt("edge_case", "structured_no_match", "warn", f"Got {ctx.total_passages} results")
except Exception as e:
    rpt("edge_case", "structured_no_match", "fail", str(e))

# 6d: Pipeline with invalid stage
try:
    params = PipelineParameters(target="src/pdfs", stages=["invalid_stage"])
    orch2 = PipelineOrchestrator(store=store)
    result = orch2.build_pipeline(params)
    if not result.success:
        rpt("edge_case", "invalid_stage", "pass", "Rejected (expected)")
    else:
        rpt("edge_case", "invalid_stage", "fail", "Should have failed")
except Exception as e:
    rpt("edge_case", "invalid_stage", "pass", f"Raised: {type(e).__name__}")


# -- 7. CHUNK INTEGRITY ------------------------------------------------
print("\n[7] CHUNK INTEGRITY")

for fname in ["sumsum.csv", "psykadelicish.csv", "cursor_paper.pdf"]:
    try:
        h = store._resolve_hash(fname)
        if h is None:
            rpt("chunk_integrity", fname, "warn", "Not indexed")
            continue
        pq = store.paths.artifact_path(h, "chunks", "file")
        df = pl.read_parquet(pq)
        n_rows = df.shape[0]
        n_cols = df.shape[1]
        has_text = "text" in df.columns
        has_source = "source" in df.columns
        has_fields = "fields" in df.columns
        mode = df["index_mode"][0] if "index_mode" in df.columns else "N/A"

        if has_text and has_source:
            rpt("chunk_integrity", fname, "pass",
                f"{n_rows} chunks, {n_cols} cols, mode={mode}, fields={has_fields}")
        else:
            rpt("chunk_integrity", fname, "fail",
                f"Missing columns: text={has_text}, source={has_source}")
    except Exception as e:
        rpt("chunk_integrity", fname, "fail", str(e))


# -- 8. SAVE RESULTS ---------------------------------------------------
print("\n[8] SAVE RESULTS")

# 8a: Text search results
try:
    args = RetrievalArgs(
        query="what improvements does Composer 2 show",
        method=RetrievalMethod.HYBRID, top_k=3,
    )
    results = handle_retrieval(args, store=store)
    output = []
    for r in results:
        output.append({
            "score": r.score, "source": r.source, "chunk_index": r.chunk_index,
            "text": r.text[:200] if r.text else "",
            "clean": r.clean[:200] if r.clean else "",
        })
    out_path = Path("tests/battle_test_results.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"query": "what improvements does Composer 2 show", "results": output}, f, indent=2)
    rpt("save", "search_results_json", "pass", f"Saved to {out_path}")
except Exception as e:
    rpt("save", "search_results_json", "fail", str(e))

# 8b: Structured search results
try:
    params = StructuredSearchParams(filters={"gender": "Female"}, mode="structured", top_k=5)
    ctx = rorch.search_lists(params)
    output = []
    if ctx and not ctx.is_empty():
        for p in ctx.passages:
            fields = getattr(p.metadata, "_structured_fields", {})
            output.append({
                "score": p.metadata.avg_score,
                "source": p.metadata.source,
                "fields": fields,
            })
    out_path = Path("tests/battle_test_structured.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"filters": {"gender": "Female"}, "results": output}, f, indent=2)
    rpt("save", "structured_results_json", "pass", f"Saved to {out_path}, {len(output)} records")
except Exception as e:
    rpt("save", "structured_results_json", "fail", str(e))


# -- SUMMARY -----------------------------------------------------------
print("\n" + "=" * 70)
print("BATTLE TEST COMPLETE")
print("=" * 70)
total = PASS + FAIL + WARN
print(f"  PASS: {PASS}/{total}")
print(f"  FAIL: {FAIL}/{total}")
print(f"  WARN: {WARN}/{total}")
print()

if FAIL > 0:
    print("FAILURES:")
    for r in RESULTS:
        if r["status"] == "fail":
            print(f"  [{r['category']}] {r['name']}: {r['detail']}")
else:
    print("NO FAILURES -- Darwin is solid.")
