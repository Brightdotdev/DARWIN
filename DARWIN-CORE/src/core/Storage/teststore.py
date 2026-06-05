# test_darwin_store.py
# ──────────────────────────────────────────────────────────────────────────────
# Full battle test for DarwinStore — runs all public API paths against /test.
# No pytest, no asserts, no frameworks. Pure Python, prints results.
# Run: python test_darwin_store.py

from __future__ import annotations

import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np

# ── Bootstrap project root if needed ────────────────────────────────────
# Adjust this if your test runner is not at repo root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Imports ──────────────────────────────────────────────────────────────

from src.core.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin, Verbosity


# ═══════════════════════════════════════════════════════════════════════════
# Test harness
# ═══════════════════════════════════════════════════════════════════════════

class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: list[str] = []

    def section(self, name: str) -> None:
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")

    def ok(self, msg: str) -> None:
        self.passed += 1
        print(f"  [OK]   {msg}")

    def fail(self, msg: str, exc: Exception | None = None) -> None:
        self.failed += 1
        detail = f" | {exc}" if exc else ""
        print(f"  [FAIL] {msg}{detail}")
        self.errors.append(f"{msg}{detail}")

    def summary(self) -> None:
        print(f"\n{'=' * 60}")
        print(f"  RESULTS: {self.passed} passed, {self.failed} failed")
        if self.errors:
            print(f"  ERRORS:")
            for e in self.errors:
                print(f"    - {e}")
        print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════════════════
# Fixture helpers
# ═══════════════════════════════════════════════════════════════════════════

def make_test_pdf(test_dir: Path, name: str = "sample.pdf") -> Path:
    """Create a minimal PDF-like file for registration."""
    pdf_path = test_dir / name
    # Minimal valid PDF header + some content
    content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"
    content += b"%% Dummy content to vary hashes: " + str(uuid.uuid4()).encode()
    pdf_path.write_bytes(content)
    return pdf_path


def make_test_txt(test_dir: Path, name: str = "notes.txt", text: str = "") -> Path:
    """Create a text file for registration."""
    txt_path = test_dir / name
    txt_path.write_text(text or f"Test document content.\nUUID: {uuid.uuid4()}\n")
    return txt_path


# ═══════════════════════════════════════════════════════════════════════════
# Main test suite
# ═══════════════════════════════════════════════════════════════════════════

def run_tests(test_dir: Path) -> None:
    t = TestRunner()
    darwin_dir = test_dir / ".darwin"
    source_dir = test_dir / "sources"

    # ── Setup ─────────────────────────────────────────────────────────────
    print(f"\nTest directory: {test_dir}")
    print(f"Darwin store:   {darwin_dir}")
    source_dir.mkdir(parents=True, exist_ok=True)

    # ── Init DarwinStore ──────────────────────────────────────────────────
    t.section("1. DarwinStore Initialization")
    try:
        store = DarwinStore(darwin_dir=darwin_dir)
        t.ok("DarwinStore initialized")
    except Exception as e:
        t.fail("DarwinStore initialization", e)
        t.summary()
        return

    # Verify internal state
    try:
        assert store._darwin_dir == darwin_dir.resolve()
        assert store.paths.root == darwin_dir.resolve()
        assert store.registry is not None
        t.ok("Internal state consistent")
    except Exception as e:
        t.fail("Internal state check", e)

    # ── File Registration ─────────────────────────────────────────────────
    t.section("2. File Registration")

    pdf1 = make_test_pdf(source_dir, "report_2024.pdf")
    pdf2 = make_test_pdf(source_dir, "report_2025.pdf")
    txt1 = make_test_txt(source_dir, "readme.txt", "Hello Darwin.\n" * 100)

    try:
        rec1 = store.register(pdf1, page_count=42, file_size=pdf1.stat().st_size)
        t.ok(f"Registered {pdf1.name} → hash={rec1.file_hash[:16]}…")
    except Exception as e:
        t.fail(f"Register {pdf1.name}", e)
        rec1 = None

    try:
        rec2 = store.register(pdf2, page_count=88)
        t.ok(f"Registered {pdf2.name} → hash={rec2.file_hash[:16]}…")
    except Exception as e:
        t.fail(f"Register {pdf2.name}", e)
        rec2 = None

    try:
        rec3 = store.register(txt1)
        t.ok(f"Registered {txt1.name} → hash={rec3.file_hash[:16]}…")
    except Exception as e:
        t.fail(f"Register {txt1.name}", e)
        rec3 = None

    # get_or_register (idempotent)
    try:
        rec1_again = store.registry.get_or_register(pdf1)
        assert rec1_again.file_hash == rec1.file_hash
        t.ok("get_or_register returns same record for existing file")
    except Exception as e:
        t.fail("get_or_register idempotency", e)

    # ── Registry Queries ──────────────────────────────────────────────────
    t.section("3. Registry Queries")

    try:
        all_records = store.registry.list_all()
        assert len(all_records) >= 3
        t.ok(f"list_all() → {len(all_records)} records")
    except Exception as e:
        t.fail("list_all()", e)

    try:
        needing = store.registry.list_needing_work()
        t.ok(f"list_needing_work() → {len(needing)} files need pipeline")
    except Exception as e:
        t.fail("list_needing_work()", e)

    try:
        stats = store.registry.stats()
        t.ok(f"registry.stats() → {stats['total_files']} files, {stats['total_pages']} pages")
    except Exception as e:
        t.fail("registry.stats()", e)

    # ── File Status ─────────────────────────────────────────────────────
    t.section("4. File Status API")

    try:
        status = store.file_status(pdf1)
        assert status["original_name"] == pdf1.name
        assert status["page_count"] == 42
        assert status["is_parsed"] is False
        t.ok(f"file_status({pdf1.name}) → {status['progress_percent']}% complete")
    except Exception as e:
        t.fail("file_status()", e)

    try:
        full_status = store.status()
        assert "darwin_dir" in full_status
        assert "stats" in full_status
        assert "files" in full_status
        t.ok(f"status() → {len(full_status['files'])} files in report")
    except Exception as e:
        t.fail("status()", e)

    # ── Meta I/O ──────────────────────────────────────────────────────────
    t.section("5. Meta.json Read/Write")

    try:
        store._update_meta(pdf1.name, "markdown", extras={"page_count": 42, "parser": "pdfplumber"})
        meta = store.load_meta(pdf1.name)
        assert meta["updates"]["markdown"] is not None
        assert meta["page_count"] == 42
        t.ok("_update_meta + load_meta roundtrip")
    except Exception as e:
        t.fail("Meta I/O", e)

    try:
        store._batch_meta_update(txt1.name, {
            "markdown": {"chunk_count": 10},
            "chunks": {"strategy": "semantic"},
        })
        meta = store.load_meta(txt1.name)
        assert "markdown" in meta["updates"]
        assert "chunks" in meta["updates"]
        t.ok("_batch_meta_update roundtrip")
    except Exception as e:
        t.fail("Batch meta update", e)

    # ── Artifact Health ───────────────────────────────────────────────────
    t.section("6. Artifact Health Checks")

    try:
        missing = store.missing_stages(pdf1.name)
        t.ok(f"missing_stages({pdf1.name}) → {missing}")
    except Exception as e:
        t.fail("missing_stages()", e)

    try:
        progress = store.progress_percent(pdf1.name)
        assert 0 <= progress <= 100
        t.ok(f"progress_percent({pdf1.name}) → {progress}%")
    except Exception as e:
        t.fail("progress_percent()", e)

    try:
        summary = store.status_summary(pdf1.name)
        t.ok(f"status_summary({pdf1.name}) → {summary}")
    except Exception as e:
        t.fail("status_summary()", e)

    try:
        integrity = store.validate_artifact_integrity(pdf1.name)
        assert "is_healthy" in integrity
        t.ok(f"validate_artifact_integrity → healthy={integrity['is_healthy']}")
    except Exception as e:
        t.fail("validate_artifact_integrity()", e)

    try:
        full_health = store.full_registry_health_check()
        t.ok(f"full_registry_health_check → {full_health['summary']['total_files']} files checked")
    except Exception as e:
        t.fail("full_registry_health_check()", e)

    # ── Search Readiness ──────────────────────────────────────────────────
    t.section("7. Search Readiness")

    try:
        searchables = store.get_searchables(pdf1.name)
        assert "kw_searchable" in searchables
        assert "semantic_searchable" in searchables
        assert "hybrid_searchable" in searchables
        t.ok(f"get_searchables → kw={searchables['kw_searchable']}, semantic={searchables['semantic_searchable']}")
    except Exception as e:
        t.fail("get_searchables()", e)

    # ── Embedding Model Queries ───────────────────────────────────────────
    t.section("8. Embedding Model Queries")



    try:
        health = store.get_embedding_health("intfloat/e5-small-v2")
        t.ok(f"get_embedding_health → up_to_date={len(health['up_to_date'])}, needs_reembed={len(health['needs_reembed'])}")
    except Exception as e:
        t.fail("get_embedding_health()", e)

    # ── Embeddings Save/Load ──────────────────────────────────────────────
    t.section("9. Embeddings Save/Load")

    try:
        fake_vectors = np.random.randn(10, 384).astype(np.float32)
        fake_vectors = fake_vectors / np.linalg.norm(fake_vectors, axis=1, keepdims=True)
        store.save_embeddings(txt1.name, fake_vectors, indices=list(range(10)))
        store.save_vector(txt1.name, fake_vectors)

        t.ok(f"save_embeddings → 10 vectors, 384d")
    except Exception as e:
        t.fail("save_embeddings()", e)

    try:
        loaded = store.load_embeddings(txt1.name, mmap=True)
        assert loaded is not None
        assert loaded.shape == (10, 384)
        t.ok(f"load_embeddings (mmap) → shape={loaded.shape}")
    except Exception as e:
        t.fail("load_embeddings()", e)

    try:
        index_map = store.load_embedding_index_map(txt1.name)
        assert index_map == list(range(10))
        t.ok(f"load_embedding_index_map → {index_map}")
    except Exception as e:
        t.fail("load_embedding_index_map()", e)

    # ── Vector Index Save/Load ────────────────────────────────────────────
    t.section("10. Vector Index Save/Load")

    try:
        store.save_vector(txt1.name, fake_vectors)
        t.ok("save_vector → usearch index written")
    except Exception as e:
        t.fail("save_vector()", e)

    try:
        index = store.load_vector(txt1.name)
        assert index is not None
        t.ok(f"load_vector → restored index")
    except Exception as e:
        t.fail("load_vector()", e)

    try:
        query = np.random.randn(384).astype(np.float32)
        query = query / np.linalg.norm(query)
        results = store.search_vector(txt1.name, query, k=5)
        assert len(results) == 5
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        t.ok(f"search_vector → {len(results)} matches")
    except Exception as e:
        t.fail("search_vector()", e)

    # ── Search Operations ───────────────────────────────────────────────────
    t.section("11. Search Operations")

    try:
        # We need records/chunks for search to work — create minimal parquet
        import pyarrow as pa
        import pyarrow.parquet as pq

        chunks_data = {
            "index": list(range(10)),
            "text": [f"Chunk {i} content here" for i in range(10)],
            "source": ["readme.txt"] * 10,
        }
        table = pa.table(chunks_data)
        chunks_path = store.paths.chunks_path(store._hash_for(txt1.name))
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, chunks_path)

        store.registry.update_pipeline(
            file_hash=store._hash_for(txt1.name),
            stage="chunks",
        )
        t.ok("Injected chunks.parquet for search testing")
    except Exception as e:
        t.fail("Chunks injection", e)

    try:
        query_vec = np.random.randn(384).astype(np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)
        rows = store.retrieve_by_vector(txt1.name, query_vec, k=3, columns=["index", "text"])
        t.ok(f"retrieve_by_vector → {len(rows)} rows")
    except Exception as e:
        t.fail("retrieve_by_vector()", e)

    try:
        rows = store.retrieve_by_numpy(txt1.name, query_vec, k=3, columns=["index", "text"])
        t.ok(f"retrieve_by_numpy → {len(rows)} rows")
    except Exception as e:
        t.fail("retrieve_by_numpy()", e)

    try:
        rows = store.search_single(query_vec, txt1.name, k=3, min_score=0.0)
        t.ok(f"search_single → {len(rows)} results")
    except Exception as e:
        t.fail("search_single()", e)

    # Multi-file search
    try:
        rows = store.search_multi(query_vec, [txt1.name, pdf1.name], top_k=5)
        t.ok(f"search_multi → {len(rows)} results across files")
    except Exception as e:
        t.fail("search_multi()", e)

    # ── BM25 / Keyword Artifacts ──────────────────────────────────────────
    t.section("12. BM25 & Stopwords")

    try:
        # Fake BM25 files
        fh = store._hash_for(txt1.name)
        bm25_path = store.paths.bm25_path(fh)
        bm25_meta = store.paths.bm25_meta_path(fh)
        bm25_tokens = store.paths.bm25_tokens_path(fh)
        stopwords_path = store.paths.stopwords_path(fh)

        bm25_path.parent.mkdir(parents=True, exist_ok=True)
        bm25_path.write_bytes(b"fake_bm25")
        bm25_meta.write_text('{"k1": 1.5, "b": 0.75}')
        bm25_tokens.write_bytes(b"fake_tokens")
        stopwords_path.write_text('["the", "and", "of"]')

        store.registry.update_pipeline(file_hash=fh, stage="bm25")
        store.registry.update_pipeline(file_hash=fh, stage="stopwords")

        assert store.is_kw_searchable(txt1.name)
        t.ok("BM25 + stopwords artifacts → is_kw_searchable=True")
    except Exception as e:
        t.fail("BM25 setup", e)

    # ── Graph / Wiki Artifacts ────────────────────────────────────────────
    t.section("13. Graph & Wiki Artifacts")

    try:
        fh = store._hash_for(txt1.name)
        graph_nodes = store.paths.graph_nodes_path(fh)
        graph_edges = store.paths.graph_edges_path(fh)
        graph_html = store.paths.graph_html_path(fh)
        wiki_path = store.paths.wiki_path(fh)

        graph_nodes.parent.mkdir(parents=True, exist_ok=True)
        graph_nodes.write_bytes(b"fake_nodes")
        graph_edges.write_bytes(b"fake_edges")
        graph_html.write_text("<html></html>")
        wiki_path.parent.mkdir(parents=True, exist_ok=True)
        wiki_path.write_bytes(b"fake_wiki")

        store.registry.update_pipeline(file_hash=fh, stage="graph")
        store.registry.update_pipeline(file_hash=fh, stage="graph_vis")
        store.registry.update_pipeline(file_hash=fh, stage="wiki")

        assert store.has_graph(txt1.name)
        assert store.has_wiki(txt1.name)
        t.ok("Graph + wiki artifacts → has_graph=True, has_wiki=True")
    except Exception as e:
        t.fail("Graph/wiki setup", e)

    # ── Stats Aggregation ─────────────────────────────────────────────────
    t.section("14. Stats Aggregation")

    try:
        total = store.total_chunks()
        t.ok(f"total_chunks() → {total}")
    except Exception as e:
        t.fail("total_chunks()", e)

    try:
        stats = store.stats()
        t.ok(f"stats() → {stats['total_files']} files, {stats['total_chunks']} chunks")
    except Exception as e:
        t.fail("stats()", e)

    try:
        graph_stats = store.get_graph_stats(txt1.name)
        t.ok(f"get_graph_stats → nodes={graph_stats['node_count']}")
    except Exception as e:
        t.fail("get_graph_stats()", e)

    try:
        wiki_stats = store.get_wiki_stats(txt1.name)
        t.ok(f"get_wiki_stats → {wiki_stats}")
    except Exception as e:
        t.fail("get_wiki_stats()", e)

    # ── Model Lists ────────────────────────────────────────────────────────
    t.section("15. Model-based Lists")

    try:
        by_model = store.list_by_model("intfloat/e5-small-v2")
        t.ok(f"list_by_model → {len(by_model)} files")
    except Exception as e:
        t.fail("list_by_model()", e)

    try:
        kw = store.list_kw_searchable()
        t.ok(f"list_kw_searchable → {len(kw)} files")
    except Exception as e:
        t.fail("list_kw_searchable()", e)

    try:
        emb = store.list_embedding_searchable()
        t.ok(f"list_embedding_searchable → {len(emb)} files")
    except Exception as e:
        t.fail("list_embedding_searchable()", e)

    try:
        hybrid = store.list_hybrid_searchable()
        t.ok(f"list_hybrid_searchable → {len(hybrid)} files")
    except Exception as e:
        t.fail("list_hybrid_searchable()", e)

    # ── Cache Invalidation ────────────────────────────────────────────────
    t.section("16. Cache Invalidation")

    try:
        store.invalidate(txt1.name)
        t.ok("invalidate() executed without error")
    except Exception as e:
        t.fail("invalidate()", e)

    # ── Registry Repair ─────────────────────────────────────────────────────
    t.section("17. Registry Repair")

    try:
        # Intentionally desync: remove a file but keep registry entry
        # (Simulated by clearing a stage timestamp manually)
        data = store.registry.load_data()
        fh = store._hash_for(pdf1.name)
        if "updates" in data["files"][fh]:
            del data["files"][fh]["updates"]["markdown"]
        store.registry.save_data(data)

        repair = store.repair_registry(pdf1.name, dry_run=True)
        t.ok(f"repair_registry(dry_run) → {repair['files_checked']} checked")
    except Exception as e:
        t.fail("repair_registry dry_run", e)

    try:
        repair = store.repair_registry(pdf1.name, dry_run=False)
        t.ok(f"repair_registry(live) → {repair['files_repaired']} repaired, {repair['files_removed']} removed")
    except Exception as e:
        t.fail("repair_registry live", e)

    try:
        repair = store.repair_registry(dry_run=True)
        t.ok(f"repair_registry(all, dry_run) → {repair['files_checked']} checked")
    except Exception as e:
        t.fail("repair_registry all", e)

    # ── Pipeline Updates ──────────────────────────────────────────────────
    t.section("18. Pipeline Stage Updates")

    try:
        rec = store.registry.update_pipeline(pdf1.name, "markdown")
        assert rec.stage_completed("markdown")
        t.ok("update_pipeline(markdown) → stage marked complete")
    except Exception as e:
        t.fail("update_pipeline()", e)

    # ── Disk Treemap ────────────────────────────────────────────────────
    t.section("19. Disk Treemap")

    try:
        tree = store.get_treemap()
        assert "name" in tree
        assert "size" in tree
        t.ok(f"get_treemap() → root={tree['name']}, size={tree['size']}")
    except Exception as e:
        t.fail("get_treemap()", e)

    try:
        sub = store.get_treemap("indexes")
        t.ok(f"get_treemap('indexes') → {sub['name']}")
    except Exception as e:
        t.fail("get_treemap subpath", e)

    try:
        store.print_treemap()
        t.ok("print_treemap() executed")
    except Exception as e:
        t.fail("print_treemap()", e)

    # ── Dashboard ─────────────────────────────────────────────────────────
    t.section("20. Dashboard")

    try:
        store.print_dashboard()
        t.ok("print_dashboard() executed")
    except Exception as e:
        t.fail("print_dashboard()", e)

    # ── Image Directory ───────────────────────────────────────────────────
    t.section("21. Image Directory")

    try:
        img_dir = store.ensure_image_dir(pdf1.name)
        assert img_dir.exists()
        assert img_dir.is_dir()
        t.ok(f"ensure_image_dir → {img_dir}")
    except Exception as e:
        t.fail("ensure_image_dir()", e)

    # ── File Removal ──────────────────────────────────────────────────────
    t.section("22. File Removal")

    try:
        removed = store.remove_file(pdf2.name)
        assert removed is True
        t.ok(f"remove_file({pdf2.name}) → success")
    except Exception as e:
        t.fail("remove_file()", e)

    try:
        rec = store.registry.find_by_name(pdf2.name)
        assert rec is None
        t.ok(f"verify removal: {pdf2.name} not in registry")
    except Exception as e:
        t.fail("verify removal", e)

    # ── Freshness Check ───────────────────────────────────────────────────
    t.section("23. Freshness Check")

    try:
        fresh = store.is_fresh(txt1.name)
        t.ok(f"is_fresh({txt1.name}) → {fresh}")
    except Exception as e:
        t.fail("is_fresh()", e)

    # ── Searchables after mutations ─────────────────────────────────────
    t.section("24. Searchables Post-Mutation")

    try:
        searchables = store.get_searchables(txt1.name)
        t.ok(f"get_searchables({txt1.name}) → hybrid={searchables['hybrid_searchable']}")
    except Exception as e:
        t.fail("get_searchables post-mutation", e)

    # ── Final Summary ───────────────────────────────────────────────────
    t.summary()


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    test_root = Path(__file__).parent / "teststore"
    darwin.configure(
        enable_websocket=False,
        enable_history=True,
        verbosity_default=Verbosity.NORMAL,
        show_banner=True,
    )

    print(f"\n{'#' * 60}")
    print(f"#  DARWIN STORE BATTLE TEST")
    print(f"#  Target: {test_root}")
    print(f"{'#' * 60}")

    try:
        run_tests(test_root)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
