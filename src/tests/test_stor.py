"""
tests for darwin store
run with: pytest tests/test_store.py -v
"""

import pytest
import json
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.core.Chunker import Chunk
from src.core.Store import DarwinStore, _sanitize, _file_hash, DARWIN_DIR


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_project(tmp_path):
    """
    Creates a temporary project directory with a fake PDF file.
    Cleaned up automatically after each test.
    """
    # fake PDF file — just needs to exist and have content for hashing
    pdf = tmp_path / "biology.pdf"
    pdf.write_bytes(b"fake pdf content for testing")

    pdf2 = tmp_path / "history.pdf"
    pdf2.write_bytes(b"fake history pdf content")

    return tmp_path


@pytest.fixture
def store(tmp_project):
    s = DarwinStore(tmp_project)
    s.init()
    return s


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            text        = "Chlorophyll absorbs red and blue light to drive photosynthesis.",
            index       = 0,
            source      = "biology.pdf",
            token_count = 12,
            is_overlap  = False,
        ),
        Chunk(
            text        = "ATP is produced during the electron transport chain.",
            index       = 1,
            source      = "biology.pdf",
            token_count = 9,
            is_overlap  = True,
        ),
    ]


@pytest.fixture
def sample_embeddings():
    # fake normalized embeddings — shape (2, 384)
    raw = np.random.randn(2, 384).astype(np.float32)
    # normalize each row
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


@pytest.fixture
def sample_graph():
    return {
        "concepts": [
            {
                "id":            0,
                "label":         "chlorophyll absorbs light to drive reactions",
                "weight":        8,
                "source_chunks": [0],
                "page":          1,
            },
            {
                "id":            1,
                "label":         "ATP stores chemical energy from electron transport",
                "weight":        7,
                "source_chunks": [1],
                "page":          2,
            },
        ],
        "similarity_matrix": [
            [1.0,  0.72],
            [0.72, 1.0],
        ],
    }


# ── Sanitize tests ─────────────────────────────────────────────────────────────

class TestSanitize:

    def test_replaces_dot(self):
        assert _sanitize("biology.pdf") == "biology_pdf"

    def test_replaces_spaces(self):
        assert _sanitize("my notes.pdf") == "my_notes_pdf"

    def test_lowercases(self):
        assert _sanitize("BiologyNotes.PDF") == "biologynotes_pdf"

    def test_collapses_multiple_underscores(self):
        assert _sanitize("bio..notes.pdf") == "bio_notes_pdf"

    def test_strips_leading_trailing_underscores(self):
        result = _sanitize(".hidden.pdf")
        assert not result.startswith("_")

    def test_handles_path_prefix(self):
        # should strip directory and only use filename
        assert _sanitize("/home/user/biology.pdf") == "biology_pdf"


# ── Init tests ─────────────────────────────────────────────────────────────────

class TestInit:

    def test_creates_darwin_dir(self, tmp_project):
        store = DarwinStore(tmp_project)
        store.init()
        assert (tmp_project / DARWIN_DIR).exists()

    def test_creates_config(self, tmp_project):
        store = DarwinStore(tmp_project)
        store.init()
        assert (tmp_project / DARWIN_DIR / "config.json").exists()

    def test_init_twice_is_safe(self, store):
        result = store.init()
        assert result == False   # already initialized

    def test_is_initialized_true_after_init(self, store):
        assert store.is_initialized()

    def test_is_initialized_false_before_init(self, tmp_project):
        store = DarwinStore(tmp_project)
        # don't call init
        assert not store.is_initialized()

    def test_config_has_defaults(self, store):
        config = store.load_config()
        assert "embedding_tier" in config
        assert "llm_mode"       in config
        assert "top_k"          in config
        assert config["created_at"] is not None


# ── Config tests ───────────────────────────────────────────────────────────────

class TestConfig:

    def test_save_and_load_config(self, store):
        config = store.load_config()
        config["top_k"] = 10
        store.save_config(config)
        reloaded = store.load_config()
        assert reloaded["top_k"] == 10

    def test_update_config_partial(self, store):
        store.update_config(llm_mode="api", llm_model="claude-opus-4-6")
        config = store.load_config()
        assert config["llm_mode"]  == "api"
        assert config["llm_model"] == "claude-opus-4-6"
        # other keys should be untouched
        assert "top_k" in config


# ── Freshness tests ────────────────────────────────────────────────────────────

class TestFreshness:

    def test_not_fresh_before_processing(self, store):
        assert not store.is_fresh("biology.pdf")

    def test_fresh_after_saving_chunks(self, store, sample_chunks, tmp_project):
        store.save_chunks("biology.pdf", sample_chunks)
        assert store.is_fresh("biology.pdf")

    def test_not_fresh_after_file_changes(self, store, sample_chunks, tmp_project):
        store.save_chunks("biology.pdf", sample_chunks)
        assert store.is_fresh("biology.pdf")

        # modify the file
        (tmp_project / "biology.pdf").write_bytes(b"completely different content now")
        assert not store.is_fresh("biology.pdf")

    def test_not_fresh_for_unknown_file(self, store):
        assert not store.is_fresh("nonexistent.pdf")


# ── Chunks tests ───────────────────────────────────────────────────────────────

class TestChunks:

    def test_save_and_load_chunks(self, store, sample_chunks):
        store.save_chunks("biology.pdf", sample_chunks)
        loaded = store.load_chunks("biology.pdf")

        assert len(loaded) == len(sample_chunks)
        assert loaded[0].text  == sample_chunks[0].text
        assert loaded[0].index == sample_chunks[0].index
        assert loaded[1].is_overlap == sample_chunks[1].is_overlap

    def test_chunk_source_preserved(self, store, sample_chunks):
        store.save_chunks("biology.pdf", sample_chunks)
        loaded = store.load_chunks("biology.pdf")
        assert all(c.source == "biology.pdf" for c in loaded)

    def test_load_chunks_missing_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.load_chunks("never_processed.pdf")

    def test_saves_meta_with_chunk_count(self, store, sample_chunks):
        store.save_chunks("biology.pdf", sample_chunks)
        status = store.status()
        file_status = next(f for f in status["files"] if "biology" in f["filename"])
        assert file_status["chunk_count"] == len(sample_chunks)

    def test_creates_file_subfolder(self, store, sample_chunks, tmp_project):
        store.save_chunks("biology.pdf", sample_chunks)
        assert (tmp_project / DARWIN_DIR / "biology_pdf").exists()


# ── Embeddings tests ───────────────────────────────────────────────────────────

class TestEmbeddings:

    def test_save_and_load_embeddings(self, store, sample_chunks, sample_embeddings):
        store.save_chunks("biology.pdf", sample_chunks)
        store.save_embeddings("biology.pdf", sample_embeddings)
        loaded = store.load_embeddings("biology.pdf")

        assert loaded.shape == sample_embeddings.shape
        np.testing.assert_allclose(loaded, sample_embeddings, rtol=1e-5)

    def test_embeddings_saved_as_npy(self, store, sample_chunks, sample_embeddings, tmp_project):
        store.save_chunks("biology.pdf", sample_chunks)
        store.save_embeddings("biology.pdf", sample_embeddings)
        assert (tmp_project / DARWIN_DIR / "biology_pdf" / "embeddings.npy").exists()

    def test_load_embeddings_missing_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.load_embeddings("never_processed.pdf")


# ── BM25 tests ─────────────────────────────────────────────────────────────────

class TestBM25:

    def test_save_and_load_bm25(self, store, sample_chunks):
        from src.core.Query import BM25Index

        store.save_chunks("biology.pdf", sample_chunks)

        index = BM25Index()
        index.build(sample_chunks)

        store.save_bm25("biology.pdf", index)
        loaded = store.load_bm25("biology.pdf")

        # verify loaded index still works
        results = loaded.search("chlorophyll light")
        assert isinstance(results, list)

    def test_load_bm25_missing_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.load_bm25("never_processed.pdf")


# ── Stopwords tests ────────────────────────────────────────────────────────────

class TestStopwords:

    def test_save_and_load_stopwords(self, store, sample_chunks):
        store.save_chunks("biology.pdf", sample_chunks)
        stopwords = {"furthermore", "however", "it", "can", "be", "seen"}
        store.save_stopwords("biology.pdf", stopwords)
        loaded = store.load_stopwords("biology.pdf")
        assert loaded == stopwords

    def test_load_missing_stopwords_returns_empty_set(self, store):
        result = store.load_stopwords("never_processed.pdf")
        assert result == set()

    def test_stopwords_saved_sorted(self, store, sample_chunks, tmp_project):
        store.save_chunks("biology.pdf", sample_chunks)
        store.save_stopwords("biology.pdf", {"zebra", "apple", "mango"})

        raw = json.loads(
            (tmp_project / DARWIN_DIR / "biology_pdf" / "stopwords.json").read_text()
        )
        assert raw["stopwords"] == sorted(["zebra", "apple", "mango"])


# ── Graph tests ────────────────────────────────────────────────────────────────

class TestGraph:

    def test_save_and_load_graph(self, store, sample_chunks, sample_graph):
        store.save_chunks("biology.pdf", sample_chunks)
        store.save_graph("biology.pdf", sample_graph)
        loaded = store.load_graph("biology.pdf")

        assert len(loaded["concepts"])          == len(sample_graph["concepts"])
        assert loaded["concepts"][0]["label"]   == sample_graph["concepts"][0]["label"]
        assert loaded["similarity_matrix"][0][1] == sample_graph["similarity_matrix"][0][1]

    def test_has_graph_false_before_build(self, store, sample_chunks):
        store.save_chunks("biology.pdf", sample_chunks)
        assert not store.has_graph("biology.pdf")

    def test_has_graph_true_after_save(self, store, sample_chunks, sample_graph):
        store.save_chunks("biology.pdf", sample_chunks)
        store.save_graph("biology.pdf", sample_graph)
        assert store.has_graph("biology.pdf")

    def test_graph_meta_written(self, store, sample_chunks, sample_graph, tmp_project):
        store.save_chunks("biology.pdf", sample_chunks)
        store.save_graph("biology.pdf", sample_graph)
        meta = store.load_graph_meta("biology.pdf")
        assert meta["concept_count"] == 2
        assert "built_at" in meta

    def test_load_graph_missing_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.load_graph("never_processed.pdf")

    def test_graphs_are_isolated_per_file(self, store, sample_chunks, sample_graph, tmp_project):
        """biology graph should not affect history graph folder"""
        store.save_chunks("biology.pdf", sample_chunks)
        store.save_graph("biology.pdf", sample_graph)

        # history has no graph yet
        assert not store.has_graph("history.pdf")


# ── Remove file tests ──────────────────────────────────────────────────────────

class TestRemoveFile:

    def test_remove_existing_file(self, store, sample_chunks, tmp_project):
        store.save_chunks("biology.pdf", sample_chunks)
        assert (tmp_project / DARWIN_DIR / "biology_pdf").exists()

        result = store.remove_file("biology.pdf")
        assert result == True
        assert not (tmp_project / DARWIN_DIR / "biology_pdf").exists()

    def test_remove_nonexistent_returns_false(self, store):
        result = store.remove_file("never_processed.pdf")
        assert result == False

    def test_other_files_unaffected_after_remove(self, store, sample_chunks, tmp_project):
        store.save_chunks("biology.pdf", sample_chunks)
        store.save_chunks("history.pdf", sample_chunks)
        store.remove_file("biology.pdf")

        # history should still be there
        assert (tmp_project / DARWIN_DIR / "history_pdf").exists()


# ── Status tests ───────────────────────────────────────────────────────────────

class TestStatus:

    def test_status_returns_dict(self, store):
        status = store.status()
        assert isinstance(status, dict)
        assert "project_dir" in status
        assert "files"       in status
        assert "config"      in status

    def test_status_empty_when_no_files(self, store):
        status = store.status()
        assert status["files"] == []

    def test_status_reflects_processed_files(self, store, sample_chunks, sample_graph):
        store.save_chunks("biology.pdf", sample_chunks)
        store.save_embeddings("biology.pdf", np.random.randn(2, 384))
        store.save_graph("biology.pdf", sample_graph)

        status = store.status()
        assert len(status["files"]) == 1

        f = status["files"][0]
        assert f["has_chunks"]     == True
        assert f["has_embeddings"] == True
        assert f["has_graph"]      == True
        assert f["chunk_count"]    == 2
        assert f["concept_count"]  == 2

    def test_status_shows_both_files(self, store, sample_chunks):
        store.save_chunks("biology.pdf", sample_chunks)
        store.save_chunks("history.pdf", sample_chunks)

        status  = store.status()
        names   = [f["filename"] for f in status["files"]]
        assert "biology.pdf" in names
        assert "history.pdf" in names