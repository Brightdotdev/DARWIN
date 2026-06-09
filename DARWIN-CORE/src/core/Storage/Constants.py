from __future__ import annotations

CORE_STAGES = ["markdown", "chunks", "embeddings", "bm25", "vector"]

# Which sub-keys inside a stage are optional (won't fail health check if missing)
OPTIONAL_ARTIFACT_KEYS = {
    "markdown": {"display"},
    "embeddings": {"index_map"},
    "images": {"dir"},
    "graph_vis": {"file"},
}

ARTIFACT_MANIFEST = {
    "images": {"dir": "images/"},
    "markdown": {"source": "markdown.md", "display": "display-markdown.md"},
    "chunks": {"file": "chunks.parquet"},
    "vector": {"index": "vectors/vector.usearch", "meta": "vectors/vector.meta.json"},
    "embeddings": {
        "file": "vectors/embeddings.f16.npy",
        "index_map": "vectors/embeddings.index_map.json",
    },
    "bm25": {
        "index": "bm25.npz",
        "meta": "bm25_meta.json",
        "tokens": "bm25_tokens.parquet",
    },
    "stopwords": {"file": "stopwords.json"},
    "graph": {"nodes": "graph/nodes.parquet", "edges": "graph/edges.parquet"},
    "graph_vis": {"file": "graph/graph.html"},
    "wiki": {"file": "wiki/wiki.json"},
    "meta": {"file": "meta.json"},
}