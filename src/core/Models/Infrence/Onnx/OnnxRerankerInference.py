import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from core.Logger.DarwinLogger import darwin


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RerankMetadata:
    """Metadata about the reranker inference run."""
    model_name: str
    repo_id: str
    vector_size: int  # 384 for MiniLM-L6
    normalize: bool
    batch_size: int
    inference_time_ms: float
    num_queries: int
    num_documents: int
    device: str
    onnx_runtime_version: str
    max_length: int


@dataclass
class RerankResult:
    """Single rerank output: scores + metadata."""
    scores: List[float]  # score per document, in original order
    metadata: RerankMetadata
    ranked_indices: List[int]  # document indices sorted by score descending
    ranked_scores: List[float]  # scores sorted descending


# ═══════════════════════════════════════════════════════════════════════════════
# Inference Engine
# ═══════════════════════════════════════════════════════════════════════════════

class OnnxRerankerInference:
    """
    MS MARCO cross-encoder ONNX reranker inference.
    Loads tokenizer + ONNX model, scores query-document pairs.
    
    Expects model directory layout from OnnxRerankerDownloader:
        model_dir/
            model.onnx
            tokenizer.json
            vocab.txt
            special_tokens_map.json
    """

    def __init__(
        self,
        model_dir: Path,
        normalize: bool = False,  # Cross-encoders don't normalize scores
        batch_size: int = 8,
        max_length: int = 512,
        repo_id: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        onnx_filename: str = "onnx/model_qint8_avx512_vnni.onnx",
    ):
        self.model_dir = Path(model_dir)
        self.normalize = normalize
        self.batch_size = batch_size
        self.max_length = max_length
        self.repo_id = repo_id
        self.onnx_filename = onnx_filename

        self._session = None
        self._tokenizer = None
        self._vector_size = 384  # MiniLM-L6-v2
        self._device = "CPU"
        
    # ── Context Manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "OnnxRerankerInference":
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.unload()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def load(self) -> "OnnxRerankerInference":
        """Load tokenizer and ONNX session. Chainable."""
        if self.is_loaded():
            darwin.info("Already loaded, skipping.")
            return self

        if not self.model_dir.exists():
            darwin.error(f"Model directory not found: {self.model_dir}")
            return self

        if not self._check_model_exists():
            darwin.error("Model files missing. Run downloader first.")
            return self

        if not self._check_integrity():
            darwin.error("Model integrity check failed. Model may be corrupted.")
            return self

        darwin.info(f"Loading model from {self.model_dir}")
        self._load_tokenizer()
        self._load_onnx_session()
        darwin.success("Model loaded successfully.")
        return self

    def unload(self) -> None:
        """Unload model and free memory."""
        if not self.is_loaded():
            darwin.info("Not loaded, nothing to unload.")
            return

        darwin.info("Unloading model...")
        self._session = None
        self._tokenizer = None
        gc.collect()
        darwin.success("Unloaded.")

    def is_loaded(self) -> bool:
        return self._session is not None and self._tokenizer is not None

    def _check_model_exists(self) -> bool:
        """Check if model files exist at all."""
        return (self.model_dir / "model.onnx").exists()

    def _check_integrity(self) -> bool:
        """Quick integrity check: files exist and have reasonable size."""
        required = [
            self.model_dir / "model.onnx",
            self.model_dir / "tokenizer.json",
        ]
        min_sizes = {
            "model.onnx": 10_000_000,
            "tokenizer.json": 100,
        }

        for p in required:
            if not p.exists():
                darwin.warning(f"Integrity fail: missing {p.name}")
                return False
            min_size = min_sizes.get(p.name, 100)
            if p.stat().st_size < min_size:
                darwin.warning(f"Integrity fail: {p.name} too small ({p.stat().st_size} bytes)")
                return False

        return True

    # ── Public inference API ─────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        documents: List[str],
        batch_size: Optional[int] = None,
    ) -> RerankResult:
        """
        Score query against each document.
        Returns scores in original document order + ranked indices.
        """
        if not self.is_loaded():
            darwin.error("Model not loaded. Call .load() first.")
            return RerankResult(
                scores=[],
                metadata=self._build_metadata("", [], 0.0),
                ranked_indices=[],
                ranked_scores=[],
            )

        if not documents:
            darwin.warning("No documents provided.")
            return RerankResult(
                scores=[],
                metadata=self._build_metadata(query, [], 0.0),
                ranked_indices=[],
                ranked_scores=[],
            )

        darwin.info(f"Reranking {len(documents)} documents for query: '{query[:50]}...'")

        # Pair query with each document
        pairs = [(query, doc) for doc in documents]

        # Run inference with batching
        spinner_id = f"rerank_{id(query)}_{len(documents)}"
        darwin.spinner_start(spinner_id, f"Reranking {len(documents)} documents")

        start = time.perf_counter()
        all_scores = []

        bs = batch_size or self.batch_size
        num_batches = (len(pairs) + bs - 1) // bs

        for batch_idx in range(0, len(pairs), bs):
            batch_num = batch_idx // bs + 1
            darwin.log(f"  → Batch {batch_num}/{num_batches}")

            batch_pairs = pairs[batch_idx : batch_idx + bs]
            scores = self._run_batch(batch_pairs)
            all_scores.extend(scores)

        inference_time_ms = (time.perf_counter() - start) * 1000

        darwin.spinner_stop(spinner_id, success=True)

        # Sort by score descending
        ranked = sorted(enumerate(all_scores), key=lambda x: x[1], reverse=True)
        ranked_indices = [idx for idx, _ in ranked]
        ranked_scores = [score for _, score in ranked]

        darwin.success(
            f"Rerank complete: {len(documents)} docs in {inference_time_ms:.1f}ms "
            f"(top score: {max(all_scores):.4f})"        )

        metadata = self._build_metadata(query, documents, inference_time_ms)

        return RerankResult(
            scores=all_scores,
            metadata=metadata,
            ranked_indices=ranked_indices,
            ranked_scores=ranked_scores,
        )

    def rerank_topk(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
        batch_size: Optional[int] = None,
    ) -> Tuple[List[int], List[float]]:
        """
        Convenience: return top-k document indices and scores.
        """
        result = self.rerank(query, documents, batch_size=batch_size)
        return result.ranked_indices[:top_k], result.ranked_scores[:top_k]

    # ── Internal: Tokenizer ──────────────────────────────────────────────────

    def _load_tokenizer(self) -> None:
        try:
            from tokenizers import Tokenizer
        except ImportError:
            darwin.error("tokenizers library required. pip install tokenizers")
            return

        tokenizer_path = self.model_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            darwin.error(f"Tokenizer not found in {self.model_dir}")
            return

        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=self.max_length)
        darwin.log(f"Tokenizer loaded from {tokenizer_path}")
    # ── Internal: ONNX Session ───────────────────────────────────────────────

    def _load_onnx_session(self) -> None:
        """Load ONNX runtime session with provider auto-selection."""
        try:
            import onnxruntime as ort
        except ImportError:
            darwin.error("onnxruntime required. pip install onnxruntime")
            return

        model_path = self.model_dir / "model.onnx"
        if not model_path.exists():
            darwin.error(f"ONNX model not found: {model_path}")
            return

        # Provider priority: CUDA → DirectML → CPU
        available = ort.get_available_providers()
        darwin.log(f"Available ONNX providers: {available}")

        preferred = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in preferred if p in available]

        if not providers:
            providers = ["CPUExecutionProvider"]

        self._device = providers[0].replace("ExecutionProvider", "")
        self._onnx_runtime_version = ort.__version__

        darwin.log(f"Using provider: {providers[0]}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers,
        )

        darwin.log(
            f"ONNX session loaded: {model_path} | "
            f"providers={providers} | vector_size={self._vector_size}"
        )

    
    def _run_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        # Encode each pair with special tokens (CLS, SEP)
        # Truncation is enabled on the tokenizer (max_length=512)
        encodings = []
        for q, d in pairs:
            enc = self._tokenizer.encode(q, d, add_special_tokens=True)
            encodings.append(enc)

        # Pad to max length in batch
        max_len = max(len(e.ids) for e in encodings)
        pad_token_id = self._tokenizer.token_to_id("[PAD]") or 0

        input_ids = np.array([
            e.ids + [pad_token_id] * (max_len - len(e.ids))
            for e in encodings
        ], dtype=np.int64)

        attention_mask = np.array([
            [1] * len(e.ids) + [0] * (max_len - len(e.ids))
            for e in encodings
        ], dtype=np.int64)

        # BERT-style token type IDs: 0 for first segment, 1 for second
        token_type_ids = np.array([
            e.type_ids + [0] * (max_len - len(e.ids))
            for e in encodings
        ], dtype=np.int64)

        outputs = self._session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        })

        logits = outputs[0]
        if logits.ndim == 2 and logits.shape[1] == 1:
            scores = logits.squeeze(axis=1).tolist()
        else:
            scores = logits[:, 0].tolist()
        return scores

    # ── Internal: Metadata ───────────────────────────────────────────────────

    def _build_metadata(
        self,
        query: str,
        documents: List[str],
        inference_time_ms: float,
    ) -> RerankMetadata:
        return RerankMetadata(
            model_name="ms-marco-MiniLM-L6-v2",
            repo_id=self.repo_id,
            vector_size=self._vector_size,
            normalize=self.normalize,
            batch_size=self.batch_size,
            inference_time_ms=round(inference_time_ms, 3),
            num_queries=1,
            num_documents=len(documents),
            device=self._device,
            onnx_runtime_version=self._onnx_runtime_version,
            max_length=self.max_length,
        )

    # ── Info ─────────────────────────────────────────────────────────────────

    def get_model_info(self) -> dict:
        """Return static info about the loaded model without running inference."""
        if not self.is_loaded():
            darwin.warning("Model not loaded")
            return {"status": "not_loaded", "model_dir": str(self.model_dir)}

        return {
            "status": "loaded",
            "model_dir": str(self.model_dir),
            "repo_id": self.repo_id,
            "vector_size": self._vector_size,
            "device": self._device,
            "onnx_runtime_version": getattr(self, "_onnx_runtime_version", "unknown"),
            "normalize": self.normalize,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
        }