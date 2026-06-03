"""
src/core/Models/ModelLoading/ONNXWrapper.py
-------------------------------------------
Robust ONNX Runtime wrappers for embeddings and reranking.
Dynamically handles input/output naming, pooling shape variants,
enforces token limits from ModelSpec, and logs via Darwin event system.

Zero heavy dependencies — uses only onnxruntime and tokenizers.
"""

import numpy as np
from pathlib import Path
from typing import List, Union, Tuple

import onnxruntime as ort
from tokenizers import Tokenizer

from src.core.Logger.DarwinLogger import darwin, LogType, LogEvent


# ------------------------------------------------------------------ #
# Helper utilities
# ------------------------------------------------------------------ #

def _get_input_map(session: ort.InferenceSession) -> dict:
    """
    Build a mapping from logical input role to actual ONNX tensor name.
    Roles: input_ids, attention_mask, token_type_ids (if present).
    """
    inputs = session.get_inputs()
    mapping = {"input_ids": inputs[0].name}

    if len(inputs) >= 2:
        mapping["attention_mask"] = inputs[1].name
    if len(inputs) >= 3:
        mapping["token_type_ids"] = inputs[2].name

    return mapping


def _needs_token_type_ids(session: ort.InferenceSession) -> bool:
    """Check if session expects token_type_ids input."""
    return len(session.get_inputs()) >= 3


def _build_token_type_ids(input_ids: np.ndarray) -> np.ndarray:
    """Create zero‑filled token_type_ids matching input_ids shape."""
    return np.zeros_like(input_ids, dtype=np.int64)


def _load_tokenizer(model_dir: Path, max_length: int) -> Tokenizer:
    """
    Load tokenizer from tokenizer.json in the model directory.
    Fails with a clear message if the file is missing — no heavy fallback.
    """
    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"tokenizer.json not found in {model_dir}. "
            f"The ONNX model export is incomplete. "
            f"Re‑download using OnnxModelDownloader or use a Xenova‑format repo."
        )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if hasattr(tokenizer, "enable_truncation"):
        tokenizer.enable_truncation(max_length=max_length)
    return tokenizer


# ------------------------------------------------------------------ #
# ONNX Embedding Wrapper
# ------------------------------------------------------------------ #

class ONNXEmbeddingWrapper:
    """
    SentenceTransformer‑compatible embedding model via ONNX Runtime.
    Detects token‑level vs. already‑pooled outputs and handles both.
    Enforces min/max token limits from ModelSpec.

    Dependencies: onnxruntime, tokenizers (no torch, no transformers).
    """

    def __init__(
        self,
        model_dir: Path,
        max_length: int = 512,
        min_length: int = 1,
        normalize: bool = True,
        enable_logging: bool = True,
    ):
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self.min_length = min_length
        self.normalize = normalize
        self.enable_logging = enable_logging

        # Locate ONNX file
        model_files = list(self.model_dir.glob("*.onnx"))
        if not model_files:
            raise FileNotFoundError(f"No .onnx file found in {model_dir}")
        self.model_path = model_files[0]

        # Load tokenizer (fast tokenizer only — no transformers fallback)
        self.tokenizer = _load_tokenizer(self.model_dir, self.max_length)

        # Create ONNX session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ort.get_available_providers()
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options,
            providers=providers,
        )

        # Build dynamic input mapping
        self.input_map = _get_input_map(self.session)
        self._needs_tti = _needs_token_type_ids(self.session)

        # Log success
        if self.enable_logging:
            darwin.log(f"ONNX embedding model loaded: {self.model_path.name}", LogType.SUCCESS,
              )

    # ------------------------------------------------------------------ #
    # Encoding interface (SentenceTransformer‑style)
    # ------------------------------------------------------------------ #
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
        **kwargs,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self._encode_batch(batch)
            all_embeddings.append(embeddings)

        result = np.vstack(all_embeddings)

        do_normalize = normalize_embeddings if normalize_embeddings else self.normalize
        if do_normalize:
            result = result / np.linalg.norm(result, axis=1, keepdims=True)

        return result

    def __call__(self, texts, **kwargs):
        return self.encode(texts, **kwargs)

    # ------------------------------------------------------------------ #
    # Internal batch encoding
    # ------------------------------------------------------------------ #
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        # Tokenize with fast tokenizer
        encodings = self.tokenizer.encode_batch(texts)
        max_len = min(max(len(e.ids) for e in encodings), self.max_length)
        input_ids = np.zeros((len(encodings), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(encodings), max_len), dtype=np.int64)
        token_counts = []

        for i, e in enumerate(encodings):
            seq_len = min(len(e.ids), self.max_length)
            input_ids[i, :seq_len] = e.ids[:seq_len]
            attention_mask[i, :seq_len] = e.attention_mask[:seq_len]
            token_counts.append(seq_len)

        # ── Enforce min_length: warn but don't pad ──
        if self.enable_logging:
            for idx, (text, count) in enumerate(zip(texts, token_counts)):
                if count < self.min_length:
                    darwin.log(f"Input too short ({count} tokens < min {self.min_length}) "
                            f"for batch item {idx}: '{text[:80]}...'",LogType.WARNING)

        # Build feed dict with dynamic names
        feed = {
            self.input_map["input_ids"]: input_ids,
            self.input_map["attention_mask"]: attention_mask,
        }
        if self._needs_tti:
            feed[self.input_map["token_type_ids"]] = _build_token_type_ids(input_ids)

        # Run inference
        outputs = self.session.run(None, feed)

        # Handle output shape
        token_output = outputs[0]
        if token_output.ndim == 3:
            # Token‑level embeddings → mean pooling
            mask = attention_mask.astype(np.float32)
            mask_expanded = np.expand_dims(mask, axis=-1)
            sum_embeddings = np.sum(token_output * mask_expanded, axis=1)
            pooled = sum_embeddings / np.clip(
                mask.sum(axis=1, keepdims=True), a_min=1e-9, a_max=None
            )
            return pooled
        else:
            # Already pooled (e.g., 2D [batch, dim])
            return token_output


# ------------------------------------------------------------------ #
# ONNX Reranker Wrapper
# ------------------------------------------------------------------ #

class ONNXRerankerWrapper:
    """
    CrossEncoder‑compatible reranker via ONNX Runtime.
    Handles binary classification logits safely.

    Dependencies: onnxruntime, tokenizers (no torch, no transformers).
    """

    def __init__(
        self,
        model_dir: Path,
        max_length: int = 512,
        min_length: int = 1,
        enable_logging: bool = True,
    ):
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self.min_length = min_length
        self.enable_logging = enable_logging

        # Locate ONNX file
        model_files = list(self.model_dir.glob("*.onnx"))
        if not model_files:
            raise FileNotFoundError(f"No .onnx file found in {model_dir}")
        self.model_path = model_files[0]

        # Create session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ort.get_available_providers()
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options,
            providers=providers,
        )

        self.input_map = _get_input_map(self.session)
        self._needs_tti = _needs_token_type_ids(self.session)

        # Load tokenizer (fast tokenizer only — no transformers fallback)
        self.tokenizer = _load_tokenizer(self.model_dir, self.max_length)

        if self.enable_logging:
            darwin.log(f"ONNX reranker loaded: {self.model_path.name}", LogType.SUCCESS)

    # ------------------------------------------------------------------ #
    # Prediction interface (CrossEncoder‑style)
    # ------------------------------------------------------------------ #
    def predict(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 32,
        **kwargs,
    ) -> np.ndarray:
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            batch_scores = self._predict_batch(batch)
            scores.extend(batch_scores)
        return np.array(scores, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Internal batch prediction
    # ------------------------------------------------------------------ #
    def _predict_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        texts = [f"{q} [SEP] {d}" for q, d in pairs]

        encodings = self.tokenizer.encode_batch(texts)
        max_len = min(max(len(e.ids) for e in encodings), self.max_length)
        input_ids = np.zeros((len(encodings), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(encodings), max_len), dtype=np.int64)
        token_counts = []

        for i, e in enumerate(encodings):
            seq_len = min(len(e.ids), self.max_length)
            input_ids[i, :seq_len] = e.ids[:seq_len]
            attention_mask[i, :seq_len] = e.attention_mask[:seq_len]
            token_counts.append(seq_len)

        # ── Enforce min_length: warn but don't pad ──
        if self.enable_logging:
            for idx, (text, count) in enumerate(zip(texts, token_counts)):
                if count < self.min_length:
                    darwin.log(f"Input too short ({count} tokens < min {self.min_length}) "
                            f"for batch item {idx}: '{text[:80]}...'",LogType.WARNING)

        feed = {
            self.input_map["input_ids"]: input_ids,
            self.input_map["attention_mask"]: attention_mask,
        }
        if self._needs_tti:
            feed[self.input_map["token_type_ids"]] = _build_token_type_ids(input_ids)

        logits = self.session.run(None, feed)[0]

        # Extract relevance score from binary or single‑output logits
        if logits.ndim == 2 and logits.shape[-1] == 2:
            scores = logits[:, 1].astype(np.float32)
        else:
            scores = logits.flatten().astype(np.float32)

        return scores.tolist()