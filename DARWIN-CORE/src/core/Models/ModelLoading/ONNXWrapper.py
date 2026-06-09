"""
src/core/Models/ModelLoading/ONNXWrapper.py
-------------------------------------------
Robust ONNX Runtime wrappers with warmup, memory-aware defaults,
and model-specific output handling (e.g. bge-m3 dense).
"""

import numpy as np
from pathlib import Path
from typing import List, Union, Tuple, Optional

import onnxruntime as ort
from tokenizers import Tokenizer

from src.core.Logger.DarwinLogger import darwin, LogType


def _get_input_map(session: ort.InferenceSession) -> dict:
    inputs = session.get_inputs()
    mapping = {"input_ids": inputs[0].name}
    if len(inputs) >= 2:
        mapping["attention_mask"] = inputs[1].name
    if len(inputs) >= 3:
        mapping["token_type_ids"] = inputs[2].name
    return mapping


def _needs_token_type_ids(session: ort.InferenceSession) -> bool:
    return len(session.get_inputs()) >= 3


def _build_token_type_ids(input_ids: np.ndarray) -> np.ndarray:
    return np.zeros_like(input_ids, dtype=np.int64)


def _load_tokenizer(model_dir: Path, max_length: int) -> Tokenizer:
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


class ONNXEmbeddingWrapper:
    """
    SentenceTransformer‑compatible embedding model via ONNX Runtime.
    Handles token‑level vs. already‑pooled outputs and model-specific flags.
    """

    def __init__(
        self,
        model_dir: Path,
        max_length: int = 512,
        min_length: int = 1,
        normalize: bool = True,
    ):
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self.min_length = min_length
        self.normalize = normalize

        model_files = list(self.model_dir.glob("*.onnx"))
        if not model_files:
            raise FileNotFoundError(f"No .onnx file found in {model_dir}")
        self.model_path = model_files[0]

        # Auto-detect bge-m3 for return_dense behavior
        model_name_lower = str(self.model_path).lower()
        self._return_dense = "bge-m3" in model_name_lower

        self.tokenizer = _load_tokenizer(self.model_dir, self.max_length)

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

        
        darwin.log(f"ONNX embedding model loaded: {self.model_path.name}", LogType.SUCCESS)

    def warmup(self, texts: Optional[List[str]] = None) -> None:
        """Run dummy inference to warm caches and avoid first-batch latency."""
        try:
            self._encode_batch(texts or ["warmup"])
        except Exception:
            pass

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: Optional[bool] = None,
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

        # ── FIX C3: Respect explicit False ─────────────────────────────
        do_normalize = normalize_embeddings if normalize_embeddings is not None else self.normalize
        if do_normalize:
            result = result / np.linalg.norm(result, axis=1, keepdims=True)

        return result

    def __call__(self, texts, **kwargs):
        return self.encode(texts, **kwargs)

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(texts)
        max_len = min(max(len(e.ids) for e in encodings), self.max_length)
        input_ids = np.zeros((len(encodings), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(encodings), max_len), dtype=np.int64)

        for i, e in enumerate(encodings):
            seq_len = min(len(e.ids), self.max_length)
            input_ids[i, :seq_len] = e.ids[:seq_len]
            attention_mask[i, :seq_len] = e.attention_mask[:seq_len]

        feed = {
            self.input_map["input_ids"]: input_ids,
            self.input_map["attention_mask"]: attention_mask,
        }
        if self._needs_tti:
            feed[self.input_map["token_type_ids"]] = _build_token_type_ids(input_ids)

        outputs = self.session.run(None, feed)

        # Handle output shape
        token_output = outputs[0]
        if token_output.ndim == 3:
            mask = attention_mask.astype(np.float32)
            mask_expanded = np.expand_dims(mask, axis=-1)
            sum_embeddings = np.sum(token_output * mask_expanded, axis=1)
            pooled = sum_embeddings / np.clip(
                mask.sum(axis=1, keepdims=True), a_min=1e-9, a_max=None
            )
            return pooled
        else:
            return token_output


class ONNXRerankerWrapper:
    """
    CrossEncoder‑compatible reranker via ONNX Runtime.
    """

    def __init__(
        self,
        model_dir: Path,
        max_length: int = 512,
        min_length: int = 1,
    ):
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self.min_length = min_length

        model_files = list(self.model_dir.glob("*.onnx"))
        if not model_files:
            raise FileNotFoundError(f"No .onnx file found in {model_dir}")
        self.model_path = model_files[0]

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

        self.tokenizer = _load_tokenizer(self.model_dir, self.max_length)

        # ── FIX C16: Discover sep_token from tokenizer ─────────────────
        raw_sep = getattr(self.tokenizer, "sep_token", None)
        if raw_sep is None:
            # Try to infer from model vocabulary
            model_id_lower = str(self.model_path).lower()
            if "llama" in model_id_lower or "mistral" in model_id_lower:
                raw_sep = "</s>"
            else:
                raw_sep = " [SEP] "
        self._sep_token = f" {raw_sep.strip()} " if raw_sep else " [SEP] "

        
        darwin.log(f"ONNX reranker loaded: {self.model_path.name}", LogType.SUCCESS)

    def warmup(self, pairs: Optional[List[Tuple[str, str]]] = None) -> None:
        try:
            self._predict_batch(pairs or [("warmup", "warmup")])
        except Exception:
            pass

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

    def _predict_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        texts = [f"{q}{self._sep_token}{d}" for q, d in pairs]

        encodings = self.tokenizer.encode_batch(texts)
        max_len = min(max(len(e.ids) for e in encodings), self.max_length)
        input_ids = np.zeros((len(encodings), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(encodings), max_len), dtype=np.int64)

        for i, e in enumerate(encodings):
            seq_len = min(len(e.ids), self.max_length)
            input_ids[i, :seq_len] = e.ids[:seq_len]
            attention_mask[i, :seq_len] = e.attention_mask[:seq_len]

        feed = {
            self.input_map["input_ids"]: input_ids,
            self.input_map["attention_mask"]: attention_mask,
        }
        if self._needs_tti:
            feed[self.input_map["token_type_ids"]] = _build_token_type_ids(input_ids)

        logits = self.session.run(None, feed)[0]

        if logits.ndim == 2 and logits.shape[-1] == 2:
            scores = logits[:, 1].astype(np.float32)
        else:
            scores = logits.flatten().astype(np.float32)

        return scores.tolist()