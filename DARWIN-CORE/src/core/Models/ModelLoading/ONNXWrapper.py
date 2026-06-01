"""
src/core/Models/ModelLoading/ONNXWrapper.py
-------------------------------------------
ONNX Runtime embedding + reranker wrappers.
Mirrors SentenceTransformer / CrossEncoder interfaces.
"""

import numpy as np
from pathlib import Path
from typing import List, Union, Tuple

import onnxruntime as ort
from tokenizers import Tokenizer


def _needs_token_type_ids(session) -> bool:
    """Check if the ONNX session requires token_type_ids as input."""
    input_names = [inp.name for inp in session.get_inputs()]
    return "token_type_ids" in input_names


def _build_token_type_ids(input_ids: np.ndarray) -> np.ndarray:
    """Create zero-filled token_type_ids matching input_ids shape."""
    return np.zeros_like(input_ids, dtype=np.int64)


class ONNXEmbeddingWrapper:
    def __init__(
        self,
        model_dir: Path,
        max_length: int = 512,
        normalize: bool = True,
    ):
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self.normalize = normalize

        model_files = list(self.model_dir.glob("*.onnx"))
        if not model_files:
            raise FileNotFoundError(f"No .onnx file found in {model_dir}")
        self.model_path = model_files[0]

        tokenizer_path = self.model_dir / "tokenizer.json"
        if tokenizer_path.exists():
            self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
            # Enable truncation so encode_batch never exceeds max_length
            if hasattr(self.tokenizer, "enable_truncation"):
                self.tokenizer.enable_truncation(max_length=self.max_length)
        else:
            self.tokenizer = self._load_transformers_tokenizer()

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ort.get_available_providers()
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options,
            providers=providers,
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self._needs_tti = _needs_token_type_ids(self.session)

    def _load_transformers_tokenizer(self):
        """Fallback if tokenizer.json isn\'t present."""
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(str(self.model_dir))
        except ImportError:
            raise RuntimeError(
                "No tokenizer.json found and transformers not available. "
                "Download a complete ONNX model export (e.g., Xenova repos)."
            )

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

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        if hasattr(self.tokenizer, "encode_batch"):
            encodings = self.tokenizer.encode_batch(texts)
            # Pad manually to max length in this batch (tokenizers lib doesn\'t pad by default)
            max_len = min(max(len(e.ids) for e in encodings), self.max_length)
            input_ids = np.zeros((len(encodings), max_len), dtype=np.int64)
            attention_mask = np.zeros((len(encodings), max_len), dtype=np.int64)
            for i, e in enumerate(encodings):
                seq_len = min(len(e.ids), self.max_length)
                input_ids[i, :seq_len] = e.ids[:seq_len]
                attention_mask[i, :seq_len] = e.attention_mask[:seq_len]
        else:
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

        feed = {
            self.input_name: input_ids,
            "attention_mask": attention_mask,
        }
        if self._needs_tti:
            feed["token_type_ids"] = _build_token_type_ids(input_ids)

        outputs = self.session.run(
            self.output_names,
            feed,
        )

        token_embeddings = outputs[0]
        mask = attention_mask.astype(np.float32)
        mask_expanded = np.expand_dims(mask, axis=-1)
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        pooled = sum_embeddings / np.clip(mask.sum(axis=1, keepdims=True), a_min=1e-9, a_max=None)
        return pooled

    def __call__(self, texts, **kwargs):
        return self.encode(texts, **kwargs)


class ONNXRerankerWrapper:
    """
    ONNX Runtime reranker wrapper.
    Matches CrossEncoder.predict() interface.
    """

    def __init__(self, model_dir: Path, max_length: int = 512):
        self.model_dir = Path(model_dir)
        self.max_length = max_length

        model_files = list(self.model_dir.glob("*.onnx"))
        if not model_files:
            raise FileNotFoundError(f"No .onnx file found in {model_dir}")

        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ort.get_available_providers()
        self.session = ort.InferenceSession(
            str(model_files[0]),
            sess_options,
            providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self._needs_tti = _needs_token_type_ids(self.session)

        # Load tokenizer
        tokenizer_path = self.model_dir / "tokenizer.json"
        if tokenizer_path.exists():
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
            if hasattr(self.tokenizer, "enable_truncation"):
                self.tokenizer.enable_truncation(max_length=self.max_length)
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))

    def predict(self, pairs: List[Tuple[str, str]], batch_size: int = 32, **kwargs) -> np.ndarray:
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            batch_scores = self._predict_batch(batch)
            scores.extend(batch_scores)
        return np.array(scores, dtype=np.float32)

    def _predict_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        texts = [f"{q} [SEP] {d}" for q, d in pairs]

        if hasattr(self.tokenizer, "encode_batch"):
            encodings = self.tokenizer.encode_batch(texts)
            max_len = min(max(len(e.ids) for e in encodings), self.max_length)
            input_ids = np.zeros((len(encodings), max_len), dtype=np.int64)
            attention_mask = np.zeros((len(encodings), max_len), dtype=np.int64)
            for i, e in enumerate(encodings):
                seq_len = min(len(e.ids), self.max_length)
                input_ids[i, :seq_len] = e.ids[:seq_len]
                attention_mask[i, :seq_len] = e.attention_mask[:seq_len]
        else:
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

        feed = {
            self.input_name: input_ids,
            "attention_mask": attention_mask,
        }
        if self._needs_tti:
            feed["token_type_ids"] = _build_token_type_ids(input_ids)

        outputs = self.session.run(
            None,
            feed,
        )

        return [float(s) for s in outputs[0].squeeze()]