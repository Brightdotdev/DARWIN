"""
src/core/Models/ModelLoading/Wrappers.py
------------------------------------------
API / PyTorch wrappers for embedding, reranking, and vision.
Now integrated with Darwin event system for logging and warnings.
"""

import base64
import logging
import warnings
import os
from pathlib import Path
from typing import List, Tuple, Union
import numpy as np

# ── Nuke litellm logging BEFORE it ever imports ──────────────────────
os.environ["LITELLM_LOG"] = "ERROR"
os.environ["LITELLM_SUPPRESS_LOGGING"] = "1"

for noisy in ("litellm", "litellm.llms.bedrock", "litellm.llms.sagemaker", "botocore"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

warnings.filterwarnings("ignore", module="litellm")

try:
    import torch
except ImportError:
    torch = None

try:
    import litellm
except ImportError:
    litellm = None

if litellm is not None:
    litellm.suppress_debug_info = True
    litellm.telemetry = False
    litellm.turn_off_message_logging = True

# ── Darwin logger (may be None if not yet initialised) ─────────────
try:
    from src.core.Logger.DarwinLogger import darwin, LogType, LogEvent
except ImportError:
    darwin = None
    LogType = None
    LogEvent = None


# ------------------------------------------------------------------ #
# Helper utilities
# ------------------------------------------------------------------ #



def _validate_non_empty_texts(texts: Union[str, List[str]]) -> List[str]:
    if isinstance(texts, str):
        texts = [texts]
    if not isinstance(texts, list) or len(texts) == 0:
        raise ValueError("texts must be a non-empty string or list of strings")
    return texts


def _safe_norm(tensor, dim: int, keepdim: bool = True):
    """L2 normalize with epsilon to avoid division by zero."""
    eps = 1e-12
    return tensor / (tensor.norm(dim=dim, keepdim=keepdim) + eps)


def _warn_short_input(texts: List[str], token_counts: List[int], min_length: int):
    """Emit Darwin warning for any input below min_length."""
    for idx, (text, count) in enumerate(zip(texts, token_counts)):
        if count < min_length:
            darwin.log(
                f"Input too short ({count} tokens < min {min_length}) "
                f"for batch item {idx}: '{text[:80]}...'",
                LogType.WARNING
            )


# ------------------------------------------------------------------ #
# HuggingFace Embedding Wrapper (PyTorch)
# ------------------------------------------------------------------ #

class HFEmbeddingWrapper:
    """
    Wraps raw HuggingFace AutoModel to match SentenceTransformer.encode() interface.
    Optionally enforces ModelSpec length limits and logs via Darwin.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device,
        max_length: int = None,
        min_length: int = 1,
        normalize: bool = True,
        enable_logging: bool = True,
    ):
        if torch is None:
            raise ImportError("torch is required for HFEmbeddingWrapper. Install the torch plugin.")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.normalize = normalize
        self.enable_logging = enable_logging

        # Use provided max/min, else fall back to model config
        self.max_length = max_length or getattr(tokenizer, "model_max_length", None) or getattr(
            model.config, "max_position_embeddings", None
        )
        self.min_length = min_length

        if self.enable_logging:
            darwin.log(
                f"HF embedding model loaded on {device}",
                LogType.SUCCESS
            )

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        **kwargs,
    ):
        texts = _validate_non_empty_texts(texts)
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            # Token count for warning (approximate: attention_mask sum)
            token_counts = inputs["attention_mask"].sum(dim=1).tolist()
            if self.enable_logging and self.min_length > 1:
                _warn_short_input(batch, token_counts, self.min_length)

            with torch.no_grad():
                outputs = self.model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1)
                if self.normalize:
                    emb = _safe_norm(emb, dim=1)

            embeddings.append(emb.cpu())

        if not embeddings:
            hidden_size = self.model.config.hidden_size
            empty = torch.empty((0, hidden_size))
            return empty.numpy() if convert_to_numpy else empty

        result = torch.cat(embeddings, dim=0)
        return result.numpy() if convert_to_numpy else result

    def __repr__(self):
        return f"HFEmbeddingWrapper(device={self.device})"


# ------------------------------------------------------------------ #
# API Embedding Wrapper
# ------------------------------------------------------------------ #

class APIEmbeddingWrapper:
    """
    Matches SentenceTransformer.encode() interface.
    Works with OpenAI, Cohere, Voyage, and any litellm-supported provider.
    Optionally warns on short inputs using Darwin.
    """

    def __init__(
        self,
        model_id: str,
        timeout: int = 30,
        num_retries: int = 3,
        max_length: int = None,        # not enforced; for logging only
        min_length: int = 1,
        enable_logging: bool = True,
    ):
        if litellm is None:
            raise ImportError("litellm is required for APIEmbeddingWrapper")
        self.model_id = model_id
        self.timeout = timeout
        self.num_retries = num_retries
        self.max_length = max_length
        self.min_length = min_length
        self.enable_logging = enable_logging

        if self.enable_logging:
            darwin.log(
                f"API embedding wrapper ready: {model_id}",
                LogType.SUCCESS
            )

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        **kwargs,
    ) -> np.ndarray:
        texts = _validate_non_empty_texts(texts)

        if not convert_to_numpy:
            warnings.warn(
                "APIEmbeddingWrapper only supports numpy output; "
                "convert_to_numpy=False is ignored.",
                stacklevel=2,
            )

        # Rough length warning (character‑based heuristic)
        if self.enable_logging and self.min_length > 1:
            for idx, text in enumerate(texts):
                if len(text.split()) < self.min_length:
                    darwin.log(
                        f"Input very short ({len(text.split())} words < min {self.min_length}) "
                        f"for batch item {idx}: '{text[:80]}...'",
                        LogType.WARNING,
                    )

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                response = litellm.embedding(
                    model=self.model_id,
                    input=batch,
                    timeout=self.timeout,
                    num_retries=self.num_retries,
                    **kwargs,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Embedding API call failed for model {self.model_id}: {e}"
                ) from e

            try:
                batch_embeddings = [d.embedding for d in response.data]
            except AttributeError as e:
                raise RuntimeError(
                    f"Unexpected response format from embedding API: {e}"
                ) from e

            all_embeddings.extend(batch_embeddings)

        result = np.array(all_embeddings, dtype=np.float32)
        return result

    def __repr__(self):
        return f"APIEmbeddingWrapper(model={self.model_id})"


# ------------------------------------------------------------------ #
# API Reranker Wrapper
# ------------------------------------------------------------------ #

class APIRerankerWrapper:
    """
    Matches CrossEncoder.predict() interface.
    Works with Cohere rerank and any litellm-supported reranker.
    Now includes a per‑query document limit to avoid API rejection.
    """

    def __init__(
        self,
        model_id: str,
        timeout: int = 30,
        num_retries: int = 3,
        max_docs_per_query: int = 1000,
        max_length: int = None,       # informational
        min_length: int = 1,
        enable_logging: bool = True,
    ):
        if litellm is None:
            raise ImportError("litellm is required for APIRerankerWrapper")
        self.model_id = model_id
        self.timeout = timeout
        self.num_retries = num_retries
        self.max_docs_per_query = max_docs_per_query
        self.max_length = max_length
        self.min_length = min_length
        self.enable_logging = enable_logging

        if self.enable_logging:
            darwin.log(
                f"API reranker wrapper ready: {model_id}",
                LogType.SUCCESS,
            )

    def predict(self, pairs: List[Tuple[str, str]], **kwargs) -> np.ndarray:
        if not pairs or not isinstance(pairs, list):
            raise ValueError("pairs must be a non-empty list of (query, document) tuples")

        # Group by query
        query_to_indices: dict[str, list[tuple[int, str]]] = {}
        for idx, (query, document) in enumerate(pairs):
            query_to_indices.setdefault(query, []).append((idx, document))

        scores = [0.0] * len(pairs)

        for query, indexed_docs in query_to_indices.items():
            # Split into chunks if exceeding max_docs_per_query
            for chunk_start in range(0, len(indexed_docs), self.max_docs_per_query):
                chunk = indexed_docs[chunk_start : chunk_start + self.max_docs_per_query]
                indices, documents = zip(*chunk) if chunk else ([], [])

                # Short document warning
                if self.enable_logging and self.min_length > 1:
                    for doc_idx, doc in enumerate(documents):
                        if len(doc.split()) < self.min_length:
                            darwin.log(
                                f"Document short ({len(doc.split())} words < min {self.min_length}) "
                                f"for query '{query[:40]}...' doc {doc_idx}",
                                LogType.WARNING,
                            )

                try:
                    response = litellm.rerank(
                        model=self.model_id,
                        query=query,
                        documents=list(documents),
                        timeout=self.timeout,
                        num_retries=self.num_retries,
                        **kwargs,
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Rerank API call failed for model {self.model_id}: {e}"
                    ) from e

                try:
                    for res, original_idx in zip(response.results, indices):
                        scores[original_idx] = res.relevance_score
                except (AttributeError, IndexError) as e:
                    raise RuntimeError(
                        f"Unexpected response format from rerank API: {e}"
                    ) from e

        return np.array(scores, dtype=np.float32)

    def __repr__(self):
        return f"APIRerankerWrapper(model={self.model_id})"


# ------------------------------------------------------------------ #
# API Vision Wrapper
# ------------------------------------------------------------------ #

class APIVisionWrapper:
    """
    Matches local vision captioning pipeline interface.
    Supports OpenAI, Gemini, OpenRouter vision models via litellm.
    Unpackable as (processor, model, device) for pipeline compatibility.
    """

    def __init__(
        self,
        model_id: str,
        timeout: int = 60,
        num_retries: int = 3,
        enable_logging: bool = True,
    ):
        if litellm is None:
            raise ImportError("litellm is required for APIVisionWrapper")
        self.model_id = model_id
        self.timeout = timeout
        self.num_retries = num_retries
        self.enable_logging = enable_logging

        if self.enable_logging:
            darwin.log(
                f"API vision wrapper ready: {model_id}",
                LogType.SUCCESS,
                
            )

    def caption(self, image_path: Union[str, Path], max_new_tokens: int = 100) -> str:
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(image_path.suffix.lower(), "image/jpeg")

        try:
            response = litellm.completion(
                model=self.model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}"
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Describe this image in one concise sentence. "
                                "Focus on what is visually present — objects, people, "
                                "text, charts, diagrams. Be specific and factual."
                            ),
                        },
                    ],
                }],
                max_tokens=max_new_tokens,
                timeout=self.timeout,
                num_retries=self.num_retries,
            )
        except Exception as e:
            raise RuntimeError(
                f"Vision API call failed for model {self.model_id}: {e}"
            ) from e

        return response.choices[0].message.content.strip()

    def __iter__(self):
        """Allow unpacking as (processor, model, device) for pipeline compatibility."""
        return iter((self, self, "api"))

    def __repr__(self):
        return f"APIVisionWrapper(model={self.model_id})"