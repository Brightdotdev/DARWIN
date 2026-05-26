"""
src/Models/wrappers.py
API and HF model wrappers — match local model interfaces exactly.
"""

import base64
import warnings
from pathlib import Path
from typing import  List, Tuple, Union
import logging
import numpy as np

import warnings
warnings.filterwarnings("ignore", message=".*bedrock-runtime.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Bedrock.*", category=UserWarning)
try:
    import torch
except ImportError:
    torch = None

try:
    import litellm
except ImportError:
    litellm = None



litellm.suppress_debug_info = True
litellm.telemetry = False
litellm.turn_off_message_logging = True

# Set root logger to ERROR to suppress verbose output
logging.getLogger("litellm").setLevel(logging.ERROR)


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


class HFEmbeddingWrapper:
    """Wraps raw HuggingFace AutoModel to match SentenceTransformer.encode() interface."""

    def __init__(self, model, tokenizer, device):
        if torch is None:
            raise ImportError("torch is required for HFEmbeddingWrapper")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        # Explicit max length from tokenizer/model config avoids silent truncation
        self.max_length = getattr(tokenizer, "model_max_length", None) or getattr(
            model.config, "max_position_embeddings", None
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

            with torch.no_grad():
                outputs = self.model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1)
                emb = _safe_norm(emb, dim=1)

            embeddings.append(emb.cpu())

        if not embeddings:
            # Empty input case – return shape (0, hidden_size)
            # We need hidden_size from model; fallback to 0 dimensions
            hidden_size = self.model.config.hidden_size
            empty = torch.empty((0, hidden_size))
            return empty.numpy() if convert_to_numpy else empty

        result = torch.cat(embeddings, dim=0)
        return result.numpy() if convert_to_numpy else result

    def __repr__(self):
        return f"HFEmbeddingWrapper(device={self.device})"


class APIEmbeddingWrapper:
    """
    Matches SentenceTransformer.encode() interface.
    Works with OpenAI, Cohere, Voyage, and any litellm-supported provider.
    """

    def __init__(self, model_id: str, timeout: int = 30, num_retries: int = 3):
        if litellm is None:
            raise ImportError("litellm is required for APIEmbeddingWrapper")
        self.model_id = model_id
        self.timeout = timeout
        self.num_retries = num_retries

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

            # litellm returns objects with .embedding attribute, not dicts
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


class APIRerankerWrapper:
    """
    Matches CrossEncoder.predict() interface.
    Works with Cohere rerank and any litellm-supported reranker.
    """

    def __init__(self, model_id: str, timeout: int = 30, num_retries: int = 3):
        if litellm is None:
            raise ImportError("litellm is required for APIRerankerWrapper")
        self.model_id = model_id
        self.timeout = timeout
        self.num_retries = num_retries

    def predict(self, pairs: List[Tuple[str, str]], **kwargs) -> np.ndarray:
        if not pairs or not isinstance(pairs, list):
            raise ValueError("pairs must be a non-empty list of (query, document) tuples")

        # Group by query to batch documents in a single API call per unique query
        query_to_indices: dict[str, list[tuple[int, str]]] = {}
        for idx, (query, document) in enumerate(pairs):
            query_to_indices.setdefault(query, []).append((idx, document))

        # scores array maintains original order
        scores = [0.0] * len(pairs)
        for query, indexed_docs in query_to_indices.items():
            indices, documents = zip(*indexed_docs) if indexed_docs else ([], [])

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
                # response.results is ordered as the input documents
                for res, original_idx in zip(response.results, indices):
                    scores[original_idx] = res.relevance_score
            except (AttributeError, IndexError) as e:
                raise RuntimeError(
                    f"Unexpected response format from rerank API: {e}"
                ) from e

        return np.array(scores, dtype=np.float32)

    def __repr__(self):
        return f"APIRerankerWrapper(model={self.model_id})"


class APIVisionWrapper:
    """
    Matches local vision captioning pipeline interface.
    Supports OpenAI, Gemini, OpenRouter vision models via litellm.
    Unpackable as (processor, model, device) for pipeline compatibility.
    """

    def __init__(self, model_id: str, timeout: int = 60, num_retries: int = 3):
        if litellm is None:
            raise ImportError("litellm is required for APIVisionWrapper")
        self.model_id = model_id
        self.timeout = timeout
        self.num_retries = num_retries

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