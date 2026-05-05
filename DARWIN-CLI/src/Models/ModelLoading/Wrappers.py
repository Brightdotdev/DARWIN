"""
src/Models/wrappers.py
API and HF model wrappers — match local model interfaces exactly.
"""

import numpy as np
from pathlib import Path
from typing import Any


class HFEmbeddingWrapper:
    """Wraps raw HuggingFace AutoModel to match SentenceTransformer.encode() interface."""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def encode(self, texts, batch_size=32, convert_to_numpy=True, **kwargs):
        import torch

        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1)
                emb = emb / emb.norm(dim=1, keepdim=True)

            embeddings.append(emb.cpu())

        result = torch.cat(embeddings, dim=0)
        return result.numpy() if convert_to_numpy else result

    def __repr__(self):
        return f"HFEmbeddingWrapper(device={self.device})"


class APIEmbeddingWrapper:
    """
    Matches SentenceTransformer.encode() interface.
    Works with OpenAI, Cohere, Voyage, and any litellm-supported provider.
    """

    def __init__(self, client, model_id: str):
        self.client = client
        self.model_id = model_id

    def encode(self, texts, batch_size=32, convert_to_numpy=True, **kwargs):
        import litellm

        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = litellm.embedding(model=self.model_id, input=batch)
            all_embeddings.extend([d["embedding"] for d in response.data])

        result = np.array(all_embeddings, dtype=np.float32)
        return result if convert_to_numpy else result.tolist()

    def __repr__(self):
        return f"APIEmbeddingWrapper(model={self.model_id})"


class APIRerankerWrapper:
    """
    Matches CrossEncoder.predict() interface.
    Works with Cohere rerank and any litellm-supported reranker.
    """

    def __init__(self, client, model_id: str):
        self.client = client
        self.model_id = model_id

    def predict(self, pairs, **kwargs):
        import litellm

        scores = []
        for query, document in pairs:
            response = litellm.rerank(
                model=self.model_id,
                query=query,
                documents=[document],
            )
            scores.append(response.results[0].relevance_score)

        return scores

    def __repr__(self):
        return f"APIRerankerWrapper(model={self.model_id})"


class APIVisionWrapper:
    """
    Matches local vision captioning pipeline interface.
    Supports OpenAI, Gemini, OpenRouter vision models via litellm.
    Unpackable as (processor, model, device) for pipeline compatibility.
    """

    def __init__(self, client, model_id: str):
        self.client = client
        self.model_id = model_id

    def caption(self, image_path, max_new_tokens: int = 100) -> str:
        """Generate a caption for an image via API."""
        import base64
        import litellm

        image_path = Path(image_path)

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        mime_map = {
            ".jpg":  "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png":  "image/png",
            ".gif":  "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(image_path.suffix.lower(), "image/jpeg")

        response = litellm.completion(
            model=self.model_id,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_data}"}
                    },
                    {
                        "type": "text",
                        "text": (
                            "Describe this image in one concise sentence. "
                            "Focus on what is visually present — objects, people, "
                            "text, charts, diagrams. Be specific and factual."
                        )
                    }
                ]
            }],
            max_tokens=max_new_tokens,
        )

        return response.choices[0].message.content.strip()

    def __iter__(self):
        """Allow unpacking as (processor, model, device) for pipeline compatibility."""
        return iter((self, self, "api"))

    def __repr__(self):
        return f"APIVisionWrapper(model={self.model_id})"