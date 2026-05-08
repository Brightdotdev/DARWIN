from __future__ import annotations

import os
import base64
import json
from pathlib import Path
from typing import Optional, List, Union
import litellm
from dataclasses import dataclass

from src.features.LLM.LLMConfig import LLMClientConfig, TierConfig
from src.core.Logger.DarwinLogger import darwin, LogType


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict
    raw: object = None


class APiClient:
    """
    Unified LLM interface for Darwin.

    Two tiers:
      - call_small(): Fast, cheap, low-latency — routing, classification
      - call_smart(): Capable, reasoning — synthesis, final answers

    Config injected via LLMClientConfig (from LLMManager).
    Handles rate-limit auto-rotation via LLMManager callback.
    """

    def __init__(self, config: LLMClientConfig):
        self.config = config
        self._validate_or_warn()

    def _validate_or_warn(self) -> None:
        errors = self.config.validate()
        for err in errors:
            darwin.log(f"[LLMClient] Config issue: {err}", LogType.WARNING)

    def _get_litellm_model(self, tier_config: TierConfig) -> str:
        model = tier_config.model
        if "/" in model:
            return model
        provider = self.config.provider
        if provider == "ollama":
            return f"ollama/{model}"
        return f"{provider}/{model}"

    def _build_chat_kwargs(self, tier_config: TierConfig) -> dict:
        kwargs = {
            "temperature": tier_config.temperature,
            "max_tokens": tier_config.max_tokens,
        }
        if tier_config.extra_body:
            kwargs["extra_body"] = tier_config.extra_body
        if self.config.provider == "ollama":
            os.environ["OLLAMA_API_BASE"] = self.config.ollama_base_url
        return kwargs

    def _call_with_rotation(
        self,
        messages: List[dict],
        tier_config: TierConfig,
        system: Optional[str] = None,
        stream: bool = False,
        _retry_count: int = 0,
    ) -> Union[LLMResponse, object]:
        """Internal: call with auto-rotation on rate limit."""
        if system:
            messages = [{"role": "system", "content": system}] + messages

        model = self._get_litellm_model(tier_config)
        kwargs = self._build_chat_kwargs(tier_config)

        darwin.log(f"[LLMClient] call → {model}", LogType.DEBUG)

        try:
            if stream:
                return litellm.completion(
                    model=model,
                    messages=messages,
                    stream=True,
                    **kwargs,
                )

            response = litellm.completion(
                model=model,
                messages=messages,
                **kwargs,
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                raw=response,
            )

        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = any(x in error_str for x in ["429", "rate limit", "too many requests"])

            if is_rate_limit and _retry_count < 3:
                # Try to rotate via manager
                try:
                    from src.features.LLM.LLMManager import LLMManager
                    rotated = LLMManager.handle_rate_limit(self.config.provider)
                    if rotated:
                        darwin.log(f"[LLMClient] Retrying after key rotation ({_retry_count + 1}/3)", LogType.WARNING)
                        return self._call_with_rotation(
                            messages=messages,
                            tier_config=tier_config,
                            system=system,
                            stream=stream,
                            _retry_count=_retry_count + 1,
                        )
                except Exception:
                    pass  # Manager not available, fail normally

            darwin.log(f"LLM call failed [{model}]: {e}", LogType.ERROR)
            raise

    # ═══════════════════════════════════════════════════════════════════════
    # PUBLIC TIER API
    # ═══════════════════════════════════════════════════════════════════════

    def call_small(
        self,
        messages: List[dict],
        system: Optional[str] = None,
        stream: bool = False,
    ) -> Union[LLMResponse, object]:
        return self._call_with_rotation(
            messages=messages,
            tier_config=self.config.fast,
            system=system,
            stream=stream,
        )

    def call_smart(
        self,
        messages: List[dict],
        system: Optional[str] = None,
        stream: bool = False,
    ) -> Union[LLMResponse, object]:
        return self._call_with_rotation(
            messages=messages,
            tier_config=self.config.smart,
            system=system,
            stream=stream,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Convenience helpers
    # ═══════════════════════════════════════════════════════════════════════

    def ask_small(self, prompt: str, system: Optional[str] = None) -> str:
        response = self.call_small(
            messages=[{"role": "user", "content": prompt}],
            system=system,
        )
        return response.content

    def ask_smart(self, prompt: str, system: Optional[str] = None) -> str:
        response = self.call_smart(
            messages=[{"role": "user", "content": prompt}],
            system=system,
        )
        return response.content

    def stream_small(self, messages: List[dict], system: Optional[str] = None):
        response = self.call_small(messages, system=system, stream=True)
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content is not None:
                yield delta.content

    def stream_smart(self, messages: List[dict], system: Optional[str] = None):
        response = self.call_smart(messages, system=system, stream=True)
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content is not None:
                yield delta.content

    # ═══════════════════════════════════════════════════════════════════════
    # Embedding
    # ═══════════════════════════════════════════════════════════════════════

    def embedding(
        self,
        text: Union[str, List[str]],
        model: Optional[str] = None,
    ) -> Union[List[float], List[List[float]]]:
        embed_model = model or self.config.smart.model
        if "/" not in embed_model:
            embed_model = f"{self.config.provider}/{embed_model}"

        response = litellm.embedding(model=embed_model, input=text)

        if isinstance(text, str):
            return response.data[0]["embedding"]
        return [d["embedding"] for d in response.data]

    def batch_embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: int = 100,
    ) -> List[List[float]]:
        embed_model = model or self.config.smart.model
        if "/" not in embed_model:
            embed_model = f"{self.config.provider}/{embed_model}"

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = litellm.embedding(model=embed_model, input=batch)
            all_embeddings.extend([d["embedding"] for d in response.data])
        return all_embeddings

    # ═══════════════════════════════════════════════════════════════════════
    # Reranking
    # ═══════════════════════════════════════════════════════════════════════

    def rerank(
        self,
        query: str,
        documents: List[str],
        model: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> List[dict]:
        rerank_model = model or self.config.smart.model
        if "/" not in rerank_model:
            rerank_model = f"{self.config.provider}/{rerank_model}"

        response = litellm.rerank(
            model=rerank_model,
            query=query,
            documents=documents,
            top_n=top_n,
        )
        return [
            {
                "index": r.index,
                "score": r.relevance_score,
                "document": r.document.get("text", "") if hasattr(r.document, "get") else str(r.document),
            }
            for r in response.results
        ]

    # ═══════════════════════════════════════════════════════════════════════
    # Vision
    # ═══════════════════════════════════════════════════════════════════════

    def vision(
        self,
        image_path: Union[str, Path],
        prompt: str = "Describe this image in one concise sentence.",
        max_tokens: int = 100,
    ) -> str:
        image_path = Path(image_path)
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        ext = image_path.suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
        }
        mime_type = mime_map.get(ext, "image/jpeg")

        response = litellm.completion(
            model=self._get_litellm_model(self.config.smart),
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def __repr__(self):
        return f"LLMClient(provider={self.config.provider}, fast={self.config.fast.model}, smart={self.config.smart.model})"