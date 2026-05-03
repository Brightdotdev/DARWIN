# src/core/LLM/LLMClient.py

import os
import base64
from pathlib import Path
from typing import Optional, List, Union
import litellm
from dataclasses import dataclass
from src.core.Logger.DarwinLogger import darwin, LogType


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict  # {"prompt_tokens": X, "completion_tokens": Y, "total_tokens": Z}
    raw: object = None


class LLMClient:
    """
    Single interface for any LLM provider via LiteLLM.
    
    Handles:
      - Chat completions (OpenAI, Anthropic, Groq, OpenRouter, etc.)
      - Embeddings (OpenAI, Cohere, Voyage, etc.)
      - Reranking (Cohere, etc.)
      - Vision (GPT-4o, Gemini, Claude, etc.)
    
    Provider format: "provider/model_name"
      "openai/gpt-4o"
      "openrouter/meta-llama/llama-4-maverick"
      "gemini/gemini-2.0-flash"
      "cohere/embed-english-v3.0"
    
    API keys: Set as environment variables (OPENAI_API_KEY, etc.)
    or LiteLLM handles them natively via os.environ.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    # ═══════════════════════════════════════════════════════════════
    # Chat
    # ═══════════════════════════════════════════════════════════════
    
    def chat(
        self,
        messages: List[dict],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Union[LLMResponse, object]:
        """Send a chat completion request. Returns LLMResponse or generator if stream=True."""
        if system:
            messages = [{"role": "system", "content": system}] + messages
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        
        if stream:
            return litellm.completion(**kwargs, stream=True)
        
        try:
            response = litellm.completion(**kwargs)
            
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
            darwin.log(f"LLM chat failed [{self.model}]: {e}", LogType.ERROR)
            raise
    
    def simple(self, prompt: str, system: Optional[str] = None) -> str:
        """Quick one-shot: string in, string out."""
        response = self.chat(
            messages=[{"role": "user", "content": prompt}],
            system=system,
        )
        return response.content
    
    def stream(self, messages: List[dict], system: Optional[str] = None):
        """Stream chunks one at a time."""
        if system:
            messages = [{"role": "system", "content": system}] + messages
        
        response = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content is not None:
                yield delta.content
    
    # ═══════════════════════════════════════════════════════════════
    # Embedding
    # ═══════════════════════════════════════════════════════════════
    
    def embedding(
        self,
        text: Union[str, List[str]],
        model: Optional[str] = None,
    ) -> Union[List[float], List[List[float]]]:
        """
        Get embeddings for text(s).
        
        Args:
            text: Single string or list of strings
            model: Override model (defaults to self.model)
        
        Returns:
            Single embedding vector or list of vectors
        """
        embed_model = model or self.model
        
        response = litellm.embedding(
            model=embed_model,
            input=text,
        )
        
        if isinstance(text, str):
            return response.data[0]["embedding"]
        return [d["embedding"] for d in response.data]
    
    def batch_embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: int = 100,
    ) -> List[List[float]]:
        """Embed large lists in batches."""
        embed_model = model or self.model
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = litellm.embedding(
                model=embed_model,
                input=batch,
            )
            
            all_embeddings.extend([d["embedding"] for d in response.data])
        
        return all_embeddings
    
    # ═══════════════════════════════════════════════════════════════
    # Reranking
    # ═══════════════════════════════════════════════════════════════
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        model: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> List[dict]:
        """
        Rerank documents by relevance to query.
        
        Returns list of dicts: [{"index": 0, "score": 0.95, "document": "..."}, ...]
        """
        rerank_model = model or self.model
        
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
    
    def rerank_scores(
        self,
        query: str,
        documents: List[str],
        model: Optional[str] = None,
    ) -> List[float]:
        """Rerank and return just the scores."""
        results = self.rerank(query, documents, model=model)
        return [r["score"] for r in sorted(results, key=lambda x: x["index"])]
    
    # ═══════════════════════════════════════════════════════════════
    # Vision (image captioning)
    # ═══════════════════════════════════════════════════════════════
    
    def vision(
        self,
        image_path: Union[str, Path],
        prompt: str = "Describe this image in one concise sentence.",
        max_tokens: int = 100,
    ) -> str:
        """
        Describe an image via vision-capable model.
        
        Supports: GPT-4o, Gemini, Claude (via OpenRouter), etc.
        """
        image_path = Path(image_path)
        
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Detect MIME type
        ext = image_path.suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(ext, "image/jpeg")
        
        response = litellm.completion(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_data}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }],
            max_tokens=max_tokens,
        )
        
        return response.choices[0].message.content.strip()
    
    # ═══════════════════════════════════════════════════════════════
    # Utility
    # ═══════════════════════════════════════════════════════════════
    
    def __repr__(self):
        return f"LLMClient(model={self.model})"
    
    @staticmethod
    def check_provider() -> str:
        """Quick check — which provider is configured?"""
        providers = {
            "OPENAI_API_KEY": "openai",
            "OPENROUTER_API_KEY": "openrouter",
            "ANTHROPIC_API_KEY": "anthropic",
            "COHERE_API_KEY": "cohere",
            "GEMINI_API_KEY": "gemini",
            "GROQ_API_KEY": "groq",
            "TOGETHER_API_KEY": "together",
            "VOYAGE_API_KEY": "voyage",
        }
        
        found = []
        for env_var, name in providers.items():
            if os.getenv(env_var):
                found.append(name)
        
        return ", ".join(found) if found else "none"


# ═══════════════════════════════════════════════════════════════
# Embedding-specific factory
# ═══════════════════════════════════════════════════════════════

def get_embedding_client(provider: str = "openai") -> LLMClient:
    """Quick factory for embedding-specific models."""
    models = {
        "openai": "openai/text-embedding-3-small",
        "cohere": "cohere/embed-english-v3.0",
        "voyage": "voyage/voyage-3-lite",
    }
    
    return LLMClient(model=models.get(provider, "openai/text-embedding-3-small"))


# ═══════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Available providers:", LLMClient.check_provider())
    
    if os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY"):
        client = LLMClient(model="openai/text-embedding-3-small")
        
        # Test embedding
        vec = client.embedding("Hello world")
        print(f"Embedding shape: {len(vec)}")
        
        # Test chat
        response = client.simple("Say hello in one word")
        print(f"Chat: {response}")
        
        # Test rerank
        scores = client.rerank_scores(
            "What is AI?",
            ["AI is artificial intelligence", "Bananas are yellow", "Machine learning is a subset of AI"]
        )
        print(f"Rerank scores: {scores}")
    
    print("✅ LLMClient ready")