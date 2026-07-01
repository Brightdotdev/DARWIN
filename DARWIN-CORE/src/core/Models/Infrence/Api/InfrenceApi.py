import asyncio
import os
from pathlib import Path
from typing import List, Optional, Union
import base64
from io import BytesIO

from PIL import Image

import litellm
from litellm import aembedding, acompletion, arerank

# Silence LiteLLM logs
os.environ.setdefault("LITELLM_LOG", "ERROR")
os.environ.setdefault("LITELLM_SUPPRESS_LOGGING", "1")
litellm.suppress_debug_info = True
litellm.telemetry = False

from src.lib.KeyManager import is_key_set, load_keys_from_keychain
from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Models.EmbeddingConfig import get_preset_with_spec



# ----------------------------------------------------------------------
# Defaults (can be overridden by env)
# ----------------------------------------------------------------------
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
DEFAULT_VISION_MODEL = "openai/gpt-4o-mini"


# ----------------------------------------------------------------------
# Helper: lookup provider from MODEL_SPECS
# ----------------------------------------------------------------------
def _get_provider(spec) -> Optional[str]:
    """Return provider name from MODEL_SPECS by model_id, or None."""

    if spec and spec.requires_provider:
        return spec.requires_provider
    darwin.log(f"Cannot determine provider for model_id: {spec.model_id}", LogType.ERROR)
    return None


# ----------------------------------------------------------------------
# Helper: image preprocessing
# ----------------------------------------------------------------------
def _image_to_base64_url(
    image: Union[str, Path, bytes, Image.Image],
    max_dimension: int = 1280,
    jpeg_quality: int = 85
) -> str:
    """Convert any image input to a base64 JPEG data URL."""
    if isinstance(image, (str, Path)):
        img = Image.open(image)
    elif isinstance(image, bytes):
        img = Image.open(BytesIO(image))
    elif isinstance(image, Image.Image):
        img = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Convert to RGB
    if img.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        bg.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Resize
    w, h = img.size
    if max(w, h) > max_dimension:
        scale = max_dimension / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    # Encode
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


# ----------------------------------------------------------------------
# Embedding
# ----------------------------------------------------------------------
async def embed(
    texts: List[str],
    model: Optional[str] = None,
    max_batch_size: Optional[int] = None
) -> List[List[float]]:
    """Async embeddings with automatic batching."""
    load_keys_from_keychain(log=False)
    if not texts:
        return []

    model = model or DEFAULT_EMBEDDING_MODEL
    darwin.log(f"Embedding {len(texts)} texts with {model}", LogType.INFO)
    preset, spec  = get_preset_with_spec(model)

    provider = _get_provider(spec)
    if not provider:
        return []
    if not is_key_set(provider):
        darwin.log(f"Embedding skipped: API key for '{provider}' not set", LogType.ERROR)
        return []

    batch_limit = max_batch_size or preset.api_batch_limit or 100

    batches = [texts[i:i + batch_limit] for i in range(0, len(texts), batch_limit)]
    if len(batches) > 1:
        darwin.log(
            f"Embedding {len(texts)} texts in {len(batches)} batches (limit={batch_limit})",
            LogType.DEBUG
        )

    all_embeddings = []
    for batch in batches:
        try:
            response = await aembedding(model=model, input=batch)
            all_embeddings.extend(item["embedding"] for item in response.data)
        except Exception as e:
            darwin.log(f"Embedding batch failed for {model}: {e}", LogType.ERROR)
            raise  # propagate so caller knows it failed

    return all_embeddings


# ----------------------------------------------------------------------
# Vision
# ----------------------------------------------------------------------
async def vision(
    image: Union[str, Path, bytes, Image.Image],
    prompt: str,
    model: Optional[str] = None,
    max_image_size: int = 1280,
    jpeg_quality: int = 85
) -> str:
    """Send image + prompt to a vision model."""
    model = model or DEFAULT_VISION_MODEL
    _, spec  = get_preset_with_spec(model)

    provider = _get_provider(spec)
    if not provider:
        return ""
    if not is_key_set(provider):
        darwin.log(f"Vision skipped: API key for '{provider}' not set", LogType.ERROR)
        return ""

    image_url = _image_to_base64_url(image, max_dimension=max_image_size, jpeg_quality=jpeg_quality)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": prompt}
        ]
    }]

    try:
        response = await acompletion(model=model, messages=messages)
        return response.choices[0].message.content
    except FileNotFoundError as e:
        darwin.log(f"Vision failed: image not found — {e}", LogType.ERROR)
        raise
    except Exception as e:
        darwin.log(f"Vision inference failed for {model}: {e}", LogType.ERROR)
        raise



# ----------------------------------------------------------------------
# Sync wrappers (single entry point, safe for both sync/async contexts)
# ----------------------------------------------------------------------
def _run_sync(coro):
    """Run a coroutine, handling both sync and async contexts."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        raise RuntimeError(
            "Cannot run sync wrapper from async context. "
            "Use the async function directly (embed(), vision(), rerank())."
        )
    return asyncio.run(coro)


def embed_sync(texts: List[str], model: Optional[str] = None, batch_size: Optional[int] = None) -> List[List[float]]:
    return _run_sync(embed(texts, model,batch_size))


def vision_sync(
    image: Union[str, Path, bytes, Image.Image],
    prompt: str,
    model: Optional[str] = None,
    max_image_size: int = 1280,
    jpeg_quality: int = 85
) -> str:
    return _run_sync(vision(image, prompt, model, max_image_size, jpeg_quality))


