import asyncio
import os
from pathlib import Path
from typing import List, Optional, Union
import base64
from io import BytesIO

from PIL import Image

import litellm
from litellm import aembedding, acompletion

# Silence LiteLLM logs
os.environ.setdefault("LITELLM_LOG", "ERROR")
os.environ.setdefault("LITELLM_SUPPRESS_LOGGING", "1")
litellm.suppress_debug_info = True
litellm.telemetry = False

from lib.KeyManager import is_key_set, load_keys_from_keychain
from core.Logger.DarwinLogger import darwin
from core.Models.EmbeddingConfig import get_preset_with_spec



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
    darwin.error(f"Cannot determine provider for {spec.model_id}")
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
        darwin.error(f"Unsupported image type: {type(image)}")
        return ""

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
    max_batch_size: Optional[int] = None,
    max_retries: int = 3,
    inter_batch_delay: float = 0.15,
) -> List[List[float]]:
    """Async embeddings with automatic batching, retry, and rate-limit pacing."""
    import random

    load_keys_from_keychain(log=False)
    if not texts:
        return []

    model = model or DEFAULT_EMBEDDING_MODEL
    darwin.info(f"Embedding {len(texts)} texts using {model}")
    preset, spec = get_preset_with_spec(model)

    provider = _get_provider(spec)
    if not provider:
        return []
    if not is_key_set(provider):
        darwin.error(f"Embedding skipped: API key for '{provider}' not set")
        return []

    batch_limit = max_batch_size or preset.api_batch_limit or 100

    batches = [texts[i:i + batch_limit] for i in range(0, len(texts), batch_limit)]
    if len(batches) > 1:
        darwin.debug(f"Embedding {len(texts)} texts in {len(batches)} batches (limit={batch_limit})")

    all_embeddings = []
    for batch_idx, batch in enumerate(batches):
        last_exc = None
        for attempt in range(max_retries + 1):
            try:
                response = await aembedding(model=model, input=batch)
                all_embeddings.extend(item["embedding"] for item in response.data)
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                is_rate_limit = "rate" in str(e).lower() or "429" in str(e)
                if attempt < max_retries:
                    wait = (2 ** attempt) + random.uniform(0, 0.5 * (2 ** attempt))
                    if is_rate_limit:
                        wait = max(wait, 2.0)
                    darwin.warning(
                        f"Embedding batch {batch_idx+1}/{len(batches)} failed "
                        f"(attempt {attempt+1}/{max_retries+1}): {e} — retrying in {wait:.1f}s"
                    )
                    await asyncio.sleep(wait)
                else:
                    darwin.error(
                        f"Embedding batch {batch_idx+1}/{len(batches)} failed after "
                        f"{max_retries+1} attempts: {e}"
                    )

        if last_exc is not None:
            darwin.error(f"Embedding aborted: all batches failed")
            return []

        if batch_idx < len(batches) - 1:
            await asyncio.sleep(inter_batch_delay)

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
        darwin.error(f"Vision skipped: API key for '{provider}' not set")
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
        darwin.error(f"Vision failed: image not found — {e}")
        return ""
    except Exception as e:
        darwin.error(f"Vision inference failed for {model}: {e}")
        return ""



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
            "Cannot call embed_sync/vision_sync from an async context. "
            "Use the async function directly (embed(), vision())."
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


