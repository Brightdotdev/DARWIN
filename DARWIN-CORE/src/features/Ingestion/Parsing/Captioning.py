# ============================================================================
# src/core/Parsing/Captioning.py
# ============================================================================

import re
from pathlib import Path
from typing import Dict
import json

from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Plugins.TorchPlugin import TorchPlugin

# ── Constants ──────────────────────────────────────────────────────────────────

CACHE_CAPTIONS = True

# ═══════════════════════════════════════════════════════════════════════════
# Caption cache helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_caption_cache_path(images_dir: Path) -> Path:
    return images_dir / ".captions_cache.json"


def load_caption_cache(images_dir: Path) -> Dict[str, str]:
    """Keys are bare filenames — never absolute paths."""
    if not CACHE_CAPTIONS:
        return {}
    cache_path = get_caption_cache_path(images_dir)
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                return {
                    k: v.strip()
                    for k, v in raw.items()
                    if isinstance(v, str) and v.strip()
                }
        except Exception as e:
            darwin.log(f"Failed to load caption cache: {e}", LogType.ERROR)
    return {}


def save_caption_cache(images_dir: Path, cache: Dict[str, str]) -> None:
    if not CACHE_CAPTIONS:
        return
    try:
        with open(get_caption_cache_path(images_dir), "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        darwin.log(f"Failed to save caption cache: {e}", LogType.WARNING)


# ═══════════════════════════════════════════════════════════════════════════
# Caption one image
# ═══════════════════════════════════════════════════════════════════════════

def caption_image(
    image_path: "str | Path",
    processor,
    model,
    device,
    cache: Dict[str, str] = None,
    preloaded_image=None,
    max_new_tokens: int = 100,
) -> str:
    """
    Local vision captioning. Requires torch plugin.
    If torch is not available, this should not be called — use APIVisionWrapper instead.
    """
    # Guard torch import at function level
    try:
        import torch
        from PIL import Image as PILImage
    except ImportError:
        darwin.log("Torch not available for captioning. Install torch plugin.", LogType.ERROR)
        return ""

    image_path = Path(image_path)
    cache_key = image_path.name
    cache = cache if cache is not None else {}

    if CACHE_CAPTIONS and cache.get(cache_key):
        return cache[cache_key]

    result = ""
    try:
        darwin.log(f"Captioning {image_path.name}", LogType.INFO)

        image = preloaded_image or PILImage.open(image_path).convert("RGB")

        model_type = model.config.model_type.lower() if hasattr(model.config, "model_type") else ""

        concise_prompt = (
            "Describe this image in one concise sentence. "
            "Focus only on what is visually present — objects, people, text, charts, diagrams. "
            "Be specific and factual. Keep it under 20 words."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": concise_prompt},
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        decoded = processor.decode(output[0], skip_special_tokens=True)

        parts = re.split(r"(?i)assistant\s*:?", decoded)
        result = parts[-1].strip() if parts else decoded.strip()

        result = _truncate_to_first_sentence(result)

    except Exception as e:
        darwin.log(f"Caption failed for {image_path.name}: {e}", LogType.ERROR)
        result = ""

    if CACHE_CAPTIONS and result:
        cache[cache_key] = result

    return result


def _truncate_to_first_sentence(text: str) -> str:
    """Truncate at first sentence-ending punctuation, collapse whitespace."""
    if not text:
        return ""
    for i, char in enumerate(text):
        if char in ".!?" and i > 0:
            if i + 1 < len(text) and text[i + 1].isdigit():
                continue
            text = text[:i + 1]
            break
    text = " ".join(text.split())
    return text.strip()