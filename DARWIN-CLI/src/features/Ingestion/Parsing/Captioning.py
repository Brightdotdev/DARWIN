# ============================================================================
# src/core/Parsing/Captioning.py
# ============================================================================

import re
from pathlib import Path
from typing import Dict
import json

from src.core.Logger.DarwinLogger import darwin, LogType

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
                # Auto-heal old poisoned caches — discard empty/falsy entries
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
    max_new_tokens: int = 100,  # CHANGED: bumped from 40 for complex diagrams
) -> str:
    import torch
    from PIL import Image as PILImage

    image_path = Path(image_path)
    cache_key  = image_path.name
    cache      = cache if cache is not None else {}

    # Truthy check — ignores old poisoned cache entries that are ""
    if CACHE_CAPTIONS and cache.get(cache_key):
        return cache[cache_key]

    result = ""
    try:
        darwin.log(f"Captioning {image_path.name}", LogType.INFO)

        image = preloaded_image or PILImage.open(image_path).convert("RGB")

        model_type = model.config.model_type.lower() if hasattr(model.config, "model_type") else ""

        # CHANGED: unified concise prompt for all model types
        concise_prompt = (
            "Describe this image in one concise sentence. "
            "Focus only on what is visually present — objects, people, text, charts, diagrams. "
            "Be specific and factual. Keep it under 20 words."
        )

        if "idefics" in model_type:
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

        else:
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

        parts  = re.split(r"(?i)assistant\s*:?", decoded)
        result = parts[-1].strip() if parts else decoded.strip()

        # CHANGED: stricter post-processing — truncate at first sentence break
        # and enforce single-line output
        result = _truncate_to_first_sentence(result)

    except Exception as e:
        darwin.log(f"Caption failed for {image_path.name}: {e}", LogType.ERROR)
        result = ""

    # Only cache successful captions — failures are retried next run
    if CACHE_CAPTIONS and result:
        cache[cache_key] = result

    return result


def _truncate_to_first_sentence(text: str) -> str:
    """Truncate at first sentence-ending punctuation, collapse whitespace."""
    if not text:
        return ""
    # Find first sentence terminator that's not inside a number (e.g., 1.5)
    for i, char in enumerate(text):
        if char in ".!?" and i > 0:
            # Check it's not part of an abbreviation or number
            if i + 1 < len(text) and text[i + 1].isdigit():
                continue
            text = text[:i + 1]
            break
    # Collapse all whitespace to single spaces
    text = " ".join(text.split())
    return text.strip()