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
                return json.load(f)
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
    max_new_tokens: int = 40,
) -> str:
    import torch
    from PIL import Image as PILImage

    image_path = Path(image_path)
    cache_key  = image_path.name
    cache      = cache if cache is not None else {}

    if CACHE_CAPTIONS and cache_key in cache:
        return cache[cache_key]

    result = ""
    try:
        darwin.log(f"Captioning {image_path.name}", LogType.INFO)

        image = preloaded_image or PILImage.open(image_path).convert("RGB")

        model_type = model.config.model_type.lower() if hasattr(model.config, "model_type") else ""

        if "idefics" in model_type:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe this image in detail."},
                    ],
                }
            ]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                )

        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text":
                            "Describe what this image shows in one or two sentences. "
                            "Focus on the content and any visible text. "
                            "If it is a diagram or chart, describe what it illustrates."
                        },
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

        if len(result) > 300 or "\n\nDescribe" in result:
            result = result.rsplit("\n", 1)[-1].strip()

    except Exception as e:
        darwin.log(f"Caption failed for {image_path.name}: {e}", LogType.ERROR)
        result = ""

    # Always write to cache — even on failure (empty string)
    if CACHE_CAPTIONS:
        cache[cache_key] = result

    return result