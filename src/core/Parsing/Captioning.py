from pathlib import Path
from typing import Dict
import json
from src.lib.Terminal import TerminalUI


# data stuff
CACHE_CAPTIONS = True


# ============================================================================
# Caption cache
# ============================================================================

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
            TerminalUI.error(f"Failed to load caption cache: {e}")
    return {}


def save_caption_cache(images_dir: Path, cache: Dict[str, str]) -> None:
    if not CACHE_CAPTIONS:
        return
    try:
        with open(get_caption_cache_path(images_dir), "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        TerminalUI.warning(f"Failed to save caption cache: {e}")


def migrate_caption_cache(images_dir: Path) -> int:
    """Migrate absolute-path cache keys to bare filenames. Safe to call repeatedly."""
    cache    = load_caption_cache(images_dir)
    migrated = {Path(k).name: v for k, v in cache.items()}
    count    = sum(1 for k in cache if Path(k).name != k)
    if count > 0:
        save_caption_cache(images_dir, migrated)
        TerminalUI.info(label="Caption Progress", value=f"Migrated {count} caption cache keys to bare filenames")
    return count


# ============================================================================
# Caption model
# ============================================================================

def caption_model_available() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def load_caption_model(model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct"):
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as ModelClass
    except ImportError:
        from transformers import AutoModelForVision2Seq as ModelClass

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32

    processor = AutoProcessor.from_pretrained(model_id)
    model     = ModelClass.from_pretrained(
        model_id, dtype=dtype, low_cpu_mem_usage=True, device_map="auto"
    )
    return processor, model, device


def caption_image(
    image_path: Path,
    processor,
    model,
    cache: Dict[str, str],
) -> str:
    """Caption one image. Cache key is bare filename — never absolute."""
    import torch

    cache_key = image_path.name
    if CACHE_CAPTIONS and cache_key in cache:
        return cache[cache_key]

    try:
        from PIL import Image as PILImage

        TerminalUI.dim(f"Captioning {image_path.name}")

        image    = PILImage.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text":
                    "Describe what this image shows in one or two sentences. "
                    "Focus on the content and any visible text. "
                    "If it is a diagram or chart, describe what it illustrates."
                },
            ],
        }]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=80, do_sample=False)

        decoded = processor.decode(output[0], skip_special_tokens=True)
        result  = decoded.split("Assistant:")[-1].strip() if "Assistant:" in decoded else decoded.strip()

        if CACHE_CAPTIONS:
            cache[cache_key] = result
        return result

    except Exception as e:
        TerminalUI.error(f"Caption failed for {image_path.name}: {e}")
        return f"image (caption failed: {e})"