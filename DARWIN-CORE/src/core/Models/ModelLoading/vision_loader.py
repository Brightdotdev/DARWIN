"""
src/Models/vision_loader.py
Local vision model loading — dispatch by model_type from config.json.
Requires torch plugin. Fails gracefully if torch is not available.
"""

import json
from pathlib import Path
from typing import Any, Tuple

from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Plugins.PluginManager import PluginRequiredError


def load_vision_model(path: str) -> Tuple[Any, Any, str]:
    """
    Load a local vision model from path.
    Returns (processor, model, device).
    Dispatches by model_type in config.json.
    """
    from src.features.Hardware.DeviceManager import DeviceManager
    from src.features.Hardware.SystemDetector import detect_system

    # Guard torch import — vision models require torch plugin
    try:
        import torch
    except ImportError:
        raise PluginRequiredError(
            "torch",
            "local vision models (SmolVLM, Moondream, etc.)"
        )

    if DeviceManager._config is None:
        DeviceManager.init(detect_system())
    cfg = DeviceManager.get()
    device = cfg.device
    dtype = torch.float16 if cfg.use_gpu else torch.float32

    config_path = Path(path) / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    model_type = config.get("model_type", "").lower()

    try:
        processor, model = _dispatch(model_type, path, dtype)
        model.to(cfg.device)
        darwin.log(f"Vision model loaded on {device}", LogType.INFO)
        return processor, model, device

    except ImportError as e:
        raise ImportError(
            f"Missing required class for '{model_type}' model. "
            f"Update transformers: pip install --upgrade transformers\n{e}"
        )


def _dispatch(model_type: str, path: str, dtype):
    """Dispatch to the correct transformers class by model_type."""
    from transformers import AutoProcessor

    if "idefics" in model_type:
        from transformers import Idefics3ForConditionalGeneration
        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        model = Idefics3ForConditionalGeneration.from_pretrained(
            path,
            dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=None,
            trust_remote_code=True,
        )

    elif "llava" in model_type:
        from transformers import LlavaForConditionalGeneration
        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        model = LlavaForConditionalGeneration.from_pretrained(
            path,
            dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=None,
            trust_remote_code=True,
        )

    elif "moondream" in model_type:
        from transformers import AutoModelForCausalLM
        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=None,
            trust_remote_code=True,
        )

    else:
        from transformers import AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            path,
            dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=None,
            trust_remote_code=True,
        )

    return processor, model