# src/core/Models/ModelInference/SmolVLMOnnxInference.py
"""
SmolVLM ONNX Inference Engine – vision-language model.
Implements the unified VisionInferenceInterface.
"""

import gc
import time
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import numpy as np
from PIL import Image

from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Models.ModeInterfaces import (
    VisionInferenceInterface,
    UnifiedVisionMetadata,
    UnifiedVisionResult,
)


class SmolVLMOnnxInference(VisionInferenceInterface):
    """
    SmolVLM ONNX inference engine.
    Loads decoder + vision + embed ONNX models and runs image-to-text generation.
    
    Expects model directory layout from SmolVLMOnnxDownloader:
        model_dir/
            decoder_model_merged_int8.onnx  (or fp32 variant)
            vision_encoder_fp32.onnx        (or other precision)
            embed_tokens_int8.onnx          (or other precision)
            tokenizer.json
            config.json
            preprocessor_config.json
            tokenizer_config.json
    """

    def __init__(
        self,
        model_id: str,  # path to model dir
        revision: str = "9181b0e",
        decoder_precision: str = "int8",
        vision_precision: str = "fp32",
        embed_precision: str = "int8",
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        super().__init__(model_id=model_id)
        self.model_dir = Path(model_id)
        self.revision = revision
        self.decoder_precision = decoder_precision
        self.vision_precision = vision_precision
        self.embed_precision = embed_precision
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self._decoder_session = None
        self._vision_session = None
        self._embed_session = None
        self._tokenizer = None
        self._preprocessor = None
        self._device = "CPU"
        self._onnx_runtime_version = "unknown"

    # ── Darwin logging ───────────────────────────────────────────────────────

    def _log(self, msg: str, level: LogType = LogType.INFO):
        darwin.log(f"[SmolVLM Inference] {msg}", level)

    # ── Context Manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "SmolVLMOnnxInference":
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.unload()

    # ── Lifecycle (Interface) ────────────────────────────────────────────────

    def load(self) -> "SmolVLMOnnxInference":
        """Load all ONNX sessions and tokenizer. Chainable."""
        if self.is_loaded():
            self._log("Already loaded, skipping.")
            return self

        if not self.model_dir.exists():
            self._log(f"Model directory not found: {self.model_dir}", LogType.ERROR)
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        if not self._check_model_exists():
            self._log("Model files missing. Run downloader first.", LogType.ERROR)
            raise FileNotFoundError("Model files missing. Run downloader first.")

        if not self._check_integrity():
            self._log("Model integrity check failed. Model may be corrupted.", LogType.ERROR)
            raise RuntimeError("Model integrity check failed. Model may be corrupted.")

        self._log(f"Loading SmolVLM from {self.model_dir}")
        self._load_tokenizer()
        self._load_preprocessor()
        self._load_onnx_sessions()
        self._log("SmolVLM loaded successfully.", LogType.SUCCESS)
        return self

    def unload(self) -> None:
        """Unload all sessions and free memory."""
        if not self.is_loaded():
            self._log("Not loaded, nothing to unload.")
            return

        self._log("Unloading SmolVLM...")
        self._decoder_session = None
        self._vision_session = None
        self._embed_session = None
        self._tokenizer = None
        self._preprocessor = None
        gc.collect()
        self._log("Unloaded.", LogType.SUCCESS)

    def is_loaded(self) -> bool:
        return all(s is not None for s in (
            self._decoder_session,
            self._vision_session,
            self._embed_session,
            self._tokenizer,
        ))

    def _check_model_exists(self) -> bool:
        suffix_decoder = "" if self.decoder_precision == "fp32" else f"_{self.decoder_precision}"
        suffix_vision = "" if self.vision_precision == "fp32" else f"_{self.vision_precision}"
        suffix_embed = "" if self.embed_precision == "fp32" else f"_{self.embed_precision}"

        decoder = self.model_dir / f"decoder_model_merged{suffix_decoder}.onnx"
        vision = self.model_dir / f"vision_encoder{suffix_vision}.onnx"
        embed = self.model_dir / f"embed_tokens{suffix_embed}.onnx"

        return decoder.exists() and vision.exists() and embed.exists()

    def _check_integrity(self) -> bool:
        suffix_decoder = "" if self.decoder_precision == "fp32" else f"_{self.decoder_precision}"
        suffix_vision = "" if self.vision_precision == "fp32" else f"_{self.vision_precision}"
        suffix_embed = "" if self.embed_precision == "fp32" else f"_{self.embed_precision}"

        required = [
            self.model_dir / f"decoder_model_merged{suffix_decoder}.onnx",
            self.model_dir / f"vision_encoder{suffix_vision}.onnx",
            self.model_dir / f"embed_tokens{suffix_embed}.onnx",
            self.model_dir / "tokenizer.json",
            self.model_dir / "config.json",
        ]

        min_sizes = {
            "decoder_model_merged": 50_000_000,
            "vision_encoder": 10_000_000,
            "embed_tokens": 5_000_000,
            "tokenizer.json": 100,
            "config.json": 100,
        }

        for p in required:
            if not p.exists():
                self._log(f"Integrity fail: missing {p.name}", LogType.WARNING)
                return False
            for prefix, min_size in min_sizes.items():
                if prefix in p.name:
                    if p.stat().st_size < min_size:
                        self._log(f"Integrity fail: {p.name} too small ({p.stat().st_size} bytes)", LogType.WARNING)
                        return False
                    break
        return True

    # ── Inference (Interface) ──────────────────────────────────────────────────

    def generate(
        self,
        image: Union[str, Path, bytes, np.ndarray, Image.Image],
        prompt: str = "Describe this image.",
        max_new_tokens: Optional[int] = None,
    ) -> UnifiedVisionResult:
        """
        Generate text from image + prompt.
        """
        if not self.is_loaded():
            self._log("Model not loaded. Call .load() first.", LogType.ERROR)
            raise RuntimeError("Model not loaded. Call .load() first.")

        img = self._load_image(image)
        img_size = img.size

        self._log(f"Generating from image {img_size} with prompt: '{prompt[:50]}...'")

        spinner_id = f"smolvlm_{id(img)}"
        darwin.spinner_start(spinner_id, "Running SmolVLM inference")

        start = time.perf_counter()

        # Preprocess image → vision features
        pixel_values = self._preprocess_image(img)

        # Encode image with vision encoder
        vision_outputs = self._vision_session.run(None, {"pixel_values": pixel_values})
        image_features = vision_outputs[0]

        # Tokenize prompt
        prompt_tokens = self._tokenizer.encode(prompt, return_tensors="np")
        prompt_ids = prompt_tokens["input_ids"]

        # Embed prompt tokens
        embed_outputs = self._embed_session.run(None, {"input_ids": prompt_ids})
        prompt_embeds = embed_outputs[0]

        # Concatenate image features + prompt embeddings
        inputs_embeds = np.concatenate([image_features, prompt_embeds], axis=1)

        # Generate tokens autoregressively
        generated_tokens = []
        current_embeds = inputs_embeds

        max_tokens = max_new_tokens or self.max_new_tokens

        for step in range(max_tokens):
            decoder_outputs = self._decoder_session.run(None, {
                "inputs_embeds": current_embeds,
                "attention_mask": np.ones((1, current_embeds.shape[1]), dtype=np.int64),
            })

            logits = decoder_outputs[0]
            next_token_logits = logits[0, -1, :]

            # Simplified greedy sampling
            next_token_id = int(np.argmax(next_token_logits))

            if next_token_id == self._tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token_id)

            next_token_array = np.array([[next_token_id]], dtype=np.int64)
            embed_outputs = self._embed_session.run(None, {"input_ids": next_token_array})
            next_embeds = embed_outputs[0]

            current_embeds = np.concatenate([current_embeds, next_embeds], axis=1)

        inference_time_ms = (time.perf_counter() - start) * 1000

        generated_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)

        darwin.spinner_stop(spinner_id, success=True)

        self._log(
            f"Generation complete: {len(generated_tokens)} tokens in {inference_time_ms:.1f}ms "
            f"on {self._device}",
            LogType.SUCCESS,
        )

        metadata = UnifiedVisionMetadata(
            model_id=self.model_id,
            inference_time_ms=round(inference_time_ms, 3),
            num_tokens_generated=len(generated_tokens),
            device=self._device,
            backend="onnx",
            image_size=img_size,
            prompt_length=len(prompt_ids[0]) if prompt_ids is not None else 0,
        )

        return UnifiedVisionResult(
            text=generated_text,
            tokens=generated_tokens,
            metadata=metadata,
        )



    # ── Introspection (Interface) ──────────────────────────────────────────────

    def get_model_info(self) -> dict:
        if not self.is_loaded():
            self._log("Model not loaded", LogType.WARNING)
            return {"status": "not_loaded", "model_dir": str(self.model_dir)}

        return {
            "status": "loaded",
            "model_dir": str(self.model_dir),
            "model_id": self.model_id,
            "revision": self.revision,
            "decoder_precision": self.decoder_precision,
            "vision_precision": self.vision_precision,
            "embed_precision": self.embed_precision,
            "device": self._device,
            "onnx_runtime_version": self._onnx_runtime_version,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "backend": "onnx",
        }

    # ── Internal: Tokenizer loading ──────────────────────────────────────────

    def _load_tokenizer(self) -> None:
        try:
            from tokenizers import Tokenizer
        except ImportError:
            self._log("tokenizers library required. pip install tokenizers", LogType.ERROR)
            raise ImportError("tokenizers library required. pip install tokenizers")

        tokenizer_path = self.model_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            self._log(f"Tokenizer not found in {self.model_dir}", LogType.ERROR)
            raise FileNotFoundError(f"Tokenizer not found in {self.model_dir}")

        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._log(f"Tokenizer loaded from {tokenizer_path}")

    def _load_preprocessor(self) -> None:
        try:
            from transformers import AutoImageProcessor
        except ImportError:
            self._log("transformers library required. pip install transformers", LogType.ERROR)
            raise ImportError("transformers library required. pip install transformers")

        preprocessor_path = self.model_dir
        if not (preprocessor_path / "preprocessor_config.json").exists():
            self._log(f"Preprocessor config not found in {self.model_dir}", LogType.ERROR)
            raise FileNotFoundError(f"Preprocessor config not found in {self.model_dir}")

        self._preprocessor = AutoImageProcessor.from_pretrained(str(preprocessor_path))
        self._log(f"Preprocessor loaded from {preprocessor_path}")

    # ── Internal: ONNX Session loading ───────────────────────────────────────

    def _load_onnx_sessions(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            self._log("onnxruntime required. pip install onnxruntime", LogType.ERROR)
            raise ImportError("onnxruntime required. pip install onnxruntime")

        self._onnx_runtime_version = ort.__version__

        available = ort.get_available_providers()
        self._log(f"Available ONNX providers: {available}")

        preferred = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in preferred if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        self._device = providers[0].replace("ExecutionProvider", "")
        self._log(f"Using provider: {providers[0]}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        suffix_decoder = "" if self.decoder_precision == "fp32" else f"_{self.decoder_precision}"
        suffix_vision = "" if self.vision_precision == "fp32" else f"_{self.vision_precision}"
        suffix_embed = "" if self.embed_precision == "fp32" else f"_{self.embed_precision}"

        decoder_path = self.model_dir / f"decoder_model_merged{suffix_decoder}.onnx"
        vision_path = self.model_dir / f"vision_encoder{suffix_vision}.onnx"
        embed_path = self.model_dir / f"embed_tokens{suffix_embed}.onnx"

        self._decoder_session = ort.InferenceSession(
            str(decoder_path), sess_options=sess_options, providers=providers,
        )
        self._log(f"Decoder loaded: {decoder_path.name}")

        self._vision_session = ort.InferenceSession(
            str(vision_path), sess_options=sess_options, providers=providers,
        )
        self._log(f"Vision encoder loaded: {vision_path.name}")

        self._embed_session = ort.InferenceSession(
            str(embed_path), sess_options=sess_options, providers=providers,
        )
        self._log(f"Embed tokens loaded: {embed_path.name}")

    # ── Internal: Image preprocessing ────────────────────────────────────────

    def _preprocess_image(self, img: Image.Image) -> np.ndarray:
        if self._preprocessor is None:
            raise RuntimeError("Preprocessor not loaded")
        inputs = self._preprocessor(images=img, return_tensors="np")
        return inputs["pixel_values"]