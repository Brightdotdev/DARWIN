#!/usr/bin/env python
# tests/test_models.py (fixed)

import sys
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.Models.Lib.ModelRegistry import ModelCategory, MODEL_SPECS
from src.core.Models.ModelManager import modelManager
from src.core.Logger.DarwinLogger import darwin, LogType


def ensure_mock_images(count: int = 3) -> Optional[Path]:
    try:
        import cv2
        import numpy as np
    except ImportError:
        darwin.log("[TEST] OpenCV not installed – skipping OCR", LogType.WARNING)
        return None
    temp_dir = Path(tempfile.mkdtemp(prefix="mock_images_"))
    for i in range(count):
        img = np.ones((200, 400, 3), dtype=np.uint8) * 255
        cv2.putText(img, f"Mock OCR Test {i+1}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, "RapidOCR is great", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        file_path = temp_dir / f"mock_{i+1}.png"
        cv2.imwrite(str(file_path), img)
    darwin.log(f"[TEST] Created {count} mock images", LogType.INFO)
    return temp_dir


def log_test_step(msg: str):
    darwin.log(f"[TEST] {msg}", LogType.INFO)


def test_embedding(model):
    texts = ["passage: Hello world", "passage: This is a test", "passage: Embeddings are useful"]
    result = model.encode(texts)          # returns UnifiedEmbeddingResult
    embeddings = result.embeddings        # numpy array
    assert len(embeddings) == len(texts)
    dims = embeddings.shape[1]
    assert dims > 0
    darwin.log(f"✅ Embedding test passed: {len(texts)} texts → {dims} dims", LogType.SUCCESS)
    return True


def test_nlp(model):
    log_test_step("Running NLP test...")
    text = "Apple Inc. is based in Cupertino, California."
    doc = model.process(text)[0]
    tokens = len(doc.tokens)
    entities = len(doc.entities)
    chunks = len(doc.noun_chunks)
    darwin.log(f"[TEST] ✅ NLP: {tokens} tokens, {entities} entities, {chunks} chunks", LogType.SUCCESS)
    return True


def test_ocr(model, images_dir: Path):
    log_test_step("Running OCR test...")
    if images_dir is None:
        darwin.log("[TEST] ⚠️ No images – skipping OCR", LogType.WARNING)
        return True
    images = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    if not images:
        darwin.log("[TEST] ⚠️ No images found", LogType.WARNING)
        return True
    for img_path in images[:2]:
        result = model.infer(img_path)
        regions = result.regions
        darwin.log(f"[TEST] OCR on {img_path.name}: {len(regions)} regions", LogType.INFO)
        if regions:
            darwin.log(f"[TEST]   First text: '{regions[0].text}'", LogType.INFO)
    darwin.log("[TEST] ✅ OCR test passed", LogType.SUCCESS)
    return True


def test_reranker(model):
    log_test_step("Running reranker test...")
    query = "What is the capital of France?"
    docs = ["Paris is the capital of France.", "Berlin is the capital of Germany.", "London is the capital of UK."]
    result = model.rerank(query, docs)
    scores = result.scores
    ranked = result.ranked_indices
    darwin.log(f"[TEST] ✅ Reranker: {len(scores)} scores, top doc index {ranked[0]}", LogType.SUCCESS)
    return True


def test_model_category(category: ModelCategory, model_id: Optional[str] = None, images_dir: Optional[Path] = None) -> bool:
    cat_name = category.name.lower()
    log_test_step(f"=== Testing {cat_name} ===")

    if model_id is None:
        candidates = [s for s in MODEL_SPECS if s.category == category and not s.requires_provider]
        if not candidates:
            darwin.log(f"[TEST] ⚠️ No local models for {cat_name}", LogType.WARNING)
            return False
        candidates.sort(key=lambda s: (s.ram_gb, s.disk_mb))
        model_id = candidates[0].model_id
    darwin.log(f"[TEST] Selected: {model_id}", LogType.INFO)

    try:
        result = modelManager.setup(category, model_id=model_id, auto_download=True, required=False)
        if not result.success:
            darwin.log(f"[TEST] ❌ Setup failed: {result.error}", LogType.ERROR)
            return False
        darwin.log(f"[TEST] Setup result: {result}", LogType.INFO)
    except Exception as e:
        darwin.log(f"[TEST] ❌ Setup exception: {e}", LogType.ERROR)
        return False

    model = modelManager.ensure(category, model_id=model_id, auto_download=False)
    if model is None:
        darwin.log(f"[TEST] ❌ Failed to load", LogType.ERROR)
        return False
    darwin.log("[TEST] ✅ Model loaded", LogType.SUCCESS)

    ok = False
    try:
        if category == ModelCategory.EMBEDDING:
            ok = test_embedding(model)
        elif category == ModelCategory.NLP:
            ok = test_nlp(model)
        elif category == ModelCategory.OCR:
            ok = test_ocr(model, images_dir)
        elif category == ModelCategory.RERANKER:
            ok = test_reranker(model)
        else:
            ok = True
    except Exception as e:
        darwin.log(f"[TEST] ❌ Inference failed: {e}", LogType.ERROR)
        ok = False

    modelManager.unload(category, model_id)
    darwin.log("[TEST] Unload complete", LogType.INFO)
    return ok


def main():
    darwin.configure()
    darwin.log("=" * 80, LogType.INFO)
    darwin.log("🚀 Model Integration Test", LogType.INFO)

    status = modelManager.initialize(auto_setup=False)
    darwin.log(status.summary, LogType.INFO)

    images_dir = ensure_mock_images(3)

    results = {}
    categories = [ModelCategory.EMBEDDING, ModelCategory.NLP, ModelCategory.OCR, ModelCategory.RERANKER]
    for cat in categories:
        ok = test_model_category(cat, images_dir=images_dir)
        results[cat] = ok
        darwin.log(f"{'✅' if ok else '❌'} {cat.name} test {'passed' if ok else 'failed'}", LogType.INFO)

    darwin.log("\n" + "=" * 80, LogType.INFO)
    darwin.log("📊 TEST SUMMARY", LogType.INFO)
    for cat, ok in results.items():
        darwin.log(f"  {cat.name}: {'✅ PASS' if ok else '❌ FAIL'}", LogType.INFO)
    failed = sum(1 for ok in results.values() if not ok)
    darwin.log(f"\n{len(results)-failed}/{len(results)} passed", LogType.INFO)

    if images_dir and images_dir.exists():
        shutil.rmtree(images_dir, ignore_errors=True)
    modelManager.unload_all()
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()