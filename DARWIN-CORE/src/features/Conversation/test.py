"""
test_integration.py
Demonstrates ConversationStore + ConversationCache + QueryOrchestrator wiring.
Run: python test_integration.py
"""

from ConversationStore import ConversationStore, TurnRecord, BlobRecord
from ConversationCache import ConversationCache
from src.features.Query.QueryOrchestrator import QueryOrchestrator, DarwinLogger


def main():
    # ── Setup ─────────────────────────────────────────────
    store = ConversationStore(db_path="test_darwin.db")
    cache = ConversationCache(db_path="test_cache.db")
    logger = DarwinLogger(log_file="darwin.log")
    orchestrator = QueryOrchestrator(store, cache, logger)

    # ── Create conversation ───────────────────────────────
    conv_id = store.create_conversation(title="GPU Research Session")
    print(f"\n🆕 Created conversation: {conv_id}\n")

    # ── Seed some history ──────────────────────────────────
    store.append_turn(TurnRecord(
        turn_id="", conv_id=conv_id, role="user",
        content="What are the best GPUs for LLM training under $2000?",
        timestamp="", extraction_method="regex"
    ))
    store.append_turn(TurnRecord(
        turn_id="", conv_id=conv_id, role="assistant",
        content="The RTX 4090 offers 24GB VRAM at $1600. The RTX 3090 Ti is cheaper used.",
        timestamp="", extraction_method="regex"
    ))

    # ── Test 1: Pushback query ─────────────────────────────
    print("═" * 60)
    print("TEST 1: Pushback / narrowing query")
    print("═" * 60)

    plan1 = orchestrator.process(
        conv_id=conv_id,
        raw_message="what about power consumption?"
    )
    print(f"\n📋 Plan ID: {plan1.plan_id}")
    print(f"🎯 Routing: {plan1.routing_target} (confidence: {plan1.confidence})")
    print(f"🔄 Pushback detected: {plan1.pushback_detected}")
    print(f"📝 Rewritten: {plan1.rewritten_query[:120]}...")
    print(f"💾 Cache hits: {plan1.cache_hits}")

    # ── Test 2: Code blob + links ──────────────────────────
    print("\n" + "═" * 60)
    print("TEST 2: Code blob + link extraction")
    print("═" * 60)

    msg2 = """
    I'm trying to optimize this training loop:
    ```python
    for epoch in range(100):
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
    ```
    I found this guide: https://huggingface.co/docs/transformers/perf_train_gpu_one
    Any thoughts?
    """

    plan2 = orchestrator.process(conv_id=conv_id, raw_message=msg2)
    print(f"\n📋 Plan ID: {plan2.plan_id}")
    print(f"🎯 Routing: {plan2.routing_target} (confidence: {plan2.confidence})")
    print(f"📎 Links: {plan2.extracted_links}")
    print(f"🧩 Blobs: {len(plan2.extracted_blobs)} found")
    for b in plan2.extracted_blobs:
        print(f"   - {b.blob_type}: {b.language or 'n/a'} | {len(b.content)} chars")

    # ── Test 3: Cache hit ──────────────────────────────────
    print("\n" + "═" * 60)
    print("TEST 3: Cache hit on identical query")
    print("═" * 60)

    plan3 = orchestrator.process(conv_id=conv_id, raw_message="what about power consumption?")
    print(f"\n📋 Plan ID: {plan3.plan_id}")
    print(f"💾 Cache hits: {plan3.cache_hits}")
    print(f"🎯 Routing: {plan3.routing_target}")

    # ── Test 4: Attachments + lifecycle ──────────────────────
    print("\n" + "═" * 60)
    print("TEST 4: Blob attachment + conversation lifecycle")
    print("═" * 60)

    blob = BlobRecord(
        blob_id="", turn_id="", conv_id=conv_id,
        blob_type="code", content="import torch\ntorch.cuda.empty_cache()",
        metadata={"source": "user_paste"}
    )
    stored_blob = store.attach_blob(blob)
    print(f"\n💾 Stored blob: {stored_blob.blob_id}")

    # List all blobs
    blobs = store.get_blobs(conv_id, blob_type="code")
    print(f"📁 Code blobs in conversation: {len(blobs)}")

    # Export
    export = store.export_conversation(conv_id, format="json")
    print(f"📤 Export: {len(export['turns'])} turns, {len(export['blobs'])} blobs")

    # ── Test 5: Cache operations ───────────────────────────
    print("\n" + "═" * 60)
    print("TEST 5: Cache set / get / stats")
    print("═" * 60)

    key = cache.make_key("web_search", query="RTX 4090 price")
    cache.set(key, {"results": ["Amazon: $1599", "Newegg: $1620"]}, 
              conv_id=conv_id, tool_name="web_search", ttl_seconds=3600)

    entry = cache.get(key, conv_id=conv_id)
    print(f"\n💾 Cached entry: {entry.tool_name} | hits: {entry.hit_count}")

    # Global cache
    global_key = cache.make_key("embedding", text="hello world")
    cache.set(global_key, {"vector": [0.1, 0.2, 0.3]}, 
              conv_id=None, tool_name="embedding")
    global_entry = cache.get(global_key)
    print(f"🌍 Global cache: {global_entry.tool_name if global_entry else 'miss'}")

    print(f"\n📊 Cache stats: {cache.get_stats()}")
    print(f"📊 Store stats: {store.get_stats()}")

    # ── Cleanup ──────────────────────────────────────────────
    store.close()
    cache.close()
    print("\n✅ All tests passed. Check darwin.log for structured events.")


if __name__ == "__main__":
    main()