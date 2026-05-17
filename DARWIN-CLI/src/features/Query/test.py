from src.features.Query.InputPreprocessor import InputPreprocessor
from src.features.Query.ClarificationGate import ClarificationGate
from src.features.Query.ResponseAssembler import ResponseAssembler
from src.features.Query.QueryOrchestrator import QueryOrchestrator, DeepSearchOptions
from src.features.LLM.LLMManager import LLMManager
from src.features.Storage.DarwinStore import DarwinStore


LLMManager.initialize()
store = DarwinStore()
client = LLMManager.get_client()

preprocessor = InputPreprocessor()
gate = ClarificationGate(client.call_small)
orch = QueryOrchestrator(store, llm_client=client.call_small)
assembler = ResponseAssembler()

user_input = "Compare the watchers in Enoch vs Genesis"
pinned = ["book_of_enoch.pdf", "CSB_Pew_Bible_2nd_Printing.pdf"]
status = store.status()
registry = list(status.get("files", {}).keys())


raw = preprocessor.process(user_input, pinned_files=pinned)
intent = gate.process(raw, file_registry=registry)

if intent.needs_clarification:
    print("\n🛑 CLARIFICATION NEEDED")
    for block in intent.followup_blocks:
        print(f"  Q: {block.question}")
        for opt in block.options:
            print(f"    [{opt.id}] {opt.label}")
else:
    options = DeepSearchOptions(
        method=intent.retrieval_mode,
        top_k=intent.top_k,
    )
    results, diagnostics = orch.search_deep(
        query=intent.clean_query,
        files=raw.pinned_files,
        options=options,
        diagnose=True,
        intent=intent,
    )

    envelope = assembler.assemble(intent, results, diagnostics)

    print(f"\n{'='*60}")
    print("PRIMARY BLOCKS")
    print(f"{'='*60}")
    for block in envelope.primary_blocks:
        print(block)
        if block.fallback_from:
            print(f"  ⚠️  Downgraded from {block.fallback_from}")

    if envelope.patches:
        print(f"\n{'='*60}")
        print("PENDING PATCHES")
        print(f"{'='*60}")
        for patch in envelope.patches:
            print(f"  [{patch.op}] {patch.reason}")

    print(f"\nPending expansion: {envelope.pending_expansion}")