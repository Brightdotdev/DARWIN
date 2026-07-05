import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.environ["LITELLM_SUPPRESS_DEBUG_INFO"] = "True"
os.environ["PYTHONIOENCODING"] = "utf-8"
import warnings

warnings.filterwarnings("ignore", module="litellm")

from src.core.Storage.DarwinStore import DarwinStore
from src.core.Pipeline.Indexing.BM25.BM25Utils import tokenize_text

store = DarwinStore(name="test")

status = store.status()
kw_files = [
    fname
    for fname, info in status.get("files", {}).items()
    if info.get("is_kw_searchable", False)
]
print(f"KW-searchable files: {kw_files}")

if kw_files:
    for fname in kw_files[:3]:
        print(f"\n{'=' * 60}")
        print(f"Examining: {fname}")

        chunks = store.query_records(fname, columns=["index", "clean"])
        print(f"Chunks count: {len(chunks)}")

        data = store.load_bm25(fname)
        if not data:
            print("BM25 data is None!")
            continue

        tokens = data.get("tokens", [])
        print(f"BM25 token chunks: {len(tokens)}")
        print(f"AvgDL: {data.get('avgdl', 0):.2f}")
        print(f"Vocab size: {data.get('vocab_size', 0)}")
        doc_freqs = data.get("doc_freqs", {})
        top = sorted(doc_freqs.items(), key=lambda x: -x[1])[:20]
        print(f"Top 20 doc_freqs:")
        for term, freq in top:
            print(f"  {term}: {freq}")

        print(f"\nChunk 0 token count: {len(tokens[0]) if tokens else 0}")
        print(f"Chunk 0 sample tokens (first 30):")
        print(f"  {tokens[0][:30] if tokens else []}")

        queries = [
            "book of enoch",
            "what is graph rag",
            "NVIDIA Nemotron",
            "Bible summary",
            "anthropic paper",
        ]
        for q in queries:
            terms = tokenize_text(q)
            print(f'\nQuery: "{q}" -> {terms}')
            for term in terms:
                if term in doc_freqs:
                    print(f'  "{term}" IN index (df={doc_freqs[term]})')
                else:
                    print(f'  "{term}" NOT in index')

        # Also check domain stopwords
        domain_stops = data.get("domain_stopwords", set())
        print(
            f"\nDomain stopwords ({len(domain_stops)}): {sorted(domain_stops)[:20]}..."
        )

        # Check if tokens actually contain query terms
        if tokens:
            found_any = False
            for term in ["book", "enoch"]:
                for ci, tok_list in enumerate(tokens[:50]):
                    if term in tok_list:
                        print(f'  "{term}" found in chunk {ci}')
                        found_any = True
            if not found_any:
                print('  "book"/"enoch" NOT found in first 50 chunks')

# Now test search_across_documents_keyword directly
print("\n\n" + "=" * 60)
print("DIRECT SEARCH TEST")
from src.core.Retrival.Keyword.KeywordSearch import search_across_documents_keyword

queries = ["book of enoch", "Bible summary"]
for q in queries:
    results = search_across_documents_keyword(
        query=q,
        filenames=kw_files,
        store=store,
        top_k=10,
    )
    print(f'\nQuery: "{q}" -> {len(results)} results')
    if results:
        for r in results[:3]:
            print(
                f"  score={r.score:.4f} source={r.source} page={r.page} text={r.text[:80]}..."
            )
    print()
else:
    print("No kw-searchable files found!")
