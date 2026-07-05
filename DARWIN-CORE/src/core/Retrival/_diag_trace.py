"""Trace search_single_document_keyword on Bible PDF with 'book of enoch'"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.environ["LITELLM_SUPPRESS_DEBUG_INFO"] = "True"
os.environ["PYTHONIOENCODING"] = "utf-8"
import warnings
warnings.filterwarnings("ignore", module="litellm")

from src.core.Storage.DarwinStore import DarwinStore
from src.core.Pipeline.Indexing.BM25.BM25Utils import tokenize_text
from src.core.Retrival.Keyword.KeywordUtilities import calculate_bm25

store = DarwinStore(name="test")

fname = "CSB_Pew_Bible_2nd_Printing.pdf"
query = "book of enoch"

# 1. Load everything like search_single_document_keyword does
all_chunks = store.query_records(fname, ["index", "text", "clean", "page", "page_end", "heading", "structural_type", "links", "overlap_direction"])
data = store.load_bm25(fname)

print(f"File: {fname}")
print(f"Chunks loaded: {len(all_chunks)}")
print(f"BM25 data loaded: {data is not None}")

if data:
    token_list = data.get("tokens", [])
    index_map = data.get("chunk_indices")
    print(f"Token chunks: {len(token_list)}")
    print(f"index_map type: {type(index_map).__name__}")
    if index_map is not None:
        print(f"index_map length: {len(index_map)}")
        print(f"index_map[:10]: {index_map[:10]}")
    else:
        print("index_map is None — will fallback to range(len(tokens))")

    chunk_by_index = {c.get("index"): c for c in all_chunks}
    print(f"chunk_by_index keys sample: {sorted(chunk_by_index.keys())[:10]}")

    query_terms = tokenize_text(query)
    print(f"Query terms: {query_terms}")

    # Trace through calculate_bm25 for a few positions
    for bm25_pos in range(min(5, len(token_list))):
        score, matched = calculate_bm25(query_terms, bm25_pos, data, 1.5, 0.75)
        print(f"  bm25_pos={bm25_pos}: score={score:.6f}, matched={matched}")
        chunk_tokens = token_list[bm25_pos]
        # Check for query terms in chunk tokens
        for qt in query_terms:
            if qt in chunk_tokens:
                print(f"    -> '{qt}' found in chunk tokens! (count={chunk_tokens.count(qt)})")

    # Check if there are ANY chunks with book or enoch
    book_count = sum(1 for t in token_list if "book" in t)
    enoch_count = sum(1 for t in token_list if "enoch" in t)
    print(f"\nChunks containing 'book': {book_count}")
    print(f"Chunks containing 'enoch': {enoch_count}")

    # Full search
    print("\n--- Full search_across_documents_keyword ---")
    from src.core.Retrival.Keyword.KeywordSearch import search_across_documents_keyword
    results = search_across_documents_keyword(
        query=query,
        filenames=[fname],
        store=store,
        top_k=10,
        min_score=0.0,
    )
    print(f"Results for {fname}: {len(results)}")
    if results:
        for r in results[:3]:
            print(f"  score={r.score:.4f} page={r.page} text={r.text[:80]}")
    else:
        print("  (empty)")
