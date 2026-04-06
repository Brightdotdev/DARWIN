from  src.core.Parser  import parse, scan_directory
from src.core.Chunker import chunk_document
from src.core.Store   import DarwinStore

store = DarwinStore("/home/user/notes")
store.init()

# parse
doc = parse("biology.pdf")

if doc.error:
    print(f"failed: {doc.error}")
else:
    # doc.text → straight into chunker
    chunks = chunk_document(doc.text, source=doc.filename)

    # store everything
    store.save_chunks(
        doc.filename,
        chunks,
        page_count = doc.page_count,   # ← page numbers for citations
        parser     = doc.strategy,
    )