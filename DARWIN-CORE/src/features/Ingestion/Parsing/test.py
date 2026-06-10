#!/usr/bin/env python3
"""
Analysis script for the document parsing pipeline – uses non‑Wikipedia URLs.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.Ingestion.Parsing.Parser import parse, parse_url, PARSER_MAP
from src.features.Ingestion.Parsing.BlobHandler import BlobHandler
from src.features.Ingestion.Parsing.lib import BlobInput
from src.core.Logger.DarwinLogger import darwin, Verbosity

darwin.configure(enable_websocket=False, verbosity_default=Verbosity.NORMAL)
darwin.start_mp_listener()

def create_sample_pdf(path: Path):
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello, this is a test PDF for parsing.", fontsize=14)
    page.insert_text((72, 100), "It contains multiple lines of text.", fontsize=12)
    page.insert_text((72, 130), "The pipeline should extract this correctly.", fontsize=12)
    doc.save(str(path))
    doc.close()

def run_test(name: str, parser_fn, **kwargs):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    try:
        doc = parser_fn(**kwargs)
    except Exception as e:
        print(f"❌ Exception: {type(e).__name__}: {e}")
        return None
    elapsed = time.perf_counter() - t0
    print(f"⏱  Elapsed: {elapsed:.2f}s")
    if doc.error:
        print(f"⚠  Error: {doc.error}")
    else:
        print(f"✅ Success")
    print(f"   Filename: {doc.filename}")
    print(f"   Page count: {doc.page_count}")
    print(f"   Display text length: {len(doc.display_text or '')} chars")
    print(f"   Embedding text length: {len(doc.embedding_text or '')} chars")
    print(f"   Images: {len(doc.images)}")
    if doc.metadata:
        safe_meta = {k: v for k, v in doc.metadata.items() if k != "links"}
        print(f"   Metadata: {safe_meta}")
    return doc

def main():
    test_dir = Path("./test_parse")
    test_dir.mkdir(exist_ok=True)
    print(f"📂 Test directory: {test_dir}")

    # 1. Local PDF
    pdf_path = test_dir / "sample.pdf"
    create_sample_pdf(pdf_path)
    run_test("Local PDF", parse,
             file_path=pdf_path, images_dir=None, caption=False, precise=False)

    # 2. URL → HTML (a simple, non‑JavaScript page)
    run_test("URL → HTML (example.com)", parse_url,
             url="https://example.com", precise=False, timeout=15.0
)

    # 3. URL → PDF (an IETF RFC – reliable PDF link)
    run_test("URL → PDF (IETF RFC)", parse_url,
             url="https://www.rfc-editor.org/rfc/rfc2616.pdf",
             precise=False, timeout=30.0
)

    # 4. Blob with plain text + link
    blob1 = BlobInput(
        content="Read this about HTTP: https://example.com",
        source="test_blob",
        title="Plain text blob",
        turn_id="t1",
        resolve_links=True,
        link_timeout=10.0,
    )
    run_test("Blob (plain text + link)", BlobHandler.process, blob=blob1)

    # 5. Blob HTML + multiple links
    blob2 = BlobInput(
        content='<html><body><h1>Links</h1><p>Check <a href="https://example.com">example</a> and <a href="https://www.ietf.org">IETF</a>.</p></body></html>',
        source="test_blob_html",
        title="HTML blob",
        turn_id="t2",
        resolve_links=True,
        link_timeout=10.0,
    )
    run_test("Blob (HTML + multiple links)", BlobHandler.process, blob=blob2)

    print("\n" + "="*60)
    print("Done. Review the extraction lengths and error messages above.")

if __name__ == "__main__":
    main()