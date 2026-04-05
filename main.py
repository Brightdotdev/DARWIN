import sys
import argparse
from pathlib import Path


from src.core.Parser import parsepdf
from src.core.Chunker import chunk_document

def main():
    parser = argparse.ArgumentParser(description="Parse a PDF file")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--strategy", choices=["light", "high"], default="light",
                        help="Extraction strategy: light (fast) or high (detailed)")
    parser.add_argument("--chunk", action="store_true",
                        help="Also chunk the parsed text")
    args = parser.parse_args()
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    print(f"Parsing: {pdf_path.name}")
    doc = parsepdf(pdf_path, strategy=args.strategy)
    print(f"Pages: {doc.page_count}")
    print(f"Characters: {len(doc.text):,}")
    if args.chunk:
        print("\nChunking document...")
        chunks = chunk_document(doc.text, source=pdf_path.name)
        print(f"Chunks created: {len(chunks)}")
        for i, chunk in enumerate(chunks[:5]):
            print(f"  [{i}] {chunk.token_count} tokens")
        if len(chunks) > 5:
            print(f"  ... and {len(chunks) - 5} more")
    else:
        print("\nText preview (first 500 chars):")
        print("-" * 40)
        print(doc.text[:500])
        print("-" * 40)
if __name__ == "__main__":
    main()