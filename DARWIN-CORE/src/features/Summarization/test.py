#!/usr/bin/env python3
"""
Battle‑test the Summarizer with multiple configurations:
  - default (no extras)
  - POS filter on
  - clustered mode
  - tight settings (min words = 6, chunk sentences = 2)
Saves a detailed report to test_summarizer_results.txt.
"""

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.Summarization.Summarizer import Summarizer
from src.core.Logger.DarwinLogger import darwin
import logging
logging.getLogger("spacy").setLevel(logging.WARNING)

darwin.configure()


def get_clean_text(url: str) -> str:
    from src.pipleline.Ingestion.Parsing.UrlFetcher import fetch_url
    from src.pipleline.Ingestion.Parsing.HtmlProcessor import extract_content

    data, _, err = fetch_url(url)
    if err:
        print(f"Fetch error: {err}")
        return ""
    text, _ = extract_content(url, html_bytes=data, precise=False)
    text = re.sub(r'(?m)^\s*\|.*$', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


def main():
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    print(f"Fetching {url}...")
    full_text = get_clean_text(url)
    if not full_text:
        print("No text retrieved.")
        return
    print(f"Original text length: {len(full_text)} characters\n")

    # Define configurations to test
    configs = [
        ("default", Summarizer()),
        ("POS filter", Summarizer(pos_filter=True)),
        ("clustered", Summarizer(clustered=True)),
        ("tight (min words=6, chunk sent=2)", Summarizer(min_sentence_words=6, chunk_sentences=2)),
    ]

    # Common test methods (we'll use summarize(5) and long(7) for each)
    methods = [
        ("summarize(5)", lambda s: s.summarize(full_text, max_sentences=5)),
        ("long(7)", lambda s: s.summarize_long(full_text, max_sentences=7)),
    ]

    output_path = Path(__file__).with_name("test_summarizer_results.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        # Full document once
        f.write("=" * 80 + "\n")
        f.write("[full document]\n")
        f.write("=" * 80 + "\n")
        f.write(full_text + "\n\n")

        for cfg_name, summarizer in configs:
            if not summarizer.available:
                print(f"Skipping {cfg_name} – summarizer unavailable")
                continue

            f.write("\n" + "#" * 80 + "\n")
            f.write(f"CONFIGURATION: {cfg_name}\n")
            f.write("#" * 80 + "\n\n")

            for method_name, method_fn in methods:
                t0 = time.perf_counter()
                result = method_fn(summarizer)
                t1 = time.perf_counter()
                f.write("=" * 80 + "\n")
                f.write(f"[{cfg_name}] {method_name}  |  {t1 - t0:.2f}s  |  {len(result)} chars\n")
                f.write("=" * 80 + "\n")
                f.write(result + "\n\n")
                print(f"{cfg_name:>35} {method_name:>20}  {t1 - t0:.2f}s  {len(result)} chars")

        # Compression ratios
        f.write("\n" + "=" * 80 + "\n")
        f.write("Compression ratios (vs original)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Original: {len(full_text)} chars\n")
        for cfg_name, summarizer in configs:
            for method_name, method_fn in methods:
                result = method_fn(summarizer)
                ratio = len(result) / len(full_text) * 100
                f.write(f"[{cfg_name}] {method_name}: {len(result)} chars ({ratio:.1f}%)\n")

    print(f"\nDetailed report saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()