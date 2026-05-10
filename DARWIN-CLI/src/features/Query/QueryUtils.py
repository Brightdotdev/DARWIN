import re
from urllib.parse import urlparse
from typing import List, Tuple, Any
from dataclasses import dataclass
import uuid


URL_REGEX = re.compile(r"https?://[^\s]+")

@dataclass
class PastedText:
    id: str
    content: str
    type: str
    meta: dict[str, Any]



def extract_and_replace_urls(text: str):
    urls  = []

    def replacer(match):
        url = match.group(0)
        parsed = urlparse(url)

        domain = parsed.netloc.replace("www.", "")
        placeholder = f"[{domain}]"

        urls.append({
            "url": url,
            "domain": domain,
            "placeholder": placeholder
        })

        return placeholder

    cleaned_text = URL_REGEX.sub(replacer, text)

    return cleaned_text, urls



def detect_blob_type(text: str) -> str:
    t = text.lower().strip()

    if "traceback" in t or "exception" in t or "error" in t:
        return "log"

    if "def " in t or "class " in t or "import " in t:
        return "code"

    if len(t) > 1500:
        return "document"

    if "?" in t and len(t) < 500:
        return "question"

    return "unknown"





def resolve_blob_references(query: str, blobs: list[dict]):
    blob_map = {b["id"]: b for b in blobs}

    def replacer(match):
        blob_id = match.group(1)
        blob = blob_map.get(blob_id)

        if not blob:
            return "[UNKNOWN_BLOB]"

        # IMPORTANT: keep full content, not type
        return blob["content"]

    return re.sub(r"\[BLOB:([a-zA-Z0-9\-]+)\]", replacer, query)


def extract_explicit_blobs(query: str):
    pattern = r"@blob(?:\[type=(\w+)\])?\s*(.*?)\s*@end"

    blobs = []

    def replacer(match):
        blob_type = match.group(1) or "unknown"
        content = match.group(2).strip()

        blob_id = str(uuid.uuid4())[:8]

        blobs.append(
            PastedText(
                id=blob_id,
                content=content,
                type=blob_type,
                meta={"source": "explicit"}
            )
        )

        return f"[BLOB:{blob_id}]"

    cleaned_query = re.sub(pattern, replacer, query, flags=re.S)

    return cleaned_query, blobs



def build_alias_map(files: list[str]) -> dict[str, str]:
    """
    Creates aliases for filenames.

    Example:
        notes.txt  -> notes.txt
        notes      -> notes.txt

        report.pdf -> report.pdf
        report     -> report.pdf
    """

    alias_map = {}

    for file in files:
        normalized = file.strip()

        # Full filename alias
        alias_map[normalized] = normalized

        # Stem alias
        if "." in normalized:
            stem = normalized.rsplit(".", 1)[0]
            alias_map[stem] = normalized

    return alias_map




def extract_file_tags(
    text: str,
    available_files: list[str]
) -> Tuple[List[str], str]:
    """
    Extracts valid @file references from text,
    replaces them with placeholders,
    and returns:
        (matched_files, cleaned_text)
    """

    alias_map = build_alias_map(available_files)

    pattern = r'@([\w.\-]+)'

    extracted = []
    seen = set()

    def replacer(match):
        token = match.group(1)

        real_file = alias_map.get(token)

        # Not a valid file reference
        if not real_file:
            return match.group(0)

        # Deduplicate extracted files
        if real_file not in seen:
            seen.add(real_file)
            extracted.append(real_file)

        # Placeholder name
        placeholder = real_file.rsplit(".", 1)[0]

        return f"[{placeholder}]"

    cleaned_text = re.sub(pattern, replacer, text)

    return extracted, cleaned_text


