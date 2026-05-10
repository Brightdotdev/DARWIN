from typing import Any
from dataclasses import dataclass, field
from enum import Enum
import uuid


from src.features.Query.QueryUtils import (
        extract_and_replace_urls,
        extract_file_tags,
        PastedText,
        extract_explicit_blobs,
        resolve_blob_references
    )
def is_searchable(filename: str) -> bool:
    from src.features.Storage.DarwinStore import DarwinStore

    store = DarwinStore()
    status = store.status()

    files = status.get("files", {})
    info = files.get(filename)

    if info is None:
        return False

    return info.get("is_hybrid_searchable", False)




def get_all_searchable_filenames() :
    
    from src.features.Storage.DarwinStore import DarwinStore

    store = DarwinStore()
    status = store.status()
    found_files = []
    files = status.get("files", {})

    for fname, info in files.items():
        if info.get("is_hybrid_searchable", False):
            print(fname)
            found_files.append(fname)

    return found_files





def validate_searchable_files(list_of_files: list[str]):
    searchable_files = []
    non_searchable_files = []

    for file in list_of_files:
        if is_searchable(file):
            searchable_files.append(file)
        else:
            non_searchable_files.append(file)

    return searchable_files, non_searchable_files



@dataclass
class RawQuery:
    query: str
    pasted_texts: list[dict[str, Any]] = field(default_factory=list)
    pinned_files: list[str] = field(default_factory=list)
    extracted_urls: list[dict[str, Any]] = field(default_factory=list)



Response_Message = {
    "indexed": "indexed",
    "files_not_indexed": "files_not_indexed"

}


@dataclass
class IndexingSuggestion:
    message: str
    files: List[str]
    action: str = "index_now"
    next_step: str = "Would you like to index these files so they can be searched?"


from dataclasses import dataclass
from typing import Any, List, Optional




class FileIntegrityStatus(str, Enum):
    INDEXED = "indexed"
    PARTIAL = "partial"
    NO_FILES = "no_files_specified"
    REQUIRES_INDEXING = "requires_indexing"

@dataclass
class FileIntegrityResult:
    status: str
    query: str
    urls: List[str]
    files_scope: List[str]
    rejected_files: List[str]
    pasted_texts: List[dict[str, Any]]

    indexing_suggestion: Optional[IndexingSuggestion] = None


def hanndle_blobs(query: str, pasted_texts: list[PastedText]):
 
    blobs: List[dict] = []

    query, explicit_blobs = extract_explicit_blobs(query)
    blobs.extend(explicit_blobs)

    for p in pasted_texts:

        blobs.append({
            "id": p.id,
            "content": p.content,
            "type": p.type,
        })

        # optional: replace inline refs if frontend uses IDs
        query = query.replace(p.content, f"[BLOB:{p.id}]")

    query = resolve_blob_references(query, blobs)

    return query, blobs








def handle_file_integrity(
    query: str,
    referenced_files: list[str],
    pasted_texts: list[PastedText]
):
    
    # -----------------------------------
    # 0. INIT BLOBS
    # -----------------------------------
    blobs: list[PastedText] = []

    # -----------------------------------
    # 1. EXTRACT EXPLICIT BLOBS
    # -----------------------------------
    query, explicit_blobs = extract_explicit_blobs(query)
    blobs.extend(explicit_blobs)

    # -----------------------------------
    # 2. ADD FRONTEND BLOBS
    # -----------------------------------
    for p in pasted_texts:
        blob = {
            "id": p.id,
            "content": p.content,
            "type": p.type
        }

        blobs.append(blob)

        # optional inline replacement
        query = query.replace(
            p.content,
            f"[BLOB:{p.id}]"
        )

    # -----------------------------------
    # 3. RESOLVE BLOB REFERENCES (STAGE 1)
    # -----------------------------------
    # IMPORTANT: this should ONLY normalize placeholders,
    # NOT expand content yet
    query = resolve_blob_references(query, blobs)

    # -----------------------------------
    # 4. FILE PROCESSING
    # -----------------------------------
    all_files = get_all_searchable_filenames()

    tagged_files, query = extract_file_tags(query, all_files)

    combined_files = list(
        dict.fromkeys(referenced_files + tagged_files)
    )

    if not combined_files:
        query, urls = extract_and_replace_urls(query)

        return {
            "status": "no_files_specified",
            "query": query,
            "urls": urls,
            "files_scope": [],
            "rejected_files": [],
            "blobs": blobs,
            "pasted_texts": pasted_texts,
        }

    searchable_files, non_searchable_files = validate_searchable_files(
        combined_files
    )

    if not searchable_files:
        query, urls = extract_and_replace_urls(query)

        return {
            "status": "requires_indexing",
            "query": query,
            "urls": urls,
            "files_scope": [],
            "rejected_files": non_searchable_files,
            "blobs": blobs,
            "pasted_texts": pasted_texts,
            "indexing_suggestion": {
                "message": "These files are not indexed yet.",
                "files": non_searchable_files,
                "action": "index_now",
                "next_step": "Would you like to index these files?"
            }
        }

    # -----------------------------------
    # 5. URL PROCESSING (AFTER EVERYTHING)
    # -----------------------------------
    query, urls = extract_and_replace_urls(query)

    # -----------------------------------
    # 6. FINAL STATUS
    # -----------------------------------
    status = "indexed" if not non_searchable_files else "partial"

    return {
        "status": status,
        "query": query,
        "urls": urls,
        "files_scope": searchable_files,
        "rejected_files": non_searchable_files,
        "blobs": blobs,
        "pasted_texts": pasted_texts
    }