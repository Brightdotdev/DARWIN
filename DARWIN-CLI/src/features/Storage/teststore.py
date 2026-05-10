
from src.features.Storage.DarwinStore import DarwinStore


def test_store() -> DarwinStore:
    store = DarwinStore()
    status = store.status()
    hybrid_filenames = [
    filename
    for filename, info in status["files"].items()
    if info.get("is_hybrid_searchable")
]

    print(hybrid_filenames)



def return_searchables(filename: str | None = None) -> bool: 
    store = DarwinStore()
    status = store.status()

    files = status.get("files", {})

    # Return all files
    if filename is None:
        return {
            fname: {
                "kw": info.get("is_kw_searchable", False),
                "hybrid": info.get("is_hybrid_searchable", False),
                "embedding": info.get("is_embedding_searchable", False),
            }
            for fname, info in files.items()
        }

    # Return single file
    info = files.get(filename)

    if not info:
        return False

    return info.get("is_hybrid_searchable", False)

def return_searchabl_files():
    store = DarwinStore()
    status = store.status()

    return {
        filename: {
            "kw": info.get("is_kw_searchable", False),
            "hybrid": info.get("is_hybrid_searchable", False),
            "embedding": info.get("is_embedding_searchable", False),
        }
        for filename, info in status.get("files", {}).items()
    }


if __name__ == "__main__":
    test_store()