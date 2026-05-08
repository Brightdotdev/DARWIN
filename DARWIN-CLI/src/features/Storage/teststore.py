
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



def return_searchables(filename: str) -> tuple[bool, bool, bool] | None:
    from src.features.Storage.DarwinStore import DarwinStore
    store = DarwinStore()
    status = store.status()

    file_info = status.get("files", {}).get(filename)

    if not file_info:
        return None

    return (
        file_info.get("is_kw_searchable", False),
        file_info.get("is_hybrid_searchable", False),
        file_info.get("is_embedding_searchable", False),
    )


if __name__ == "__main__":
    test_store()