from src.core.Storage.DarwinStore import DarwinStore


def _validated_status(store: DarwinStore, auto_repair: bool = False) -> dict:
    """
    Get store.status() but validated against actual artifact files.
    
    Args:
        store: DarwinStore instance
        auto_repair: If True, fix the registry automatically when lies are found
    
    Returns:
        Same structure as store.status() but with corrected boolean flags
    """
    # Run the integrity check across ALL files
    health = store.full_registry_health_check()
    
    # If there are broken files and auto_repair is on, fix them
    if auto_repair and health["summary"]["broken"] > 0:
        store.repair_registry()  # This fixes everything
    
    # Get the status (now accurate if we repaired)
    status = store.status()
    files = status.get("files", {})
    
    # Overlay validation results onto the status
    for filename, integrity in health["all_files"].items():
        if filename in files:
            # Any stage that's "actually_missing" means the boolean was a lie
            if integrity["actually_missing"]:
                files[filename]["integrity_issues"] = integrity["actually_missing"]
                
                # Correct the booleans based on what's actually missing
                if "markdown" in integrity["actually_missing"]:
                    files[filename]["is_parsed"] = False
                if "chunks" in integrity["actually_missing"]:
                    files[filename]["is_chunked"] = False
                if "embeddings" in integrity["actually_missing"]:
                    files[filename]["is_embedded"] = False
                if "bm25" in integrity["actually_missing"]:
                    files[filename]["is_kw_searchable"] = False
                if "vector" in integrity["actually_missing"]:
                    files[filename]["is_embedding_searchable"] = False
                
                # Recalculate hybrid (requires ALL of the above)
                files[filename]["is_hybrid_searchable"] = (
                    files[filename].get("is_kw_searchable", False) and
                    files[filename].get("is_embedding_searchable", False)
                )
            else:
                files[filename]["integrity_issues"] = []
    
    status["integrity_summary"] = {
        "total_files": health["summary"]["total_files"],
        "healthy": health["summary"]["healthy"],
        "broken": health["summary"]["broken"],
        "auto_repaired": auto_repair and health["summary"]["broken"] > 0,
    }
    
    return status


def test_store():
    """Quick test that validates integrity."""
    store = DarwinStore()
    
    # Get validated status (read-only check)
    status = _validated_status(store, auto_repair=False)
    
    hybrid_filenames = [
        filename
        for filename, info in status["files"].items()
        if info.get("is_hybrid_searchable")
    ]
    
    print(f"\n📊 Integrity Summary: {status['integrity_summary']}")
    
    if status["integrity_summary"]["broken"] > 0:
        print(f"\n⚠️  WARNING: {status['integrity_summary']['broken']} files have integrity issues!")
        for filename, info in status["files"].items():
            if info.get("integrity_issues"):
                print(f"   📄 {filename}: registry claimed done, but missing: {info['integrity_issues']}")
    
    print(f"\n✅ Actually hybrid-searchable files ({len(hybrid_filenames)}):")
    for name in hybrid_filenames:
        print(f"   - {name}")
    
    return hybrid_filenames


def return_searchables(filename: str | None = None, validate: bool = True) -> dict | bool:
    """
    Get searchability status for files, with optional integrity validation.
    
    Args:
        filename: Specific file to check, or None for all files
        validate: If True, verify registry against actual artifact files
    
    Returns:
        For single file: bool (is_hybrid_searchable)
        For all files: dict of filename -> {kw, hybrid, embedding, integrity_issues}
    """
    store = DarwinStore()
    
    if validate:
        status = _validated_status(store, auto_repair=False)
    else:
        status = store.status()
    
    files = status.get("files", {})
    
    # Return all files
    if filename is None:
        return {
            fname: {
                "kw": info.get("is_kw_searchable", False),
                "hybrid": info.get("is_hybrid_searchable", False),
                "embedding": info.get("is_embedding_searchable", False),
                "integrity_issues": info.get("integrity_issues", []),
            }
            for fname, info in files.items()
        }
    
    # Return single file
    info = files.get(filename)
    if not info:
        return False
    
    return info.get("is_hybrid_searchable", False)


def return_searchable_files(validate: bool = True, auto_repair: bool = False) -> dict:
    """
    Get all files with their searchability status.
    
    Args:
        validate: Run integrity check against artifacts
        auto_repair: If True AND validate is True, auto-fix registry lies
    
    Returns:
        dict of filename -> {kw, hybrid, embedding, integrity_issues}
    """
    store = DarwinStore()
    
    if validate:
        status = _validated_status(store, auto_repair=auto_repair)
    else:
        status = store.status()
    
    return {
        filename: {
            "kw": info.get("is_kw_searchable", False),
            "hybrid": info.get("is_hybrid_searchable", False),
            "embedding": info.get("is_embedding_searchable", False),
            "integrity_issues": info.get("integrity_issues", []),
        }
        for filename, info in status.get("files", {}).items()
    }


def get_truly_searchable_files() -> list[str]:
    """
    Get list of filenames that are ACTUALLY hybrid-searchable
    (validated against real artifact files).
    """
    store = DarwinStore()
    status = _validated_status(store, auto_repair=True)  # Fix any lies
    
    return [
        filename
        for filename, info in status["files"].items()
        if info.get("is_hybrid_searchable")
    ]


def get_files_for_pipeline() -> dict:
    """
    Get files that need pipeline work, but validated.
    Returns dict with:
        - needs_parsing: files missing markdown artifacts
        - needs_chunking: files with markdown but no chunks
        - needs_embedding: files with chunks but no embeddings
    """
    store = DarwinStore()
    status = _validated_status(store, auto_repair=True)
    
    needs_parsing = []
    needs_chunking = []
    needs_embedding = []
    needs_bm25 = []
    
    for filename, info in status["files"].items():
        issues = info.get("integrity_issues", [])
        
        if not info.get("is_parsed") or "markdown" in issues:
            needs_parsing.append(filename)
        elif not info.get("is_chunked") or "chunks" in issues:
            needs_chunking.append(filename)
        elif not info.get("is_embedded") or "embeddings" in issues:
            needs_embedding.append(filename)
        elif not info.get("is_kw_searchable") or "bm25" in issues:
            needs_bm25.append(filename)
    
    return {
        "needs_parsing": needs_parsing,
        "needs_chunking": needs_chunking,
        "needs_embedding": needs_embedding,
        "needs_bm25": needs_bm25,
        "total_needing_work": len(needs_parsing) + len(needs_chunking) + 
                              len(needs_embedding) + len(needs_bm25),
    }


if __name__ == "__main__":
    # Basic test
    hybrid = test_store()
    
    print("\n" + "="*60)
    print("VALIDATED FILE STATUS")
    print("="*60)
    
    files = return_searchable_files(validate=True)
    for name, status in files.items():
        icon = "✅" if status["hybrid"] else "⏳"
        issues = f" (⚠️ {status['integrity_issues']})" if status["integrity_issues"] else ""
        print(f"  {icon} {name}: kw={status['kw']}, hybrid={status['hybrid']}{issues}")
    
    # Show pipeline needs
    print("\n" + "="*60)
    print("PIPELINE NEEDS")
    print("="*60)
    pipeline = get_files_for_pipeline()
    for stage, files in pipeline.items():
        if stage != "total_needing_work" and files:
            print(f"  {stage}: {files}")
    print(f"  Total needing work: {pipeline['total_needing_work']}")