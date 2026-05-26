
# Quick test
if __name__ == "__main__":
    from src.core.Storage.DarwinStore import DarwinStore
    from src.core.Logger.DarwinLogger import darwin, LogType, Verbosity


    from src.features.Knowledge.main import handle_knowledge,KnowledgePipelineArgs
    store = DarwinStore()
    files =["book_of_enoch.pdf", "CSB_Pew_Bible_2nd_Printing.pdf"]
    
    darwin.configure(
        enable_websocket=False,
        enable_history=True,
        verbosity_default=Verbosity.NORMAL
    )
    
    knowledge_args = KnowledgePipelineArgs(
            files=files,
            visualize=True,
            inspect=True,
            rerun=True,
        )

    handle_knowledge(knowledge_args, store)

    for f in files:
        filewiki = store.load_wiki(f)
        f"{f} has {len(filewiki.pages)} pages"
        for page in filewiki.pages:
            f"{page.concept} has {page.frequency} chunks"