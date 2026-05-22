
# Quick test
if __name__ == "__main__":
    from src.features.Storage.DarwinStore import DarwinStore
    from src.features.Knowledge.WikiOrchestrator import WikiOrchestrator
    
    store = DarwinStore()
    orchestrator = WikiOrchestrator(store)
    query = "who are the watchers"
    context = orchestrator.search(
        query=query,
        filename="book_of_enoch.pdf",
        top_k=3,
    )
    print(context)
    prompt = orchestrator.build_llm_prompt(context, query)
    print(prompt)