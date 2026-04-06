"""
src/ui/Cli.py
-------------
Textual-based TUI for Darwin with Kilo Code style.

States:
    1. Not onboarded → Show onboarding wizard
    2. Onboarded but no indexes → Show indexing progress  
    3. Ready → Show search interface
"""

from pathlib import Path
from typing import AsyncGenerator, Optional
import asyncio
import time

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal, Center, ScrollableContainer
from textual.widgets import Header, Footer, Input, Button, Static, Label, LoadingIndicator, ProgressBar, ListView, ListItem
from textual.screen import Screen
from textual.reactive import reactive
from textual import events
from rich.text import Text
from rich.panel import Panel

from src.core.Query import SearchResult
from src.core.Store import DarwinStore
from src.core.Onboarding import needs_onboarding
from src.lib.Terminal import TerminalUI


# ── ASCII Art ──────────────────────────────────────────────────────────────────

DARWIN_ASCII = [
    "  ██████╗  █████╗ ██████╗ ██╗    ██╗██╗███╗   ██╗",
    "  ██╔══██╗██╔══██╗██╔══██╗██║    ██║██║████╗  ██║",
    "  ██║  ██║███████║██████╔╝██║ █╗ ██║██║██╔██╗ ██║",
    "  ██║  ██║██╔══██║██╔══██╗██║███╗██║██║██║╚██╗██║",
    "  ██████╔╝██║  ██║██║  ██║╚███╔███╔╝██║██║ ╚████║",
    "  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝╚═╝  ╚═══╝",
]


class CenteredASCII(Static):
    """Centered ASCII art widget"""
    
    def render(self) -> Text:
        text = Text()
        for line in DARWIN_ASCII:
            text.append(line + "\n", style="cyan bold")
        return text


# ── Onboarding Screen (Pure Textual - No Questionary) ─────────────────────────

class ProviderSelectScreen(Screen):
    """LLM Provider selection"""
    
    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label("Step 1/3: Select LLM Provider", classes="title")
            yield Label("Choose your language model provider:", classes="subtitle")
            yield ListView(id="provider-list")
            yield Horizontal(
                Button("Cancel", id="cancel", variant="default"),
                Button("Next", id="next", variant="primary"),
                classes="button-row"
            )
    
    def on_mount(self) -> None:
        list_view = self.query_one("#provider-list")
        providers = [
            ("OpenRouter", "recommended — free models available"),
            ("Ollama", "fully local, no key needed"),
            ("Anthropic", "Claude models, needs API key"),
            ("OpenAI", "GPT models, needs API key"),
            ("DeepSeek", "DeepSeek models, needs API key"),
            ("None", "skip LLM — raw graph only"),
        ]
        for provider, note in providers:
            list_view.append(ListItem(Label(f"{provider}\n[dim]{note}[/dim]")))
    
    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.app.selected_provider = event.item.index
        self.app.push_screen(ModelSelectScreen())


class ModelSelectScreen(Screen):
    """Model selection based on provider"""
    
    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label("Step 1/3: Select Model", classes="title")
            yield Label("Choose the specific model:", classes="subtitle")
            yield ListView(id="model-list")
            yield Horizontal(
                Button("Back", id="back", variant="default"),
                Button("Next", id="next", variant="primary"),
                classes="button-row"
            )
    
    def on_mount(self) -> None:
        list_view = self.query_one("#model-list")
        
        # Models based on selected provider
        models = {
            0: [  # OpenRouter
                ("Llama 3.2 3B", "free tier, fast, good for extraction"),
                ("Llama 3.1 8B", "free tier, smarter, slightly slower"),
                ("Gemma 3 4B", "free tier, Google model"),
                ("Claude Haiku", "paid, fast and cheap"),
                ("GPT-4o Mini", "paid, fast, good quality"),
            ],
            1: [  # Ollama
                ("Llama 3.2 (3B)", "run: ollama pull llama3.2"),
                ("Llama 3.1 (8B)", "run: ollama pull llama3.1"),
                ("Mistral 7B", "run: ollama pull mistral"),
            ],
            2: [  # Anthropic
                ("Claude Haiku", "fast, cheap, great for extraction"),
                ("Claude Sonnet", "smarter, costs more"),
            ],
            3: [  # OpenAI
                ("GPT-4o Mini", "fast, cheap, good quality"),
                ("GPT-4o", "smarter, costs more"),
            ],
            4: [  # DeepSeek
                ("DeepSeek V3", "fast, very affordable"),
                ("DeepSeek R1", "reasoning model, slower"),
            ],
            5: [],  # None
        }
        
        for model, note in models.get(self.app.selected_provider, []):
            list_view.append(ListItem(Label(f"{model}\n[dim]{note}[/dim]")))
    
    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.app.selected_model = event.item.index
        self.app.push_screen(EmbeddingSelectScreen())


class EmbeddingSelectScreen(Screen):
    """Embedding model selection"""
    
    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label("Step 2/3: Select Embedding Model", classes="title")
            yield Label("Converts text to vectors for semantic search:", classes="subtitle")
            yield ListView(id="embedding-list")
            yield Horizontal(
                Button("Back", id="back", variant="default"),
                Button("Next", id="next", variant="primary"),
                classes="button-row"
            )
    
    def on_mount(self) -> None:
        list_view = self.query_one("#embedding-list")
        models = [
            ("e5-small", "light — 8GB RAM, default, fast", "intfloat/e5-small-v2"),
            ("BGE-M3", "medium — 16GB RAM, better quality", "BAAI/bge-m3"),
        ]
        for model, note, model_id in models:
            list_view.append(ListItem(Label(f"{model}\n[dim]{note}[/dim]")))
            self.app.embedding_options = {model: model_id for model, note, model_id in models}
    
    def on_list_view_selected(self, event: ListView.Selected) -> None:
        selected_label = event.item.children[0].renderable.split('\n')[0]
        self.app.selected_embedding = self.app.embedding_options.get(selected_label, "intfloat/e5-small-v2")
        self.app.push_screen(RerankerSelectScreen())


class RerankerSelectScreen(Screen):
    """Reranker selection"""
    
    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label("Step 3/3: Select Reranker", classes="title")
            yield Label("Improves search result ordering:", classes="subtitle")
            yield ListView(id="reranker-list")
            yield Horizontal(
                Button("Back", id="back", variant="default"),
                Button("Complete", id="complete", variant="primary"),
                classes="button-row"
            )
    
    def on_mount(self) -> None:
        list_view = self.query_one("#reranker-list")
        models = [
            ("TinyBERT", "light — fast, low memory, default", "cross-encoder/ms-marco-TinyBERT-L-2-v2"),
            ("MiniLM L6", "medium — better quality, slightly slower", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        ]
        for model, note, model_id in models:
            list_view.append(ListItem(Label(f"{model}\n[dim]{note}[/dim]")))
            self.app.reranker_options = {model: model_id for model, note, model_id in models}
    
    def on_list_view_selected(self, event: ListView.Selected) -> None:
        selected_label = event.item.children[0].renderable.split('\n')[0]
        self.app.selected_reranker = self.app.reranker_options.get(selected_label, "cross-encoder/ms-marco-TinyBERT-L-2-v2")
        self.app.complete_onboarding()


class OnboardingScreen(Screen):
    """Main onboarding container"""
    
    CSS = """
    #dialog {
        width: 60%;
        height: 70%;
        border: solid #89b4fa;
        background: #1e1e2e;
        padding: 2;
        margin: 2;
    }
    
    .title {
        text-align: center;
        text-style: bold;
        color: #89b4fa;
        margin: 1 0;
    }
    
    .subtitle {
        text-align: center;
        color: #6c7086;
        margin-bottom: 2;
    }
    
    .button-row {
        margin-top: 2;
        align: center;
    }
    
    Button {
        margin: 0 1;
    }
    
    ListView {
        height: 60%;
        margin: 1 0;
    }
    
    ListItem {
        padding: 1;
    }
    
    ListItem:hover {
        background: #313244;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield CenteredASCII()
        yield ProviderSelectScreen()


# ── Indexing Screen ────────────────────────────────────────────────────────────

class IndexingScreen(Screen):
    """Shows progress during file indexing"""
    
    def compose(self) -> ComposeResult:
        with Container(id="indexing-container"):
            yield CenteredASCII()
            yield Label("Indexing Documents", classes="title")
            yield Label("Processing files in your project...", classes="subtitle")
            yield ProgressBar(id="overall-progress", total=100)
            yield Label("", id="current-file")
            yield Label("", id="file-progress", classes="dim")
            yield Label("", id="stats", classes="dim")
    
    async def on_mount(self) -> None:
        """Start indexing process"""
        overall_bar = self.query_one("#overall-progress")
        current_file_label = self.query_one("#current-file")
        file_progress_label = self.query_one("#file-progress")
        stats_label = self.query_one("#stats")
        
        # Get all files to index
        from src.core.Parser import SUPPORTED_TYPES
        
        directory = self.app.project_dir
        
        all_files = [
            p for p in directory.iterdir()
            if p.is_file()
            and p.suffix.lower() in SUPPORTED_TYPES
            and not p.name.startswith(".")
        ]
        
        if not all_files:
            stats_label.update("No supported files found. Add PDF, TXT, or MD files to your project.")
            await asyncio.sleep(2)
            self.app.push_screen(SearchScreen())
            return
        
        overall_bar.total = len(all_files)
        indexed_count = 0
        cached_count = 0
        error_count = 0
        
        for i, file_path in enumerate(all_files):
            filename = file_path.name
            current_file_label.update(f"📄 {filename}")
            
            # Check freshness
            if self.app.store.is_fresh(filename):
                cached_count += 1
                file_progress_label.update("✓ Cached (unchanged)")
            else:
                file_progress_label.update("Processing...")
                
                try:
                    # Parse
                    from src.core.Parser import parse
                    doc = parse(file_path, strategy="light")
                    
                    if doc.error:
                        error_count += 1
                        file_progress_label.update(f"✗ {doc.error}")
                    else:
                        # Chunk
                        from src.core.Chunker import chunk_document
                        chunks = chunk_document(doc.text, source=filename)
                        
                        # Embed
                        from src.core.Embedder import Embedder
                        embedder = Embedder()
                        result = embedder.embed(chunks)
                        embeddings = [ec.embedding for ec in result.embedded_chunks]
                        
                        # BM25
                        from src.core.Query import BM25Index
                        bm25 = BM25Index()
                        bm25.build(chunks)
                        
                        # Save
                        import numpy as np
                        self.app.store.save_chunks(filename, chunks, page_count=doc.page_count, parser="light")
                        self.app.store.save_embeddings(filename, np.array(embeddings))
                        self.app.store.save_bm25(filename, bm25)
                        self.app.store.save_stopwords(filename, bm25._domain_stopwords)
                        
                        indexed_count += 1
                        file_progress_label.update("✓ Indexed")
                        
                except Exception as e:
                    error_count += 1
                    file_progress_label.update(f"✗ {str(e)[:50]}")
            
            overall_bar.advance(1)
            stats_label.update(f"📊 Indexed: {indexed_count} | Cached: {cached_count} | Errors: {error_count}")
            await asyncio.sleep(0.1)  # Allow UI to update
        
        stats_label.update(f"✅ Complete! Indexed {indexed_count} files, cached {cached_count}, errors {error_count}")
        await asyncio.sleep(1.5)
        self.app.push_screen(SearchScreen())


# ── Search Screen ──────────────────────────────────────────────────────────────

class SearchScreen(Screen):
    """Main search interface - Kilo Code style"""
    
    CSS = """
    Screen {
        background: #1e1e2e;
    }
    
    #search-container {
        width: 100%;
        height: 100%;
        padding: 2;
    }
    
    #ascii-container {
        height: auto;
        margin-bottom: 2;
    }
    
    CenteredASCII {
        text-align: center;
        width: 100%;
    }
    
    .search-bar-container {
        margin: 1 4;
        border: solid #89b4fa;
        background: #313244;
    }
    
    Input {
        background: #313244;
        color: #cdd6f4;
        padding: 1;
    }
    
    Input:focus {
        border: solid #89b4fa;
    }
    
    .quick-actions {
        margin: 1 4;
        height: auto;
    }
    
    .action-button {
        margin: 0 1;
        background: #181825;
        color: #89b4fa;
    }
    
    .action-button:hover {
        background: #89b4fa;
        color: #1e1e2e;
    }
    
    #results-container {
        margin: 1 2;
        height: 60%;
        overflow-y: auto;
    }
    
    .result-card {
        margin: 1 0;
        padding: 1;
        border: solid #313244;
        background: #181825;
    }
    
    .result-card:hover {
        background: #1e1e2e;
        border: solid #89b4fa;
    }
    
    .result-title {
        text-style: bold;
        color: #89b4fa;
    }
    
    .result-score {
        color: #a6e3a1;
        margin-left: 1;
    }
    
    .result-snippet {
        color: #cdd6f4;
        margin-top: 1;
    }
    
    .result-meta {
        color: #6c7086;
        margin-top: 1;
    }
    
    .status-bar {
        height: 3;
        background: #181825;
        padding: 0 1;
        margin-top: 1;
    }
    
    .dim {
        color: #6c7086;
    }
    """
    
    def compose(self) -> ComposeResult:
        with Container(id="search-container"):
            with Container(id="ascii-container"):
                yield CenteredASCII()
            
            yield Label("🔍 Search Your Knowledge Base", classes="title")
            
            with Container(classes="search-bar-container"):
                yield Input(placeholder="Ask a question...", id="search-input")
            
            with Horizontal(classes="quick-actions"):
                yield Button("Search", variant="primary", id="search-btn", classes="action-button")
                yield Button("Config", id="config-btn", classes="action-button")
                yield Button("Re-index", id="reindex-btn", classes="action-button")
                yield Button("Quit", id="quit-btn", classes="action-button")
            
            with ScrollableContainer(id="results-container"):
                yield Label("Enter a query above to search your documents", classes="dim")
            
            with Container(classes="status-bar"):
                yield self.render_status()
    
    def render_status(self) -> Static:
        """Render status bar with project info"""
        status = self.app.store.status()
        files_count = len(status["files"])
        chunks_total = sum(f["chunk_count"] for f in status["files"])
        
        return Static(
            f"📁 {files_count} files  |  📄 {chunks_total} chunks  |  "
            f"💾 {status['config'].get('embedding_model', 'e5-small')}  |  "
            f"Ctrl+P: Search  |  Ctrl+C: Config  |  Ctrl+Q: Quit",
            classes="dim"
        )
    
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in search input"""
        if event.input.id == "search-input" and event.value:
            await self.perform_search(event.value)
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks"""
        if event.button.id == "search-btn":
            query = self.query_one("#search-input").value
            if query:
                await self.perform_search(query)
        elif event.button.id == "config-btn":
            self.app.run_config_screen()
        elif event.button.id == "reindex-btn":
            self.app.push_screen(IndexingScreen())
        elif event.button.id == "quit-btn":
            self.app.exit()
    
    async def perform_search(self, query: str) ->  AsyncGenerator[SearchResult, None]:
        """Execute search and display results"""
        results_container = self.query_one("#results-container")
        
        # Show loading
        await results_container.remove_children()
        await results_container.mount(LoadingIndicator())
        
        try:
            # Run search in thread
            from src.core.Search import index_directory, build_pipeline
            
            # Get all chunks
            all_chunks = await asyncio.to_thread(
                index_directory,
                self.app.store,
                self.app.project_dir,
                "light",
                None
            )
            
            if not all_chunks:
                await results_container.remove_children()
                await results_container.mount(Label("No indexed files found. Run re-index first.", classes="dim"))
                return
            
            # Build pipeline
            pipeline = await asyncio.to_thread(build_pipeline, self.app.store, all_chunks)
            
            # Search
            result = await asyncio.to_thread(pipeline.query, query, top_k=5)
            
            # Display results
            await results_container.remove_children()
            
            if not result.results:
                await results_container.mount(Label("No results found. Try a different query.", classes="dim"))
                return
            
            for r in result.results:
                with Container(classes="result-card"):
                    yield Static(
                        f"#{r.rank + 1}  {r.chunk.source}  (score: {r.rerank_score:.3f})",
                        classes="result-title"
                    )
                    yield Static(r.chunk.text[:300] + ("..." if len(r.chunk.text) > 300 else ""), classes="result-snippet")
                    yield Static(
                        f"Semantic: {r.semantic_score:.3f}  |  BM25: {r.bm25_score:.1f}  |  RRF: {r.rrf_score:.4f}",
                        classes="result-meta"
                    )
            
        except Exception as e:
            await results_container.remove_children()
            await results_container.mount(Label(f"Error: {str(e)}", classes="dim"))
    
    async def on_key(self, event: events.Key) -> None:
        """Handle keyboard shortcuts"""
        if event.key == "ctrl+c":
            self.app.run_config_screen()
        elif event.key == "ctrl+q":
            self.app.exit()
        elif event.key == "ctrl+p":
            self.query_one("#search-input").focus()


# ── Main Application ───────────────────────────────────────────────────────────

class DarwinApp(App):
    """Main Darwin TUI Application"""
    
    CSS = """
    .title {
        text-align: center;
        text-style: bold;
        color: #89b4fa;
        margin: 1 0;
    }
    
    .subtitle {
        text-align: center;
        color: #6c7086;
        margin-bottom: 2;
    }
    
    #indexing-container {
        width: 100%;
        height: 100%;
        padding: 2;
    }
    
    ProgressBar {
        margin: 1 4;
    }
    """
    
    def __init__(self, project_dir: Path):
        super().__init__()
        self.project_dir = project_dir
        self.store = DarwinStore(project_dir)
        self.store.init()
        
        # Onboarding state
        self.selected_provider = 0
        self.selected_model = 0
        self.selected_embedding = "intfloat/e5-small-v2"
        self.selected_reranker = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
        self.embedding_options = {}
        self.reranker_options = {}
    
    def complete_onboarding(self) -> None:
        """Save onboarding config and continue"""
        # Map provider index to ID
        provider_map = {
            0: "openrouter",
            1: "ollama", 
            2: "anthropic",
            3: "openai",
            4: "deepseek",
            5: "none",
        }
        
        # Map model index to ID (simplified)
        model_map = {
            (0,0): "openrouter/meta-llama/llama-3.2-3b-instruct:free",
            (0,1): "openrouter/meta-llama/llama-3.1-8b-instruct:free",
            (0,2): "openrouter/google/gemma-3-4b-it:free",
            (0,3): "openrouter/anthropic/claude-haiku-4-5",
            (0,4): "openrouter/openai/gpt-4o-mini",
            (1,0): "ollama/llama3.2",
            (1,1): "ollama/llama3.1",
            (1,2): "ollama/mistral",
            (2,0): "claude-haiku-4-5-20251001",
            (2,1): "claude-sonnet-4-6",
            (3,0): "gpt-4o-mini",
            (3,1): "gpt-4o",
            (4,0): "deepseek/deepseek-chat",
            (4,1): "deepseek/deepseek-reasoner",
        }
        
        provider_id = provider_map.get(self.selected_provider, "none")
        model_id = model_map.get((self.selected_provider, self.selected_model), "")
        
        if provider_id == "none":
            llm_mode = "none"
        elif provider_id == "ollama":
            llm_mode = "local"
        else:
            llm_mode = "api"
        
        config = self.store.load_config()
        config.update({
            "llm_mode": llm_mode,
            "llm_provider": provider_id,
            "llm_model": model_id,
            "llm_key_name": f"{provider_id}_api_key" if provider_id != "ollama" else "",
            "embedding_model": self.selected_embedding,
            "reranker_model": self.selected_reranker,
            "onboarded": True,
        })
        self.store.save_config(config)
        
        # Move to indexing screen
        self.push_screen(IndexingScreen())
    
    def run_config_screen(self) -> None:
        """Simple config using Textual dialog"""
        def update_config(choice):
            if choice == "Re-index":
                self.push_screen(IndexingScreen())
            elif choice == "Quit":
                self.exit()
        
        # For now, just re-index
        self.push_screen(IndexingScreen())
    
    def on_mount(self) -> None:
        """Determine which screen to show based on state"""
        if needs_onboarding(self.store):
            self.push_screen(OnboardingScreen())
        else:
            # Check if any files are indexed
            status = self.store.status()
            has_indexed = any(
                f["has_chunks"] and f["has_embeddings"] and f["has_bm25"]
                for f in status["files"]
            )
            
            if not has_indexed:
                self.push_screen(IndexingScreen())
            else:
                self.push_screen(SearchScreen())


def run_tui(project_dir: str | Path) -> None:
    """Entry point for TUI mode"""
    app = DarwinApp(Path(project_dir))
    app.run()


if __name__ == "__main__":
    import sys
    project = sys.argv[1] if len(sys.argv) > 1 else "."
    run_tui(project)