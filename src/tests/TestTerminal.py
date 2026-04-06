from src.lib.Terminal import TerminalUI

# Bold text
TerminalUI.bold("IMPORTANT: This is critical information")
TerminalUI.bold_inline("Loading ")
TerminalUI.dim_inline("please wait...")

# Color shortcuts
TerminalUI.cyan("python main.py start")
TerminalUI.yellow("Your query: photosynthesis")
TerminalUI.green("✓ Indexing complete")

# Combined formatting
TerminalUI.bold("Step 1:")
TerminalUI.dim(" (required)")

# Panel for important messages
TerminalUI.panel(
    "Your graph has been built successfully!\nRun 'python main.py graph --file document.pdf' to view it.",
    title="Success",
    border_style="green"
)

# Rule for section separation
TerminalUI.rule("Search Results")
TerminalUI.blank()
TerminalUI.dim("Found 5 results")

# Practical example for onboarding
TerminalUI.header("LLM Configuration", level=2)
TerminalUI.step("Checking API key")
TerminalUI.done()
TerminalUI.success("OpenRouter API key found")
TerminalUI.bold("Selected model:")
TerminalUI.dim("Llama 3.2 3B (free tier)")