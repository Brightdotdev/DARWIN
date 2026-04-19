import re
import unicodedata


# Regex patterns for semantic cleaning
_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^()]+|\([^()]*\))*\)")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_IMAGE_RE = re.compile(r"!\[.*?\]\(.*?\)")
_CODE_RE = re.compile(r"```.*?```|`[^`\n]+`", re.DOTALL)
_HTML_RE = re.compile(r"</?\w+[^>]*>|<!--.*?-->", re.DOTALL)
_MARKER_RE = re.compile(r"==>[^<]*?<==")


def prepare_embedding_text(
    text: str, 
    model_name: str,
    preserve_line_breaks: bool = True
) -> str:
    """
    Prepare text for embedding based on model capabilities.
    
    Args:
        text: Raw markdown/formatted text
        model_name: Model identifier from PRESETS keys
        preserve_line_breaks: Keep paragraph structure (recommended)
    
    Returns:
        Cleaned text optimized for the specific embedding model
    """
    
    # Normalize Unicode and line endings first
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\r\n?", "\n", text)
    model_config = get_model_config(model_name)
    # Determine cleaning strategy
    if model_config["category"] == "plain_text":
        return _clean_plain_text(text, preserve_line_breaks)
    elif model_config["category"]== "semantic_markdown":
        return _clean_semantic_markdown(text, preserve_line_breaks)
    else:
        # Unknown model - use conservative semantic cleaning
        return _clean_semantic_markdown(text, preserve_line_breaks)


def _clean_plain_text(text: str, preserve_line_breaks: bool = True) -> str:
    """
    Aggressive cleaning for models that don't understand formatting.
    Used for: e5-*, embed-english, embed-multilingual, ada-002
    """
    
    # Remove code blocks and inline code completely
    text = _CODE_RE.sub("", text)
    
    # Remove images completely
    text = _IMAGE_RE.sub("", text)
    
    # Links: keep text, remove URL
    text = _LINK_RE.sub(r"\1", text)
    
    # Strip formatting markers
    text = _BOLD_RE.sub(r"\1", text)
    text = _ITALIC_RE.sub(r"\1", text)
    
    # Remove HTML and custom markers
    text = _HTML_RE.sub("", text)
    text = _MARKER_RE.sub("", text)
    
    # Headers: remove # but keep text
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    
    # Lists: convert to simple bullets
    text = re.sub(r"^\s*[-*]\s+", "• ", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "• ", text, flags=re.MULTILINE)
    
    # Remove image placeholders
    text = re.sub(r"\[Image[^\]]*\]", "", text)
    
    # Remove special characters
    text = re.sub(r"[#>~|\\]", " ", text)
    text = re.sub(r"-{3,}", " ", text)
    
    # Table formatting: simplify
    text = re.sub(r"\s*\|\s*", " ", text)
    
    # Whitespace normalization
    if preserve_line_breaks:
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"\n\s+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
    else:
        text = re.sub(r"\s+", " ", text)
    
    text = re.sub(r"[ \t]{2,}", " ", text)
    
    return text.strip()


def _clean_semantic_markdown(text: str, preserve_line_breaks: bool = True) -> str:
    """
    Preserve semantic structure for models that understand markdown.
    Used for: bge-*, gritlm-*, voyage, text-embedding-3-*
    """
    
    # Keep code blocks - models trained on GitHub understand them
    # Just normalize spacing around them
    text = re.sub(r"```(\w*)\n", r"```\1\n", text)
    
    # Images: replace with semantic placeholder
    text = _IMAGE_RE.sub(_extract_image_semantic, text)
    
    # Links: keep as "text (url)" format
    text = _LINK_RE.sub(r"\1 (\2)", text)
    
    # Keep bold and italic - they signal emphasis
    # Just ensure proper formatting
    text = re.sub(r"\*\*\s+", "**", text)
    text = re.sub(r"\s+\*\*", "**", text)
    
    # Headers: normalize spacing after #
    text = re.sub(r"^(#{1,6})\s*", r"\1 ", text, flags=re.MULTILINE)
    
    # Lists: keep markers but normalize
    text = re.sub(r"^\s*[-*]\s+", "- ", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*(\d+)\.\s+", r"\1. ", text, flags=re.MULTILINE)
    
    # Tables: keep markdown table format (models understand it)
    # Just clean up alignment markers
    text = re.sub(r"\|[-\s:]+(\|[-\s:]+)+\|", "", text)  # Remove separator rows
    
    # Remove only truly useless elements
    text = _HTML_RE.sub("", text)
    text = _MARKER_RE.sub("", text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    
    # Remove problematic characters but keep markdown syntax
    text = re.sub(r"[>~\\]", " ", text)  # Keep # * - [ ] ( ) | etc.
    
    # Clean up image placeholders from our own system
    text = re.sub(r"\[Image:\d+\]", "", text)
    
    # Whitespace normalization (preserve structure)
    if preserve_line_breaks:
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
    else:
        text = re.sub(r"\s+", " ", text)
    
    return text.strip()


def _extract_image_semantic(match: re.Match) -> str:
    """
    Extract semantic information from image markdown for models
    that can use the context.
    """
    img_text = match.group(0)
    
    # Try to extract alt text/caption
    alt_match = re.search(r'!\[(.*?)\]', img_text)
    
    if alt_match and alt_match.group(1):
        alt_text = alt_match.group(1).strip()
        if alt_text and not alt_text.isspace():
            return f"[Image: {alt_text}]"
    
    # If it's our custom format with caption
    caption_match = re.search(r'Figure:\s*(.+?)(?:\n|$)', img_text)
    if caption_match:
        return f"[Image: {caption_match.group(1)}]"
    
    return "[Image]"


# Advanced: Model-specific token limits and chunking hints
def get_model_config(model_name: str) -> dict:
    """Get recommended chunking parameters for each model."""
    
    PRESETS = {
    
"intfloat/e5-small-v2":    {"max_tokens": 480,  "min_tokens": 120, "overlap": 3,  "category": "plain_text"},
"GritLM/GritLM-7B":              {"max_tokens": 3800, "min_tokens": 300, "overlap": 4,  "category": "semantic_markdown"},
"BAAI/bge-m3":    {"max_tokens": 8000, "min_tokens": 400, "overlap": 5,  "category": "semantic_markdown"},
    }
    return PRESETS.get(model_name, {"max_tokens": 512, "min_tokens": 100, "overlap": 3, "category": "plain_text"})



