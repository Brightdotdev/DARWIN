


# ============================================================================
# Constants
# ============================================================================
from src.lib.Terminal import TerminalUI
import psutil
import torch



DEFAULT_MODEL      = "intfloat/e5-small-v2"
DEFAULT_BATCH_SIZE = 64




def get_optimal_batch_size(requested_size: int = 32) -> int:
    """
    Detects LIVE hardware availability to set a safe batch size.
    """
    # 1. GPU Check (High Priority)
    if torch.cuda.is_available():
        vram_free = torch.cuda.mem_get_info()[0] / (1024**3) # Free VRAM in GB
        if vram_free > 4: return 128
        return 64

    # 2. CPU/RAM Check (The "Chrome" Protector)
    # We look at 'available' RAM, not 'total'
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    cpu_usage = psutil.cpu_percent(interval=0.1)

    # Logic: If system is already struggling, go small
    if available_gb < 2.0 or cpu_usage > 80:
        TerminalUI.warn("System resources low! Throttling batch size for stability.")
        return 16

    # For your 12GB Dell (usually has 4-6GB available with apps open)
    if available_gb > 4.0:
        return 64
    
    return requested_size # Fallback to 32

# ============================================================================
# Model loader — lazy, cached per session
# ============================================================================

_model_cache: dict = {}


def load_model(model_id: str):
    if model_id in _model_cache:
        return _model_cache[model_id]

    import torch # Add this import
    from sentence_transformers import SentenceTransformer

    # Auto-detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t = TerminalUI.timer(f"load_model:{model_id.split('/')[-1]}")
    TerminalUI.dim(f"Loading {model_id} on [bold]{device.upper()}[/bold]")

    # Pass the device to the model
    model = SentenceTransformer(model_id, device=device)
    _model_cache[model_id] = model

    t.done()
    return model

