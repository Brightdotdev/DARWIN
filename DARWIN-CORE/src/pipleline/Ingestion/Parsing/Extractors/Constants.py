from typing import Final

# --- Image extraction limits ---
MIN_IMAGE_SIZE_PX:     Final[int] = 100
MAX_IMAGES_PER_PAGE:   Final[int] = 10
MAX_IMAGES_PER_DOC:    Final[int] = 200

# --- Heading detection thresholds ---
HEADING_SIZE_RATIO:    Final[float] = 1.20
CAPTION_SIZE_RATIO:    Final[float] = 0.85
LARGE_HEADING_RATIO:   Final[float] = 1.50

# --- Table detection heuristics ---
TABLE_X_SPREAD:        Final[float] = 0.45
TABLE_MIN_SPANS:       Final[int] = 6
TABLE_MIN_COLS:        Final[int] = 3
TABLE_MIN_ROWS:        Final[int] = 3
TABLE_Y_GAP_RATIO:     Final[float] = 2.5
COLUMN_BUCKETS_FALLBACK: Final[int] = 2