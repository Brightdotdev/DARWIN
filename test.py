

from src.core.Storage.DarwinStore import DarwinStore
from src.core.Models.EmbeddingConfig import get_preset
from src.core.Models import  setup


setup()
store = DarwinStore()
preset = get_preset()
print(f'Current model: {preset.model_name}')
print(preset)
print(f'Has embeddings: {store.has_embeddings("FF_ONE_HUNDRED_WAYS_TO_USE_AI.pdf")}')
print(f'Has chunks: {store.has_chunks("FF_ONE_HUNDRED_WAYS_TO_USE_AI.pdf")}')

# Check what files actually exist
from pathlib import Path
import os
darwin = Path.home() / 'AppData' / 'Local' / 'Darwin' / 'indexes'
for d in darwin.iterdir():
    if d.is_dir():
        files = list(d.glob('*'))
        if any('embed' in f.name for f in files):
            print(f'\n{d.name}:')
            for f in sorted(files):
                print(f'  {f.name}')