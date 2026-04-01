from pathlib import Path

text = Path('.agent/sources/CSB_Pew_Bible_2nd_Printing.md').read_text(encoding='utf-8')
print(f'Characters: {len(text):,}')
print(f'Words:      {len(text.split()):,}')
print(f'Lines:      {len(text.splitlines()):,}')