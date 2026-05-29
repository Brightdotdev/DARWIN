from src.features.Ingestion.Chunking.ChunkerUtils import tokens

with open('Darwin.md', 'r') as f:
    darwin_prompt = f.read()
    
    token_count = tokens(darwin_prompt)
    print(f"Darwin prompt token count: {token_count}")