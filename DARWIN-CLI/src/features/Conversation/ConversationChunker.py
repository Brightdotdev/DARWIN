"""
Conversation chunking logic.
Preserves turn boundaries, roles, temporal sequence.
"""

from typing import List, Tuple

from src.features.Conversation.ConversationModule import Turn
from src.features.Conversation.ConversationRAG.types import ConvoChunk


def chunk_conversation(turns: List[Turn]) -> List[ConvoChunk]:
    """
    Chunk turns for embedding.
    
    Rules:
    1. Combine consecutive user turns (clarifications, follow-ups)
    2. Assistant turns usually standalone (they're answers)
    3. Tool turns combine with invoking assistant turn
    4. Split on topic shift (heuristic: role change + content length)
    5. Max 3 turns per chunk, max 500 chars
    6. Preserve chronological order
    """
    if not turns:
        return []
    
    chunks = []
    buffer: List[Turn] = []
    
    for i, turn in enumerate(turns):
        # Determine if we should flush buffer
        should_flush = False
        
        if not buffer:
            buffer.append(turn)
            continue
        
        last = buffer[-1]
        
        # Rule 1: Role change user→assistant or assistant→user
        if last.role != turn.role:
            if turn.role == "assistant":
                # Flush before assistant answer
                should_flush = True
            elif turn.role == "user" and last.role == "user":
                # Combine user turns if under limit
                if len(buffer) >= 3:
                    should_flush = True
            elif turn.role == "tool":
                # Combine tool with previous assistant
                buffer.append(turn)
                continue
        
        # Rule 2: Content length overflow
        current_len = sum(len(t.content) for t in buffer)
        if current_len + len(turn.content) > 500:
            should_flush = True
        
        # Rule 3: Max turns per chunk
        if len(buffer) >= 3:
            should_flush = True
        
        if should_flush and buffer:
            chunks.append(_create_chunk(buffer, len(chunks)))
            buffer = [turn]
        else:
            buffer.append(turn)
    
    # Flush remaining
    if buffer:
        chunks.append(_create_chunk(buffer, len(chunks)))
    
    # Post-process: classify claims and preferences
    for chunk in chunks:
        _classify_chunk(chunk)
    
    return chunks


def _create_chunk(turns: List[Turn], chunk_idx: int) -> ConvoChunk:
    """Create ConvoChunk from buffer of turns."""
    combined_text = " ".join([
        f"{t.role}: {t.content}" for t in turns
    ])
    
    # Clean for embedding (strip artifacts, normalize)
    clean_text = _clean_for_embedding(combined_text)
    
    # Determine role
    roles = [t.role for t in turns]
    if len(set(roles)) == 1:
        role = roles[0]
    else:
        role = "mixed"
    
    return ConvoChunk(
        text=combined_text,
        clean=clean_text,
        index=chunk_idx,
        source=turns[0].metadata.thread_id,
        turn_range=(turns[0].metadata.get("turn_idx", 0), turns[-1].metadata.get("turn_idx", 0)),
        role=role,
        thread_id=turns[0].metadata.thread_id,
        timestamp=turns[0].timestamp,
        turn_ids=[t.turn_id for t in turns],
    )


def _clean_for_embedding(text: str) -> str:
    """Normalize text for embedding."""
    # Strip artifact references
    import re
    text = re.sub(r'artifact://\w+', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text.strip()


def _classify_chunk(chunk: ConvoChunk) -> None:
    """Heuristic classification for contradiction detection."""
    text_lower = chunk.clean.lower()
    
    # Preference patterns
    preference_markers = ["like", "love", "hate", "prefer", "want", "need", "should"]
    if any(m in text_lower for m in preference_markers):
        chunk.is_preference_statement = True
    
    # Factual claim patterns
    factual_markers = ["is", "are", "was", "were", "has", "have", "equals", "total"]
    if any(m in text_lower for m in factual_markers) and not chunk.is_preference_statement:
        chunk.is_factual_claim = True