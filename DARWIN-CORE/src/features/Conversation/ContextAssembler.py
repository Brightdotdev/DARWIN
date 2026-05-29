import json
from typing import Dict, Any, List

from src.features.Conversation.ConversationConfig import ConversationConfig


class ContextAssembler:
    def __init__(self, config: ConversationConfig):
        self.config = config
        self._token_estimate = 0

    def assemble(
        self,
        system_prompt: str,
        memories: List[Dict[str, Any]],
        turns: List[Dict[str, Any]],
        user_message: str,
        available_tools: List[str],
    ) -> List[Dict[str, Any]]:
        messages = []

        tool_list = ", ".join(available_tools)
        system_content = f"{system_prompt}\n\nAvailable tools: {tool_list}"
        messages.append({"role": "system", "content": system_content})

        if memories:
            memory_msg = self._format_memories(memories)
            messages.append({"role": "user", "name": "memory", "content": memory_msg})
            messages.append({"role": "assistant", "content": "<ack>Noted.</ack>"})

        recent_turns = turns[-self.config.max_turns_before_summary:] if len(turns) > self.config.max_turns_before_summary else turns

        for turn in recent_turns:
            messages.append({"role": "user", "content": turn.get("user_message", "")})
            messages.append({"role": "assistant", "content": self._format_turn_response(turn)})

        messages.append({"role": "user", "content": user_message})

        self._token_estimate = sum(self._estimate_tokens(m.get("content", "")) for m in messages)

        if self._token_estimate > self.config.summary_trigger_tokens:
            messages = self._compress(messages, turns)

        return messages

    def _format_memories(self, memories: List[Dict[str, Any]]) -> str:
        if self.config.memory_format == "claude_xml":
            return self._format_claude_xml(memories)
        elif self.config.memory_format == "json":
            return self._format_json(memories)
        return self._format_bullet(memories)

    def _format_claude_xml(self, memories: List[Dict[str, Any]]) -> str:
        lines = ["<memory>"]
        for i, mem in enumerate(memories, 1):
            claims = mem.get("text", [])
            if isinstance(claims, list):
                claims = "; ".join(claims)

            lines.append(f'  <memory_entry index="{i}" similarity="{mem.get("similarity", 0):.2f}" conversation="{mem.get("conversation_id", "unknown")[:8]}...">')
            lines.append(f'    <claims>{claims}</claims>')
            if mem.get("preferences"):
                lines.append(f'    <preferences>{json.dumps(mem["preferences"])}</preferences>')
            lines.append(f'  </memory_entry>')
        lines.append("</memory>")
        return "\n".join(lines)

    def _format_json(self, memories: List[Dict[str, Any]]) -> str:
        return json.dumps({"memories": memories}, indent=2)

    def _format_bullet(self, memories: List[Dict[str, Any]]) -> str:
        lines = ["Relevant context from previous conversations:"]
        for mem in memories:
            claims = mem.get("text", [])
            if isinstance(claims, list):
                claims = "; ".join(claims[:2])
            lines.append(f"- {claims}")
        return "\n".join(lines)

    def _format_turn_response(self, turn: Dict[str, Any]) -> str:
        claims = turn.get("extracted_claims", [])
        if claims:
            return f"<turn_summary intent='{turn.get('intent', 'general')}'>{'; '.join(claims[:2])}</turn_summary>"
        return "<turn_summary>Previous discussion</turn_summary>"

    def _compress(self, messages: List[Dict[str, Any]], turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(messages) <= 5:
            return messages

        system = messages[0]
        memory_pair = messages[1:3] if len(messages) > 2 and messages[1].get("name") == "memory" else []
        user_msg = messages[-1]

        middle = messages[len(system) + len(memory_pair):-1]

        summary_claims = []
        for t in turns[:-2]:
            summary_claims.extend(t.get("extracted_claims", []))

        summary = f"<conversation_summary>Previous {len(middle)} messages summarized: " + "; ".join(summary_claims[:5]) + "</conversation_summary>"

        compressed = [system]
        if memory_pair:
            compressed.extend(memory_pair)
        compressed.append({"role": "assistant", "content": summary})
        compressed.append(user_msg)

        return compressed

    def _estimate_tokens(self, text: str) -> int:
        return max(len(text) // 4, 1)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "estimated_tokens": self._token_estimate,
            "budget": self.config.effective_budget(),
            "under_budget": self._token_estimate < self.config.effective_budget(),
        }