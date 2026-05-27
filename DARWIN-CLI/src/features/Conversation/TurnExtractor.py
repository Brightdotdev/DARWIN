import json
import re
from typing import Dict, Any, List, Optional

from src.features.Conversation.ConversationConfig import ConversationConfig
from src.features.Indexing.Embedding.Embedder import embed_text
from src.core.Logger.DarwinLogger import darwin, LogType


class TurnExtractor:
    def __init__(self, config: ConversationConfig):
        self.config = config
        self._llm_client = None

    def extract(self, turn_data: Dict[str, Any]) -> Dict[str, Any]:
        combined_text = self._combine_turn_text(turn_data)
        regex_result = self._regex_extract(combined_text)

        if self.config.budget_mode:
            return self._budget_fallback(combined_text, regex_result)

        if regex_result["confidence"] >= self.config.min_confidence_for_llm:
            return self._confident_regex_result(combined_text, regex_result)

        llm_result = self._llm_extract(combined_text)
        merged = self._merge_results(regex_result, llm_result)

        return {
            **merged,
            "embedding": embed_text(combined_text[:500]),
            "extraction_method": "llm_enhanced",
            "confidence": 1.0,
        }

    def _combine_turn_text(self, turn_data: Dict[str, Any]) -> str:
        parts = [
            turn_data.get("user_message", ""),
            turn_data.get("assistant_plan", ""),
            turn_data.get("final_response", ""),
        ]
        text = " ".join(parts)
        text = re.sub(r'<[^>]+>', ' ', text)
        return text.strip()

    def _regex_extract(self, text: str) -> Dict[str, Any]:
        claims = self._extract_claims(text)
        prefs = self._extract_preferences(text)
        intent = self._classify_intent(text)

        confidence = 0.3
        confidence += min(len(claims) * 0.15, 0.4)
        confidence += 0.2 if prefs.get("format") else 0
        confidence += 0.1 if intent != "general" else 0

        return {
            "claims": claims,
            "preferences": prefs,
            "intent": intent,
            "confidence": min(confidence, 0.95),
        }

    def _extract_claims(self, text: str) -> List[str]:
        sentences = re.split(r'[.!?]\s+', text)
        claims = []
        for s in sentences:
            s = s.strip()
            if len(s) < 20:
                continue
            if re.search(r'\d+|\$|is a|are the|will be|concluded that|found that', s):
                claims.append(s[:200])
        return claims[:5]

    def _extract_preferences(self, text: str) -> Dict[str, Any]:
        prefs = {}
        if any(w in text.lower() for w in ["table", "chart", "graph", "visual"]):
            prefs["format"] = "visual"
        elif any(w in text.lower() for w in ["summary", "brief", "short"]):
            prefs["format"] = "concise"
        else:
            prefs["format"] = "detailed"

        if any(w in text.lower() for w in ["cheap", "budget", "lowest price", "affordable"]):
            prefs["price_sensitivity"] = "high"

        domains = []
        if any(w in text.lower() for w in ["gpu", "inference", "training", "model"]):
            domains.append("hardware")
        if any(w in text.lower() for w in ["llm", "transformer", "attention", "world model"]):
            domains.append("ai_research")
        prefs["domains"] = domains

        return prefs

    def _classify_intent(self, text: str) -> str:
        text_lower = text.lower()
        if any(w in text_lower for w in ["synthesis", "overview", "summary of", "direction of"]):
            return "synthesis"
        if any(w in text_lower for w in ["price", "cost", "cheap", "buy"]):
            return "pricing"
        if any(w in text_lower for w in ["how to", "steps", "guide", "tutorial"]):
            return "how_to"
        return "general"

    def _budget_fallback(self, combined_text: str, regex_result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "claims": regex_result.get("claims", [combined_text[:300]]),
            "preferences": regex_result.get("preferences", {}),
            "intent": regex_result.get("intent", "general"),
            "embedding": embed_text(combined_text[:500]),
            "extraction_method": "regex_budget",
            "confidence": regex_result.get("confidence", 0.3),
        }

    def _confident_regex_result(self, combined_text: str, regex_result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **regex_result,
            "embedding": embed_text(combined_text[:500]),
            "extraction_method": "regex_confident",
        }

    def _llm_extract(self, text: str) -> Dict[str, Any]:
        if self._llm_client is None:
            from src.features.LLM.LLMManager import LLMManager
            LLMManager.initialize()
            manager = LLMManager()
            self._llm_client = manager.get_client()

        prompt = f"""Extract from this conversation turn:
- Key claims (factual statements, max 5)
- User preferences (format, tone, domain interests)
- Intent (synthesis, pricing, how_to, general)

Text: {text[:2000]}

Output JSON: {{"claims": [...], "preferences": {{...}}, "intent": "..."}}"""
        
        messages = [{"role": "user", "content": prompt}]
        

        try:
            response = self._llm_client.call_small(messages=messages, max_tokens=300)
            if response is None or not response.json:
                darwin.log("LLM returned no response for turn extraction", LogType.WARNING)
                return {"claims": [], "preferences": {}, "intent": "general"}
            
            return json.loads(response.json)
        except Exception as e:
            darwin.log(f"LLM extraction failed: {e}", LogType.WARNING)
            return {"claims": [], "preferences": {}, "intent": "general"}

    def _merge_results(self, regex: Dict[str, Any], llm: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "claims": llm.get("claims") if llm.get("claims") else regex.get("claims", []),
            "preferences": {**regex.get("preferences", {}), **llm.get("preferences", {})},
            "intent": llm.get("intent", regex.get("intent", "general")),
        }