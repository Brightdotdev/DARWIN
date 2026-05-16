# research_agent_tagteam.py
import json
import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Any, Literal
from dataclasses import dataclass
from playwright.sync_api import sync_playwright

# LiteLLM gives you a unified interface to all models
import litellm
from litellm import completion

# Configure LiteLLM to use your Ollama models
# Ollama runs at localhost:11434 by default

class LiteLLMRouter:
    """Route tasks to appropriate models"""
    
    # Model configurations
    MODELS = {
        "fast": "ollama/qwen-model",  # Small, fast, cheap
        "smart": "ollama/fast-gemma",               # Bigger, slower, smarter
        # You can also mix in cloud models via LiteLLM:
        # "gpt": "gpt-4o-mini",  # needs OPENAI_API_KEY
        # "gemini": "gemini/gemini-1.5-flash",  # needs GEMINI_API_KEY
    }
    
    @staticmethod
    def is_simple_task(task: str, context: str = "") -> bool:
        """Determine if a task is simple enough for the fast model"""
        
        # Simple classifiers (you can also use an LLM for this)
        simple_indicators = [
            "format", "parse", "extract", "search", "summarize short",
            "list", "count", "find", "check if", "does this contain"
        ]
        
        complex_indicators = [
            "reason", "analyze", "synthesize", "compare and contrast",
            "evaluate", "critique", "what does this mean", "implications",
            "trade-offs", "which is better", "why"
        ]
        
        task_lower = task.lower()
        
        # If it has complex indicators, route to smart model
        for indicator in complex_indicators:
            if indicator in task_lower:
                return False
        
        # If it has simple indicators, route to fast model
        for indicator in simple_indicators:
            if indicator in task_lower:
                return True
        
        # Default: short context/small input → fast, long/complex → smart
        if len(task) < 200 and len(context) < 1000:
            return True
        return False
    
    @staticmethod
    def think(prompt: str, task_type: Literal["simple", "hard"] = None) -> str:
        """Route to the right model"""
        
        # Auto-detect if not specified
        if task_type is None:
            task_type = "simple" if LiteLLMRouter.is_simple_task(prompt) else "hard"
        
        model = LiteLLMRouter.MODELS["fast"] if task_type == "simple" else LiteLLMRouter.MODELS["smart"]
        
        print(f"🧠 Routing to: {model} ({task_type})")
        
        try:
            response = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3 if task_type == "simple" else 0.7,
                max_tokens=500 if task_type == "simple" else 2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️ Model failed: {e}. Falling back to fast model.")
            # Fallback
            response = completion(
                model=LiteLLMRouter.MODELS["fast"],
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content


class ResearchAgent:
    def __init__(self, profile_path: str = "./agent_profile"):
        self.profile_path = profile_path
        self.router = LiteLLMRouter()
        self.browser_context = None
        self.findings = []  # Store findings for markdown export
        self.setup_db()
        
    def setup_db(self):
        self.conn = sqlite3.connect("agent_memory.db")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS research_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                timestamp TEXT,
                markdown_path TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS findings (
                session_id INTEGER,
                step INTEGER,
                model_used TEXT,
                insight TEXT,
                source TEXT
            )
        """)
    
    # ========== TOOLS ==========
    
    def tool_search_web(self, query: str) -> Dict:
        """Search DuckDuckGo - simple task, fast model"""
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            # Use fast model to parse/format results if needed
            return {"results": results}
    
    def tool_wikipedia(self, query: str) -> str:
        """Get Wikipedia summary - simple extraction"""
        import wikipedia
        return wikipedia.summary(query, sentences=5)
    
    def tool_read_url(self, url: str) -> str:
        """Read URL content - may need smart model for analysis"""
        if not self.browser_context:
            playwright = sync_playwright().start()
            self.browser_context = playwright.chromium.launch_persistent_context(
                user_data_dir=self.profile_path,
                headless=False
            )
        page = self.browser_context.new_page()
        page.goto(url, timeout=30000)
        
        if "captcha" in page.content().lower():
            print("\n🔐 CAPTCHA detected! Please solve in browser window.")
            input("Press Enter after solving...")
        
        text = page.locator("body").inner_text()[:8000]
        page.close()
        return text
    
    # ========== PLANNER WITH TAG TEAM ==========
    
    def think(self, user_query: str, previous_actions: List[Dict]) -> Dict:
        """LLM decides next action - uses fast model for planning, smart for complex decisions"""
        
        # Build context
        context = f"""
User query: {user_query}
Previous actions: {json.dumps(previous_actions[-3:])}
Available tools: search_web, read_url, wikipedia, google_fact_check, complete
"""
        
        # Planning is relatively simple - use fast model
        decision_prompt = f"""
{context}

Decide the next action. Return JSON: {{"action": tool_name, "input": value, "reasoning": "why"}}
"""
        
        response = self.router.think(decision_prompt, task_type="simple")
        
        # Parse response
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            decision = json.loads(response[start:end])
            return decision
        except:
            return {"action": "search_web", "input": user_query, "reasoning": "fallback"}
    
    def analyze_finding(self, raw_data: str, context: str) -> str:
        """Extract insights from raw data - use smart model for analysis"""
        
        analysis_prompt = f"""
You are an expert analyst. Extract the key insights from this research data.

Research context: {context}
Raw data: {raw_data[:3000]}

Provide a concise, insightful summary (2-4 sentences) focusing on what's most relevant.
"""
        return self.router.think(analysis_prompt, task_type="hard")
    
    def generate_final_answer(self, user_query: str, findings: List[Dict]) -> str:
        """Synthesize all findings into final answer - definitely a hard task"""
        
        synthesis_prompt = f"""
User asked: "{user_query}"

Here are all the research findings:
{json.dumps(findings, indent=2)[:8000]}

Write a comprehensive, well-structured answer that:
1. Directly answers the question
2. Cites specific sources
3. Highlights conflicting viewpoints if any
4. Ends with a clear conclusion

Use markdown formatting (headings, bullet points, bold for key claims).
"""
        return self.router.think(synthesis_prompt, task_type="hard")
    
    def save_to_markdown(self, query: str, final_answer: str, findings: List[Dict]) -> str:
        """Save the entire research session to a markdown file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_{timestamp}.md"
        
        # Create markdown content
        markdown = f"""# Research Report: {query}

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Model used for synthesis:** {LiteLLMRouter.MODELS["smart"]}

---

## Executive Summary

{final_answer}

---

## Research Methodology

| Step | Model Used | Action | Insight |
|------|-----------|--------|---------|
"""
        
        for f in findings:
            markdown += f"| {f.get('step', '?')} | {f.get('model', 'N/A')} | {f.get('action', 'N/A')} | {f.get('insight', '')[:100]} |\n"
        
        markdown += f"""
---

## Raw Findings

"""
        for i, f in enumerate(findings):
            markdown += f"### Finding {i+1}: {f.get('action', 'Unknown')}\n"
            markdown += f"**Model used:** {f.get('model', 'N/A')}\n\n"
            markdown += f"{f.get('insight', 'No insight')}\n\n"
            markdown += f"<details>\n<summary>Raw data</summary>\n\n```\n{f.get('raw', '')[:1000]}\n```\n</details>\n\n"
            markdown += "---\n\n"
        
        # Save file
        os.makedirs("research_reports", exist_ok=True)
        path = f"research_reports/{filename}"
        with open(path, "w", encoding="utf-8") as f:
            f.write(markdown)
        
        print(f"\n📄 Research saved to: {path}")
        return path
    
    # ========== MAIN RESEARCH LOOP ==========
    
    def research(self, user_query: str, max_steps: int = 8) -> Dict:
        """Main loop with tag team models"""
        
        print(f"\n{'='*60}")
        print(f"🔍 RESEARCH AGENT (Tag Team Mode)")
        print(f"📝 Query: {user_query}")
        print(f"⚡ Fast model: {LiteLLMRouter.MODELS['fast']}")
        print(f"🧠 Smart model: {LiteLLMRouter.MODELS['smart']}")
        print(f"{'='*60}\n")
        
        actions_log = []
        findings = []
        
        for step in range(max_steps):
            print(f"\n--- Step {step + 1}/{max_steps} ---")
            
            # 1. Plan (simple model)
            decision = self.think(user_query, actions_log)
            print(f"💭 Plan: {decision.get('reasoning', 'No reasoning')}")
            print(f"🎯 Action: {decision.get('action', 'unknown')}")
            
            if decision.get('action') == "complete":
                print("\n✅ Agent believes it has enough information.")
                break
            
            # 2. Execute
            action = decision.get('action', 'search_web')
            action_input = decision.get('input', user_query)
            
            print(f"⚡ Executing: {action}({action_input[:50]}...)")
            
            if action == "search_web":
                result = self.tool_search_web(action_input)
                raw_data = json.dumps(result)
                model_used = "fast"
            elif action == "wikipedia":
                result = self.tool_wikipedia(action_input)
                raw_data = result
                model_used = "fast"
            elif action == "read_url":
                result = self.tool_read_url(action_input)
                raw_data = result
                model_used = "fast" if len(result) < 2000 else "hard"  # Long content → smart
            else:
                raw_data = f"Unknown action: {action}"
                model_used = "fast"
            
            # 3. Analyze (smart model for complex, fast for simple)
            print(f"🧠 Analyzing with: {model_used}")
            insight = self.analyze_finding(raw_data, user_query)
            print(f"📖 Insight: {insight[:200]}...")
            
            # Store
            finding = {
                "step": step + 1,
                "action": action,
                "model": model_used,
                "insight": insight,
                "raw": raw_data[:1000]
            }
            findings.append(finding)
            
            actions_log.append({
                "step": step + 1,
                "action": action,
                "result": insight[:200]
            })
        
        # 4. Generate final answer (smart model)
        print(f"\n{'='*60}")
        print("📝 Generating final answer (using smart model)...")
        print(f"{'='*60}")
        
        final_answer = self.generate_final_answer(user_query, findings)
        
        # 5. Save to markdown
        markdown_path = self.save_to_markdown(user_query, final_answer, findings)
        
        return {
            "query": user_query,
            "answer": final_answer,
            "findings": findings,
            "markdown_path": markdown_path,
            "steps_taken": len(actions_log),
            "models_used": {
                "fast": LiteLLMRouter.MODELS["fast"],
                "smart": LiteLLMRouter.MODELS["smart"]
            }
        }


# ========== RUN ==========
if __name__ == "__main__":
    agent = ResearchAgent()
    
    result = agent.research("What are the disadvantages of caffeine?")
    
    print("\n" + "="*60)
    print("FINAL ANSWER")
    print("="*60)
    print(result['answer'])
    print("\n" + "="*60)
    print(f"✅ Markdown saved to: {result['markdown_path']}")