from loguru import logger
from groq import Groq
import json
import re

from backend.app.config import get_settings
from backend.graph.state import ResearchState

settings = get_settings()


def run_planner(state: ResearchState) -> ResearchState:
    """
    Planner Agent — ReAct style.
    Decomposes the user query into subqueries and picks a retrieval strategy.

    Why ReAct: The planner must reason about query complexity before deciding
    how many subqueries to generate and which retrieval strategy fits best.
    Simple queries need one subquery and vector search.
    Complex multi-entity queries need several subqueries and hybrid retrieval.
    """
    client = Groq(api_key=settings.groq_api_key)
    query = state["query"]

    logger.info(f"[Planner] Query: {query[:80]}")

    prompt = f"""You are a research planning agent. Analyze the query and create a retrieval plan.

Query: {query}

Think step by step:
1. What is the core information need?
2. Can this be broken into sub-questions?
3. Which strategy fits best: vector (semantic similarity), graph (entity relationships), or hybrid (both)?

Return ONLY valid JSON. No markdown. No explanation.

{{
  "reasoning": "brief reasoning about query complexity",
  "subqueries": ["subquery1", "subquery2"],
  "strategy": "vector|graph|hybrid",
  "complexity": "simple|moderate|complex"
}}"""

    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        plan = json.loads(raw)

        subqueries = plan.get("subqueries", [query])
        if not subqueries:
            subqueries = [query]

        strategy = plan.get("strategy", state["retrieval_strategy"])
        reasoning = plan.get("reasoning", "")

        logger.info(f"[Planner] Strategy: {strategy}, Subqueries: {len(subqueries)}")

        state["subqueries"] = subqueries
        state["retrieval_plan"] = plan
        state["retrieval_strategy"] = strategy
        state["agent_decisions"].append(
            {
                "agent": "Planner",
                "decision": f"Strategy: {strategy}, {len(subqueries)} subqueries",
                "reasoning": reasoning,
                "confidence": 0.9,
            }
        )
        return state

    except Exception as e:
        logger.warning(f"[Planner] Failed, using defaults: {e}")
        state["subqueries"] = [query]
        state["retrieval_plan"] = {"strategy": state["retrieval_strategy"]}
        state["agent_decisions"].append(
            {
                "agent": "Planner",
                "decision": "Used default plan due to error",
                "reasoning": str(e),
                "confidence": 0.5,
            }
        )
        return state
