from loguru import logger
from groq import Groq
import json
import re

from backend.app.config import get_settings
from backend.graph.state import ResearchState

settings = get_settings()
MAX_REFLECTION_ITERATIONS = 2


def run_reflection(state: ResearchState) -> ResearchState:
    """
    Reflection Agent — ReAct style.
    Evaluates whether the retrieved evidence is sufficient to answer the query.
    Can trigger another retrieval cycle by setting reflection_passed=False.

    Why ReAct: This agent must reason about evidence quality and decide
    whether to loop back or proceed — a genuine workflow control decision.
    """
    if not state.get("enable_reflection", True):
        state["reflection_passed"] = True
        state["reflection_output"] = "Reflection disabled."
        return state

    iterations = state.get("reflection_iterations", 0)
    if iterations >= MAX_REFLECTION_ITERATIONS:
        logger.info("[Reflection] Max iterations reached — proceeding.")
        state["reflection_passed"] = True
        state["reflection_output"] = "Max reflection iterations reached."
        return state

    client = Groq(api_key=settings.groq_api_key)
    query = state["query"]
    chunks = state.get("evidence_chunks", [])

    evidence_text = "\n\n".join(
        [f"[Source: {c.get('source_file','?')} p.{c.get('page_num',0)}]\n{c.get('text','')}"
         for c in chunks[:5]]
    )

    prompt = f"""You are a research quality evaluator.

Query: {query}

Retrieved Evidence:
{evidence_text[:3000]}

Evaluate:
1. Does the evidence directly address the query?
2. Are there obvious gaps or missing information?
3. Is the evidence sufficient to write a confident answer?

Return ONLY valid JSON. No markdown.

{{
  "sufficient": true|false,
  "confidence": 0.0-1.0,
  "gaps": ["gap1", "gap2"],
  "reasoning": "brief explanation"
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
        result = json.loads(raw)

        sufficient = result.get("sufficient", True)
        confidence = result.get("confidence", 0.7)
        reasoning = result.get("reasoning", "")
        gaps = result.get("gaps", [])

        logger.info(f"[Reflection] Sufficient: {sufficient}, Confidence: {confidence}")

        state["reflection_passed"] = sufficient
        state["reflection_output"] = reasoning
        state["reflection_iterations"] = iterations + 1
        state["confidence_score"] = confidence
        state["agent_decisions"].append(
            {
                "agent": "Reflection",
                "decision": "Sufficient" if sufficient else f"Gaps found: {gaps}",
                "reasoning": reasoning,
                "confidence": confidence,
            }
        )
        return state

    except Exception as e:
        logger.warning(f"[Reflection] Failed, defaulting to pass: {e}")
        state["reflection_passed"] = True
        state["reflection_output"] = "Reflection error — proceeding."
        state["reflection_iterations"] = iterations + 1
        return state
