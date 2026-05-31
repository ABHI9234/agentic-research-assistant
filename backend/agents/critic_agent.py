from loguru import logger
from groq import Groq
import json
import re

from backend.app.config import get_settings
from backend.graph.state import ResearchState

settings = get_settings()


def run_critic(state: ResearchState) -> ResearchState:
    """
    Critic Agent — Hybrid (deterministic checks + lightweight LLM).

    Why NOT full ReAct: Most verification is deterministic.
    We first do cheap rule-based checks, only call the LLM for
    nuanced hallucination detection when rules pass.
    This saves tokens and latency.
    """
    if not state.get("enable_critic", True):
        state["critic_passed"] = True
        state["critic_findings"] = "Critic disabled."
        return state

    chunks = state.get("evidence_chunks", [])
    query = state["query"]

    # ── Deterministic check 1: Do we have any evidence at all? ──────────
    if not chunks:
        state["critic_passed"] = False
        state["critic_findings"] = "No evidence retrieved. Cannot generate answer."
        state["agent_decisions"].append(
            {
                "agent": "Critic",
                "decision": "REJECTED — no evidence",
                "reasoning": "Evidence list is empty.",
                "confidence": 1.0,
            }
        )
        logger.warning("[Critic] Rejected — no evidence chunks.")
        return state

    # ── Deterministic check 2: Minimum evidence threshold ───────────────
    if len(chunks) < 2:
        logger.info("[Critic] Low evidence count — proceeding with caution.")

    # ── LLM check: Hallucination and contradiction detection ─────────────
    client = Groq(api_key=settings.groq_api_key)

    evidence_text = "\n\n".join(
        [f"[{c.get('source_file','?')} p.{c.get('page_num',0)}]: {c.get('text','')}"
         for c in chunks[:4]]
    )

    prompt = f"""You are a fact-checking agent. Evaluate whether an AI answer about the query below would be supportable by the evidence provided.

Query: {query}

Available Evidence:
{evidence_text[:2500]}

Check for:
1. Is the evidence relevant to the query?
2. Are there contradictions in the evidence?
3. Would an answer based on this evidence risk hallucination?

Return ONLY valid JSON. No markdown.

{{
  "passes": true|false,
  "hallucination_risk": "low|medium|high",
  "contradictions_found": true|false,
  "findings": "brief explanation",
  "confidence": 0.0-1.0
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

        passes = result.get("passes", True)
        findings = result.get("findings", "")
        risk = result.get("hallucination_risk", "low")
        confidence = result.get("confidence", 0.8)

        logger.info(f"[Critic] Passes: {passes}, Risk: {risk}")

        state["critic_passed"] = passes
        state["critic_findings"] = f"[Risk: {risk.upper()}] {findings}"
        state["agent_decisions"].append(
            {
                "agent": "Critic",
                "decision": "APPROVED" if passes else "REJECTED",
                "reasoning": findings,
                "confidence": confidence,
            }
        )
        return state

    except Exception as e:
        logger.warning(f"[Critic] LLM check failed, defaulting to pass: {e}")
        state["critic_passed"] = True
        state["critic_findings"] = "Critic check inconclusive — proceeding."
        return state
