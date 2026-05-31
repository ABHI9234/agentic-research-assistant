from loguru import logger
from groq import Groq

from backend.app.config import get_settings
from backend.graph.state import ResearchState

settings = get_settings()


def run_report(state: ResearchState) -> ResearchState:
    """
    Report Generation Agent — Structured generation (NOT ReAct).

    Why NOT ReAct: By this point the evidence is verified and cited.
    The agent's job is synthesis only — no tool use or iterative reasoning needed.
    We give it the evidence and citations and it produces the final answer.
    """
    client = Groq(api_key=settings.groq_api_key)
    query = state["query"]
    chunks = state.get("evidence_chunks", [])
    citations = state.get("citations", [])

    evidence_text = "\n\n".join(
        [f"[Source {i+1}: {c.get('source_file','?')} p.{c.get('page_num',0)}]\n{c.get('text','')}"
         for i, c in enumerate(chunks[:6])]
    )

    citation_text = "\n".join(
        [f"[{c['citation_id']}] {c['source_file']}, p.{c['page_num']}"
         for c in citations]
    )

    prompt = f"""You are a research report writer. Using ONLY the evidence provided, answer the query comprehensively.

Query: {query}

Evidence:
{evidence_text[:3500]}

Citations available:
{citation_text}

Write a research answer that:
1. Directly answers the query
2. References sources where relevant (use citation IDs)
3. Is factual, clear, and well-structured
4. Acknowledges if evidence is limited

Answer:"""

    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=settings.groq_max_tokens,
        )
        answer = response.choices[0].message.content.strip()
        usage = response.usage

        # Build markdown report
        citations_md = "\n".join(
            [f"- [{c['citation_id']}] **{c['source_file']}**, p.{c['page_num']}"
             for c in citations]
        )
        report_markdown = f"""## Research Report

**Query:** {query}

---

### Answer

{answer}

---

### Sources

{citations_md if citations_md else '_No sources available._'}
"""

        token_usage = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        }

        logger.info(f"[Report] Answer generated — {token_usage['total_tokens']} tokens")

        state["answer"] = answer
        state["report_markdown"] = report_markdown
        state["token_usage"] = token_usage
        state["agent_decisions"].append(
            {
                "agent": "Report",
                "decision": "Report generated",
                "reasoning": f"Synthesized {len(chunks)} evidence chunks into final answer.",
                "confidence": state.get("confidence_score", 0.8),
            }
        )
        return state

    except Exception as e:
        logger.error(f"[Report] Generation failed: {e}")
        state["answer"] = "Report generation failed. Please try again."
        state["report_markdown"] = "# Error\n\nReport generation failed."
        state["error"] = str(e)
        return state
