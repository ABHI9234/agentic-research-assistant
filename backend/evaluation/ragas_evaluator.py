from typing import List, Dict
from loguru import logger
from groq import Groq
import re
from backend.app.config import get_settings

settings = get_settings()


def get_groq_client():
    return Groq(api_key=settings.groq_api_key)


def _extract_score(raw: str) -> float:
    """Pull the first float found in the response, robust to reasoning leakage."""
    if not raw:
        raise ValueError("empty response")
    match = re.search(r'(\d*\.?\d+)', raw.strip())
    if not match:
        raise ValueError(f"no number found in: {raw!r}")
    return float(match.group(1))


def score_faithfulness(answer: str, contexts: List[str]) -> float:
    client = get_groq_client()
    context_text = "\n\n".join(contexts[:3])

    prompt = f"""You are an evaluation judge. Score how faithful the answer is to the provided context.

Context:
{context_text[:2000]}

Answer:
{answer[:1000]}

Rate faithfulness from 0.0 to 1.0 where:
1.0 = every claim in the answer is supported by the context
0.0 = the answer contains claims not found in the context

Return ONLY a number between 0.0 and 1.0. Nothing else."""

    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
            reasoning_effort="low",
        )
        return round(min(max(_extract_score(response.choices[0].message.content), 0.0), 1.0), 3)
    except Exception as e:
        logger.warning(f"Faithfulness scoring failed: {e}")
        return 0.0


def score_answer_relevancy(question: str, answer: str) -> float:
    client = get_groq_client()

    prompt = f"""You are an evaluation judge. Score how relevant the answer is to the question.

Question: {question}

Answer: {answer[:1000]}

Rate relevancy from 0.0 to 1.0 where:
1.0 = answer directly and completely addresses the question
0.0 = answer is completely irrelevant to the question

Return ONLY a number between 0.0 and 1.0. Nothing else."""

    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
            reasoning_effort="low",
        )
        return round(min(max(_extract_score(response.choices[0].message.content), 0.0), 1.0), 3)
    except Exception as e:
        logger.warning(f"Answer relevancy scoring failed: {e}")
        return 0.0


def score_context_precision(question: str, contexts: List[str]) -> float:
    if not contexts:
        return 0.0

    client = get_groq_client()
    relevant = 0

    for context in contexts[:5]:
        prompt = f"""Is this context relevant to answering the question?

Question: {question}
Context: {context[:500]}

Answer with only YES or NO."""

        try:
            response = client.chat.completions.create(
                model=settings.groq_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
                reasoning_effort="low",
            )
            answer = (response.choices[0].message.content or "").strip().upper()
            if "YES" in answer:
                relevant += 1
        except Exception as e:
            logger.warning(f"Context precision check failed: {e}")

    return round(relevant / len(contexts[:5]), 3)


def score_context_recall(question: str, answer: str, contexts: List[str]) -> float:
    client = get_groq_client()
    context_text = "\n\n".join(contexts[:3])

    prompt = f"""You are an evaluation judge. Score how much of the answer is supported by the context.

Question: {question}
Context: {context_text[:2000]}
Answer: {answer[:1000]}

Rate context recall from 0.0 to 1.0 where:
1.0 = all parts of the answer can be found in or inferred from the context
0.0 = none of the answer is supported by the context

Return ONLY a number between 0.0 and 1.0. Nothing else."""

    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
            reasoning_effort="low",
        )
        return round(min(max(_extract_score(response.choices[0].message.content), 0.0), 1.0), 3)
    except Exception as e:
        logger.warning(f"Context recall scoring failed: {e}")
        return 0.0


async def evaluate_single_question(question: str, strategy: str) -> Dict:
    from backend.app.schemas.query_schemas import ResearchRequest, RetrievalStrategy
    from backend.app.services.retrieval_service import run_research_query

    try:
        request = ResearchRequest(
            query=question,
            retrieval_strategy=RetrievalStrategy(strategy),
            top_k=5,
            graph_depth=2,
            enable_reflection=False,
            enable_critic=False,
        )
        result = await run_research_query(request)

        answer = result.answer
        contexts = [c.text for c in result.evidence_chunks]

        faithfulness = score_faithfulness(answer, contexts)
        relevancy = score_answer_relevancy(question, answer)
        precision = score_context_precision(question, contexts)
        recall = score_context_recall(question, answer, contexts)
        hallucination_rate = round(1.0 - faithfulness, 3)

        return {
            "strategy": strategy,
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "context_precision": precision,
            "context_recall": recall,
            "hallucination_rate": hallucination_rate,
            "avg_latency_ms": result.latency_ms,
        }

    except Exception as e:
        logger.error(f"Evaluation failed for question '{question}': {e}")
        return {
            "strategy": strategy,
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "hallucination_rate": 1.0,
            "avg_latency_ms": 0.0,
            "error": str(e),
        }


async def run_evaluation(questions: List[str], strategies: List[str]) -> List[Dict]:
    strategy_results = {}

    for strategy in strategies:
        scores = []
        for question in questions:
            logger.info(f"Evaluating: '{question[:50]}' with {strategy}")
            score = await evaluate_single_question(question, strategy)
            scores.append(score)

        if scores:
            avg = {
                "strategy": strategy,
                "faithfulness": round(sum(s["faithfulness"] for s in scores) / len(scores), 3),
                "answer_relevancy": round(sum(s["answer_relevancy"] for s in scores) / len(scores), 3),
                "context_precision": round(sum(s["context_precision"] for s in scores) / len(scores), 3),
                "context_recall": round(sum(s["context_recall"] for s in scores) / len(scores), 3),
                "hallucination_rate": round(sum(s["hallucination_rate"] for s in scores) / len(scores), 3),
                "avg_latency_ms": round(sum(s["avg_latency_ms"] for s in scores) / len(scores), 1),
            }
            strategy_results[strategy] = avg

    return list(strategy_results.values())
