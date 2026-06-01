from fastapi import APIRouter, HTTPException
from loguru import logger
from backend.app.schemas.eval_schemas import EvalRequest, EvalResponse, StrategyMetrics, EvalStrategy
from backend.evaluation.ragas_evaluator import run_evaluation
import uuid
from datetime import datetime

router = APIRouter()


@router.get("/status")
async def eval_status():
    return {"status": "Evaluation module ready. POST /api/eval/run to start."}


@router.post("/run", response_model=EvalResponse)
async def run_eval(request: EvalRequest):
    """
    Runs RAGAS evaluation across selected retrieval strategies.
    For each question, runs the full 6-agent pipeline and scores with:
    Faithfulness, Context Precision, Context Recall, Answer Relevancy
    """
    try:
        strategies = [s.value for s in request.strategies]
        logger.info(f"Starting evaluation: {len(request.questions)} questions, strategies: {strategies}")

        raw_results = await run_evaluation(
            questions=request.questions,
            strategies=strategies,
        )

        results = []
        best_score = -1
        best_strategy = strategies[0]

        for r in raw_results:
            avg_score = (
                r["faithfulness"] +
                r["answer_relevancy"] +
                r["context_precision"] +
                r["context_recall"]
            ) / 4

            if avg_score > best_score:
                best_score = avg_score
                best_strategy = r["strategy"]

            results.append(StrategyMetrics(
                strategy=EvalStrategy(r["strategy"]),
                faithfulness=r["faithfulness"],
                context_precision=r["context_precision"],
                context_recall=r["context_recall"],
                answer_relevancy=r["answer_relevancy"],
                hallucination_rate=r["hallucination_rate"],
                avg_latency_ms=r["avg_latency_ms"],
            ))

        return EvalResponse(
            eval_id=str(uuid.uuid4())[:8],
            total_questions=len(request.questions),
            results=results,
            best_strategy=EvalStrategy(best_strategy),
            summary=f"Evaluated {len(request.questions)} question(s) across {len(strategies)} strategies. Best: {best_strategy} (avg score: {best_score:.2f})",
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
