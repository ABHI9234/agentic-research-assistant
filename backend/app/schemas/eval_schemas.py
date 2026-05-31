from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from enum import Enum


class EvalStrategy(str, Enum):
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


class EvalRequest(BaseModel):
    questions: List[str] = Field(..., min_length=1)
    ground_truths: Optional[List[str]] = None
    strategies: List[EvalStrategy] = [
        EvalStrategy.VECTOR,
        EvalStrategy.GRAPH,
        EvalStrategy.HYBRID,
    ]


class StrategyMetrics(BaseModel):
    strategy: EvalStrategy
    faithfulness: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    answer_relevancy: float = 0.0
    hallucination_rate: float = 0.0
    avg_latency_ms: float = 0.0


class EvalResponse(BaseModel):
    eval_id: str
    total_questions: int
    results: List[StrategyMetrics]
    best_strategy: EvalStrategy
    summary: str
    timestamp: str


class EvalResultsResponse(BaseModel):
    evals: List[EvalResponse]
    total_runs: int
