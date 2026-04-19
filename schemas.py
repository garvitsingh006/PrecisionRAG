from pydantic import BaseModel, Field
from typing import Optional
from db_models import Run


class NewEvaluationInput(BaseModel):
    question: str
    config_id: str


class ScoreDetail(BaseModel):
    label: Optional[str]
    score: Optional[float]
    reason: Optional[str]


class EvaluationOut(BaseModel):
    confidence: Optional[float]
    support: Optional[ScoreDetail]
    usefulness: Optional[ScoreDetail]
    retrieval_relevance: Optional[float]


class PipelineOut(BaseModel):
    retrieval_used: Optional[bool]
    web_search_used: Optional[bool]
    hallucination_retries: Optional[int]
    usefulness_retries: Optional[int]


class PerformanceOut(BaseModel):
    latency_ms: Optional[float]


class ExperimentOut(BaseModel):
    model: Optional[str]
    embedding_model: Optional[str]
    chunk_size: Optional[int]
    chunk_overlap: Optional[int]
    top_k: Optional[int]
    temperature: Optional[float]


class EvaluationResponse(BaseModel):
    answer: Optional[str]
    evaluation: Optional[EvaluationOut]
    pipeline: Optional[PipelineOut]
    performance: Optional[PerformanceOut]
    experiment: Optional[ExperimentOut]


def to_evaluation_response(run: Run) -> EvaluationResponse:
    evaluation = run.evaluation

    support = None
    usefulness = None

    if evaluation:
        for m in evaluation.metric_scores:
            if m.type == "support":
                support = ScoreDetail(
                    label=m.label,
                    score=m.score,
                    reason=m.reason
                )
            elif m.type == "usefulness":
                usefulness = ScoreDetail(
                    label=m.label,
                    score=m.score,
                    reason=m.reason
                )

    return EvaluationResponse(
        answer=run.answer,

        evaluation=EvaluationOut(
            confidence=evaluation.confidence if evaluation else None,
            retrieval_relevance=evaluation.retrieval_relevance if evaluation else None,
            support=support,
            usefulness=usefulness,
        ) if evaluation else None,

        pipeline=PipelineOut(
            retrieval_used=run.retrieval_used,
            web_search_used=run.web_search_used,
            hallucination_retries=run.hallucination_retries,
            usefulness_retries=run.usefulness_retries,
        ),

        performance=PerformanceOut(
            latency_ms=round(run.latency_ms or 0, 1)
        ),

        experiment=ExperimentOut(
            model=run.config.model if run.config else None,
            embedding_model=run.config.embedding_model if run.config else None,
            chunk_size=run.config.chunk_size if run.config else None,
            chunk_overlap=run.config.chunk_overlap if run.config else None,
            top_k=run.config.top_k if run.config else None,
            temperature=run.config.temperature if run.config else None,
        )
    )



class ConfigCreateInput(BaseModel):
    id: str
    model: str
    embedding_model: str
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    top_k: int = Field(gt=0)
    temperature: float = Field(ge=0, le=2)

class ConfigResponse(BaseModel):
    id: str
    model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    temperature: float

    class Config:
        from_attributes = True
