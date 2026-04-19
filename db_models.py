from sqlalchemy import TIMESTAMP, Column, ForeignKey, Integer, String, Numeric, Boolean, text, Text, Float, DateTime, UniqueConstraint
import datetime
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Run(Base):
    __tablename__ = "runs"

    id = Column(Integer, primary_key=True)
    status = Column(String, default="pending")  
    question = Column(Text)
    answer = Column(Text)

    retrieval_query = Column(Text)
    retrieval_used = Column(Boolean)
    web_search_used = Column(Boolean)
    hallucination_retries = Column(Integer)
    usefulness_retries = Column(Integer)

    latency_ms = Column(Float)

    config_id = Column(String, ForeignKey("experiment_configs.id"), index=True)
    config = relationship("ExperimentConfig")

    evaluation = relationship("Evaluation", back_populates="run", uselist=False)

    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('now()'))


class Evaluation(Base):
    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("runs.id"), index=True)

    run = relationship("Run", back_populates="evaluation")

    confidence = Column(Float)
    retrieval_relevance = Column(Float)

    metric_scores = relationship("MetricScore", back_populates="evaluation")


class MetricScore(Base):
    __tablename__ = "metric_scores"

    id = Column(Integer, primary_key=True)
    evaluation_id = Column(Integer, ForeignKey("evaluations.id"), index=True)

    evaluation = relationship("Evaluation", back_populates="metric_scores")

    type = Column(String)  # "support" or "usefulness"
    label = Column(String)
    score = Column(Float)
    reason = Column(Text)

    __table_args__ = (
        UniqueConstraint('evaluation_id', 'type', name='uq_eval_metric_type'),
    )


class ExperimentConfig(Base):
    __tablename__ = "experiment_configs"

    id = Column(String, primary_key=True)  # "test1", "exp_v2"

    model = Column(String)
    embedding_model = Column(String)
    chunk_size = Column(Integer)
    chunk_overlap = Column(Integer)
    top_k = Column(Integer)
    temperature = Column(Float)