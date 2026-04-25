from fastapi import HTTPException
from models.precision_rag import State
from db_models import Run, Evaluation, MetricScore, ExperimentConfig
from sqlalchemy.orm import Session
import schemas

def compute_confidence(state):

    s = state.get("sup_score", 0.0)
    u = state.get("usefulness_score", 0.0)
    r = state.get("retrieved_docs_relevance_score", 0.0)

    return round((s * 0.5 + u * 0.3 + r * 0.2), 3)

def run_new_evaluation(input: schemas.NewEvaluationInput, db: Session, workflow):

    # 1. Create new run
    new_run = Run(
        retrieval_used=False,
        web_search_used=False,
        hallucination_retries=0,
        usefulness_retries=0,
        latency_ms=0,
        config_id=input.config_id,
        status="running",
        question=input.question
    )

    db.add(new_run)
    db.commit()
    db.refresh(new_run)

    try:
        # 2. Load config from DB
        exp_config = db.query(ExperimentConfig).filter_by(id=input.config_id).first()

        langgraph_config = {
            "configurable": {
                "thread_id": new_run.id
            }
        }

        initial_state: State = {
            "question": input.question,
            "retrieval_query": "",
            "answer": "",
            "corrective_retrieval_attempted": False,
            "run_config": {
                "model": exp_config.model,
                "embedding_model": exp_config.embedding_model,
                "chunk_size": exp_config.chunk_size,
                "chunk_overlap": exp_config.chunk_overlap,
                "top_k": exp_config.top_k,
                "temperature": exp_config.temperature,
            }
        }

        final_state = workflow.invoke(initial_state, langgraph_config)
        final_state["confidence"] = compute_confidence(final_state)

        # 3. Update run
        new_run.answer = final_state.get("answer")
        new_run.retrieval_query = final_state.get("retrieval_query")
        new_run.retrieval_used = final_state.get("should_retrieve")
        new_run.web_search_used = bool(final_state.get("websearch_docs"))
        new_run.hallucination_retries = final_state.get("hallucination_retries")
        new_run.usefulness_retries = final_state.get("usefulness_retries")
        new_run.latency_ms = final_state.get("latency_ms", 0)
        new_run.status = "completed"

        # 4. Create evaluation
        evaluation = Evaluation(
            run_id=new_run.id,
            confidence=final_state.get("confidence"),
            retrieval_relevance=final_state.get("retrieved_docs_relevance_score"),
        )
        db.add(evaluation)
        db.commit()
        db.refresh(evaluation)

        # 5. Add metric scores
        metrics = [
            MetricScore(
                evaluation_id=evaluation.id,
                type="support",
                label=final_state.get("is_sup"),
                score=final_state.get("sup_score"),
                reason=final_state.get("sup_reason"),
            ),
            MetricScore(
                evaluation_id=evaluation.id,
                type="usefulness",
                label=final_state.get("is_useful"),
                score=final_state.get("usefulness_score"),
                reason=final_state.get("usefulness_reason"),
            )
        ]

        db.add_all(metrics)
        db.commit()

    except Exception as e:
        # 6. Handle failure (this is why resume works later)
        new_run.status = "failed"
        new_run.answer = str(e)  # optional, for debugging
        db.commit()

        raise e  # let FastAPI handle response

    db.refresh(new_run)
    return new_run


def resume_evaluation(run_id: int, db: Session, workflow):
    # 1. Fetch run
    run = db.query(Run).filter(Run.id == run_id).first()

    if not run:
        raise Exception("Run not found")

    if run.status == "completed":
        return run  # already done, don't waste tokens

    if run.status == "running":
        return run  # already in progress (optional: handle differently)

    # 2. Resume execution
    try:
        run.status = "running"
        db.commit()

        langgraph_config = {
            "configurable": {
                "thread_id": run.id
            }
        }

        # No initial_state → resumes from last PostgreSQL checkpoint
        final_state = workflow.invoke(None, langgraph_config)
        final_state["confidence"] = compute_confidence(final_state)

        # 3. Update run
        run.answer = final_state.get("answer")
        run.retrieval_used = final_state.get("should_retrieve")
        run.web_search_used = bool(final_state.get("websearch_docs"))
        run.hallucination_retries = final_state.get("hallucination_retries")
        run.usefulness_retries = final_state.get("usefulness_retries")
        run.latency_ms = final_state.get("latency_ms", 0)
        run.status = "completed"

        # 4. Handle evaluation (create or update)
        evaluation = run.evaluation

        if not evaluation:
            evaluation = Evaluation(
                run_id=run.id,
                confidence=final_state.get("confidence"),
                retrieval_relevance=final_state.get("retrieved_docs_relevance_score"),
            )
            db.add(evaluation)
            db.commit()
            db.refresh(evaluation)
        else:
            evaluation.confidence = final_state.get("confidence")
            evaluation.retrieval_relevance = final_state.get("retrieved_docs_relevance_score")
            db.commit()

        # 5. Upsert metric scores
        for metric_type, label, score, reason in [
            ("support", final_state.get("is_sup"), final_state.get("sup_score"), final_state.get("sup_reason")),
            ("usefulness", final_state.get("is_useful"), final_state.get("usefulness_score"), final_state.get("usefulness_reason")),
        ]:
            existing = db.query(MetricScore).filter_by(
                evaluation_id=evaluation.id,
                type=metric_type
            ).first()

            if existing:
                existing.label = label
                existing.score = score
                existing.reason = reason
            else:
                db.add(MetricScore(
                    evaluation_id=evaluation.id,
                    type=metric_type,
                    label=label,
                    score=score,
                    reason=reason,
                ))

        db.commit()

    except Exception as e:
        run.status = "failed"
        run.answer = str(e)
        db.commit()
        raise e

    db.refresh(run)
    return run


def create_config(input: schemas.ConfigCreateInput, db: Session) -> ExperimentConfig:
    existing = db.query(ExperimentConfig).filter_by(id=input.id).first()

    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Config '{input.id}' already exists"
        )

    config = ExperimentConfig(
        id=input.id,
        model=input.model,
        embedding_model=input.embedding_model,
        chunk_size=input.chunk_size,
        chunk_overlap=input.chunk_overlap,
        top_k=input.top_k,
        temperature=input.temperature,
    )

    db.add(config)
    db.commit()
    db.refresh(config)

    return config