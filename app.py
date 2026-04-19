from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session
from db import get_db, engine
from db_models import Base, Run, ExperimentConfig
import schemas
from utils import run_new_evaluation, resume_evaluation, create_config
from langgraph.checkpoint.postgres import PostgresSaver
from models.precision_rag import build_graph, build_or_load_retriever
import os

Base.metadata.create_all(bind=engine)

app = FastAPI()
DB_URI_GRAPH = os.getenv("DB_URI_GRAPH")


@app.on_event("startup")
def startup():
    global workflow
    global checkpointer_cm
    global checkpointer

    
    checkpointer_cm = PostgresSaver.from_conn_string(DB_URI_GRAPH)
    checkpointer = checkpointer_cm.__enter__()
    checkpointer.setup()
    workflow = build_graph(checkpointer)

@app.on_event("shutdown")
def shutdown():
    checkpointer_cm.__exit__(None, None, None)


@app.get("/")
def home():
    return {
        "response": "Welcome to Precision RAG made by Garvit Singh"
    }

@app.post("/evaluation/new", response_model=schemas.EvaluationResponse)
def new(input: schemas.NewEvaluationInput, db: Session = Depends(get_db)):
    config = db.query(ExperimentConfig).filter_by(id=input.config_id).first()

    if not config:
        raise HTTPException(
            status_code=400,
            detail=f"Config '{input.config_id}' does not exist"
    )
    run = run_new_evaluation(input, db, workflow)
    return schemas.to_evaluation_response(run)

@app.post("/evaluation/resume/{run_id}", response_model=schemas.EvaluationResponse)
def resume(run_id: int, db: Session = Depends(get_db)):
    run = db.query(Run).filter(Run.id == run_id).first()

    if not run:
        return {"error": "Run not found"}

    if run.status == "completed":
        return {"message": "Already completed", "run_id": run.id}

    if run.status == "running":
        return {"message": "Already running", "run_id": run.id}

    # Only resume if failed or pending
    if run.status not in ["failed", "pending"]:
        return {"error": f"Cannot resume run with status '{run.status}'"}

    updated_run = resume_evaluation(run_id, db, workflow)
    return schemas.to_evaluation_response(updated_run)

@app.post("/config/new", response_model=schemas.ConfigResponse)
def new_config(input: schemas.ConfigCreateInput, db: Session = Depends(get_db)):
    config = create_config(input, db)
    return config