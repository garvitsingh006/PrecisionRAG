from fastapi import Depends, FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from db import get_db, engine
from db_models import Base, Run, ExperimentConfig
import schemas
from utils import run_new_evaluation, resume_evaluation, create_config
from langgraph.checkpoint.postgres import PostgresSaver
from models.precision_rag import build_graph, build_or_load_retriever, _retriever_cache
import os
import cloudinary
import cloudinary.uploader

Base.metadata.create_all(bind=engine)

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_methods=["*"],
    allow_headers=["*"],
)
DB_URI_GRAPH = os.getenv("DB_URI_GRAPH")

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
)


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


@app.get("/configs", response_model=list[schemas.ConfigResponse])
def list_configs(db: Session = Depends(get_db)):
    return db.query(ExperimentConfig).all()


@app.post("/upload-doc")
def upload_doc(file: UploadFile = File(...)):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(BASE_DIR, "models", "docs")
    os.makedirs(docs_dir, exist_ok=True)

    filename = file.filename if file.filename.lower().endswith(".pdf") else file.filename + ".pdf"
    dest_path = os.path.join(docs_dir, filename)

    with open(dest_path, "wb") as f:
        f.write(file.file.read())

    # Invalidate the in-memory retriever cache so the next run rebuilds with the new doc
    _retriever_cache.clear()

    return {"message": f"Document '{filename}' saved.", "filename": filename}