import time
from typing import List, TypedDict, Optional
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.output_parsers import OutputFixingParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os

load_dotenv()

MAX_HALLU_RETRIES = 3
MAX_USEFUL_RETRIES = 3
UPPER_TH = 0.7
LOWER_TH = 0.3


def get_model(model_id: str, temperature: float):
    llm = HuggingFaceEndpoint(
        repo_id=model_id,
        task="text-generation",
        provider="novita",
        max_new_tokens=5000,
        temperature=temperature,
    )
    return ChatHuggingFace(llm=llm)




### Document loading, chunking, embedding, and retrieval function
_retriever_cache: dict = {}
_parent_chunks_cache: dict = {}  # ADD THIS

def build_or_load_retriever(chunk_size, chunk_overlap, top_k, embedding_model):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(BASE_DIR, "docs")

    pdf_files = sorted([
        f for f in os.listdir(folder_path)
        if f.endswith(".pdf") or f.endswith(".PDF")
    ]) if os.path.exists(folder_path) else []

    if not pdf_files:
        return None

    cache_key = (tuple(pdf_files), chunk_size, chunk_overlap, embedding_model)

    if cache_key in _retriever_cache:
        print("Retriever cache hit.")
        return _retriever_cache[cache_key].as_retriever(search_kwargs={"k": top_k})

    print("Building FAISS index with parent-document retrieval...")
    docs = []
    for file in pdf_files:
        loader = PyPDFLoader(os.path.join(folder_path, file))
        docs.extend(loader.load())

    # Parent chunks — bigger, used for answer generation
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 3,
        chunk_overlap=chunk_overlap * 2,
    )
    parent_chunks = parent_splitter.split_documents(docs)

    # Child chunks — smaller, used for retrieval
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    child_docs = []
    for parent_id, parent in enumerate(parent_chunks):
        children = child_splitter.split_documents([parent])
        for child in children:
            child.metadata["parent_id"] = parent_id  # link to parent
            child_docs.append(child)

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        cache_folder=os.path.join(BASE_DIR, "..", "cache")
    )

    vector_store = FAISS.from_documents(child_docs, embeddings)
    _retriever_cache[cache_key] = vector_store
    _parent_chunks_cache[cache_key] = parent_chunks  # STORE PARENTS
    return vector_store.as_retriever(search_kwargs={"k": top_k})

### State
class RunConfig(TypedDict):
    model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    temperature: float


class State(TypedDict):
    question: str
    retrieval_query: str
    run_config: RunConfig

    should_retrieve: bool
    corrective_retrieval_attempted: bool

    websearch_query: str
    websearch_docs: List[Document]

    docs: List[Document]
    good_docs: List[Document]
    evaluation_result: Literal["incorrect", "correct", "ambiguous"]

    strips: List[str]
    kept_strips: List[str]
    refined_context: str

    answer: str

    # Post Generation checking
    is_sup: Literal["fully_supported", "partially_supported", "no_support"]
    sup_score: float
    sup_reason: str
    hallucination_retries: int
    max_hallucination_retries: int

    is_useful: Literal["useful", "not_useful"]
    usefulness_score: float
    usefulness_reason: str

    usefulness_retries: int
    max_usefulness_retries: int

    # Model Evaluation Metrices
    confidence: float
    retrieved_docs_relevance_score: float
    latency_ms: float




# -------------------------------- SELF_RAG_NODE_1: Decide Retrieval --------------------------------
class RetrieveDecision(BaseModel):
    should_retrieve: Annotated[bool,  Field(..., description="True if external documents are required to be retrieved to answer the question, else False")]

retrieve_decision_parser = PydanticOutputParser(pydantic_object=RetrieveDecision)
retrieve_decision_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval decision system.\n"
            "Return JSON matching schema: {format_instructions}\n\n"

            "Set should_retrieve = True if:\n"
            "- The query mentions a specific year, event, version (e.g., 2026, latest, current)\n"
            "- The query asks for official rules, manuals, or recent updates\n"
            "- The answer may have changed over time\n"
            "- You are NOT 100% certain the answer is static knowledge\n\n"

            "Set should_retrieve = False only if:\n"
            "- The question is purely conceptual, timeless, or definitional\n\n"

            "When in doubt, ALWAYS choose True.\n"
            "Be conservative. Prefer retrieving over guessing."
        ),
        (
            "human",
            "Question: {question}"
        )
    ]
).partial(format_instructions=retrieve_decision_parser.get_format_instructions())

def decide_retrieval(state: State) -> State:
    model = get_model(state["run_config"]["model"], state["run_config"]["temperature"])
    retrieve_decision_chain = retrieve_decision_prompt | model | retrieve_decision_parser
    decision: RetrieveDecision = retrieve_decision_chain.invoke({"question": state["question"]})
    print(f"Retrieval decided: {decision.should_retrieve}")
    return {"should_retrieve": decision.should_retrieve}

# -------------------------------- SELF_RAG_NODE_2: Direct Generation --------------------------------
class DirectGenerationSchema(BaseModel):
    answer: str

direct_generation_parser = PydanticOutputParser(pydantic_object=DirectGenerationSchema)
direct_generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the following question using your own intrinsic knowledge, follow the following formatting instructions: {format_instructions}"),
        ("human", "Question: {question}")
    ]
).partial(format_instructions=direct_generation_parser.get_format_instructions())

def direct_generate(state: State) -> State:
    print("Generating Directly...")
    model = get_model(state["run_config"]["model"], state["run_config"]["temperature"])
    direct_generation_chain = direct_generation_prompt | model | direct_generation_parser
    response = direct_generation_chain.invoke({"question": state["question"]})
    return {"answer": response.answer}


# -------------------------------- CRAG_NODE_1: Retrieve Docs from DB --------------------------------
def retrieve(state: State) -> State:
    q = state["retrieval_query"]
    cfg = state["run_config"]
    print("Retrieving docs...")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(BASE_DIR, "docs")
    pdf_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".pdf") or f.endswith(".PDF")])
    cache_key = (tuple(pdf_files), cfg["chunk_size"], cfg["chunk_overlap"], cfg["embedding_model"])

    retriever = build_or_load_retriever(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        top_k=cfg["top_k"],
        embedding_model=cfg["embedding_model"],
    )
    if retriever is None:
        return {"docs": []}

    child_hits = retriever.invoke(q)

    # Swap each child for its parent
    parent_chunks = _parent_chunks_cache.get(cache_key, [])
    if not parent_chunks:
        return {"docs": child_hits}  # fallback if cache miss

    seen_parent_ids = set()
    result_docs = []
    for child in child_hits:
        pid = child.metadata.get("parent_id")
        if pid is not None and pid not in seen_parent_ids:
            seen_parent_ids.add(pid)
            result_docs.append(parent_chunks[pid])
        elif pid is None:
            result_docs.append(child)

    print(f"Retrieved {len(child_hits)} child chunks → expanded to {len(result_docs)} parent chunks")
    return {"docs": result_docs}

# -------------------------------- CRAG_NODE_2: Evaluate Retrieved Docs --------------------------------
class DocScore(BaseModel):
    index: int = Field(..., description="0-based index of the document chunk")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score between 0.0 and 1.0")
    reasoning: str = Field(..., description="One-line reasoning for the score")

class BatchEvalSchema(BaseModel):
    scores: List[DocScore] = Field(..., description="One entry per document chunk, in the same order as provided")

batch_eval_parser = PydanticOutputParser(pydantic_object=BatchEvalSchema)
batch_eval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert document relevance evaluator for a RAG pipeline.\n\n"
            "You will be given a QUESTION and a numbered list of document chunks.\n"
            "Your task is to score each chunk on how useful it is for answering the question.\n\n"
            "Scoring rules:\n"
            "- 1.0 : chunk contains a direct answer or decisive evidence\n"
            "- 0.7–0.9 : chunk contains important facts clearly needed to answer the question\n"
            "- 0.4–0.6 : chunk contains background or partial information that is somewhat useful\n"
            "- 0.1–0.3 : chunk is loosely related but unlikely to help\n"
            "- 0.0 : chunk is completely unrelated to the question\n\n"
            "Important:\n"
            "- A chunk does NOT need to directly answer the question to score high.\n"
            "- If the question asks for a comparison, any chunk about ANY entity in the question is useful.\n"
            "- Rules, definitions, background context, and supporting evidence all count as relevant.\n"
            "- Be consistent: similar chunks should receive similar scores.\n"
            "- You MUST return exactly one score object per chunk, in the same order, with the correct 0-based index.\n\n"
            "{format_instructions}"
        ),
        (
            "human",
            "QUESTION: {question}\n\n"
            "DOCUMENT CHUNKS:\n{chunks}"
        )
    ]
).partial(format_instructions=batch_eval_parser.get_format_instructions())

def evaluate_retrieved_docs(state: State) -> State:
    docs = state["docs"]
    if not docs:
        return {"good_docs": [], "evaluation_result": "incorrect", "retrieved_docs_relevance_score": 0.0}

    def clean_text(text):
        return " ".join(text.split())

    numbered_chunks = "\n\n".join(
        f"[{i}] {clean_text(d.page_content)}" for i, d in enumerate(docs)
    )

    model = get_model(state["run_config"]["model"], state["run_config"]["temperature"])
    fixing_parser = OutputFixingParser.from_llm(parser=batch_eval_parser, llm=model)
    chain = batch_eval_prompt | model | fixing_parser

    print(f"Evaluating {len(docs)} retrieved docs in a single call...")
    result: BatchEvalSchema = chain.invoke({"question": state["question"], "chunks": numbered_chunks})

    scores_by_index = {s.index: s for s in result.scores}
    good_docs: List[Document] = []
    all_scores: List[float] = []

    for i, doc in enumerate(docs):
        score_obj = scores_by_index.get(i)
        score = score_obj.score if score_obj else 0.0
        all_scores.append(score)
        if score > LOWER_TH:
            good_docs.append(doc)

    avg_score = sum(all_scores) / len(all_scores)

    if any(s > UPPER_TH for s in all_scores):
        evaluation_result = "correct"
    elif all(s < LOWER_TH for s in all_scores):
        evaluation_result = "incorrect"
    else:
        evaluation_result = "ambiguous"

    print(f"Doc scores: {all_scores} → {evaluation_result}")
    return {
        "good_docs": good_docs,
        "evaluation_result": evaluation_result,
        "retrieved_docs_relevance_score": round(avg_score, 3)
    }


# -------------------------------- CRAG_NODE_3: Web Search --------------------------------
class RewriteSchema(BaseModel):
        websearch_query: str
        
rewrite_parser = PydanticOutputParser(pydantic_object=RewriteSchema)
rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at rewriting questions into effective web search queries."
            "Rewrite the question into a concise search query that will retrieve relevant information."
            "Answer in the following format: {format_instructions}"
        ),
        (
            "human",
            "Question: {question}"
        )
    ]
).partial(format_instructions=rewrite_parser.get_format_instructions())

tavily = TavilySearchResults(max_results=5)

def search_on_web(query):
    q = query  # no query rewrite
    results = tavily.invoke({"query": q})  # no knowledge selection

    web_docs = []
    for r in results or []:

        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "") or r.get("snippet", "")
        
        text = f"TITLE: {title}\nCONTENT:\n{content}"

        web_docs.append(Document(page_content=text, metadata={"url": url, "title": title}))

    return web_docs

def web_search(state: State) -> State:
    model = get_model(state["run_config"]["model"], state["run_config"]["temperature"])
    rewrite_chain = rewrite_prompt | model | rewrite_parser
    try:
        rewritten = rewrite_chain.invoke({"question": state["question"]})
        websearch_query = rewritten.websearch_query
    except Exception:
        websearch_query = state["question"]
    websearch_docs = search_on_web(websearch_query)
    return {"websearch_docs": websearch_docs, "websearch_query": websearch_query}


# -------------------------------- CRAG_NODE_4: Refine --------------------------------
class BatchFilterSchema(BaseModel):
    decisions: List[bool] = Field(
        ...,
        description="Ordered list of true/false decisions, one per chunk. true = keep, false = drop."
    )

batch_filter_parser = PydanticOutputParser(pydantic_object=BatchFilterSchema)
batch_filter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a precision context filter for a RAG pipeline.\n\n"
            "You will be given a QUESTION and a numbered list of text chunks extracted from documents.\n"
            "For each chunk, decide whether it should be KEPT or DROPPED from the context that will be\n"
            "used to generate an answer.\n\n"
            "Keep a chunk (true) if ANY of the following apply:\n"
            "- It contains facts, figures, dates, names, or events directly relevant to the question\n"
            "- It provides necessary background or definitions needed to understand the answer\n"
            "- It contains partial evidence that, combined with other chunks, helps answer the question\n"
            "- It narrows down or constrains the answer in a meaningful way\n\n"
            "Drop a chunk (false) if ALL of the following apply:\n"
            "- It is completely off-topic relative to the question\n"
            "- It contains only boilerplate, headers, page numbers, or formatting artifacts\n"
            "- Removing it would not affect the quality of the final answer at all\n\n"
            "Rules:\n"
            "- Be INCLUSIVE rather than exclusive — when in doubt, keep the chunk\n"
            "- You MUST return exactly one boolean per chunk, in the same order as provided\n"
            "- The length of 'decisions' must equal the number of chunks\n"
            "- Do not merge, reorder, or skip any chunks\n\n"
            "{format_instructions}"
        ),
        (
            "human",
            "QUESTION: {question}\n\n"
            "TEXT CHUNKS:\n{chunks}"
        )
    ]
).partial(format_instructions=batch_filter_parser.get_format_instructions())

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def refine(state: State) -> State:
    q = state["question"]

    if not state["docs"]:
        return {"refined_context": "", "kept_strips": [], "strips": []}

    docs_to_refine: List[Document] = (
        state["good_docs"] if state["evaluation_result"] == "correct"
        else state["good_docs"] + state["websearch_docs"]
    )

    context = "\n\n".join(d.page_content for d in docs_to_refine).strip()
    strips = recursive_splitter.split_text(context)

    numbered_chunks = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(strips))

    model = get_model(state["run_config"]["model"], state["run_config"]["temperature"])
    fixing_parser = OutputFixingParser.from_llm(parser=batch_filter_parser, llm=model)
    chain = batch_filter_prompt | model | fixing_parser

    print(f"Refining {len(strips)} strips in a single call...")
    result: BatchFilterSchema = chain.invoke({"question": q, "chunks": numbered_chunks})

    decisions = result.decisions
    # guard against length mismatch from the model
    if len(decisions) != len(strips):
        decisions = decisions[:len(strips)] + [True] * max(0, len(strips) - len(decisions))

    kept = [s for s, keep in zip(strips, decisions) if keep]
    refined_context = "\n".join(kept) if kept else "\n".join(strips[:5])

    print(f"Kept {len(kept)}/{len(strips)} strips")
    return {
        "refined_context": refined_context,
        "kept_strips": kept,
        "strips": strips
    }


# -------------------------------- CRAG_NODE_5: Generate --------------------------------
class GenerateSchema(BaseModel):
    answer: str

generate_parser = PydanticOutputParser(pydantic_object=GenerateSchema)
generate_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional research assistant."
            "You have to answer the question given based on the context given to you."
            "If the  context is  insufficient to answer the question, do not hallucinate and just say 'I don't know'"
            "Answer in the following json format only: {format_instructions}"
        ),
        (
            "human",
            "Question: {question}\n\nRefined Context:\n{refined_context}"
        )
    ]
).partial(format_instructions=generate_parser.get_format_instructions())

def generate_answer(state: State) -> State:
    if not state["refined_context"]:
        return {"answer": "I don't know because there was no refined context given to me"}
    print("Generating response based on refined context...")
    model = get_model(state["run_config"]["model"], state["run_config"]["temperature"])
    generate_chain = generate_prompt | model | generate_parser
    response = generate_chain.invoke({"question": state["question"], "refined_context": state["refined_context"]})
    return {"answer": response.answer}


# -------------------------------- SELF_RAG_NODE_3: Check hallucination --------------------------------
class Is_Sup_Decision(BaseModel):
    is_sup: Literal["fully_supported", "partially_supported", "no_support"]
    sup_score: Annotated[float, Field(..., description="supported score, how much is this answer supported by the context, it is a number ranging from 0.0 to 1.0, and should satisfy the decision made by you in 'is_sup' field", ge=0.0, le=1.0)]
    sup_reason: Annotated[str, Field(..., description="The reason why you chose the sup_score and is_sup as they are right now")]

is_sup_parser = PydanticOutputParser(pydantic_object=Is_Sup_Decision)
is_sup_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are verifying whether the ANSWER is supported by the CONTEXT.\n"
            "Return JSON with key: is_sup\nsup_score\nsup_reason\n\n"
            "is_sup must be one of: fully_supported, partially_supported, no_support.\n\n"
            "How to decide is_sup:\n"
            "- fully_supported:\n"
            "  Every meaningful claim is explicitly supported by CONTEXT, and the ANSWER does NOT introduce\n"
            "  any qualitative/interpretive words that are not present in CONTEXT.\n"
            "  (Examples of disallowed words unless present in CONTEXT: culture, generous, robust, designed to,\n"
            "  supports professional development, best-in-class, employee-first, etc.)\n\n"
            "- partially_supported:\n"
            "  The core facts are supported, BUT the ANSWER includes ANY abstraction, interpretation, or qualitative\n"
            "  phrasing not explicitly stated in CONTEXT (e.g., calling policies 'culture', saying leave is 'generous',\n"
            "  or inferring outcomes like 'supports professional development').\n\n"
            "- no_support:\n"
            "  The key claims are not supported by CONTEXT.\n\n"
            "sup_score must be between 0.0 and 1.0\n"
            "How to decide sup_score:\n"
            "if is_sup is 'fully_supported', the sup_score must range from 0.8 to 1.0 (including both ends)\n"
            "if is_sup is 'no_support', the sup_score must range from 0.0 to 0.2 (including both ends)\n"
            "if is_sup is 'partially_supported', the sup_score must range from 0.2 to 0.8 (excluding both ends), depending on to how much extent is the answer supported by the context\n\n"
            "sup_reason must justify the reason behind choosing the specific is_sup and sup_score values.\n\n"
            "Rules:\n"
            "- Be strict: if you see ANY unsupported qualitative/interpretive phrasing, choose partially_supported.\n"
            "Follow the following output format: {format_instructions}"
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Answer:\n{answer}\n\n"
            "Refined Context:\n{refined_context}\n"
        ),
    ]
).partial(format_instructions=is_sup_parser.get_format_instructions())

def is_sup(state: State) -> State:
    print("Checking Hallucination")
    model = get_model(state["run_config"]["model"], state["run_config"]["temperature"])
    is_sup_chain = is_sup_prompt | model | is_sup_parser
    decision: Is_Sup_Decision = is_sup_chain.invoke({"question": state["question"], "answer": state["answer"], "refined_context": state["refined_context"]})
    return {"is_sup": decision.is_sup, "sup_reason": decision.sup_reason, "sup_score": decision.sup_score}


# -------------------------------- SELF_RAG_NODE_4: Revise Answer --------------------------------
class ReviseSchema(BaseModel):
    answer: str

revise_parser = PydanticOutputParser(pydantic_object=ReviseSchema)
revise_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a STRICT reviser.\n\n"
            "You must output based on the following format:\n\n"
            "FORMAT: {format_instructions}\n"
            "Rules:\n"
            "- Use ONLY the CONTEXT.\n"
            "- Remove any information not supported by the CONTEXT.\n"
            "- Only keep statements directly supported by the CONTEXT.\n"
            "Rewrite the answer so that every statement is directly supported by the Refined Context. "
            "Delete unsupported statements. Do not add new facts.\n"
            "Prefer copying phrases directly from the CONTEXT instead of paraphrasing."
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Answer:\n{answer}\n\n"
            "Refined Context:\n{refined_context}\n\n"
        )
    ]
).partial(format_instructions=revise_parser.get_format_instructions())

def revise_answer(state: State) -> State:
    print("Revising answer...")
    model = get_model(state["run_config"]["model"], state["run_config"]["temperature"])
    revise_chain = revise_prompt | model | revise_parser
    response = revise_chain.invoke({"question": state["question"], "answer": state["answer"], "refined_context": state["refined_context"]})
    return {"answer": response.answer, "hallucination_retries": state.get("hallucination_retries", 0) + 1}


# -------------------------------- SELF_RAG_NODE_5: Check if answer is useful --------------------------------
class IsUSEDecision(BaseModel):
    is_useful: Literal["useful", "not_useful"]
    reason: str = Field(..., description="Short reason in 1 line explaining why the answer is useful/not_useful.")
    score: Annotated[float, Field(..., description="Usefullness score ranging from 0.0 to 1.0 depending on how much useful the answer is for the question asked by the user.", ge=0.0, le=1.0)]

is_use_parser = PydanticOutputParser(pydantic_object=IsUSEDecision)
is_use_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are judging USEFULNESS of the ANSWER for the QUESTION.\n\n"
            "Goal:\n"
            "- Decide if the answer actually addresses what the user asked.\n\n"
            "Return JSON with keys: is_useful, reason, score.\n"
            "is_use must be one of: useful, not_useful.\n\n"
            "Rules:\n"
            "- useful: The answer directly answers the question or provides the requested specific info.\n"
            "- not_useful: The answer is generic, off-topic, or only gives related background without answering.\n"
            "- Do NOT use outside knowledge.\n"
            "- Do NOT re-check grounding. Only check: 'Did we answer the question?'\n\n"
            "For 'reason':\n"
            "- Keep reason to 1 or 2-3 short lines, up to your discretion.\n"
            "For 'score':\n"
            "- It must range from 0.0 to 1.0 (both ends included)"
            "- make sure your reason and score and is_use, all align in the same direction of thinking and justify each other"
            "Output Format: {format_instructions}"
        ),
        (
            "human",
            "Question:\n{question}\n\nAnswer:\n{answer}"
        )
    ]
).partial(format_instructions=is_use_parser.get_format_instructions())

def is_useful(state: State) -> State:
    print("Checking usefulness of the answer...")
    model = get_model(state["run_config"]["model"], state["run_config"]["temperature"])
    is_use_chain = is_use_prompt | model | is_use_parser
    decision: IsUSEDecision = is_use_chain.invoke({"question": state["question"], "answer": state["answer"]})
    return {"is_useful": decision.is_useful, "usefulness_reason": decision.reason, "usefulness_score": decision.score}


# -------------------------------- CRAG_NODE_2b / SELF_RAG_NODE_6: Rewrite Retrieval Query --------------------------------
# Used in two situations:
#   1. corrective: docs were ambiguous after first retrieval — rewrite before falling back to web search
#   2. usefulness loop: full pipeline completed but answer was not useful — rewrite and re-run everything

class RewriteDecision(BaseModel):
    retrieval_query: str = Field(
        ...,
        description="Rewritten query optimized for vector retrieval."
    )

rewrite_for_retrieval_parser = PydanticOutputParser(pydantic_object=RewriteDecision)
rewrite_for_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a retrieval query specialist for a RAG pipeline.\n\n"
            "You will be given a QUESTION, the PREVIOUS RETRIEVAL QUERY that was used, "
            "the REASON the previous retrieval failed, and optionally a PREVIOUS ANSWER.\n\n"
            "Your job is to rewrite the retrieval query so it finds better, more relevant documents "
            "from the vector store on the next attempt.\n\n"
            "Rewriting strategy based on failure reason:\n"
            "- 'ambiguous_docs': the previous query was too broad or vague — use more specific "
            "terminology, proper nouns, or domain-specific keywords; try a different angle\n"
            "- 'not_useful_answer': the retrieved docs led to an answer that didn't address the question — "
            "focus on the specific aspect of the question that was missed; rephrase around the core intent\n\n"
            "Rules:\n"
            "- Keep the query between 6 and 20 words\n"
            "- Preserve key entities and proper nouns from the original question\n"
            "- Remove filler words\n"
            "- Do NOT answer the question — only produce the retrieval query\n"
            "- Return ONLY valid JSON in exactly this format: {{\"retrieval_query\": \"your rewritten query\"}}\n\n"
            "{format_instructions}"
        ),
        (
            "human",
            "QUESTION: {question}\n\n"
            "PREVIOUS RETRIEVAL QUERY: {retrieval_query}\n\n"
            "FAILURE REASON: {rewrite_reason}\n\n"
            "PREVIOUS ANSWER (if any): {answer}"
        )
    ]
).partial(format_instructions=rewrite_for_retrieval_parser.get_format_instructions())


def _do_rewrite(state: State) -> str:
    """Shared rewrite logic. Returns the new retrieval query string."""
    model = get_model(state["run_config"]["model"], state["run_config"]["temperature"])
    chain = rewrite_for_retrieval_prompt | model | rewrite_for_retrieval_parser
    try:
        decision: RewriteDecision = chain.invoke({
            "question": state["question"],
            "retrieval_query": state.get("retrieval_query", ""),
            "rewrite_reason": state.get("rewrite_reason", ""),
            "answer": state.get("answer", ""),
        })
        return decision.retrieval_query
    except Exception as e:
        print("Rewrite failed, falling back to original question:", e)
        return state["question"]


def corrective_rewrite(state: State) -> State:
    """Called when evaluation_result == ambiguous. Rewrites query, re-retrieves once before web search."""
    print("Ambiguous docs — corrective rewrite before web search...")
    query = _do_rewrite({**state, "rewrite_reason": "ambiguous_docs"})
    return {
        "retrieval_query": query,
        "corrective_retrieval_attempted": True,
        "docs": [],
        "good_docs": [],
        "evaluation_result": "",
    }


def rewrite_query(state: State) -> State:
    """Called when answer was not useful. Rewrites query and resets full pipeline state."""
    print("Answer not useful — rewriting query for full re-run...")
    query = _do_rewrite({**state, "rewrite_reason": "not_useful_answer"})
    return {
        "retrieval_query": query,
        "corrective_retrieval_attempted": False,
        "usefulness_retries": state.get("usefulness_retries", 0) + 1,
        # Reset full pipeline state for clean re-run
        "docs": [],
        "good_docs": [],
        "websearch_query": "",
        "websearch_docs": [],
        "evaluation_result": "",
        "strips": [],
        "kept_strips": [],
        "refined_context": "",
        "answer": "",
        "is_sup": "",
        "hallucination_retries": 0,
        "usefulness_retries": 0,
        "max_hallucination_retries": MAX_HALLU_RETRIES,
        "is_useful": "",
        "usefulness_reason": "",
        "max_usefulness_retries": MAX_USEFUL_RETRIES,
    }


# -------------------------------- SELF_RAG_NODE_7: No answer found --------------------------------
def no_answer(state: State) -> State:
    return {
        "answer": "No answer found, please try asking a different question!"
    }


# -------------------------------- ROUTES --------------------------------

## Condition based on decide_retrieval
def route_should_retrieve(state: State):
    if state["should_retrieve"]:
        return "should_retrieve"
    else:
        return "direct_generate"
    

## Condition based on evaluation
def route(state: State):
    if state["evaluation_result"] == "correct":
        return "good"
    if state["evaluation_result"] == "ambiguous" and not state.get("corrective_retrieval_attempted", False):
        return "ambiguous"
    return "bad"


## Condition based on hallucination
def route_hallucination(state: State)-> Literal["accept_answer", "revise_answer"]:
    if state["hallucination_retries"] >= state["max_hallucination_retries"]:
        return "accept_answer"
    if state["is_sup"] == "fully_supported":
        return "accept_answer"
    else:
        return "revise_answer"


## Condition based on usefulness
def route_usefullness(state: State):
    if state["is_useful"] == "useful":
        return "useful"
    if state["usefulness_retries"] >= state["max_usefulness_retries"]:
        return "no_answer"
    return "rewrite_query"






def build_graph(checkpointer):
    g = StateGraph(State)

    g.add_node("decide_retrieval", decide_retrieval)
    g.add_node("direct_generate", direct_generate)
    g.add_node("retrieve", retrieve)
    g.add_node("evaluate_retrieved_docs", evaluate_retrieved_docs)
    g.add_node("corrective_rewrite", corrective_rewrite)
    g.add_node("web_search", web_search)
    g.add_node("refine", refine)
    g.add_node("generate_answer", generate_answer)
    g.add_node("is_sup", is_sup)
    g.add_node("revise_answer", revise_answer)
    g.add_node("is_useful", is_useful)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("no_answer", no_answer)


    g.add_edge(START, "decide_retrieval")
    g.add_conditional_edges("decide_retrieval", route_should_retrieve, {
        "should_retrieve": "rewrite_query",
        "direct_generate": "direct_generate"
    })
    g.add_edge("direct_generate", END)

    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("retrieve", "evaluate_retrieved_docs")
    g.add_conditional_edges("evaluate_retrieved_docs", route, {
        "good": "refine",
        "ambiguous": "corrective_rewrite",
        "bad": "web_search"
    })
    g.add_edge("corrective_rewrite", "retrieve")
    g.add_edge("web_search", "refine")
    g.add_edge("refine", "generate_answer")

    g.add_edge("generate_answer", "is_sup")
    g.add_conditional_edges("is_sup", route_hallucination, {
        "accept_answer": "is_useful",
        "revise_answer": "revise_answer"
    })
    g.add_edge("revise_answer", "is_sup")
    g.add_conditional_edges("is_useful", route_usefullness, {
        "useful": END,
        "no_answer": "no_answer",
        "rewrite_query": "rewrite_query"
    })
    g.add_edge("no_answer", END)
    return g.compile(checkpointer=checkpointer)

# async with AsyncPostgresSaver.from_conn_string(DB_URI_GRAPH) as checkpointer:
#         await checkpointer.setup()  # creates tables on first run, safe to call always
#         precision_rag = g.compile(checkpointer=checkpointer)
#         initial_state : State = {
#             "question": "What was the judge's verdict and it's reasoning in this case?",
#             "retrieval_query": "",
#             "max_hallucination_retries": MAX_HALLU_RETRIES,
#             "max_usefulness_retries": MAX_USEFUL_RETRIES,
#             "hallucination_retries": 0,
#             "usefulness_retries": 0,
#             "answer": ""
#         }
#         start = time.time()
#         final_state = await precision_rag.ainvoke(None, config=config)
#         final_state["latency_ms"] = (time.time() - start) * 1000
### Metrics

# api_payload = {
#     "answer": final_state["answer"],
    
#     "evaluation": {
#         "confidence": final_state.get("confidence"),
#         "support": {
#             "label": final_state.get("is_sup"),
#             "score": final_state.get("sup_score"),
#             "reason": final_state.get("sup_reason"),
#         },
#         "usefulness": {
#             "label": final_state.get("is_useful"),
#             "score": final_state.get("usefulness_score"),
#             "reason": final_state.get("usefulness_reason"),
#         },
#         "retrieval_relevance": final_state.get("retrieved_docs_relevance_score"),
#     },
    
#     "pipeline": {
#         "retrieval_used": final_state.get("should_retrieve"),
#         "web_search_used": bool(final_state.get("websearch_docs")),
#         "hallucination_retries": final_state.get("hallucination_retries"),
#         "usefulness_retries": final_state.get("usefulness_retries"),
#     },
    
#     "performance": {
#         "latency_ms": round(final_state.get("latency_ms", 0), 1),
#     },
    
#     "experiment": {
#         "model": "deepseek-r1",
#         "embedding_model": "all-MiniLM-L6-v2",
#         "chunk_size": 700,
#         "chunk_overlap": 150,
#         "top_k": 5,
#         "temperature": 0.2,
#     }
# }