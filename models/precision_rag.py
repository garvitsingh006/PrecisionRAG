import time
from typing import List, TypedDict
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.output_parsers import OutputFixingParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os

load_dotenv()




# Essentials
MAX_HALLU_RETRIES = 3
MAX_USEFUL_RETRIES = 3
DB_URI_GRAPH = os.getenv("DB_URI_GRAPH")
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3",
    task="text-generation",
    provider="novita",
    max_new_tokens=5000
)
model = ChatHuggingFace(llm=llm)

UPPER_TH = 0.7
LOWER_TH = 0.3




### Document loading, chunking, embedding, and retrieval function
def build_or_load_retriever():
    if os.path.exists("faiss_index"):
        print("Loading FAISS index from disk...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="./cache"
        )
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        return vector_store.as_retriever(search_kwargs={"k": 4})

    print("Building FAISS index...")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(BASE_DIR, "docs")

    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf") or file.endswith(".PDF"):
            print(f"Loading: {file}")
            loader = PyPDFLoader(os.path.join(folder_path, file))
            docs.extend(loader.load())

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150
    ).split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="./cache"
    )

    vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local("faiss_index")

    return vector_store.as_retriever(search_kwargs={"k": 4})

retriever = build_or_load_retriever()

### State
class State(TypedDict):
    question: str
    retrieval_query: str

    should_retrieve: bool
    
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
retrieve_decision_chain = retrieve_decision_prompt | model | retrieve_decision_parser

def decide_retrieval(state: State) -> State:
    decision : RetrieveDecision = retrieve_decision_chain.invoke({"question": state["question"]})
    print(f"Retrieval decided: {decision.should_retrieve}")
    return {
        "should_retrieve": decision.should_retrieve
    }

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
direct_generation_chain = direct_generation_prompt | model | direct_generation_parser

def direct_generate(state: State) -> State:
    print("Generating Directly...")
    response = direct_generation_chain.invoke({"question": state["question"]})
    return {
        "answer": response.answer
    }


# -------------------------------- CRAG_NODE_1: Retrieve Docs from DB --------------------------------
def retrieve(state: State) -> State:
    q = state["retrieval_query"]
    print("Retrieving docs...")
    docs = retriever.invoke(q)
    return {
        "docs": docs
    }


# -------------------------------- CRAG_NODE_2: Evaluate Retrieved Docs --------------------------------
class EvaluateSchema(BaseModel):
    score: Annotated[float, Field(..., description="Relevancy score of the text with respect to answering the question", ge=0.0, le=1.0)]
    reasoning: Annotated[str, Field(..., description="A one-line reasoning for the given score")]

evaluate_docs_parser = PydanticOutputParser(pydantic_object=EvaluateSchema)
evaluate_fixing_parser = OutputFixingParser.from_llm(
    parser=evaluate_docs_parser,
    llm=model
)
evaluate_docs_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a document relevance evaluator.

            Your job is to score how useful a piece of text is for answering the question.

            Important rules:
            - A document does NOT need to directly answer the question to be relevant.
            - If the question asks for comparison, information about ANY entity mentioned in the question is useful.
            - Background information, rules, definitions, or explanations about those entities should receive a high score.
            - Only give low scores if the text is unrelated to the topic.

            Scoring guide:
            1.0 = contains a direct answer or very strong evidence
            0.7–0.9 = contains important information needed to answer the question
            0.4–0.6 = somewhat useful background information
            0.0–0.3 = unrelated

            Return JSON only.
            If you do not return valid JSON, your response is useless.

            {format_instructions}
            """
        ),
        ("human", "Text: {text}\n\n Question: {question}")
    ]
).partial(format_instructions=evaluate_docs_parser.get_format_instructions())

evaluate_docs_chain = evaluate_docs_prompt | model | evaluate_fixing_parser

def evaluate_retrieved_docs(state: State) -> State:
    docs = state["docs"]
    good_docs : List[Document] = []
    results : List[EvaluateSchema] = []
    evaluation_result = ""

    if not docs:
        evaluation_result = "incorrect"
        return {
            "good_docs": [],
            "evaluation_result": evaluation_result
        }

    # invoke to evaluate docs
    print("Evaluating retrieved docs...")
    def clean_text(text):
        return " ".join(text.split())
    for d in docs:
        results.append(evaluate_docs_chain.invoke({"text": clean_text(d.page_content), "question": state["question"]}))

    print("RAW MODEL OUTPUT:", results)
    
    # add good_docs
    for doc, res in zip(docs, results):
        if res.score > LOWER_TH:
            good_docs.append(doc)

    # check scores
    # case 1: atleast one score above 0.7
    if any(r.score > UPPER_TH for r in results):
        evaluation_result = "correct"

    # case 2: all scores below 0.3
    elif all(r.score < LOWER_TH for r in results):
        evaluation_result = "incorrect"

    # case 3: atleast one ambiguous score         
    else:
        evaluation_result="ambiguous"

    all_scores = [result.score for result in results]
    scores = [s for s in all_scores]  # collect the float scores you already compute
    avg_score = sum(scores) / len(scores) if scores else 0.0

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
rewrite_chain = rewrite_prompt | model | rewrite_parser

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
    # rewrite query
    try:
        rewritten = rewrite_chain.invoke({"question": state["question"]})
        websearch_query = rewritten.websearch_query
    except Exception:
        websearch_query = state["question"]  # fallback
    
    websearch_docs = search_on_web(websearch_query)

    return {
        "websearch_docs": websearch_docs,
        "websearch_query": websearch_query
    }


# -------------------------------- CRAG_NODE_4: Refine --------------------------------
class KeepOrDrop(BaseModel):
    keep: bool

filter_parser = PydanticOutputParser(pydantic_object = KeepOrDrop)
filter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict relevance filter.\n"
            "Return keep=True only if the text chunk helps answer the question.\n"
            "The chunk may contain multiple sentences.\n"
            "Output only JSON.\n"
            "Follow the following output format: {format_instructions}"
        ),
        ("human", "Question: {question}\n\n Chunk:\n{chunk}"),
    ]
).partial(format_instructions=filter_parser.get_format_instructions())

filter_chain = filter_prompt | model | filter_parser

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def refine(state: State) -> State:
    q = state["question"]

    if not state["docs"]:
        return {
            "refined_context": "",
            "kept_strips": [],
            "strips": []
        }

    docs_to_be_refined : List[Document] = []
    if state["evaluation_result"] == "correct":
        docs_to_be_refined = state["good_docs"]
    else:
        docs_to_be_refined = state["good_docs"] + state["websearch_docs"]

    context = "\n\n".join(d.page_content for d in docs_to_be_refined).strip()

    # Decompose
    strips = recursive_splitter.split_text(context)

    print("Refining context...")

    # Filter
    inputs = [{"question": q, "chunk": s} for s in strips]
    results = filter_chain.batch(inputs)
    kept = [s for s, r in zip(strips, results) if r.keep]

    # Recompose
    refined_context = "\n".join(kept)

    if not kept:
        refined_context = "\n".join(strips[:5])

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
).partial(format_instructions = generate_parser.get_format_instructions())
generate_chain = generate_prompt | model | generate_parser

def generate_answer(state: State) -> State:
    if not state["refined_context"]:
        return {"answer": "I don't know because there was no refined context given to me"}
    
    print("Generating response based on refined context...")

    response = generate_chain.invoke({"question": state["question"], "refined_context": state["refined_context"]})

    answer = response.answer
    return {
        "answer": answer
    }


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
is_sup_chain = is_sup_prompt | model | is_sup_parser

def is_sup(state: State) -> State:
    print("Checking Hallucination")
    decision: Is_Sup_Decision = is_sup_chain.invoke({"question": state["question"], "answer": state["answer"], "refined_context": state["refined_context"]})
    return {
        "is_sup": decision.is_sup,
        "sup_reason": decision.sup_reason,
        "sup_score": decision.sup_score
    }


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
revise_chain = revise_prompt | model | revise_parser

def revise_answer(state: State) -> State:
    print("Revising answer...")
    response = revise_chain.invoke({"question": state["question"], "answer": state["answer"], "refined_context": state["refined_context"]})
    return {
        "answer": response.answer,
        "hallucination_retries": state.get("hallucination_retries", 0) + 1
    }


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
is_use_chain = is_use_prompt | model | is_use_parser

def is_useful(state: State) -> State:
    print("Checking usefulness of the answer...")
    decision: IsUSEDecision = is_use_chain.invoke({"question": state["question"], "answer": state["answer"]})
    return {
        "is_useful": decision.is_useful,
        "usefulness_reason": decision.reason,
        "usefulness_score": decision.score
    }


# -------------------------------- SELF_RAG_NODE_6: Rewrite question --------------------------------
class RewriteDecision(BaseModel):
    retrieval_query: str = Field(
        ...,
        description="Rewritten query optimized for vector retrieval against internal company PDFs."
    )

rewrite_for_retrieval_parser = PydanticOutputParser(pydantic_object=RewriteDecision)
rewrite_for_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's QUESTION into a query optimized for vector retrieval over INTERNAL company PDFs.\n\n"
            "Rules:\n"
            "- Keep it short (6–16 words).\n"
            "- Preserve key entities.\n"
            "- Remove filler words.\n"
            "- Do NOT answer the question.\n\n"
            "IMPORTANT:\n"
            "- Return ONLY valid JSON.\n"
            "- Do NOT return schema.\n"
            "- Do NOT explain anything.\n"
            "- Output EXACTLY in this format:\n"
            "{\"retrieval_query\": \"your rewritten query\"}\n\n"
            "{format_instructions}"
        ),
        (
            "human",
            "QUESTION:\n{question}\n\n"
            "Previous retrieval query (if any):\n{retrieval_query}\n\n"
            "Answer (if any):\n{answer}"
        )
    ]
).partial(format_instructions=rewrite_for_retrieval_parser.get_format_instructions())
rewrite_for_retrieval_chain = rewrite_for_retrieval_prompt | model | rewrite_for_retrieval_parser

def rewrite_query(state: State) -> State:
    print("Rewriting Question...")
    try:
        decision: RewriteDecision = rewrite_for_retrieval_chain.invoke({
            "question": state["question"],
            "retrieval_query": state["retrieval_query"],
            "answer": state["answer"]
        })
        query = decision.retrieval_query

    except Exception as e:
        print("Rewrite failed, falling back:", e)
        query = state["question"]  # fallback
    return {
        "retrieval_query": query,
        "usefulness_retries": state.get("usefulness_retries", 0) + 1,

        # Resetting entire state
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
def route(state: State) -> State:
    if state["evaluation_result"] == "correct":
        return "good"
    else:
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
        "bad": "web_search"
    })
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