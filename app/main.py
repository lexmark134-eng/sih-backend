import os
import tempfile
from typing import Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import ChatRequest, ChatResponse, TutorState
from app.state_machine import tutor_graph
from app.vectorstore import ingest_curriculum_file, seed_misconceptions
from app.problem_bank import add_problem, add_problem_auto_solve

app = FastAPI(title="Hallucination-Resistant Socratic Tutor")

# Allow your teammate's frontend (different laptop/origin) to call this API.
# Tighten allow_origins to the actual frontend URL before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory per-student hint history (topic -> running avg hint level needed).
# Swap for Redis/Postgres later; kept simple so it runs anywhere for the demo.
_student_hint_history: Dict[str, Dict[str, float]] = {}

# In-memory per-session conversation history: session_id -> list of
# {"role": "user"|"assistant", "content": "..."}. This is what makes /chat
# a real multi-turn conversation instead of forgetting everything between
# requests. Same caveat as above — swap for a real store before deploying
# beyond a single demo/dev server, since this resets on restart and doesn't
# survive multiple server workers.
_session_history: Dict[str, list] = {}
_MAX_STORED_TURNS = 30  # keep the last N messages per session, trim older


def _starting_hint_level(student_id: str, topic: str) -> int:
    history = _student_hint_history.get(student_id, {})
    avg = history.get(topic)
    if avg is None:
        return 1
    return max(1, min(round(avg), 4))


def _update_hint_history(student_id: str, topic: str, used_level: int):
    history = _student_hint_history.setdefault(student_id, {})
    prev = history.get(topic, used_level)
    history[topic] = (prev + used_level) / 2  # simple running average


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    starting_level = _starting_hint_level(req.student_id, req.topic)
    history = _session_history.get(req.session_id, [])

    state = TutorState(
        session_id=req.session_id,
        student_id=req.student_id,
        topic=req.topic,
        student_message=req.message,
        image_base64=req.image_base64,
        hint_level=starting_level,
        conversation_history=history,
    )

    result = tutor_graph.invoke(state)
    # LangGraph's merged output can omit keys that stayed at their default
    # (version-dependent behavior), so use .get() with the same defaults
    # TutorState itself defines, rather than assuming every key is present.
    final_reply = result.get("final_reply") or result.get("draft_reply") or ""
    faithfulness_score = result.get("faithfulness_score", 0.0)
    hint_level = result.get("hint_level", starting_level)
    misconception = result.get("detected_misconception")
    was_regenerated = result.get("was_regenerated", False)
    answer_leaked = result.get("answer_leaked", False)
    sources = list({c.source for c in result.get("retrieved_chunks", [])})

    _update_hint_history(req.student_id, req.topic, hint_level)

    # Append this turn to session memory so the next /chat call for the
    # same session_id continues the conversation instead of starting over.
    updated_history = history + [
        {"role": "user", "content": req.message},
        {"role": "assistant", "content": final_reply},
    ]
    _session_history[req.session_id] = updated_history[-_MAX_STORED_TURNS:]

    return ChatResponse(
        session_id=req.session_id,
        reply=final_reply,
        hint_level=hint_level,
        detected_misconception=misconception,
        faithfulness_score=round(faithfulness_score, 3),
        grounded_sources=sources,
        was_regenerated=was_regenerated,
        answer_leak_intercepted=answer_leaked and was_regenerated,
    )


@app.post("/ingest")
async def ingest(file: UploadFile = File(...), source_label: str = Form(...)):
    """Upload a verified curriculum file (pdf/docx/txt) to embed into the store."""
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        n_chunks = ingest_curriculum_file(tmp_path, source_label)
    finally:
        os.remove(tmp_path)
    return {"status": "ok", "chunks_added": n_chunks, "source": source_label}


@app.post("/misconceptions/seed")
def seed(topic: str, misconceptions: list[dict]):
    """
    Body example:
    {
      "topic": "quadratic_equations",
      "misconceptions": [
        {"pattern": "student drops the negative root when solving x^2=9",
         "explanation": "Forgetting both +/- roots when taking a square root",
         "targeted_hint": "Ask: are there other values that, when squared, also give this result?"}
      ]
    }
    """
    seed_misconceptions(topic, misconceptions)
    return {"status": "ok", "topic": topic, "count": len(misconceptions)}


@app.post("/problems/add")
def add_problem_endpoint(topic: str, problem_text: str, canonical_answer: str, variable: str = "x"):
    """
    Add a problem with a manually-supplied canonical answer, e.g.:
    topic=quadratic_equations
    problem_text="Solve x^2 - 9 = 0"
    canonical_answer="x = 3, x = -3"
    """
    result = add_problem(topic, problem_text, canonical_answer, variable)
    return {"status": "ok", **result}


@app.post("/problems/add_auto_solve")
def add_problem_auto_solve_endpoint(topic: str, problem_text: str, equation_str: str, variable: str = "x"):
    """
    Add a problem where the canonical answer is computed by sympy instead of
    supplied manually, e.g.:
    equation_str="x**2 - 9 = 0"  ->  sympy solves it, not the LLM.
    """
    result = add_problem_auto_solve(topic, problem_text, equation_str, variable)
    if result is None:
        raise HTTPException(status_code=400, detail="sympy could not solve the given equation")
    return {"status": "ok", **result}


@app.post("/chat/reset")
def reset_session(session_id: str):
    """Clears stored conversation history for a session — useful for a
    'start over' button on the frontend, or when testing repeatedly."""
    _session_history.pop(session_id, None)
    return {"status": "ok", "session_id": session_id}


@app.get("/health")
def health():
    return {"status": "ok"}
