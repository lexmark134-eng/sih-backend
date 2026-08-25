"""
Most "Socratic tutor" projects implement the whole behavior as one system
prompt: "never give the answer directly, ask leading questions instead."
That degrades over a multi-turn conversation and is impossible to debug —
you can't tell WHY the tutor slipped and gave the answer away on turn 6.

Here, Socratic behavior is an explicit graph with named nodes and typed
state (see schemas.TutorState), so each step is independently inspectable,
testable, and swappable:

    classify_turn -> retrieve_curriculum -> detect_misconception
        -> generate_reply -> check_faithfulness -> [regenerate?] -> finalize

The faithfulness gate (node 5) is what prevents a hallucinated "fact" from
ever reaching the student, regardless of what the generation node produced.
"""
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from app.config import settings
from app.schemas import TutorState, TurnType, RetrievedChunk
from app.vectorstore import get_curriculum_store, get_misconception_store
from app.faithfulness import score_faithfulness
from app.problem_bank import match_problem
from app.answer_leak import check_answer_leak

_llm = ChatOllama(model=settings.CHAT_MODEL, temperature=0.4, base_url=settings.OLLAMA_BASE_URL)


def _extract_text(content) -> str:
    """
    Gemini responses (via langchain_google_genai) sometimes return .content
    as a plain string, and sometimes as a list of content blocks like
    [{"type": "text", "text": "..."}]. Normalize either shape to plain text
    so downstream code (pydantic fields, faithfulness checks, leak checks)
    never has to care which one it got.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return "".join(parts)
    return str(content)


_SOCRATIC_SYSTEM = """You are a Socratic tutor for STEM students. 
CRITICAL RULE: You must NEVER provide direct answers, definitions, explanations, expansions or code to the user. If a user asks for a definition (e.g., 'What is photosynthesis?'), you must refuse to define it directly. Instead, respond by asking a guiding question to check their current understanding. 
Violating this rule defeats the purpose of the tutoring system. 

Rules, no exceptions:
- NEVER state the final numeric/symbolic answer to the student's problem.
- NEVER write out the full solved solution or code blocks, even partially, if it reveals the next required step directly.
- SUCCESS STATE CONFIRMATION: If the student explicitly states the correct final answer, you MUST confirm they are correct (e.g., "Yes, exactly!").
- NO MOVING THE GOALPOSTS: Once the student correctly solves the problem, acknowledge their success and stop asking guiding questions.
- SCOPE ENFORCEMENT: Strictly adhere to the initial constraints of the problem. If the user introduces unnecessary complexity or concepts outside the original scope, gently reject the over-complication and steer them back to the simplest solution.
- DOMAIN ANCHORING: If the user asks a subjective or conversational question, do NOT engage in general discussion. Immediately redirect the question back to the concrete STEM/programming task.
- SCAFFOLDING DOWN: If the student says "I don't know," you MUST lower the difficulty. Validate their confusion briefly, take a step back, and ask a highly specific question about one tiny, foundational piece of the current step.
- THE "NO SANDWICH" RULE: NEVER combine an explanation with a question. 
- NO MINI-LECTURES: If your response contains a factual explanation of the topic before the student gets it right, you have failed. 
- STRICT DEFINITION HANDLING: If the user prompts with "Define X", your ONLY response must be a question reflecting it back to them.
- MAXIMUM BREVITY: Keep your responses to 1-3 sentences maximum. State your guiding question immediately.
- Ask ONE guiding question or give ONE small nudge per turn, calibrated to hint level {hint_level}.
- Ground every factual statement you make ONLY in the CURRICULUM CONTEXT below.
- If a MISCONCEPTION is identified, gently target it by name/pattern without lecturing.
{forbidden_answer_clause}

CURRICULUM CONTEXT:
{context}

DETECTED MISCONCEPTION (may be empty):
{misconception}
"""

_FORBIDDEN_ANSWER_TEMPLATE = """- This problem has a known correct answer that you must NOT state or imply
  in any form (not as "x = ...", not spelled out in words, not as a bare
  number matching it). If the student states this value themselves, you may
  confirm correctness in principle without repeating the numeric value."""


def classify_turn(state: TutorState) -> TutorState:
    """Cheap heuristic + LLM fallback: is the student asking, or attempting an answer?"""
    msg = state.student_message.strip()
    has_math_content = any(ch.isdigit() for ch in msg) or "=" in msg
    ends_as_question = msg.endswith("?")
    if ends_as_question and not has_math_content:
        state.turn_type = TurnType.QUESTION
    else:
        state.turn_type = TurnType.ANSWER_ATTEMPT
    return state


def retrieve_curriculum(state: TutorState) -> TutorState:
    store = get_curriculum_store()
    results = store.similarity_search_with_relevance_scores(
        f"{state.topic}: {state.student_message}", k=settings.RETRIEVAL_K
    )
    state.retrieved_chunks = [
        RetrievedChunk(content=doc.page_content, source=doc.metadata.get("source", "unknown"), score=score)
        for doc, score in results
    ]
    return state


def detect_misconception(state: TutorState) -> TutorState:
    if state.turn_type != TurnType.ANSWER_ATTEMPT:
        return state
    store = get_misconception_store()
    results = store.similarity_search_with_relevance_scores(state.student_message, k=1)
    if results and results[0][1] > 0.75:  # relevance threshold
        doc, _ = results[0]
        state.detected_misconception = doc.metadata.get("explanation", doc.page_content)
    return state


def match_against_problem_bank(state: TutorState) -> TutorState:
    """
    If the student's message matches a known problem, pull its canonical
    answer (computed independently via sympy — see problem_bank.py). This is
    what check_answer_leak downstream verifies the draft reply against. If
    nothing matches, canonical_answer stays None and the leak check is a
    harmless no-op for this turn.
    """
    match = match_problem(
        state.student_message, state.topic, threshold=settings.PROBLEM_MATCH_THRESHOLD
    )
    if match:
        state.matched_problem_text = match["problem_text"]
        state.canonical_answer = match["canonical_answer"]
    return state


def _build_reply(state: TutorState, strict: bool = False) -> str:
    context = "\n---\n".join(c.content for c in state.retrieved_chunks) or "No context retrieved."
    # Deliberately never insert the actual canonical_answer value into the
    # prompt — the model only knows a forbidden value EXISTS, never what it
    # is. That way there's nothing for it to regurgitate even if it slips;
    # the leak check downstream still guards against it independently
    # producing/deriving the same value itself.
    forbidden_clause = _FORBIDDEN_ANSWER_TEMPLATE if state.canonical_answer else ""
    system = _SOCRATIC_SYSTEM.format(
        hint_level=state.hint_level,
        max_hint=settings.MAX_HINT_LEVEL,
        context=context,
        misconception=state.detected_misconception or "none",
        forbidden_answer_clause=forbidden_clause,
    )
    if strict:
        system += (
            "\n\nSTRICT MODE: Your previous draft either included unsupported "
            "claims or came too close to stating the forbidden answer. This "
            "time, only state things directly present in CURRICULUM CONTEXT, "
            "prefer asking a question over asserting anything, and avoid "
            "writing out any specific numeric or symbolic final value."
        )

    # Include prior turns so the tutor actually remembers the conversation
    # instead of treating every message as a fresh, context-free question.
    # Capped to the last 12 turns to keep prompts from growing unbounded
    # over a long session.
    messages = [("system", system)]
    for turn in state.conversation_history[-12:]:
        role = turn.get("role", "user")
        messages.append((role, turn.get("content", "")))
    messages.append(("user", state.student_message))

    resp = _llm.invoke(messages)
    return _extract_text(resp.content)


def generate_reply(state: TutorState) -> TutorState:
    state.draft_reply = _build_reply(state, strict=False)
    return state


def check_faithfulness(state: TutorState) -> TutorState:
    score, unsupported = score_faithfulness(state.draft_reply, state.retrieved_chunks)
    state.faithfulness_score = score
    return state


def check_leak(state: TutorState) -> TutorState:
    """
    Independent, non-LLM check: does the draft reply contain the canonical
    answer (or something mathematically equivalent to it)? This runs
    regardless of what the faithfulness check found — a reply can be
    perfectly grounded in the curriculum AND still leak the answer.
    """
    leaked, value = check_answer_leak(state.draft_reply, state.canonical_answer)
    state.answer_leaked = leaked
    state.leaked_value = value
    return state


def regenerate_strict(state: TutorState) -> TutorState:
    state.draft_reply = _build_reply(state, strict=True)
    state.was_regenerated = True
    state.regeneration_count += 1
    score, _ = score_faithfulness(state.draft_reply, state.retrieved_chunks)
    state.faithfulness_score = score
    leaked, value = check_answer_leak(state.draft_reply, state.canonical_answer)
    state.answer_leaked = leaked
    state.leaked_value = value
    return state


def finalize(state: TutorState) -> TutorState:
    # Last-resort backstop: if we've already retried once and the draft
    # STILL leaks the answer, don't ever send the leaking text to the
    # student — fall back to a safe, hardcoded scaffolding question instead
    # of trusting a third LLM call to finally get it right.
    if state.answer_leaked and state.regeneration_count >= 1:
        state.final_reply = (
            "Let's slow down for a second. Instead of the final value, "
            "can you walk me through the very first step you took, and why?"
        )
    else:
        state.final_reply = state.draft_reply
    return state


def _safety_router(state: TutorState) -> str:
    needs_retry = (
        state.faithfulness_score < settings.FAITHFULNESS_THRESHOLD or state.answer_leaked
    )
    if needs_retry and state.regeneration_count == 0:
        return "regenerate"
    return "finalize"


def build_tutor_graph():
    graph = StateGraph(TutorState)
    graph.add_node("classify_turn", classify_turn)
    graph.add_node("retrieve_curriculum", retrieve_curriculum)
    graph.add_node("detect_misconception", detect_misconception)
    graph.add_node("match_problem_bank", match_against_problem_bank)
    graph.add_node("generate_reply", generate_reply)
    graph.add_node("check_faithfulness", check_faithfulness)
    graph.add_node("check_leak", check_leak)
    graph.add_node("regenerate_strict", regenerate_strict)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("classify_turn")
    graph.add_edge("classify_turn", "retrieve_curriculum")
    graph.add_edge("retrieve_curriculum", "detect_misconception")
    graph.add_edge("detect_misconception", "match_problem_bank")
    graph.add_edge("match_problem_bank", "generate_reply")
    graph.add_edge("generate_reply", "check_faithfulness")
    graph.add_edge("check_faithfulness", "check_leak")
    graph.add_conditional_edges(
        "check_leak",
        _safety_router,
        {"regenerate": "regenerate_strict", "finalize": "finalize"},
    )
    graph.add_edge("regenerate_strict", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


tutor_graph = build_tutor_graph()
