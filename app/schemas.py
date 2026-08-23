from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class TurnType(str, Enum):
    QUESTION = "question"
    ANSWER_ATTEMPT = "answer_attempt"


class ChatRequest(BaseModel):
    session_id: str
    student_id: str
    message: str
    # Optional base64 image (e.g. photo of handwritten work) for the
    # multimodal path. Frontend sends this alongside the text message.
    image_base64: Optional[str] = None
    topic: str = Field(..., description="e.g. 'quadratic_equations', 'newtons_third_law'")


class RetrievedChunk(BaseModel):
    content: str
    source: str
    score: float


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    hint_level: int
    detected_misconception: Optional[str] = None
    faithfulness_score: float
    grounded_sources: List[str]
    # True if the response had to be regenerated because it failed the
    # faithfulness check OR the answer-leak check on the first pass —
    # surfaced for transparency (great demo material: show the judge the
    # exact moment a leak got intercepted).
    was_regenerated: bool = False
    answer_leak_intercepted: bool = False


class StudentProfile(BaseModel):
    student_id: str
    topic_hint_history: dict = Field(default_factory=dict)  # topic -> avg hints needed


class TutorState(BaseModel):
    """
    The full state object that flows through the LangGraph graph.
    Each node reads/writes this instead of everything being crammed
    into one prompt string.
    """
    session_id: str
    student_id: str
    topic: str
    student_message: str
    image_base64: Optional[str] = None

    # Prior turns in this session, oldest first, e.g.
    # [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    # Passed in by main.py so the model actually remembers the conversation
    # instead of treating every message as a fresh, context-free question.
    conversation_history: List[dict] = Field(default_factory=list)

    turn_type: Optional[TurnType] = None
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    detected_misconception: Optional[str] = None

    # Problem-bank matching — canonical_answer was computed independently
    # of the LLM (see app/problem_bank.py), and is what answer_leak.py
    # checks every draft reply against.
    matched_problem_text: Optional[str] = None
    canonical_answer: Optional[str] = None

    hint_level: int = 1
    draft_reply: Optional[str] = None
    faithfulness_score: float = 0.0
    answer_leaked: bool = False
    leaked_value: Optional[str] = None
    was_regenerated: bool = False
    regeneration_count: int = 0
    final_reply: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
