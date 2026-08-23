# Hallucination-Resistant Socratic Tutor — Backend

## What makes this different from a standard RAG tutor

1. **Measurable faithfulness, not just a claim.** Every reply is split into
   individual factual claims (`app/faithfulness.py`) and each claim is
   checked against the retrieved curriculum chunks. A reply only reaches the
   student if it clears `FAITHFULNESS_THRESHOLD`; otherwise it's regenerated
   in strict mode. The score is returned in the API response so it can be
   shown live in the UI (e.g. "grounded: 0.92, sources: [Ch.4 p.12]").

2. **Socratic behavior as an explicit graph**, not one big prompt
   (`app/state_machine.py`, built with LangGraph):
   `classify_turn → retrieve_curriculum → detect_misconception →
   generate_reply → check_faithfulness → (regenerate?) → finalize`
   Each node is independently testable and debuggable.

3. **Misconception library.** A second Chroma collection stores known
   student error patterns per topic. When a student's attempt matches one,
   the tutor targets it specifically instead of giving a generic hint.

4. **Adaptive hint level.** `main.py` keeps a lightweight per-student,
   per-topic running average of hint levels needed, so a student who's
   struggled with a topic before starts at a gentler level next time.

## Setup

1. Install Ollama from https://ollama.com (Windows/Mac/Linux installer, not a pip package).
2. Pull the models this project uses:
   ```
   ollama pull mistral
   ollama pull nomic-embed-text
   ollama pull llava
   ```
3. Leave the Ollama app running in the background (it serves on `http://localhost:11434` automatically).
4. Then:
   ```bash
   pip install -r requirements.txt
   cp .env.example .env
   uvicorn app.main:app --reload --port 8000
   ```

No API key needed — everything runs locally, which is also what makes the
"open-source" claim in the project title true end-to-end. Note: `mistral`
is used by default because it's Apache 2.0 licensed (genuinely
unrestricted open source); `llama3.1` is a fine alternative but carries
Meta's Community License with field-of-use restrictions, worth mentioning
if a judge asks about licensing.

## Endpoints

- `POST /ingest` — upload a curriculum file (pdf/docx/txt) with a `source_label`
- `POST /misconceptions/seed` — seed known misconception patterns for a topic
- `POST /chat` — main tutoring endpoint, called by the frontend
- `GET /health`

## For your frontend teammate

`POST /chat` expects:
```json
{
  "session_id": "abc123",
  "student_id": "student_42",
  "message": "I got x = 3",
  "topic": "quadratic_equations",
  "image_base64": null
}
```
Returns `reply`, `hint_level`, `detected_misconception`, `faithfulness_score`,
`grounded_sources`, `was_regenerated` — all worth surfacing in the UI, not
just `reply`.

## Not yet wired up (next steps)

- Vision model call for `image_base64` (currently accepted but unused —
  add a node in `state_machine.py` that calls `VISION_MODEL` when an image
  is present and merges its OCR/description into `student_message`)
- Persistent (non-in-memory) hint history store
- `scripts/seed_misconceptions.py` with real seed data per topic
