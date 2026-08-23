
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", "./chroma_db")
    CURRICULUM_COLLECTION: str = "curriculum"
    MISCONCEPTION_COLLECTION: str = "misconceptions"
    PROBLEM_BANK_COLLECTION: str = "problem_bank"
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "mistral")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    VISION_MODEL: str = os.getenv("VISION_MODEL", "llava")
    FAITHFULNESS_THRESHOLD: float = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.7"))
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "4"))
    MAX_HINT_LEVEL: int = int(os.getenv("MAX_HINT_LEVEL", "4"))
    PROBLEM_MATCH_THRESHOLD: float = float(os.getenv("PROBLEM_MATCH_THRESHOLD", "0.78"))
    LEAK_NUMERIC_TOLERANCE: float = float(os.getenv("LEAK_NUMERIC_TOLERANCE", "1e-6"))
settings = Settings()
