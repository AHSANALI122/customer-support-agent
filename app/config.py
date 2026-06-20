from dotenv import load_dotenv
import os

load_dotenv()


class Settings:
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./support_agent.db")
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    model_name: str = os.getenv("MODEL_NAME", "gemini-2.0-flash")
    embed_model_name: str = os.getenv("EMBED_MODEL_NAME", "models/text-embedding-004")
    chroma_path: str = os.getenv("CHROMA_PATH", "./chroma_db")
    admin_token: str = os.getenv("ADMIN_TOKEN", "")
    # Below this relevance score (0–1, higher = more relevant) a policy match is
    # treated as low-confidence so the agent hedges instead of answering firmly.
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    # Verification throttle (F8): after this many failed order/email lookups
    # within the window below, order lookups are paused for the session so an
    # attacker can't make unlimited guesses at order_id/email combinations.
    verification_max_attempts: int = int(os.getenv("VERIFICATION_MAX_ATTEMPTS", "5"))
    verification_window_minutes: int = int(os.getenv("VERIFICATION_WINDOW_MINUTES", "10"))


settings = Settings()
