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


settings = Settings()
