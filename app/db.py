from sqlmodel import SQLModel, Session, create_engine
from app.config import settings

engine = create_engine(settings.database_url, echo=False)


def init_db():
    import app.models  # noqa: F401 — registers all models with SQLModel metadata
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
