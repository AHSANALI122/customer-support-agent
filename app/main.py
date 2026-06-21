import logging
import os
from contextlib import asynccontextmanager

import chromadb
from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session, text

from app.api.analytics import router as analytics_router
from app.api.chat import router as chat_router
from app.api.feedback import router as feedback_router
from app.api.knowledge_base import router as knowledge_base_router
from app.api.orders import router as orders_router
from app.api.tickets import router as tickets_router
from app.config import settings
from app.db import engine, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # Fail loudly at startup if the DB is unreachable (F13). SQLAlchemy's
    # create_engine is lazy, so this is the first actual connection attempt.
    try:
        with Session(engine) as session:
            session.exec(text("SELECT 1"))
    except Exception as e:
        logging.critical("Database is unreachable at startup: %s", e)
        raise
    init_db()
    yield


app = FastAPI(title="Customer Support Agent", lifespan=lifespan)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Let FastAPI's built-in handler deal with validation errors (clean 422).
    if isinstance(exc, RequestValidationError):
        return await request_validation_exception_handler(request, exc)
    logging.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={"error": "Something went wrong, please try again."},
    )


app.include_router(orders_router)
app.include_router(chat_router)
app.include_router(tickets_router)
app.include_router(feedback_router)
app.include_router(analytics_router)
app.include_router(knowledge_base_router)


@app.get("/health")
def health():
    result = {"db": "ok", "llm": "ok", "vector_store": "ok"}

    # DB check
    try:
        with Session(engine) as session:
            session.exec(text("SELECT 1"))
    except Exception:
        result["db"] = "error"

    # LLM check — list models to validate the Groq API key
    try:
        from groq import Groq
        client = Groq(api_key=settings.groq_api_key)
        models = list(client.models.list().data)
        if not models:
            raise ValueError("No models returned")
    except Exception:
        result["llm"] = "error"

    # Vector store check
    try:
        if not os.path.isdir(settings.chroma_path):
            raise FileNotFoundError("chroma_path directory does not exist")
        chromadb.PersistentClient(path=settings.chroma_path)
    except Exception:
        result["vector_store"] = "error"

    status = "ok" if all(v == "ok" for v in result.values()) else "error"
    return {"status": status, **result}


# Serve the frontend — mounted last so API routes always take priority.
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
