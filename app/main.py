import logging
import os
from contextlib import asynccontextmanager

import chromadb
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session, text

from app.api.chat import router as chat_router
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
    init_db()
    yield


app = FastAPI(title="Customer Support Agent", lifespan=lifespan)
app.include_router(orders_router)
app.include_router(chat_router)
app.include_router(tickets_router)


@app.get("/health")
def health():
    result = {"db": "ok", "llm": "ok", "vector_store": "ok"}

    # DB check
    try:
        with Session(engine) as session:
            session.exec(text("SELECT 1"))
    except Exception:
        result["db"] = "error"

    # LLM check — list models (free, no quota cost) to validate the key
    try:
        from google import genai as google_genai
        client = google_genai.Client(api_key=settings.google_api_key)
        models = list(client.models.list())
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
