"""Admin endpoint for uploading PDF policy documents (beyond F14 spec).

POST /admin/knowledge-base/upload  — accepts a PDF, extracts text, saves it
as a markdown file in data/policies/, and re-runs the ingest pipeline so the
new content is immediately searchable.
"""

import logging
import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pypdf import PdfReader

from app.api.deps import verify_admin
from app.rag.ingest import ingest

router = APIRouter(prefix="/admin/knowledge-base", tags=["admin"])

POLICIES_DIR = Path(__file__).parent.parent.parent / "data" / "policies"


def _pdf_to_text(data: bytes) -> str:
    import io
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return "\n\n".join(pages)


def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    return re.sub(r"[^\w\-]", "_", stem).strip("_") or "uploaded_policy"


@router.post("/upload", dependencies=[Depends(verify_admin)])
async def upload_policy_pdf(file: UploadFile = File(...)):
    """Upload a PDF policy document and immediately ingest it into the vector store.

    Returns the filename saved and the number of chunks added.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        text = _pdf_to_text(data)
    except Exception:
        logging.exception("Failed to extract text from PDF %s", file.filename)
        raise HTTPException(status_code=422, detail="Could not extract text from the PDF.")

    if not text.strip():
        raise HTTPException(status_code=422, detail="No readable text found in the PDF.")

    stem = _safe_stem(file.filename)
    dest = POLICIES_DIR / f"{stem}.md"
    POLICIES_DIR.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    logging.info("Saved policy document: %s", dest)

    try:
        ingest()
    except Exception:
        logging.exception("Ingest failed after uploading %s", file.filename)
        raise HTTPException(status_code=500, detail="File saved but ingest failed — check server logs.")

    return {"saved_as": dest.name, "message": "Uploaded and ingested successfully."}
