from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import settings

POLICIES_DIR = Path(__file__).parent.parent.parent / "data" / "policies"
COLLECTION_NAME = "support_policies"


def load_docs() -> list[Document]:
    return [
        Document(page_content=f.read_text(encoding="utf-8"), metadata={"source": f.name})
        for f in sorted(POLICIES_DIR.glob("*.md"))
    ]


def chunk_docs(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)


def build_chunk_ids(chunks: list[Document]) -> list[str]:
    counter: dict[str, int] = {}
    ids = []
    for chunk in chunks:
        src = chunk.metadata["source"]
        idx = counter.get(src, 0)
        ids.append(f"{src}::{idx}")
        counter[src] = idx + 1
    return ids


def ingest() -> None:
    print("Loading policy documents...")
    docs = load_docs()
    if not docs:
        print(f"No markdown files found in {POLICIES_DIR}")
        return

    chunks = chunk_docs(docs)
    ids = build_chunk_ids(chunks)
    print(f"  {len(docs)} document(s) -> {len(chunks)} chunk(s)")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.embed_model_name,
        google_api_key=settings.google_api_key,
    )
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=settings.chroma_path,
    )

    existing_ids = set(vectorstore.get()["ids"])
    new_pairs = [(chunk, id_) for chunk, id_ in zip(chunks, ids) if id_ not in existing_ids]

    if not new_pairs:
        print("All chunks already ingested — nothing to do.")
        return

    new_chunks, new_ids = zip(*new_pairs)
    skipped = len(chunks) - len(new_chunks)
    print(f"  Adding {len(new_chunks)} new chunk(s)" + (f", skipping {skipped} existing" if skipped else "") + "...")
    vectorstore.add_documents(list(new_chunks), ids=list(new_ids))
    print("Ingestion complete.")


if __name__ == "__main__":
    ingest()
