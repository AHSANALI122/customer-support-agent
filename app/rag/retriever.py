from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings

COLLECTION_NAME = "support_policies"


def get_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=settings.embed_model_name)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=settings.chroma_path,
    )


def get_retriever(k: int = 4) -> VectorStoreRetriever:
    return get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def search_with_scores(query: str, k: int = 4) -> list[tuple[Document, float]]:
    """Top-k policy chunks paired with relevance scores (0–1, higher = more
    relevant). Lets callers judge confidence in a match, which the plain
    retriever's documents alone don't expose."""
    return get_vectorstore().similarity_search_with_relevance_scores(query, k=k)
