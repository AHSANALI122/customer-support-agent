from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import settings

COLLECTION_NAME = "support_policies"


def get_vectorstore() -> Chroma:
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.embed_model_name,
        google_api_key=settings.google_api_key,
    )
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
