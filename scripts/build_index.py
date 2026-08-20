from pathlib import Path
import json
import os
import shutil
import time
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "legal_chunks.json"
INDEX_REPORT_PATH = PROJECT_ROOT / "data" / "chunks" / "indexing_report.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_env_value(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)

    if value is None or not value.strip():
        raise ValueError(f"Missing required environment variable: {name}")

    return value.strip()


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Chroma metadata values must be simple types:
    str, int, float, bool, or None.

    Purpose:
    - Avoid nested dictionaries/lists inside Chroma metadata.
    - Keep citation metadata searchable and easy to return later.
    """

    clean_metadata = {}

    for key, value in metadata.items():
        if value is None:
            clean_metadata[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            clean_metadata[key] = value
        else:
            clean_metadata[key] = json.dumps(value, ensure_ascii=False)

    return clean_metadata


def chunk_to_document(chunk: dict[str, Any]) -> Document:
    """
    Convert one chunk dictionary into a LangChain Document.

    page_content:
    - The text that will be embedded and searched.

    metadata:
    - The information needed for citations, filtering, and debugging.
    """

    metadata = {
        "chunk_id": chunk["chunk_id"],
        "canonical_id": chunk["canonical_id"],
        "law_id": chunk["law_id"],
        "law_name_ar": chunk["law_name_ar"],
        "law_name_en": chunk["law_name_en"],
        "article_number": int(chunk["article_number"]),
        "article_heading_ar": chunk["article_heading_ar"],
        "chunk_number": int(chunk["chunk_number"]),
        "total_chunks_for_article": int(chunk["total_chunks_for_article"]),
        "source_language": chunk["source_language"],
        "source_file": chunk["source_file"],
        "status": chunk["status"],
        "text_characters": int(chunk["text_characters"]),
    }

    return Document(
        page_content=chunk["text"],
        metadata=sanitize_metadata(metadata),
    )


def build_documents(chunks: list[dict[str, Any]]) -> tuple[list[Document], list[str]]:
    documents = []
    ids = []

    for chunk in chunks:
        if not chunk.get("indexable", True):
            continue

        documents.append(chunk_to_document(chunk))
        ids.append(chunk["chunk_id"])

    return documents, ids


def reset_vector_db(db_dir: Path) -> None:
    """
    Delete old vector database before rebuilding.

    Purpose:
    - Avoid mixing old chunks with new chunks.
    - Make indexing deterministic.
    """

    if db_dir.exists():
        shutil.rmtree(db_dir)


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Missing chunks file: {CHUNKS_PATH}\n"
            "Run scripts/chunk_documents.py first."
        )

    openai_api_key = get_env_value("OPENAI_API_KEY")
    embedding_model = get_env_value("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    chroma_db_dir = PROJECT_ROOT / get_env_value("CHROMA_DB_DIR", "vector_db/chroma")
    collection_name = get_env_value("COLLECTION_NAME", "saudi_legal_rag")

    chunks = load_json(CHUNKS_PATH)
    documents, ids = build_documents(chunks)

    if not documents:
        raise ValueError("No documents found for indexing.")

    print("Building vector database index...")
    print("=" * 80)
    print(f"Chunks loaded:       {len(chunks)}")
    print(f"Documents to index:  {len(documents)}")
    print(f"Embedding model:     {embedding_model}")
    print(f"Chroma directory:    {chroma_db_dir.relative_to(PROJECT_ROOT)}")
    print(f"Collection name:     {collection_name}")
    print("-" * 80)

    duplicate_ids = sorted({item for item in ids if ids.count(item) > 1})

    if duplicate_ids:
        raise ValueError(f"Duplicate chunk IDs found: {duplicate_ids[:10]}")

    start_time = time.perf_counter()

    reset_vector_db(chroma_db_dir)

    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        api_key=openai_api_key,
    )

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        ids=ids,
        collection_name=collection_name,
        persist_directory=str(chroma_db_dir),
    )


    elapsed_seconds = round(time.perf_counter() - start_time, 2)

    # Quick verification search
    sample_query = "مكافأة نهاية الخدمة"
    sample_results = vector_store.similarity_search_with_score(sample_query, k=3)

    sample_results_report = []

    for document, score in sample_results:
        sample_results_report.append(
            {
                "chunk_id": document.metadata.get("chunk_id"),
                "law_id": document.metadata.get("law_id"),
                "article_number": document.metadata.get("article_number"),
                "score": float(score),
                "preview": document.page_content[:300],
            }
        )

    by_law = {}

    for document in documents:
        law_id = document.metadata["law_id"]

        if law_id not in by_law:
            by_law[law_id] = {
                "documents": 0,
                "articles": set(),
            }

        by_law[law_id]["documents"] += 1
        by_law[law_id]["articles"].add(document.metadata["canonical_id"])

    by_law_report = {
        law_id: {
            "documents": stats["documents"],
            "articles": len(stats["articles"]),
        }
        for law_id, stats in by_law.items()
    }

    report = {
        "chunks_file": str(CHUNKS_PATH.relative_to(PROJECT_ROOT)),
        "chroma_db_dir": str(chroma_db_dir.relative_to(PROJECT_ROOT)),
        "collection_name": collection_name,
        "embedding_model": embedding_model,
        "documents_indexed": len(documents),
        "unique_ids": len(set(ids)),
        "duplicate_ids_count": len(duplicate_ids),
        "elapsed_seconds": elapsed_seconds,
        "by_law": by_law_report,
        "sample_query": sample_query,
        "sample_results": sample_results_report,
        "status": "success",
    }

    save_json(INDEX_REPORT_PATH, report)

    print("Vector database indexing completed")
    print("=" * 80)
    print(f"Documents indexed:   {len(documents)}")
    print(f"Unique IDs:          {len(set(ids))}")
    print(f"Elapsed seconds:     {elapsed_seconds}")
    print("-" * 80)

    print("By law:")
    for law_id, stats in by_law_report.items():
        print(
            f"  {law_id}: articles={stats['articles']}, documents={stats['documents']}"
        )

    print("-" * 80)
    print("Sample semantic search:")
    print(f"Query: {sample_query}")

    for item in sample_results_report:
        print(
            f"  chunk_id={item['chunk_id']} | "
            f"law={item['law_id']} | "
            f"article={item['article_number']} | "
            f"score={item['score']}"
        )

    print("-" * 80)
    print(f"Index report saved to: {INDEX_REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()