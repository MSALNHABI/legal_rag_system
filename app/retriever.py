from pathlib import Path
import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "legal_chunks.json"


class LegalRetriever:
    """
    Retriever for the Saudi legal RAG system.

    Supports:
    1. Semantic search using OpenAI embeddings + Chroma.
    2. Keyword search using BM25.
    3. Exact article lookup for queries like "المادة 80".
    4. Weighted hybrid search using Reciprocal Rank Fusion.

    Purpose:
    - Semantic search captures meaning.
    - Keyword search captures exact legal terms.
    - Exact article lookup handles legal article-number queries.
    - Weighted hybrid search combines them while protecting strong semantic performance.
    """

    def __init__(self) -> None:
        load_dotenv(PROJECT_ROOT / ".env")

        self.openai_api_key = self._get_env_value("OPENAI_API_KEY")
        self.embedding_model = self._get_env_value(
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-large",
        )
        self.chroma_db_dir = PROJECT_ROOT / self._get_env_value(
            "CHROMA_DB_DIR",
            "vector_db/chroma",
        )
        self.collection_name = self._get_env_value(
            "COLLECTION_NAME",
            "saudi_legal_rag",
        )

        if not self.chroma_db_dir.exists():
            raise FileNotFoundError(
                f"Vector database not found: {self.chroma_db_dir}\n"
                "Run scripts/build_index.py first."
            )

        if not CHUNKS_PATH.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {CHUNKS_PATH}\n"
                "Run scripts/chunk_documents.py first."
            )

        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            api_key=self.openai_api_key,
        )

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.chroma_db_dir),
        )

        self.chunks = self._load_chunks()

        self.chunk_by_id = {
            chunk["chunk_id"]: chunk
            for chunk in self.chunks
        }

        self.bm25_corpus_tokens = [
            self._tokenize_for_keyword_search(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(self.bm25_corpus_tokens)

    def _get_env_value(self, name: str, default: str | None = None) -> str:
        value = os.getenv(name, default)

        if value is None or not value.strip():
            raise ValueError(f"Missing required environment variable: {name}")

        return value.strip()

    def _load_chunks(self) -> list[dict[str, Any]]:
        chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

        return [
            chunk
            for chunk in chunks
            if chunk.get("indexable") is True
        ]

    def _normalize_arabic_text(self, text: str) -> str:
        """
        Normalize Arabic text.

        Purpose:
        - Remove tashkeel.
        - Remove tatweel.
        - Normalize common Arabic spelling variants.
        """

        text = text.lower()

        text = re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", text)
        text = text.replace("ـ", "")

        replacements = {
            "أ": "ا",
            "إ": "ا",
            "آ": "ا",
            "ى": "ي",
            "ة": "ه",
            "ؤ": "و",
            "ئ": "ي",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    def _tokenize_for_keyword_search(self, text: str) -> list[str]:
        """
        Convert Arabic/English legal text into BM25 tokens.
        """

        text = self._normalize_arabic_text(text)

        text = re.sub(r"[^0-9A-Za-z\u0600-\u06FF]+", " ", text)

        tokens = [
            token.strip()
            for token in text.split()
            if len(token.strip()) >= 2
        ]

        return tokens

    def _arabic_digits_to_english(self, text: str) -> str:
        arabic_digits = "٠١٢٣٤٥٦٧٨٩"
        english_digits = "0123456789"

        translation_table = str.maketrans(arabic_digits, english_digits)
        return text.translate(translation_table)

    def _detect_law_hint(self, query: str) -> str | None:
        """
        Detect whether the user explicitly refers to Labor Law or Social Insurance Law.
        """

        normalized = self._normalize_arabic_text(query)

        if "التامينات" in normalized or "التامين الاجتماعي" in normalized:
            return "social_insurance_law"

        if "نظام العمل" in normalized:
            return "labor_law"

        return None

    def _extract_article_number_from_query(self, query: str) -> int | None:
        """
        Extract article number from queries such as:
        - المادة 80
        - مادة ٨٠
        - Article 80
        """

        normalized = self._normalize_arabic_text(query)
        normalized = self._arabic_digits_to_english(normalized)

        arabic_match = re.search(
            r"(?:الماده|ماده)\s*\(?\s*(\d{1,3})\s*\)?",
            normalized,
        )

        if arabic_match:
            return int(arabic_match.group(1))

        english_match = re.search(
            r"\barticle\s+(\d{1,3})\b",
            normalized,
            flags=re.IGNORECASE,
        )

        if english_match:
            return int(english_match.group(1))

        return None

    def _format_chunk_result(
        self,
        chunk: dict[str, Any],
        rank: int,
        score: float,
        retrieval_method: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = {
            "rank": rank,
            "score": float(score),
            "retrieval_method": retrieval_method,
            "chunk_id": chunk["chunk_id"],
            "canonical_id": chunk["canonical_id"],
            "law_id": chunk["law_id"],
            "law_name_ar": chunk["law_name_ar"],
            "law_name_en": chunk["law_name_en"],
            "article_number": chunk["article_number"],
            "article_heading_ar": chunk["article_heading_ar"],
            "chunk_number": chunk["chunk_number"],
            "total_chunks_for_article": chunk["total_chunks_for_article"],
            "source_file": chunk["source_file"],
            "text": chunk["text"],
            "citation": {
                "law_name_ar": chunk["law_name_ar"],
                "law_name_en": chunk["law_name_en"],
                "article_number": chunk["article_number"],
                "article_heading_ar": chunk["article_heading_ar"],
                "chunk_id": chunk["chunk_id"],
            },
        }

        if extra:
            result.update(extra)

        return result

    def exact_article_search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """
        Directly retrieve an article if the query contains an article number.

        Purpose:
        - Legal users often ask by article number.
        - This avoids retrieval mistakes for short queries like "المادة 80".
        """

        article_number = self._extract_article_number_from_query(query)

        if article_number is None:
            return []

        law_hint = self._detect_law_hint(query)

        matching_chunks = []

        for chunk in self.chunks:
            if int(chunk["article_number"]) != article_number:
                continue

            if law_hint and chunk["law_id"] != law_hint:
                continue

            matching_chunks.append(chunk)

        matching_chunks = sorted(
            matching_chunks,
            key=lambda item: (
                item["law_id"],
                int(item["article_number"]),
                int(item["chunk_number"]),
            ),
        )

        results = []

        for rank, chunk in enumerate(matching_chunks[:top_k], start=1):
            results.append(
                self._format_chunk_result(
                    chunk=chunk,
                    rank=rank,
                    score=1.0,
                    retrieval_method="exact_article",
                    extra={
                        "exact_article_match": True,
                        "semantic_rank": None,
                        "keyword_rank": None,
                        "semantic_score": None,
                        "keyword_score": None,
                        "matched_by": ["exact_article"],
                    },
                )
            )

        return results

    def semantic_search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """
        Run semantic search against Chroma.

        Note:
        - Chroma score is usually distance.
        - Lower score usually means more similar.
        """

        query = query.strip()

        if not query:
            return []

        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
        )

        retrieved_chunks = []

        for rank, (document, score) in enumerate(results, start=1):
            metadata = document.metadata
            chunk_id = metadata.get("chunk_id")
            chunk = self.chunk_by_id.get(chunk_id)

            if not chunk:
                continue

            retrieved_chunks.append(
                self._format_chunk_result(
                    chunk=chunk,
                    rank=rank,
                    score=float(score),
                    retrieval_method="semantic",
                )
            )

        return retrieved_chunks

    def keyword_search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """
        Run BM25 keyword search.
        """

        query = query.strip()

        if not query:
            return []

        query_tokens = self._tokenize_for_keyword_search(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        retrieved_chunks = []
        rank = 1

        for index in ranked_indices:
            score = float(scores[index])

            if score <= 0:
                continue

            chunk = self.chunks[index]

            retrieved_chunks.append(
                self._format_chunk_result(
                    chunk=chunk,
                    rank=rank,
                    score=score,
                    retrieval_method="keyword",
                )
            )

            rank += 1

            if len(retrieved_chunks) >= top_k:
                break

        return retrieved_chunks

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        semantic_k: int = 20,
        keyword_k: int = 20,
        rrf_k: int = 60,
        semantic_weight: float = 2.0,
        keyword_weight: float = 1.0,
    ) -> list[dict[str, Any]]:
        """
        Weighted hybrid search.

        Strategy:
        1. If query contains a direct article number, boost the exact article.
        2. Run semantic search.
        3. Run BM25 keyword search.
        4. Combine semantic + keyword using weighted Reciprocal Rank Fusion.

        Weighted RRF:
            score = semantic_weight / (rrf_k + semantic_rank)
                  + keyword_weight  / (rrf_k + keyword_rank)
        """

        query = query.strip()

        if not query:
            return []

        exact_results = self.exact_article_search(query=query, top_k=top_k)

        semantic_results = self.semantic_search(query=query, top_k=semantic_k)
        keyword_results = self.keyword_search(query=query, top_k=keyword_k)

        fused: dict[str, dict[str, Any]] = {}

        def add_results(
            results: list[dict[str, Any]],
            source_name: str,
            source_weight: float,
        ) -> None:
            for item in results:
                chunk_id = item["chunk_id"]
                rank = int(item["rank"])
                rrf_score = source_weight / (rrf_k + rank)

                if chunk_id not in fused:
                    chunk = self.chunk_by_id[chunk_id]

                    fused[chunk_id] = {
                        "chunk": chunk,
                        "rrf_score": 0.0,
                        "semantic_rank": None,
                        "keyword_rank": None,
                        "semantic_score": None,
                        "keyword_score": None,
                        "matched_by": [],
                    }

                fused[chunk_id]["rrf_score"] += rrf_score

                if source_name == "semantic":
                    fused[chunk_id]["semantic_rank"] = rank
                    fused[chunk_id]["semantic_score"] = item["score"]

                if source_name == "keyword":
                    fused[chunk_id]["keyword_rank"] = rank
                    fused[chunk_id]["keyword_score"] = item["score"]

                if source_name not in fused[chunk_id]["matched_by"]:
                    fused[chunk_id]["matched_by"].append(source_name)

        add_results(semantic_results, "semantic", semantic_weight)
        add_results(keyword_results, "keyword", keyword_weight)

        ranked_items = sorted(
            fused.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

        exact_chunk_ids = {
            item["chunk_id"]
            for item in exact_results
        }

        final_results = []

        # Exact article matches always come first.
        for item in exact_results:
            final_results.append(item)

        for item in ranked_items:
            chunk = item["chunk"]

            if chunk["chunk_id"] in exact_chunk_ids:
                continue

            final_results.append(
                self._format_chunk_result(
                    chunk=chunk,
                    rank=0,
                    score=float(item["rrf_score"]),
                    retrieval_method="hybrid_weighted_rrf",
                    extra={
                        "exact_article_match": False,
                        "semantic_rank": item["semantic_rank"],
                        "keyword_rank": item["keyword_rank"],
                        "semantic_score": item["semantic_score"],
                        "keyword_score": item["keyword_score"],
                        "matched_by": item["matched_by"],
                        "semantic_weight": semantic_weight,
                        "keyword_weight": keyword_weight,
                    },
                )
            )

            if len(final_results) >= top_k:
                break

        # Reassign final ranks after exact-article boosting.
        for rank, item in enumerate(final_results[:top_k], start=1):
            item["rank"] = rank

        return final_results[:top_k]

    def retrieve_semantic_chunk_ids(self, query: str, top_k: int = 10) -> list[str]:
        results = self.semantic_search(query=query, top_k=top_k)
        return [item["chunk_id"] for item in results if item.get("chunk_id")]

    def retrieve_keyword_chunk_ids(self, query: str, top_k: int = 10) -> list[str]:
        results = self.keyword_search(query=query, top_k=top_k)
        return [item["chunk_id"] for item in results if item.get("chunk_id")]

    def retrieve_hybrid_chunk_ids(self, query: str, top_k: int = 10) -> list[str]:
        results = self.hybrid_search(query=query, top_k=top_k)
        return [item["chunk_id"] for item in results if item.get("chunk_id")]


LegalSemanticRetriever = LegalRetriever