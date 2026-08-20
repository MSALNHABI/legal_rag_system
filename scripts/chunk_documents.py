from pathlib import Path
import json
import re
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CANONICAL_ARTICLES_PATH = PROJECT_ROOT / "data" / "processed" / "articles" / "canonical_articles.json"

CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"
CHUNKS_OUTPUT_PATH = CHUNKS_DIR / "legal_chunks.json"
REPORT_OUTPUT_PATH = CHUNKS_DIR / "chunking_report.json"


MAX_CHUNK_CHARACTERS = 2500
MIN_CHUNK_CHARACTERS = 80
CHUNK_OVERLAP_PARAGRAPHS = 1


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_paragraphs(text: str) -> list[str]:
    """
    Split article text into paragraph-like units.

    Purpose:
    - Legal text often contains paragraphs, numbered clauses, or bullet-like lines.
    - We preserve these boundaries instead of cutting randomly.
    """

    text = normalize_text(text)

    if not text:
        return []

    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]

    paragraphs = []

    for line in raw_lines:
        # Split long lines when they contain Arabic semicolon or period-like legal clauses.
        if len(line) > MAX_CHUNK_CHARACTERS:
            parts = re.split(r"(?<=[.؟؛])\s+", line)
            for part in parts:
                part = normalize_text(part)
                if part:
                    paragraphs.append(part)
        else:
            paragraphs.append(line)

    return paragraphs


def build_chunk_text(article: dict[str, Any], body_part: str) -> str:
    """
    Build the text that will be embedded.

    We include law name and article heading because it improves retrieval.
    """

    law_name = article["law_name_ar"]
    article_heading = article["article_heading_ar"]
    article_number = article["article_number"]

    return normalize_text(
        f"{law_name}\n"
        f"{article_heading}\n"
        f"رقم المادة: {article_number}\n\n"
        f"{body_part}"
    )


def split_article_into_body_parts(article_text: str) -> list[str]:
    """
    Split a long article into body parts.

    Strategy:
    - If the article is short, keep it as one body part.
    - If it is long, group paragraphs until max character limit is reached.
    - Add small paragraph overlap for continuity.
    """

    article_text = normalize_text(article_text)

    if len(article_text) <= MAX_CHUNK_CHARACTERS:
        return [article_text]

    paragraphs = split_into_paragraphs(article_text)

    if not paragraphs:
        return []

    chunks = []
    current_paragraphs = []
    current_length = 0

    index = 0

    while index < len(paragraphs):
        paragraph = paragraphs[index]
        paragraph_length = len(paragraph)

        if current_paragraphs and current_length + paragraph_length > MAX_CHUNK_CHARACTERS:
            chunks.append(normalize_text("\n".join(current_paragraphs)))

            if CHUNK_OVERLAP_PARAGRAPHS > 0:
                current_paragraphs = current_paragraphs[-CHUNK_OVERLAP_PARAGRAPHS:]
                current_length = sum(len(item) for item in current_paragraphs)
            else:
                current_paragraphs = []
                current_length = 0

            continue

        # If a single paragraph is still too long, hard split it safely.
        if paragraph_length > MAX_CHUNK_CHARACTERS:
            start = 0
            while start < paragraph_length:
                end = start + MAX_CHUNK_CHARACTERS
                part = paragraph[start:end].strip()
                if part:
                    if current_paragraphs:
                        chunks.append(normalize_text("\n".join(current_paragraphs)))
                        current_paragraphs = []
                        current_length = 0

                    chunks.append(part)
                start = end

            index += 1
            continue

        current_paragraphs.append(paragraph)
        current_length += paragraph_length
        index += 1

    if current_paragraphs:
        chunks.append(normalize_text("\n".join(current_paragraphs)))

    return [chunk for chunk in chunks if chunk.strip()]


def create_chunk_id(canonical_id: str, chunk_number: int) -> str:
    return f"{canonical_id}__chunk_{chunk_number:03d}"


def create_chunks_for_article(article: dict[str, Any]) -> list[dict[str, Any]]:
    body_parts = split_article_into_body_parts(article["text_ar"])

    chunks = []
    total_parts = len(body_parts)

    for index, body_part in enumerate(body_parts, start=1):
        chunk_text = build_chunk_text(article, body_part)
        chunk_id = create_chunk_id(article["canonical_id"], index)

        chunk = {
            "chunk_id": chunk_id,
            "canonical_id": article["canonical_id"],
            "law_id": article["law_id"],
            "law_name_ar": article["law_name_ar"],
            "law_name_en": article["law_name_en"],
            "article_number": article["article_number"],
            "article_heading_ar": article["article_heading_ar"],
            "chunk_number": index,
            "total_chunks_for_article": total_parts,
            "text": chunk_text,
            "body_text": body_part,
            "text_characters": len(chunk_text),
            "body_characters": len(body_part),
            "source_language": article["source_language"],
            "source_file": article["source_file"],
            "status": article["status"],
            "indexable": article["indexable"],
            "citation": {
                "law_name_ar": article["law_name_ar"],
                "law_name_en": article["law_name_en"],
                "article_number": article["article_number"],
                "article_heading_ar": article["article_heading_ar"],
            },
            "metadata": {
                "law_id": article["law_id"],
                "article_number": article["article_number"],
                "canonical_id": article["canonical_id"],
                "chunk_number": index,
                "source_language": article["source_language"],
                "status": article["status"],
            },
        }

        chunks.append(chunk)

    return chunks


def validate_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    duplicate_chunk_ids = sorted(
        {
            chunk_id
            for chunk_id in chunk_ids
            if chunk_ids.count(chunk_id) > 1
        }
    )

    empty_chunks = [
        chunk["chunk_id"]
        for chunk in chunks
        if not chunk.get("text", "").strip()
    ]

    short_chunks = [
        {
            "chunk_id": chunk["chunk_id"],
            "text_characters": chunk["text_characters"],
        }
        for chunk in chunks
        if chunk["text_characters"] < MIN_CHUNK_CHARACTERS
    ]

    oversized_chunks = [
        {
            "chunk_id": chunk["chunk_id"],
            "text_characters": chunk["text_characters"],
        }
        for chunk in chunks
        if chunk["text_characters"] > MAX_CHUNK_CHARACTERS + 300
    ]

    missing_metadata = []

    required_fields = [
        "chunk_id",
        "canonical_id",
        "law_id",
        "article_number",
        "text",
        "citation",
        "metadata",
    ]

    for chunk in chunks:
        for field in required_fields:
            if field not in chunk:
                missing_metadata.append(
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "missing_field": field,
                    }
                )

    return {
        "duplicate_chunk_ids_count": len(duplicate_chunk_ids),
        "duplicate_chunk_ids": duplicate_chunk_ids[:30],
        "empty_chunks_count": len(empty_chunks),
        "empty_chunks": empty_chunks[:30],
        "short_chunks_count": len(short_chunks),
        "short_chunks_sample": short_chunks[:30],
        "oversized_chunks_count": len(oversized_chunks),
        "oversized_chunks_sample": oversized_chunks[:30],
        "missing_metadata_count": len(missing_metadata),
        "missing_metadata_sample": missing_metadata[:30],
        "ready_for_indexing": (
            len(duplicate_chunk_ids) == 0
            and len(empty_chunks) == 0
            and len(missing_metadata) == 0
        ),
    }


def main() -> None:
    if not CANONICAL_ARTICLES_PATH.exists():
        raise FileNotFoundError(
            f"Missing canonical articles file: {CANONICAL_ARTICLES_PATH}\n"
            "Run scripts/resolve_current_articles.py first."
        )

    canonical_articles = load_json(CANONICAL_ARTICLES_PATH)

    indexable_articles = [
        article
        for article in canonical_articles
        if article.get("indexable") is True
    ]

    all_chunks = []

    print("Creating legal retrieval chunks...")
    print("=" * 80)
    print(f"Canonical articles loaded: {len(canonical_articles)}")
    print(f"Indexable articles:        {len(indexable_articles)}")
    print("-" * 80)

    articles_split_into_multiple_chunks = []

    for article in indexable_articles:
        chunks = create_chunks_for_article(article)
        all_chunks.extend(chunks)

        if len(chunks) > 1:
            articles_split_into_multiple_chunks.append(
                {
                    "canonical_id": article["canonical_id"],
                    "law_id": article["law_id"],
                    "article_number": article["article_number"],
                    "chunks_created": len(chunks),
                    "text_characters": article["text_characters"],
                }
            )

    validation = validate_chunks(all_chunks)

    by_law = {}

    for chunk in all_chunks:
        law_id = chunk["law_id"]

        if law_id not in by_law:
            by_law[law_id] = {
                "chunks": 0,
                "articles": set(),
            }

        by_law[law_id]["chunks"] += 1
        by_law[law_id]["articles"].add(chunk["canonical_id"])

    by_law_report = {
        law_id: {
            "chunks": stats["chunks"],
            "articles": len(stats["articles"]),
        }
        for law_id, stats in by_law.items()
    }

    report = {
        "source_file": str(CANONICAL_ARTICLES_PATH.relative_to(PROJECT_ROOT)),
        "output_file": str(CHUNKS_OUTPUT_PATH.relative_to(PROJECT_ROOT)),
        "max_chunk_characters": MAX_CHUNK_CHARACTERS,
        "chunk_overlap_paragraphs": CHUNK_OVERLAP_PARAGRAPHS,
        "canonical_articles_loaded": len(canonical_articles),
        "indexable_articles": len(indexable_articles),
        "chunks_created": len(all_chunks),
        "by_law": by_law_report,
        "articles_split_into_multiple_chunks_count": len(articles_split_into_multiple_chunks),
        "articles_split_into_multiple_chunks_sample": articles_split_into_multiple_chunks[:50],
        "average_chunk_characters": round(
            sum(chunk["text_characters"] for chunk in all_chunks) / max(len(all_chunks), 1),
            2,
        ),
        "validation": validation,
    }

    save_json(CHUNKS_OUTPUT_PATH, all_chunks)
    save_json(REPORT_OUTPUT_PATH, report)

    print("Chunking completed")
    print("=" * 80)
    print(f"Chunks created:                       {len(all_chunks)}")

    for law_id, stats in by_law_report.items():
        print(
            f"{law_id}: articles={stats['articles']}, chunks={stats['chunks']}"
        )

    print(f"Articles split into multiple chunks:  {len(articles_split_into_multiple_chunks)}")
    print(f"Average chunk characters:             {report['average_chunk_characters']}")
    print("-" * 80)
    print(f"Duplicate chunk IDs:                  {validation['duplicate_chunk_ids_count']}")
    print(f"Empty chunks:                         {validation['empty_chunks_count']}")
    print(f"Short chunks:                         {validation['short_chunks_count']}")
    print(f"Oversized chunks:                     {validation['oversized_chunks_count']}")
    print(f"Missing metadata:                     {validation['missing_metadata_count']}")
    print(f"Ready for indexing:                   {validation['ready_for_indexing']}")
    print("-" * 80)
    print(f"Chunks saved to:                      {CHUNKS_OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Report saved to:                      {REPORT_OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()