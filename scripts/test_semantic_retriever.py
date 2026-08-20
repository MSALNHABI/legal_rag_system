from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app.retriever import LegalSemanticRetriever


TEST_QUERIES = [
    "متى يستحق العامل مكافأة نهاية الخدمة؟",
    "ما هي حالات إصابة العمل؟",
    "متى يجوز لصاحب العمل إنهاء العقد؟",
    "ما المقصود بالأجر في نظام العمل؟",
    "ما هي اشتراكات التأمينات الاجتماعية؟",
]


def print_results(query: str, results: list[dict]) -> None:
    print("=" * 100)
    print(f"Query: {query}")
    print("-" * 100)

    for item in results:
        preview = item["text"].replace("\n", " ")
        preview = preview[:250] + "..." if len(preview) > 250 else preview

        print(
            f"Rank {item['rank']} | "
            f"score={item['score']:.4f} | "
            f"chunk_id={item['chunk_id']} | "
            f"law={item['law_id']} | "
            f"article={item['article_number']}"
        )
        print(f"Preview: {preview}")
        print("-" * 100)


def main() -> None:
    retriever = LegalSemanticRetriever()

    for query in TEST_QUERIES:
        results = retriever.semantic_search(query=query, top_k=5)
        print_results(query, results)


if __name__ == "__main__":
    main()