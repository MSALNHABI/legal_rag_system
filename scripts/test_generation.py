from pathlib import Path
import sys
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app.retriever import LegalRetriever
from app.generator import LegalAnswerGenerator


OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "generation_test_results.txt"


TEST_QUERIES = [
    "متى يستحق العامل مكافأة نهاية الخدمة؟",
    "متى يجوز لصاحب العمل فسخ العقد دون مكافأة أو تعويض؟",
    "ما المقصود بإصابة العمل؟",
    "كم نسبة الاشتراك في فرع المعاشات؟",
    "ما هي ضريبة الشركات في السعودية؟",
]


def format_retrieved_chunks(retrieved_chunks: list[dict]) -> str:
    lines = []

    for chunk in retrieved_chunks:
        lines.append(
            f"Rank {chunk['rank']} | "
            f"chunk_id={chunk['chunk_id']} | "
            f"law={chunk['law_id']} | "
            f"article={chunk['article_number']} | "
            f"method={chunk['retrieval_method']}"
        )

    return "\n".join(lines)


def format_citations(citations: list[dict]) -> str:
    if not citations:
        return "لا توجد استشهادات."

    lines = []

    for citation in citations:
        lines.append(
            f"[{citation['label']}] "
            f"{citation['law_name_ar']}، "
            f"المادة {citation['article_number']}، "
            f"chunk_id={citation['chunk_id']}"
        )

    return "\n".join(lines)


def main() -> None:
    retriever = LegalRetriever()
    generator = LegalAnswerGenerator()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report_sections = []

    header = f"""
اختبار توليد الإجابات القانونية
Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Total queries: {len(TEST_QUERIES)}
Retrieval top_k: 10
{"=" * 100}
""".strip()

    report_sections.append(header)

    for index, query in enumerate(TEST_QUERIES, start=1):
        retrieved_chunks = retriever.hybrid_search(
            query=query,
            top_k=10,
            semantic_k=20,
            keyword_k=20,
        )

        result = generator.generate_answer(
            question=query,
            retrieved_chunks=retrieved_chunks,
        )

        section = f"""
{"=" * 100}
Test Case {index}
{"=" * 100}

Question:
{query}

{"-" * 100}
Answer:
{result["answer"]}

{"-" * 100}
Citations:
{format_citations(result["citations"])}

{"-" * 100}
Retrieved Chunks:
{format_retrieved_chunks(retrieved_chunks)}

{"-" * 100}
Grounded: {result["grounded"]}
Model: {result["model"]}
""".strip()

        report_sections.append(section)

    final_report = "\n\n".join(report_sections)

    OUTPUT_PATH.write_text(final_report, encoding="utf-8-sig")

    print("Generation test completed.")
    print(f"Results saved to: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()