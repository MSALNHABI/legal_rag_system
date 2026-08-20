from pathlib import Path
import json
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARTICLES_DIR = PROJECT_ROOT / "data" / "processed" / "articles"
REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "parsed_articles_quality_report.json"


ARTICLE_FILES = [
    "labor_law_ar_articles_raw.json",
    "labor_law_en_articles_raw.json",
    "social_insurance_law_ar_articles_raw.json",
    "social_insurance_law_en_articles_raw.json",
]


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def group_by_article_number(articles: list[dict]) -> dict[int, list[dict]]:
    grouped = defaultdict(list)

    for article in articles:
        grouped[int(article["article_number"])].append(article)

    return dict(grouped)


def get_short_body_articles(articles: list[dict], min_chars: int = 40) -> list[dict]:
    short_articles = []

    for article in articles:
        body = article.get("body", "")
        if len(body.strip()) < min_chars:
            short_articles.append(
                {
                    "article_id": article["article_id"],
                    "article_number": article["article_number"],
                    "article_heading": article["article_heading"],
                    "body_characters": len(body.strip()),
                    "body_preview": body[:120],
                }
            )

    return short_articles


def inspect_article_file(file_name: str) -> dict:
    path = ARTICLES_DIR / file_name

    if not path.exists():
        raise FileNotFoundError(f"Missing article file: {path}")

    articles = load_json(path)
    grouped = group_by_article_number(articles)

    duplicate_groups = {
        article_number: versions
        for article_number, versions in grouped.items()
        if len(versions) > 1
    }

    amended_articles = [
        article
        for article in articles
        if article.get("is_amendment_version") is True
    ]

    short_body_articles = get_short_body_articles(articles)

    article_numbers = sorted(grouped.keys())

    missing_number_gaps = []
    if article_numbers:
        expected_numbers = set(range(min(article_numbers), max(article_numbers) + 1))
        actual_numbers = set(article_numbers)
        missing_number_gaps = sorted(expected_numbers - actual_numbers)

    duplicate_summary = []

    for article_number, versions in sorted(duplicate_groups.items()):
        duplicate_summary.append(
            {
                "article_number": article_number,
                "versions_count": len(versions),
                "headings": [version["article_heading"] for version in versions],
                "version_ids": [version["article_id"] for version in versions],
                "body_lengths": [version.get("body_characters", 0) for version in versions],
            }
        )

    return {
        "file": str(path.relative_to(PROJECT_ROOT)),
        "law_id": articles[0]["law_id"] if articles else None,
        "language": articles[0]["source_language"] if articles else None,
        "total_records": len(articles),
        "unique_article_numbers": len(grouped),
        "min_article_number": min(article_numbers) if article_numbers else None,
        "max_article_number": max(article_numbers) if article_numbers else None,
        "duplicate_article_number_groups": len(duplicate_groups),
        "amendment_versions": len(amended_articles),
        "short_body_articles_count": len(short_body_articles),
        "missing_number_gaps_count": len(missing_number_gaps),
        "missing_number_gaps": missing_number_gaps[:50],
        "duplicate_summary_sample": duplicate_summary[:30],
        "short_body_articles_sample": short_body_articles[:30],
    }


def main() -> None:
    reports = []

    print("Inspecting parsed article JSON files...")
    print("=" * 80)

    for file_name in ARTICLE_FILES:
        try:
            report = inspect_article_file(file_name)
            reports.append({**report, "status": "success"})

            print(f"File: {report['file']}")
            print(f"Law: {report['law_id']} | Language: {report['language']}")
            print(f"Total records:             {report['total_records']}")
            print(f"Unique article numbers:    {report['unique_article_numbers']}")
            print(f"Article range:             {report['min_article_number']} → {report['max_article_number']}")
            print(f"Duplicate article groups:  {report['duplicate_article_number_groups']}")
            print(f"Amendment versions:        {report['amendment_versions']}")
            print(f"Short body articles:       {report['short_body_articles_count']}")
            print(f"Missing number gaps:       {report['missing_number_gaps_count']}")

            if report["duplicate_summary_sample"]:
                print("\nDuplicate / version groups sample:")
                for item in report["duplicate_summary_sample"][:10]:
                    print(
                        f"  Article {item['article_number']} | "
                        f"versions={item['versions_count']} | "
                        f"headings={item['headings']}"
                    )

            if report["short_body_articles_sample"]:
                print("\nShort body articles sample:")
                for item in report["short_body_articles_sample"][:10]:
                    print(
                        f"  Article {item['article_number']} | "
                        f"{item['article_heading']} | "
                        f"chars={item['body_characters']}"
                    )

            if report["missing_number_gaps"]:
                print("\nMissing article numbers sample:")
                print(f"  {report['missing_number_gaps'][:20]}")

            print("-" * 80)

        except Exception as error:
            reports.append(
                {
                    "file": file_name,
                    "status": "failed",
                    "error": str(error),
                }
            )

            print(f"Failed: {file_name}")
            print(f"Error: {error}")
            print("-" * 80)

    REPORT_PATH.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("Parsed article inspection completed")
    print(f"Report saved to: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()