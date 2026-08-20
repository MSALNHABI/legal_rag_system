from pathlib import Path
import json
import re
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEXT_DIR = PROJECT_ROOT / "data" / "processed" / "text"
ARTICLES_DIR = PROJECT_ROOT / "data" / "processed" / "articles"
REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "article_parsing_report.json"


LAW_CONFIGS = [
    {
        "input_file": "labor_law.txt",
        "law_id": "labor_law",
        "law_name_ar": "نظام العمل",
        "law_name_en": "Labor Law",
        "language": "ar",
    },
    {
        "input_file": "social_insurance_law.txt",
        "law_id": "social_insurance_law",
        "law_name_ar": "نظام التأمينات الاجتماعية",
        "law_name_en": "Social Insurance Law",
        "language": "ar",
    },
    {
        "input_file": "labor_law_en.txt",
        "law_id": "labor_law",
        "law_name_ar": "نظام العمل",
        "law_name_en": "Labor Law",
        "language": "en",
    },
    {
        "input_file": "social_insurance_law_en.txt",
        "law_id": "social_insurance_law",
        "law_name_ar": "نظام التأمينات الاجتماعية",
        "law_name_en": "Social Insurance Law",
        "language": "en",
    },
]


def normalize_spaces(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_trailing_colon(text: str) -> str:
    return re.sub(r"\s*[:：]\s*$", "", text).strip()


def normalize_arabic_heading_base(heading: str) -> str:
    """
    Convert:
        المادة الثانية(التعديل الاول):
    into:
        المادة الثانية

    Purpose:
    - The amendment versions should belong to the same base article.
    """

    heading = normalize_spaces(heading)
    heading = remove_trailing_colon(heading)

    heading = re.sub(
        r"\s*\(?\s*التعديل\s+(الأول|الاول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر)\s*\)?",
        "",
        heading,
        flags=re.IGNORECASE,
    )

    heading = normalize_spaces(heading)
    heading = remove_trailing_colon(heading)

    return heading


def extract_arabic_amendment_label(heading: str) -> Optional[str]:
    match = re.search(
        r"التعديل\s+(الأول|الاول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر)",
        heading,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return normalize_spaces(match.group(0))


def detect_arabic_article_heading(line: str) -> Optional[dict]:
    """
    Detect Arabic article headings.

    Examples:
        المادة الأولى :
        المادة الثانية
        المادة الثانية(التعديل الاول)
        المادة الخامسة(التعديل الاول):
    """

    line = normalize_spaces(line)

    if not line:
        return None

    # Avoid treating long body sentences as headings.
    if len(line) > 140:
        return None

    match = re.match(r"^المادة\s+(.+?)\s*[:：]?$", line)

    if not match:
        return None

    heading = remove_trailing_colon(line)
    base_heading = normalize_arabic_heading_base(heading)
    amendment_label = extract_arabic_amendment_label(heading)

    return {
        "article_heading": heading,
        "article_heading_base": base_heading,
        "amendment_label": amendment_label,
        "inline_body": "",
    }


def detect_english_article_heading(line: str) -> Optional[dict]:
    """
    Detect English article headings.

    Examples:
        Article 1
        Article 2
        Article 15: Something
    """

    line = normalize_spaces(line)

    if not line:
        return None

    # Ignore PDF page markers.
    if re.match(r"^\[Page\s+\d+\]$", line, flags=re.IGNORECASE):
        return None

    if len(line) > 140:
        return None

    match = re.match(
        r"^Article\s+(\d+)\s*[:：\-–—.]?\s*(.*)$",
        line,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    article_number = int(match.group(1))
    inline_body = normalize_spaces(match.group(2))

    heading = f"Article {article_number}"

    return {
        "article_heading": heading,
        "article_heading_base": heading,
        "article_number": article_number,
        "amendment_label": None,
        "inline_body": inline_body,
    }


def detect_article_heading(line: str, language: str) -> Optional[dict]:
    if language == "ar":
        return detect_arabic_article_heading(line)

    if language == "en":
        return detect_english_article_heading(line)

    raise ValueError(f"Unsupported language: {language}")


def make_output_filename(law_id: str, language: str) -> str:
    return f"{law_id}_{language}_articles_raw.json"


def finalize_article(
    current_article: Optional[dict],
    body_lines: list[str],
    articles: list[dict],
) -> None:
    if current_article is None:
        return

    body = normalize_spaces("\n".join(body_lines))

    current_article["body"] = body
    current_article["body_characters"] = len(body)
    current_article["body_lines"] = len([line for line in body.splitlines() if line.strip()])
    current_article["has_body"] = bool(body)

    articles.append(current_article)


def parse_file(config: dict) -> tuple[list[dict], dict]:
    input_path = TEXT_DIR / config["input_file"]

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input text file: {input_path}")

    text = input_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    law_id = config["law_id"]
    language = config["language"]

    articles: list[dict] = []
    current_article: Optional[dict] = None
    body_lines: list[str] = []

    # Arabic headings do not contain numeric article numbers.
    # We assign numbers deterministically based on first appearance of each base heading.
    arabic_base_heading_to_number: dict[str, int] = {}

    # Count versions for each article number.
    version_counter: dict[int, int] = {}

    preamble_lines = []

    for line in lines:
        clean_line = normalize_spaces(line)

        if not clean_line:
            continue

        heading_info = detect_article_heading(clean_line, language)

        if heading_info:
            finalize_article(current_article, body_lines, articles)
            body_lines = []

            if language == "ar":
                base_heading = heading_info["article_heading_base"]

                if base_heading not in arabic_base_heading_to_number:
                    arabic_base_heading_to_number[base_heading] = len(arabic_base_heading_to_number) + 1

                article_number = arabic_base_heading_to_number[base_heading]

            else:
                article_number = heading_info["article_number"]

            version_counter[article_number] = version_counter.get(article_number, 0) + 1
            version_number = version_counter[article_number]

            article_id = (
                f"{law_id}__{language}__article_{article_number:03d}"
                f"__version_{version_number:02d}"
            )

            current_article = {
                "article_id": article_id,
                "law_id": law_id,
                "law_name_ar": config["law_name_ar"],
                "law_name_en": config["law_name_en"],
                "source_language": language,
                "source_file": str(input_path.relative_to(PROJECT_ROOT)),
                "article_number": article_number,
                "article_heading": heading_info["article_heading"],
                "article_heading_base": heading_info["article_heading_base"],
                "amendment_label": heading_info.get("amendment_label"),
                "version_number": version_number,
                "is_amendment_version": heading_info.get("amendment_label") is not None,
            }

            inline_body = heading_info.get("inline_body", "")
            if inline_body:
                body_lines.append(inline_body)

        else:
            if current_article is None:
                preamble_lines.append(clean_line)
            else:
                body_lines.append(clean_line)

    finalize_article(current_article, body_lines, articles)

    empty_body_count = sum(1 for article in articles if not article["has_body"])
    amendment_count = sum(1 for article in articles if article["is_amendment_version"])

    report = {
        "input_file": str(input_path.relative_to(PROJECT_ROOT)),
        "law_id": law_id,
        "language": language,
        "total_articles_parsed": len(articles),
        "empty_body_articles": empty_body_count,
        "amendment_versions": amendment_count,
        "preamble_lines_before_first_article": len(preamble_lines),
        "first_article": articles[0]["article_heading"] if articles else None,
        "last_article": articles[-1]["article_heading"] if articles else None,
        "output_file": str((ARTICLES_DIR / make_output_filename(law_id, language)).relative_to(PROJECT_ROOT)),
    }

    return articles, report


def main() -> None:
    ARTICLES_DIR.mkdir(parents=True, exist_ok=True)

    all_articles = []
    reports = []

    print("Parsing text files into article-level JSON...")
    print("=" * 80)

    for config in LAW_CONFIGS:
        try:
            articles, report = parse_file(config)

            output_path = ARTICLES_DIR / make_output_filename(
                config["law_id"],
                config["language"],
            )

            output_path.write_text(
                json.dumps(articles, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            all_articles.extend(articles)
            reports.append({**report, "status": "success"})

            print(f"Parsed: {report['input_file']}")
            print(f"Law:    {report['law_id']} | Language: {report['language']}")
            print(f"Articles parsed:      {report['total_articles_parsed']}")
            print(f"Amendment versions:   {report['amendment_versions']}")
            print(f"Empty body articles:  {report['empty_body_articles']}")
            print(f"First article:        {report['first_article']}")
            print(f"Last article:         {report['last_article']}")
            print(f"Output:               {report['output_file']}")
            print("-" * 80)

        except Exception as error:
            error_report = {
                "input_file": config["input_file"],
                "law_id": config["law_id"],
                "language": config["language"],
                "status": "failed",
                "error": str(error),
            }

            reports.append(error_report)

            print(f"Failed: {config['input_file']}")
            print(f"Error:  {error}")
            print("-" * 80)

    combined_output_path = ARTICLES_DIR / "all_articles_raw.json"
    combined_output_path.write_text(
        json.dumps(all_articles, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    REPORT_PATH.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    successful = sum(1 for report in reports if report["status"] == "success")
    failed = sum(1 for report in reports if report["status"] == "failed")

    print("=" * 80)
    print("Article parsing completed")
    print(f"Successful files: {successful}")
    print(f"Failed files:     {failed}")
    print(f"Total articles:   {len(all_articles)}")
    print(f"Combined output:  {combined_output_path.relative_to(PROJECT_ROOT)}")
    print(f"Report saved to:  {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()