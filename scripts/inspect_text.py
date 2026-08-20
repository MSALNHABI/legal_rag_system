from pathlib import Path
import json
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEXT_DIR = PROJECT_ROOT / "data" / "processed" / "text"
REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "text_inspection_report.json"


ARABIC_ARTICLE_PATTERNS = [
    r"^المادة\s+.+",
    r"^مادة\s+.+",
    r"^\(?\s*المادة\s+.+\)?$",
]

ENGLISH_ARTICLE_PATTERNS = [
    r"^Article\s+\d+.*",
    r"^ARTICLE\s+\d+.*",
    r"^Art\.\s*\d+.*",
]


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def get_basic_stats(text: str) -> dict:
    lines = text.splitlines()
    non_empty_lines = [line for line in lines if line.strip()]

    return {
        "characters": len(text),
        "total_lines": len(lines),
        "non_empty_lines": len(non_empty_lines),
        "average_line_length": round(
            sum(len(line) for line in non_empty_lines) / max(len(non_empty_lines), 1),
            2,
        ),
    }


def find_matches(text: str, patterns: list[str], max_results: int = 40) -> list[str]:
    matches = []

    for line in text.splitlines():
        clean_line = line.strip()

        if not clean_line:
            continue

        for pattern in patterns:
            if re.match(pattern, clean_line, flags=re.IGNORECASE):
                matches.append(clean_line)
                break

        if len(matches) >= max_results:
            break

    return matches


def get_first_lines(text: str, count: int = 40) -> list[str]:
    lines = []

    for line in text.splitlines():
        clean_line = line.strip()
        if clean_line:
            lines.append(clean_line)

        if len(lines) >= count:
            break

    return lines


def detect_language_hint(filename: str, text: str) -> str:
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    english_chars = len(re.findall(r"[A-Za-z]", text))

    if "_ar" in filename.lower() or arabic_chars > english_chars:
        return "arabic"

    if "_en" in filename.lower() or english_chars > arabic_chars:
        return "english"

    return "unknown"


def inspect_file(path: Path) -> dict:
    text = read_text_file(path)
    language_hint = detect_language_hint(path.name, text)

    arabic_article_matches = find_matches(text, ARABIC_ARTICLE_PATTERNS)
    english_article_matches = find_matches(text, ENGLISH_ARTICLE_PATTERNS)

    return {
        "file": str(path.relative_to(PROJECT_ROOT)),
        "language_hint": language_hint,
        "stats": get_basic_stats(text),
        "first_lines": get_first_lines(text),
        "arabic_article_heading_examples": arabic_article_matches,
        "english_article_heading_examples": english_article_matches,
        "arabic_article_heading_count_sample": len(arabic_article_matches),
        "english_article_heading_count_sample": len(english_article_matches),
    }


def main() -> None:
    if not TEXT_DIR.exists():
        print(f"Text folder does not exist: {TEXT_DIR}")
        print("Run scripts/extract_text.py first.")
        return

    text_files = sorted(TEXT_DIR.glob("*.txt"))

    if not text_files:
        print(f"No TXT files found in: {TEXT_DIR}")
        print("Run scripts/extract_text.py first.")
        return

    report = []

    print("Inspecting extracted text files...")
    print("=" * 80)

    for path in text_files:
        result = inspect_file(path)
        report.append(result)

        print(f"File: {result['file']}")
        print(f"Language hint: {result['language_hint']}")
        print(f"Characters: {result['stats']['characters']:,}")
        print(f"Total lines: {result['stats']['total_lines']:,}")
        print(f"Non-empty lines: {result['stats']['non_empty_lines']:,}")

        print("\nFirst lines:")
        for line in result["first_lines"][:10]:
            print(f"  {line}")

        print("\nArabic article heading examples:")
        if result["arabic_article_heading_examples"]:
            for line in result["arabic_article_heading_examples"][:10]:
                print(f"  {line}")
        else:
            print("  None found")

        print("\nEnglish article heading examples:")
        if result["english_article_heading_examples"]:
            for line in result["english_article_heading_examples"][:10]:
                print(f"  {line}")
        else:
            print("  None found")

        print("-" * 80)

    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("Inspection completed")
    print(f"Report saved to: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()