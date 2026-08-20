from pathlib import Path
import json
import re
from collections import defaultdict
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARTICLES_DIR = PROJECT_ROOT / "data" / "processed" / "articles"

INPUT_FILES = [
    ARTICLES_DIR / "labor_law_ar_articles_raw.json",
    ARTICLES_DIR / "social_insurance_law_ar_articles_raw.json",
]

CANONICAL_OUTPUT_PATH = ARTICLES_DIR / "canonical_articles.json"
REPORT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "canonical_resolution_report.json"


def load_json(path: Path) -> list[dict[str, Any]]:
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


def normalize_arabic_for_detection(text: str) -> str:
    """
    Normalize Arabic text for legal status detection.

    Purpose:
    - Remove tashkeel so words like أُلغيت become الغيت.
    - Remove tatweel.
    - Normalize common Arabic letter forms.
    """

    text = normalize_text(text)

    # Remove Arabic tashkeel.
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", text)

    # Remove tatweel.
    text = text.replace("ـ", "")

    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ى": "ي",
        "ة": "ه",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def is_probably_repealed(text: str) -> bool:
    """
    Detect articles that should not be indexed.

    Important:
    - Some articles mention that they were amended:
      "عدلت هذه المادة لتكون بالنص الآتي"
      These must stay active.
    - Articles that only say they were deleted or repealed should be excluded.
    """

    clean = normalize_arabic_for_detection(text)
    beginning = clean[:250]

    active_amendment_signals = [
        "عدلت هذه الماده لتكون بالنص الاتي",
        "وعدلت هذه الماده لتكون بالنص الاتي",
        "ليكون بالنص الاتي",
        "لتكون بالنص الاتي",
        "لتصبح بالنص الاتي",
    ]

    if any(signal in clean for signal in active_amendment_signals):
        return False

    direct_repeal_signals = [
        "حذفت هذه الماده",
        "الغيت هذه الماده",
        "تلغي هذه الماده",
        "يلغي هذه الماده",
        "هذه الماده ملغاه",
        "هذه الماده محذوفه",
    ]

    if any(signal in beginning for signal in direct_repeal_signals):
        return True

    repeal_patterns = [
        r"\bملغاه\b",
        r"\bملغي\b",
        r"\bالغيت\b",
        r"\bتلغي\b",
        r"\bحذفت\b",
        r"\bمحذوفه\b",
        r"\bمحذوف\b",
    ]

    has_repeal_signal = any(
        re.search(pattern, clean)
        for pattern in repeal_patterns
    )

    if not has_repeal_signal:
        return False

    # Only mark short articles as repealed if they mainly contain deletion/repeal wording.
    if len(clean) <= 300:
        return True

    return False


def group_articles_by_law_and_number(
    articles: list[dict[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped = defaultdict(list)

    for article in articles:
        law_id = article["law_id"]
        article_number = int(article["article_number"])
        grouped[(law_id, article_number)].append(article)

    return dict(grouped)


def choose_current_version(versions: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Choose the latest version of an article.

    The parser assigned version_number based on document order:
    version_01 = original article
    version_02 = first amendment
    version_03 = second amendment

    Therefore, the highest version_number is treated as the current version.
    """

    return sorted(
        versions,
        key=lambda item: int(item.get("version_number", 1)),
    )[-1]


def build_canonical_article(
    selected: dict[str, Any],
    versions: list[dict[str, Any]],
) -> dict[str, Any]:
    law_id = selected["law_id"]
    article_number = int(selected["article_number"])
    body = normalize_text(selected.get("body", ""))

    repealed = is_probably_repealed(body)

    canonical_id = f"{law_id}__article_{article_number:03d}"

    version_ids = [item["article_id"] for item in versions]
    version_headings = [item["article_heading"] for item in versions]

    return {
        "canonical_id": canonical_id,
        "law_id": law_id,
        "law_name_ar": selected["law_name_ar"],
        "law_name_en": selected["law_name_en"],
        "article_number": article_number,
        "article_heading_ar": selected["article_heading_base"],
        "current_article_heading_ar": selected["article_heading"],
        "text_ar": body,
        "text_characters": len(body),
        "status": "repealed" if repealed else "active",
        "indexable": not repealed and bool(body),
        "source_language": "ar",
        "source_file": selected["source_file"],
        "selected_raw_article_id": selected["article_id"],
        "selected_version_number": int(selected.get("version_number", 1)),
        "selected_amendment_label": selected.get("amendment_label"),
        "total_versions_found": len(versions),
        "all_version_ids": version_ids,
        "all_version_headings": version_headings,
        "selection_strategy": "highest_version_number_from_arabic_source",
    }


def main() -> None:
    all_raw_articles: list[dict[str, Any]] = []

    print("Resolving Arabic articles into canonical current corpus...")
    print("=" * 80)

    for input_file in INPUT_FILES:
        if not input_file.exists():
            raise FileNotFoundError(f"Missing input file: {input_file}")

        articles = load_json(input_file)
        all_raw_articles.extend(articles)

        print(f"Loaded: {input_file.relative_to(PROJECT_ROOT)}")
        print(f"Records: {len(articles)}")
        print("-" * 80)

    grouped = group_articles_by_law_and_number(all_raw_articles)

    canonical_articles = []
    resolution_details = []

    for (law_id, article_number), versions in sorted(grouped.items()):
        selected = choose_current_version(versions)
        canonical = build_canonical_article(selected, versions)

        canonical_articles.append(canonical)

        resolution_details.append(
            {
                "canonical_id": canonical["canonical_id"],
                "law_id": law_id,
                "article_number": article_number,
                "versions_found": len(versions),
                "selected_raw_article_id": selected["article_id"],
                "selected_heading": selected["article_heading"],
                "selected_version_number": selected.get("version_number"),
                "selected_amendment_label": selected.get("amendment_label"),
                "status": canonical["status"],
                "indexable": canonical["indexable"],
                "text_preview": canonical["text_ar"][:180],
            }
        )

    active_count = sum(1 for item in canonical_articles if item["status"] == "active")
    repealed_count = sum(1 for item in canonical_articles if item["status"] == "repealed")
    indexable_count = sum(1 for item in canonical_articles if item["indexable"])
    multi_version_count = sum(1 for item in canonical_articles if item["total_versions_found"] > 1)

    by_law = defaultdict(lambda: {"articles": 0, "active": 0, "repealed": 0, "indexable": 0})

    repealed_articles = []

    for article in canonical_articles:
        law = article["law_id"]
        by_law[law]["articles"] += 1

        if article["status"] == "active":
            by_law[law]["active"] += 1

        if article["status"] == "repealed":
            by_law[law]["repealed"] += 1
            repealed_articles.append(
                {
                    "canonical_id": article["canonical_id"],
                    "law_id": article["law_id"],
                    "article_number": article["article_number"],
                    "preview": article["text_ar"][:180],
                }
            )

        if article["indexable"]:
            by_law[law]["indexable"] += 1

    report = {
        "input_files": [str(path.relative_to(PROJECT_ROOT)) for path in INPUT_FILES],
        "output_file": str(CANONICAL_OUTPUT_PATH.relative_to(PROJECT_ROOT)),
        "total_raw_records_loaded": len(all_raw_articles),
        "canonical_articles": len(canonical_articles),
        "active_articles": active_count,
        "repealed_articles": repealed_count,
        "indexable_articles": indexable_count,
        "articles_with_multiple_versions": multi_version_count,
        "by_law": dict(by_law),
        "repealed_articles": repealed_articles,
        "resolution_details_sample": resolution_details[:50],
        "all_resolution_details": resolution_details,
    }

    save_json(CANONICAL_OUTPUT_PATH, canonical_articles)
    save_json(REPORT_OUTPUT_PATH, report)

    print("=" * 80)
    print("Canonical corpus created")
    print(f"Raw records loaded:              {len(all_raw_articles)}")
    print(f"Canonical articles:              {len(canonical_articles)}")
    print(f"Articles with multiple versions: {multi_version_count}")
    print(f"Active articles:                 {active_count}")
    print(f"Repealed articles:               {repealed_count}")
    print(f"Indexable articles:              {indexable_count}")

    print("\nBy law:")
    for law_id, stats in by_law.items():
        print(
            f"  {law_id}: "
            f"articles={stats['articles']}, "
            f"active={stats['active']}, "
            f"repealed={stats['repealed']}, "
            f"indexable={stats['indexable']}"
        )

    print("\nRepealed/deleted articles detected:")
    for article in repealed_articles[:40]:
        print(
            f"  {article['law_id']} article {article['article_number']}: "
            f"{article['preview']}"
        )

    if len(repealed_articles) > 40:
        print(f"  ... and {len(repealed_articles) - 40} more")

    print(f"\nOutput saved to: {CANONICAL_OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Report saved to: {REPORT_OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()