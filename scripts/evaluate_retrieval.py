from pathlib import Path
import json
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app.retriever import LegalRetriever


TEST_SET_PATH = PROJECT_ROOT / "data" / "evaluation" / "test_set.json"
RESULTS_OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_evaluation_results.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0

    top_k = set(retrieved_ids[:k])
    hits = len(top_k & set(relevant_ids))

    return hits / len(relevant_ids)


def first_relevant_rank(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> int | None:
    relevant_set = set(relevant_ids)

    for index, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_set:
            return index

    return None


def mrr_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    rank = first_relevant_rank(
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
        k=k,
    )

    if rank is None:
        return 0.0

    return 1.0 / rank


def top_1_accuracy(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> float:
    if not retrieved_ids or not relevant_ids:
        return 0.0

    return 1.0 if retrieved_ids[0] in set(relevant_ids) else 0.0


def evaluate_method(
    test_set: list[dict[str, Any]],
    method_name: str,
    retrieve_function,
    k: int = 10,
) -> dict[str, Any]:
    per_query_results = []

    recall_scores = []
    mrr_scores = []
    top_1_scores = []
    first_ranks = []

    for item in test_set:
        query_id = item["id"]
        query = item["query"]
        relevant_ids = item["relevant_chunk_ids"]

        retrieved_ids = retrieve_function(query, k)

        recall_score = recall_at_k(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            k=k,
        )

        mrr_score = mrr_at_k(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            k=k,
        )

        top_1_score = top_1_accuracy(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
        )

        first_rank = first_relevant_rank(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            k=k,
        )

        recall_scores.append(recall_score)
        mrr_scores.append(mrr_score)
        top_1_scores.append(top_1_score)

        if first_rank is not None:
            first_ranks.append(first_rank)

        missed_ids = [
            relevant_id
            for relevant_id in relevant_ids
            if relevant_id not in retrieved_ids[:k]
        ]

        hit_ids = [
            relevant_id
            for relevant_id in relevant_ids
            if relevant_id in retrieved_ids[:k]
        ]

        per_query_results.append(
            {
                "id": query_id,
                "query": query,
                "relevant_chunk_ids": relevant_ids,
                "retrieved_chunk_ids": retrieved_ids,
                "recall_at_10": round(recall_score, 4),
                "mrr_at_10": round(mrr_score, 4),
                "top_1_accuracy": round(top_1_score, 4),
                "first_relevant_rank": first_rank,
                "hit_ids": hit_ids,
                "missed_ids": missed_ids,
                "passed": recall_score > 0,
            }
        )

    query_count = max(len(test_set), 1)

    mean_recall = sum(recall_scores) / query_count
    mean_mrr = sum(mrr_scores) / query_count
    mean_top_1 = sum(top_1_scores) / query_count

    mean_first_relevant_rank = (
        sum(first_ranks) / len(first_ranks)
        if first_ranks
        else None
    )

    perfect_queries = sum(1 for score in recall_scores if score == 1.0)
    zero_recall_queries = sum(1 for score in recall_scores if score == 0.0)

    return {
        "method": method_name,
        "k": k,
        "queries_evaluated": len(test_set),
        "mean_recall_at_10": round(mean_recall, 4),
        "mean_mrr_at_10": round(mean_mrr, 4),
        "top_1_accuracy": round(mean_top_1, 4),
        "mean_first_relevant_rank": (
            round(mean_first_relevant_rank, 4)
            if mean_first_relevant_rank is not None
            else None
        ),
        "perfect_queries": perfect_queries,
        "zero_recall_queries": zero_recall_queries,
        "per_query_results": per_query_results,
    }


def print_summary(result: dict[str, Any]) -> None:
    print(f"Method: {result['method']}")
    print(f"Queries evaluated:          {result['queries_evaluated']}")
    print(f"Mean Recall@10:             {result['mean_recall_at_10']}")
    print(f"Mean MRR@10:                {result['mean_mrr_at_10']}")
    print(f"Top-1 Accuracy:             {result['top_1_accuracy']}")
    print(f"Mean First Relevant Rank:   {result['mean_first_relevant_rank']}")
    print(f"Perfect queries:            {result['perfect_queries']}")
    print(f"Zero recall:                {result['zero_recall_queries']}")

    failed = [
        item
        for item in result["per_query_results"]
        if item["recall_at_10"] == 0.0
    ]

    if failed:
        print("\nZero-recall queries:")
        for item in failed:
            print(f"  {item['id']} | {item['query']}")
            print(f"    Relevant:  {item['relevant_chunk_ids']}")
            print(f"    Retrieved: {item['retrieved_chunk_ids'][:10]}")

    print("-" * 80)


def main() -> None:
    if not TEST_SET_PATH.exists():
        raise FileNotFoundError(
            f"Missing test set: {TEST_SET_PATH}"
        )

    test_set = load_json(TEST_SET_PATH)

    if len(test_set) < 20:
        raise ValueError(
            f"Test set must contain at least 20 queries. Found: {len(test_set)}"
        )

    retriever = LegalRetriever()

    print("Evaluating retrieval methods...")
    print("=" * 80)

    semantic_result = evaluate_method(
        test_set=test_set,
        method_name="semantic",
        retrieve_function=retriever.retrieve_semantic_chunk_ids,
        k=10,
    )

    keyword_result = evaluate_method(
        test_set=test_set,
        method_name="keyword_bm25",
        retrieve_function=retriever.retrieve_keyword_chunk_ids,
        k=10,
    )

    hybrid_result = evaluate_method(
        test_set=test_set,
        method_name="hybrid_weighted_rrf",
        retrieve_function=retriever.retrieve_hybrid_chunk_ids,
        k=10,
    )

    methods = [
        semantic_result,
        keyword_result,
        hybrid_result,
    ]

    best_by_recall = max(methods, key=lambda item: item["mean_recall_at_10"])
    best_by_mrr = max(methods, key=lambda item: item["mean_mrr_at_10"])
    best_by_top_1 = max(methods, key=lambda item: item["top_1_accuracy"])

    all_results = {
        "test_set_file": str(TEST_SET_PATH.relative_to(PROJECT_ROOT)),
        "k": 10,
        "methods": methods,
        "summary": {
            "semantic_mean_recall_at_10": semantic_result["mean_recall_at_10"],
            "keyword_mean_recall_at_10": keyword_result["mean_recall_at_10"],
            "hybrid_mean_recall_at_10": hybrid_result["mean_recall_at_10"],
            "semantic_mrr_at_10": semantic_result["mean_mrr_at_10"],
            "keyword_mrr_at_10": keyword_result["mean_mrr_at_10"],
            "hybrid_mrr_at_10": hybrid_result["mean_mrr_at_10"],
            "semantic_top_1_accuracy": semantic_result["top_1_accuracy"],
            "keyword_top_1_accuracy": keyword_result["top_1_accuracy"],
            "hybrid_top_1_accuracy": hybrid_result["top_1_accuracy"],
            "best_method_by_recall": best_by_recall["method"],
            "best_method_by_mrr": best_by_mrr["method"],
            "best_method_by_top_1": best_by_top_1["method"],
        },
    }

    save_json(RESULTS_OUTPUT_PATH, all_results)

    print_summary(semantic_result)
    print_summary(keyword_result)
    print_summary(hybrid_result)

    print("=" * 80)
    print("Evaluation completed")
    print(f"Results saved to: {RESULTS_OUTPUT_PATH.relative_to(PROJECT_ROOT)}")

    print("\nSummary:")
    print(f"Semantic Recall@10:     {semantic_result['mean_recall_at_10']}")
    print(f"Keyword Recall@10:      {keyword_result['mean_recall_at_10']}")
    print(f"Hybrid Recall@10:       {hybrid_result['mean_recall_at_10']}")
    print(f"Semantic MRR@10:        {semantic_result['mean_mrr_at_10']}")
    print(f"Keyword MRR@10:         {keyword_result['mean_mrr_at_10']}")
    print(f"Hybrid MRR@10:          {hybrid_result['mean_mrr_at_10']}")
    print(f"Semantic Top-1 Accuracy:{semantic_result['top_1_accuracy']}")
    print(f"Keyword Top-1 Accuracy: {keyword_result['top_1_accuracy']}")
    print(f"Hybrid Top-1 Accuracy:  {hybrid_result['top_1_accuracy']}")
    print(f"Best by Recall:         {best_by_recall['method']}")
    print(f"Best by MRR:            {best_by_mrr['method']}")
    print(f"Best by Top-1:          {best_by_top_1['method']}")


if __name__ == "__main__":
    main()