from pathlib import Path
import json
from collections import Counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOG_PATH = PROJECT_ROOT / "logs" / "rag_queries.jsonl"
REPORT_PATH = PROJECT_ROOT / "data" / "evaluation" / "logging_check_report.json"


def load_logs() -> list[dict[str, Any]]:
    if not LOG_PATH.exists():
        return []

    records = []

    for line_number, line in enumerate(LOG_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()

        if not line:
            continue

        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            records.append(
                {
                    "invalid_json": True,
                    "line_number": line_number,
                    "raw_line": line,
                }
            )

    return records


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    logs = load_logs()

    if not logs:
        print("No logs found.")
        print("Run FastAPI and ask at least one question through /chat or Streamlit.")
        return

    valid_logs = [
        record
        for record in logs
        if not record.get("invalid_json")
    ]

    invalid_logs = [
        record
        for record in logs
        if record.get("invalid_json")
    ]

    endpoint_counts = Counter(
        record.get("endpoint", "unknown")
        for record in valid_logs
    )

    latency_values = [
        float(record["latency_ms"])
        for record in valid_logs
        if "latency_ms" in record
    ]

    average_latency = (
        sum(latency_values) / len(latency_values)
        if latency_values
        else 0
    )

    records_missing_required_fields = []

    for record in valid_logs:
        missing_fields = []

        for field in ["query", "retrieved_chunk_ids", "latency_ms"]:
            if field not in record:
                missing_fields.append(field)

        if missing_fields:
            records_missing_required_fields.append(
                {
                    "timestamp_utc": record.get("timestamp_utc"),
                    "endpoint": record.get("endpoint"),
                    "missing_fields": missing_fields,
                }
            )

    latest_logs = valid_logs[-10:]

    report = {
        "log_file": str(LOG_PATH.relative_to(PROJECT_ROOT)),
        "total_log_lines": len(logs),
        "valid_log_records": len(valid_logs),
        "invalid_log_records": len(invalid_logs),
        "endpoint_counts": dict(endpoint_counts),
        "average_latency_ms": round(average_latency, 2),
        "records_missing_required_fields": records_missing_required_fields,
        "logging_requirement_passed": len(records_missing_required_fields) == 0 and len(valid_logs) > 0,
        "latest_logs_sample": latest_logs,
    }

    save_json(REPORT_PATH, report)

    print("Logging inspection completed")
    print("=" * 80)
    print(f"Log file:                     {LOG_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Total log lines:              {len(logs)}")
    print(f"Valid log records:            {len(valid_logs)}")
    print(f"Invalid log records:          {len(invalid_logs)}")
    print(f"Endpoint counts:              {dict(endpoint_counts)}")
    print(f"Average latency ms:           {round(average_latency, 2)}")
    print(f"Missing required field cases: {len(records_missing_required_fields)}")
    print(f"Requirement passed:           {report['logging_requirement_passed']}")
    print(f"Report saved to:              {REPORT_PATH.relative_to(PROJECT_ROOT)}")

    print("\nLatest logs:")
    print("-" * 80)

    for record in latest_logs:
        print(f"Endpoint: {record.get('endpoint')}")
        print(f"Query: {record.get('query')}")
        print(f"Latency: {record.get('latency_ms')} ms")
        print(f"Retrieved chunks: {record.get('retrieved_chunk_ids')}")
        print("-" * 80)


if __name__ == "__main__":
    main()