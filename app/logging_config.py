from pathlib import Path
from datetime import datetime, timezone
import json
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs"
QUERY_LOG_PATH = LOGS_DIR / "rag_queries.jsonl"


def write_query_log(event: dict[str, Any]) -> None:
    """
    Write query/retrieval/generation logs as JSONL.

    This supports the assignment requirement:
    - Log query
    - Log retrieved chunks
    - Log latency
    """

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **event,
    }

    with QUERY_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(event, ensure_ascii=False, default=str) + "\n"
        )