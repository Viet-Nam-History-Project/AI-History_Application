"""Backfill retrieval facets on existing Neo4j AIChunk nodes without AI calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from src.api.neo4j_repository import Neo4jKnowledgeRepository


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repository = Neo4jKnowledgeRepository()
    try:
        repository.verify_connectivity()
        result = repository.backfill_chunk_facets(
            args.batch_size,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        repository.close()


if __name__ == "__main__":
    main()
