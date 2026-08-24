"""Run the stable cross-topic RAG regression set against local FastAPI.

This script deliberately keeps expected facts in an eval fixture, never in
the runtime pipeline.  It reports lexical concept coverage as a smoke signal;
an administrator still owns the historical review verdict.
"""

from __future__ import annotations

import argparse
import json
import os
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "evals" / "history_rag_cases.json"


def _load_env(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key, value)


def _key(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.casefold())
    return " ".join(
        "".join(char for char in normalized if unicodedata.category(char) != "Mn")
        .replace("đ", "d")
        .split()
    )


def _ask(base_url: str, admin_key: str, question: str, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/admin/retrieval/debug",
        data=json.dumps({"question": question, "include_citations": True}).encode(),
        headers={"Content-Type": "application/json", "X-Admin-Key": admin_key},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--show-answer",
        action="store_true",
        help="Print the generated answer after each compact result.",
    )
    parser.add_argument(
        "--show-diagnostics",
        action="store_true",
        help="Print retrieval requirements, mappings and citation excerpts.",
    )
    args = parser.parse_args()
    _load_env(ROOT / ".env")
    admin_key = os.environ.get("AI_ADMIN_API_KEY", "")
    if not admin_key:
        raise SystemExit("AI_ADMIN_API_KEY is missing")
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[: args.limit]

    failures = 0
    for case in cases:
        result = _ask(args.base_url, admin_key, case["question"], args.timeout)
        answer_key = _key(result.get("answer") or "")
        concept_checks = [
            any(_key(option) in answer_key for option in alternatives)
            for alternatives in case.get("requiredConcepts", [])
        ]
        missing_concepts = [
            alternatives
            for alternatives, passed_concept in zip(
                case.get("requiredConcepts", []), concept_checks, strict=True
            )
            if not passed_concept
        ]
        requirement_count = len(
            (result.get("retrieval") or {}).get("semantic_requirements") or {}
        )
        passed = all(concept_checks) and requirement_count >= int(
            case.get("minimumRequirements") or 1
        )
        failures += int(not passed)
        print(json.dumps({
            "id": case["id"],
            "pass": passed,
            "concepts": f"{sum(concept_checks)}/{len(concept_checks)}",
            "missingConcepts": missing_concepts,
            "requirements": requirement_count,
            "confidence": round(float(result.get("confidence") or 0), 3),
            "verification": (result.get("retrieval") or {}).get(
                "claim_verification_status"
            ),
        }, ensure_ascii=False))
        if args.show_answer:
            print(result.get("answer") or "")
        if args.show_diagnostics:
            retrieval = result.get("retrieval") or {}
            print(json.dumps({
                "requirements": retrieval.get("semantic_requirements"),
                "requirementEvidence": retrieval.get("requirement_evidence"),
                "missing": retrieval.get("missing_facets"),
                "limitations": retrieval.get("coverage_gate_limitations"),
                "citations": [
                    {
                        "id": item.get("chunk_id"),
                        "title": item.get("source_title"),
                        "excerpt": item.get("excerpt"),
                    }
                    for item in result.get("citations") or []
                ],
            }, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
