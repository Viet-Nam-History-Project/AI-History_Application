"""Read administrator-reviewed static history content as grounded evidence.

The mobile app consumes the same versioned JSON through Firebase Hosting.  A
chat request may therefore retrieve from this controlled corpus as well as
from PDF/Neo4j.  No historical answer is embedded in code: the manifest,
hashes and published records remain the source of truth.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen

from ...entity_resolution import normalize_search_key


LOGGER = logging.getLogger(__name__)

_STOP_WORDS = {
    "ai", "bao", "cac", "cho", "cua", "duoc", "gi", "hay", "khi",
    "la", "nao", "noi", "o", "ra", "tai", "the", "thoi", "vao",
    "va", "ve", "vay",
}
_IGNORED_FIELDS = {
    "images", "videos", "covermediaref", "latlon", "link", "status",
    "updated at", "publishedat", "sortorder", "slug", "id",
    "periodslug", "stageslug",
}
_FIELD_LABELS = {
    "title": "Sự kiện",
    "smalltitle": "Khái quát",
    "summary": "Tóm tắt",
    "description": "Mô tả",
    "startdate": "Thời gian bắt đầu",
    "enddate": "Thời gian kết thúc",
    "details": "Diễn biến",
    "warcause": "Nguyên nhân",
    "meaning": "Ý nghĩa",
    "forces": "Lực lượng",
    "result": "Kết quả",
    "warsummary": "Diễn biến",
    "diadiem": "Địa điểm",
    "content": "Nội dung",
    "detail": "Chi tiết",
    "object": "Đối tượng",
    "vn": "Phía Việt Nam",
    "usallies": "Phía đối phương",
}


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        token
        for token in normalize_search_key(value).split()
        if len(token) >= 2 and token not in _STOP_WORDS
    ))


def _year(value: object) -> int | None:
    match = re.search(r"(?<!\d)(1[0-9]{3}|20[0-9]{2})(?!\d)", str(value or ""))
    return int(match.group(1)) if match else None


def _display_date(value: object) -> str:
    raw = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return raw
    return f"{parsed.day}/{parsed.month}/{parsed.year}"


def _flatten(value: object, field: str = "") -> list[str]:
    normalized_field = normalize_search_key(field)
    if normalized_field in _IGNORED_FIELDS or value in (None, "", [], {}):
        return []
    label = _FIELD_LABELS.get(normalized_field, "")
    if isinstance(value, dict):
        lines: list[str] = []
        for key, child in value.items():
            lines.extend(_flatten(child, str(key)))
        return lines
    if isinstance(value, list):
        lines: list[str] = []
        for child in value:
            lines.extend(_flatten(child, field))
        return lines
    rendered = _display_date(value) if normalized_field.endswith("date") else str(value).strip()
    return [f"{label}: {rendered}" if label else rendered]


def _event_text(record: dict[str, Any]) -> str:
    ordered_fields = (
        "title", "smallTitle", "summary", "description", "startDate",
        "endDate", "details", "warCause", "meaning", "content", "object",
    )
    def compact(value: object) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip(" .;,:\n\t")

    def locations(value: object) -> list[str]:
        found: list[str] = []
        if isinstance(value, dict):
            place = value.get("diadiem")
            if isinstance(place, dict) and compact(place.get("content")):
                found.append(compact(place.get("content")))
            for child in value.values():
                found.extend(locations(child))
        elif isinstance(value, list):
            for child in value:
                found.extend(locations(child))
        return list(dict.fromkeys(found))

    # Preserve a compact, same-record evidence window.  The verifier works on
    # bounded sentences; emitting every JSON field as an isolated line would
    # separate the subject from its date/place even though all values belong
    # to the same reviewed event document.
    compact_parts = [
        compact(record.get("title")),
        compact(record.get("smallTitle")),
        compact(record.get("summary") or record.get("description")),
    ]
    event_locations = locations(record.get("content"))
    if event_locations:
        compact_parts.append("tại " + ", ".join(event_locations))
    if record.get("startDate"):
        compact_parts.append(
            "bắt đầu ngày " + _display_date(record["startDate"])
        )

    lines: list[str] = [
        "Bản ghi sự kiện: "
        + ", ".join(part for part in compact_parts if part)
        + "."
    ]
    for field in ordered_fields:
        lines.extend(_flatten(record.get(field), field))
    return "\n".join(dict.fromkeys(line for line in lines if line))


@dataclass(frozen=True, slots=True)
class _Snapshot:
    version: str
    loaded_at: float
    records: tuple[dict[str, Any], ...]


class PublishedContentRepository:
    """Search a verified Firebase Hosting manifest with a bounded TTL cache."""

    def __init__(
        self,
        *,
        enabled: bool,
        manifest_path: str = "",
        manifest_url: str = "",
        cache_ttl_seconds: int = 300,
    ) -> None:
        self.enabled = bool(enabled)
        self.manifest_path = Path(manifest_path).expanduser() if manifest_path else None
        self.manifest_url = manifest_url.strip()
        self.cache_ttl_seconds = max(0, int(cache_ttl_seconds))
        self._snapshot: _Snapshot | None = None
        self._lock = threading.Lock()

    def _read_bytes(self, location: str) -> bytes:
        parsed = urlparse(location)
        if parsed.scheme:
            if parsed.scheme != "https":
                raise ValueError("Published content URL must use HTTPS")
            with urlopen(location, timeout=6) as response:  # noqa: S310
                data = response.read(8_000_001)
        else:
            data = Path(location).read_bytes()
        if len(data) > 8_000_000:
            raise ValueError("Published content file exceeds 8 MB")
        return data

    def _manifest_location(self) -> str:
        if self.manifest_path and self.manifest_path.is_file():
            return str(self.manifest_path)
        if self.manifest_url:
            return self.manifest_url
        raise FileNotFoundError("No published-content manifest is available")

    @staticmethod
    def _eligible_file(logical_path: str) -> bool:
        return (
            logical_path.startswith("periods/")
            and logical_path.endswith("/events.json")
            and "/persons/" not in logical_path
        )

    def _file_location(self, manifest_location: str, relative_url: str) -> str:
        if urlparse(manifest_location).scheme:
            return urljoin(manifest_location, relative_url)
        return str(Path(manifest_location).parent / relative_url)

    def _load(self) -> _Snapshot:
        manifest_location = self._manifest_location()
        manifest = json.loads(self._read_bytes(manifest_location))
        version = str(manifest.get("contentVersion") or "").strip()
        files = manifest.get("files") or {}
        if not version or not isinstance(files, dict):
            raise ValueError("Invalid published-content manifest")

        records: list[dict[str, Any]] = []
        for logical_path, descriptor in files.items():
            if not self._eligible_file(str(logical_path)):
                continue
            if not isinstance(descriptor, dict):
                continue
            relative_url = str(descriptor.get("url") or "")
            expected_hash = str(descriptor.get("sha256") or "")
            if not relative_url or not expected_hash:
                continue
            payload = self._read_bytes(
                self._file_location(manifest_location, relative_url)
            )
            if hashlib.sha256(payload).hexdigest() != expected_hash:
                raise ValueError(f"Hash mismatch for {logical_path}")
            decoded = json.loads(payload)
            if not isinstance(decoded, list):
                continue
            for record in decoded:
                if not isinstance(record, dict) or record.get("status") != "published":
                    continue
                text = _event_text(record)
                if not text:
                    continue
                record_id = str(record.get("id") or record.get("slug") or "")
                records.append({
                    "id": f"published:{logical_path}:{record_id}",
                    "text": text,
                    "heading": str(record.get("title") or ""),
                    "sequence": int(record.get("sortOrder") or 0),
                    "pageStart": None,
                    "pageEnd": None,
                    "facets": [],
                    "yearStart": _year(record.get("startDate")),
                    "yearEnd": _year(record.get("endDate")),
                    "sourcePriority": 200,
                    "extractionStatus": "published",
                    "sourceId": f"published-content:{version}",
                    "sourceTitle": "Kho nội dung lịch sử đã xuất bản",
                    "trustLevel": "official",
                    "relatedEntities": [],
                    "relatedEntityIds": [],
                    "publishedContentPath": str(logical_path),
                })
        return _Snapshot(version, time.monotonic(), tuple(records))

    def _records(self) -> tuple[dict[str, Any], ...]:
        if not self.enabled:
            return ()
        now = time.monotonic()
        current = self._snapshot
        if current and now - current.loaded_at < self.cache_ttl_seconds:
            return current.records
        with self._lock:
            current = self._snapshot
            if current and now - current.loaded_at < self.cache_ttl_seconds:
                return current.records
            try:
                self._snapshot = self._load()
            except Exception as error:  # fail open to PDF/Neo4j
                LOGGER.warning("Cannot refresh published content: %s", error)
                if current:
                    self._snapshot = _Snapshot(
                        current.version,
                        now,
                        current.records,
                    )
                return current.records if current else ()
            return self._snapshot.records

    @staticmethod
    def _score(question: str, record: dict[str, Any]) -> float:
        query_tokens = _tokens(question)
        if not query_tokens:
            return 0.0
        text_key = normalize_search_key(
            f"{record.get('heading') or ''} {record.get('text') or ''}"
        )
        heading_key = normalize_search_key(record.get("heading") or "")
        hits = sum(token in text_key for token in query_tokens)
        title_hits = sum(token in heading_key for token in query_tokens)
        coverage = hits / len(query_tokens)
        title_coverage = title_hits / len(query_tokens)
        subject_phrase = " ".join(query_tokens[: min(5, len(query_tokens))])
        phrase_bonus = 0.12 if subject_phrase and subject_phrase in text_key else 0.0
        if hits < 2 or coverage < 0.28:
            return 0.0
        return min(1.0, 0.72 * coverage + 0.20 * title_coverage + phrase_bonus)

    def search(self, question: str, limit: int) -> list[dict[str, Any]]:
        ranked: list[dict[str, Any]] = []
        for original in self._records():
            score = self._score(question, original)
            if score <= 0:
                continue
            item = dict(original)
            item.update({
                "score": round(score, 6),
                "publishedContentScore": round(score, 6),
                "channels": ["published_content"],
                "directEvidence": True,
                "entityMatch": False,
                "matchedPhrases": [],
            })
            ranked.append(item)
        ranked.sort(
            key=lambda item: (float(item["score"]), item.get("sourcePriority", 0)),
            reverse=True,
        )
        return ranked[: max(1, int(limit))]

    def retrieve_hybrid(
        self,
        question: str,
        embedding: list[float],
        candidate_k: int,
        *,
        exact_phrases: list[str] | None = None,
        primary_entity: str = "",
    ) -> dict[str, Any]:
        del embedding, exact_phrases, primary_entity
        items = self.search(question, candidate_k)
        return {
            "items": items,
            "candidateCount": len(items),
            "queryTerms": list(_tokens(question)),
            "exactMatchCount": 0,
            "channels": ["published_content"] if items else [],
        }

    def retrieve_subject_evidence(
        self,
        subjects: list[str],
        *,
        limit: int = 240,
    ) -> list[dict[str, Any]]:
        return self.search(" ".join(subjects), limit) if subjects else []

    def retrieve_timeline_windows(
        self,
        question: str,
        windows: list[tuple[int, int]],
        *,
        per_window: int = 4,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for start, end in windows:
            eligible = [
                item for item in self.search(question, per_window * 4)
                if (item.get("yearStart") or start) <= end
                and (item.get("yearEnd") or end) >= start
            ][:per_window]
            for rank, item in enumerate(eligible, start=1):
                item.update({
                    "timelineWindowStart": start,
                    "timelineWindowEnd": end,
                    "timelineWindowRank": rank,
                })
                results.append(item)
        return results


def _dedupe(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for item in items:
        key = str(item.get("id") or "")
        if not key:
            continue
        previous = selected.get(key)
        if previous is None or float(item.get("score") or 0) > float(previous.get("score") or 0):
            selected[key] = item
    return list(selected.values())


class CompositeKnowledgeRepository:
    """Delegate mutations to Neo4j and merge both grounded read corpora."""

    def __init__(self, primary: Any, published: PublishedContentRepository) -> None:
        self.primary = primary
        self.published = published

    def __getattr__(self, name: str) -> Any:
        return getattr(self.primary, name)

    @staticmethod
    def _rank(
        items: Iterable[dict[str, Any]],
        limit: int,
        *,
        query: str = "",
    ) -> list[dict[str, Any]]:
        merged = _dedupe(items)
        query_tokens = _tokens(query)
        for item in merged:
            raw = float(item.get("score") or 0)
            text_key = normalize_search_key(
                f"{item.get('heading') or ''} {item.get('text') or ''}"
            )
            lexical_coverage = (
                sum(token in text_key for token in query_tokens)
                / len(query_tokens)
                if query_tokens else 0.0
            )
            authority_bonus = (
                0.12
                if "published_content" in (item.get("channels") or [])
                else 0.0
            )
            # Scores emitted by vector/full-text/JSON channels are not on the
            # same scale. Re-check question-to-text coverage before combining
            # them, so a normalized but irrelevant vector hit cannot outrank
            # an exact reviewed record.
            item["compositeScore"] = (
                raw * 0.55 + lexical_coverage * 0.45 + authority_bonus
            )
        merged.sort(key=lambda item: float(item["compositeScore"]), reverse=True)
        maximum = max((float(item["compositeScore"]) for item in merged), default=1.0)
        for item in merged:
            item["score"] = round(float(item["compositeScore"]) / maximum, 6)
        return merged[:limit]

    def retrieve_hybrid(self, question: str, embedding: list[float], candidate_k: int, **kwargs: Any) -> dict[str, Any]:
        primary = self.primary.retrieve_hybrid(question, embedding, candidate_k, **kwargs)
        published = self.published.retrieve_hybrid(question, embedding, candidate_k, **kwargs)
        items = self._rank(
            [*primary.get("items", []), *published.get("items", [])],
            candidate_k,
            query=question,
        )
        return {
            "items": items,
            "candidateCount": int(primary.get("candidateCount") or 0) + int(published.get("candidateCount") or 0),
            "queryTerms": list(dict.fromkeys([*primary.get("queryTerms", []), *published.get("queryTerms", [])])),
            "exactMatchCount": int(primary.get("exactMatchCount") or 0) + int(published.get("exactMatchCount") or 0),
            "channels": list(dict.fromkeys([*primary.get("channels", []), *published.get("channels", [])])),
        }

    def retrieve_subject_evidence(self, subjects: list[str], *, limit: int = 240) -> list[dict[str, Any]]:
        return self._rank([
            *self.primary.retrieve_subject_evidence(subjects, limit=limit),
            *self.published.retrieve_subject_evidence(subjects, limit=limit),
        ], limit, query=" ".join(subjects))

    def retrieve_timeline_windows(self, question: str, windows: list[tuple[int, int]], *, per_window: int = 4) -> list[dict[str, Any]]:
        return self._rank([
            *self.primary.retrieve_timeline_windows(question, windows, per_window=per_window),
            *self.published.retrieve_timeline_windows(question, windows, per_window=per_window),
        ], max(1, len(windows) * per_window), query=question)
