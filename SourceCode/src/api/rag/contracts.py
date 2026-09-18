"""Small, stable contracts for the unified RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


RAG_MODES = {
    "direct_fact",
    "explanatory_rag",
    "comparison",
    "timeline_evolution",
    "graph_multihop",
    "reliability_and_conversation",
}


# These are answer-shape contracts, not historical-domain labels.  The model
# planner uses them for any question that asks for participating people,
# social groups, organisations, forces or sides in an event.
PARTICIPANT_GROUP_ANSWER_TYPES = {
    "participant_groups",
    "participating_forces",
    "participant_inventory",
}
PARTICIPANT_ROLE_ANSWER_TYPES = {
    "participant_roles",
    "force_roles",
    "participation_roles",
}


def is_participant_group_type(answer_type: str) -> bool:
    return str(answer_type or "").strip().casefold() in PARTICIPANT_GROUP_ANSWER_TYPES


def is_participant_role_type(answer_type: str) -> bool:
    return str(answer_type or "").strip().casefold() in PARTICIPANT_ROLE_ANSWER_TYPES


def is_participant_inventory_plan(plan: "UnifiedPlan") -> bool:
    return any(
        is_participant_group_type(item.answer_type)
        or is_participant_role_type(item.answer_type)
        for item in plan.requirements
    )


@dataclass(frozen=True, slots=True)
class AtomicRequirement:
    id: str
    question: str
    answer_type: str = "fact"
    required: bool = True
    time_window_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any], index: int) -> "AtomicRequirement":
        return cls(
            id=str(value.get("id") or f"r{index + 1}"),
            question=str(value.get("question") or "").strip(),
            answer_type=str(value.get("answerType") or "fact").strip(),
            required=bool(value.get("required", True)),
            time_window_ids=tuple(dict.fromkeys(
                str(item).strip()
                for item in (value.get("timeWindowIds") or [])
                if str(item).strip()
            )),
        )


@dataclass(frozen=True, slots=True)
class TimeWindow:
    id: str
    start: int
    end: int
    label: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any], index: int) -> "TimeWindow | None":
        try:
            start = int(value.get("start"))
            end = int(value.get("end"))
        except (TypeError, ValueError):
            return None
        if start > end:
            start, end = end, start
        return cls(
            id=str(value.get("id") or f"w{index + 1}").strip(),
            start=start,
            end=end,
            label=str(value.get("label") or "").strip(),
        )


@dataclass(frozen=True, slots=True)
class UnifiedPlan:
    standalone_question: str
    mode: str
    subject: str
    requirements: tuple[AtomicRequirement, ...]
    retrieval_queries: tuple[str, ...]
    scope: str = ""
    date_range: str = ""
    year_start: int | None = None
    year_end: int | None = None
    time_windows: tuple[TimeWindow, ...] = ()
    output_style: str = "direct"
    requires_verification: bool = False
    ambiguity: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any], original: str) -> "UnifiedPlan":
        raw_requirements = value.get("requirements")
        raw_requirements = (
            raw_requirements if isinstance(raw_requirements, list) else []
        )
        requirements = tuple(
            requirement
            for index, item in enumerate(raw_requirements[:8])
            if isinstance(item, dict)
            for requirement in [AtomicRequirement.from_dict(item, index)]
            if requirement.question
        )
        if not requirements:
            requirements = (AtomicRequirement("r1", original),)
        mode = str(value.get("mode") or "explanatory_rag")
        if mode not in RAG_MODES:
            mode = "explanatory_rag"
        queries = tuple(dict.fromkeys(
            str(item).strip()
            for item in (value.get("retrievalQueries") or [])[:12]
            if str(item).strip()
        ))
        standalone = str(value.get("standaloneQuestion") or original).strip()
        raw_windows = value.get("timeWindows")
        raw_windows = raw_windows if isinstance(raw_windows, list) else []
        time_windows = tuple(
            window
            for index, item in enumerate(raw_windows[:6])
            if isinstance(item, dict)
            for window in [TimeWindow.from_dict(item, index)]
            if window is not None and window.id
        )
        year_start = value.get("yearStart")
        year_end = value.get("yearEnd")
        try:
            year_start = int(year_start) if year_start is not None else None
        except (TypeError, ValueError):
            year_start = None
        try:
            year_end = int(year_end) if year_end is not None else None
        except (TypeError, ValueError):
            year_end = None
        if year_start is not None and year_end is not None and year_start > year_end:
            year_start, year_end = year_end, year_start
        if time_windows:
            year_start = min(window.start for window in time_windows)
            year_end = max(window.end for window in time_windows)
        return cls(
            standalone_question=standalone,
            mode=mode,
            subject=str(value.get("subject") or "").strip(),
            requirements=requirements,
            retrieval_queries=queries or (standalone,),
            scope=str(value.get("scope") or "").strip(),
            date_range=str(value.get("dateRange") or "").strip(),
            year_start=year_start,
            year_end=year_end,
            time_windows=time_windows,
            output_style=str(value.get("outputStyle") or "direct").strip(),
            requires_verification=bool(value.get("requiresVerification")),
            ambiguity=str(value.get("ambiguity") or "").strip(),
        )


@dataclass(slots=True)
class EvidenceItem:
    id: str
    text: str
    heading: str = ""
    source_id: str = ""
    source_title: str = ""
    page_start: int | None = None
    page_end: int | None = None
    score: float = 0.0
    facets: list[str] = field(default_factory=list)
    related_entities: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    trust_level: str = ""
    source_priority: int = 0
    curated: bool = False
    year_start: int | None = None
    year_end: int | None = None

    @classmethod
    def from_record(cls, value: dict[str, Any]) -> "EvidenceItem | None":
        item_id = str(value.get("id") or value.get("chunkId") or "").strip()
        text = str(value.get("text") or "").strip()
        if not item_id or not text:
            return None
        return cls(
            id=item_id,
            text=text,
            heading=str(value.get("heading") or "").strip(),
            source_id=str(value.get("sourceId") or "").strip(),
            source_title=str(value.get("sourceTitle") or "").strip(),
            page_start=value.get("pageStart"),
            page_end=value.get("pageEnd"),
            score=float(value.get("score") or 0.0),
            facets=list(value.get("facets") or []),
            related_entities=list(value.get("relatedEntities") or []),
            channels=list(value.get("channels") or []),
            trust_level=str(value.get("trustLevel") or "").strip(),
            source_priority=int(value.get("sourcePriority") or 0),
            curated=(
                bool(value.get("directEvidence"))
                or "published_content" in (value.get("channels") or [])
            ),
            year_start=value.get("yearStart") if isinstance(value.get("yearStart"), int) else None,
            year_end=value.get("yearEnd") if isinstance(value.get("yearEnd"), int) else None,
        )

    def overlaps(self, start: int, end: int) -> bool | None:
        """Return temporal compatibility, or None when the item is undated."""
        if self.year_start is None and self.year_end is None:
            return None
        item_start = self.year_start if self.year_start is not None else self.year_end
        item_end = self.year_end if self.year_end is not None else self.year_start
        return bool(
            item_start is not None
            and item_end is not None
            and item_start <= end
            and item_end >= start
        )

    def prompt_payload(self, max_chars: int = 1800) -> dict[str, Any]:
        return {
            "id": self.id,
            "heading": self.heading,
            "text": self.text[:max_chars],
            "source": self.source_title,
            "page": self.page_start,
            "score": round(self.score, 4),
            "channels": self.channels,
            "trustLevel": self.trust_level,
            "sourcePriority": self.source_priority,
            "curated": self.curated,
            "yearStart": self.year_start,
            "yearEnd": self.year_end,
        }


@dataclass(frozen=True, slots=True)
class EvidenceSelection:
    evidence_ids: tuple[str, ...]
    requirement_evidence: dict[str, list[str]]
    missing_requirements: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    answer: str
    claims: tuple[dict[str, Any], ...] = ()

    @property
    def covered_requirement_ids(self) -> tuple[str, ...]:
        """Return requirements explicitly covered by generated claims."""
        return tuple(dict.fromkeys(
            str(requirement_id)
            for claim in self.claims
            for requirement_id in claim.get("requirementIds") or []
            if str(requirement_id)
        ))


@dataclass(frozen=True, slots=True)
class VerificationResult:
    status: str
    answer: str
    unsupported_claims: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    missing_requirement_ids: tuple[str, ...] = ()
