"""Unified and intentionally short RAG orchestration."""

from __future__ import annotations

import re
import time
from dataclasses import replace
from typing import Any, Callable

from ..models import (
    ChatRequest,
    ChatResponse,
    Citation,
    ContentSuggestion,
    RetrievalDiagnostics,
)
from ..entity_resolution import normalize_search_key
from .contracts import EvidenceItem, GeneratedAnswer, UnifiedPlan, VerificationResult
from .evidence import EvidenceTools, ModelEvidenceSelector
from .generator import GroundedAnswerModel, remove_redundant_opening_heading
from .planner import UnifiedQueryPlanner
from .verifier import SemanticEvidenceVerifier, should_verify


class GroundedRagPipeline:
    """Coordinate six stages; semantic decisions remain model-owned."""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        embedding_model: str,
        repository: Any,
        settings: Any,
        prompt_provider: Callable[[], dict[str, str]],
        query_logger: Callable[[dict[str, Any]], None],
    ) -> None:
        self.planner = UnifiedQueryPlanner(client, model)
        self.tools = EvidenceTools(client, embedding_model, repository)
        self.repository = repository
        self.selector = ModelEvidenceSelector(client, model)
        self.generator = GroundedAnswerModel(client, model)
        self.verifier = SemanticEvidenceVerifier(client, model)
        self.settings = settings
        self.prompt_provider = prompt_provider
        self.query_logger = query_logger

    @staticmethod
    def _citations(items: list[EvidenceItem], selected_ids: tuple[str, ...]) -> list[Citation]:
        selected = set(selected_ids)
        return [
            Citation(
                chunk_id=item.id,
                source_id=item.source_id,
                source_title=item.source_title or item.source_id or "Nguồn lịch sử",
                page_start=item.page_start,
                page_end=item.page_end,
                excerpt=item.text[:420],
                score=item.score,
                facets=item.facets,
            )
            for item in items
            if item.id in selected
        ]

    @staticmethod
    def _content_suggestions(
        plan: UnifiedPlan,
        items: list[EvidenceItem],
        selected_ids: tuple[str, ...],
        answer: str,
        *,
        limit: int = 12,
    ) -> list[ContentSuggestion]:
        """Build trusted in-app links from verified published-content hits."""
        if plan.suggestion_type == "none":
            return []
        selected = set(selected_ids)
        suggestions: list[ContentSuggestion] = []
        seen: set[str] = set()
        normalized_answer = answer.casefold()
        normalized_answer_key = normalize_search_key(answer)
        person_titles_in_answer = {
            item.heading.casefold()
            for item in items
            if item.published_content_type == "person"
            and item.heading
            and item.heading.casefold() in normalized_answer
        }
        event_title_positions = {
            item.id: normalized_answer_key.find(normalize_search_key(item.heading))
            for item in items
            if item.published_content_type == "event" and item.heading
        }
        generic_event_tokens = {
            "chien", "dich", "tran", "su", "kien", "thang", "dien",
            "ra", "nhu", "the", "nao", "ve", "cua",
        }

        def subject_similarity(item: EvidenceItem) -> float:
            subject_tokens = {
                token for token in normalize_search_key(
                    plan.subject or plan.standalone_question
                ).split()
                if token not in generic_event_tokens
                and len(token) >= 2
                and not token.isdigit()
            }
            title_tokens = {
                token for token in normalize_search_key(item.heading).split()
                if token not in generic_event_tokens
                and len(token) >= 2
                and not token.isdigit()
            }
            if not subject_tokens or not title_tokens:
                return 0.0
            overlap = len(subject_tokens & title_tokens)
            return (2.0 * overlap) / (len(subject_tokens) + len(title_tokens))

        event_candidates = [
            item for item in items
            if item.published_content_type == "event"
        ]
        primary_event_id = ""
        if plan.suggestion_type == "event" and event_candidates:
            best_event = max(
                event_candidates,
                key=lambda item: (subject_similarity(item), item.score),
            )
            if subject_similarity(best_event) >= 0.55:
                primary_event_id = best_event.id

        content_limit = min(limit, {
            "person": 12,
            "event": 4,
            "period": 3,
        }.get(plan.suggestion_type, limit))

        def suggestion_rank(value: EvidenceItem) -> tuple[int, int, int, float]:
            heading = value.heading.casefold()
            position = normalized_answer.find(heading) if heading else -1
            if value.published_content_type == "event":
                event_position = event_title_positions.get(value.id, -1)
                return (
                    0 if value.id == primary_event_id else 1,
                    event_position if event_position >= 0 else 1_000_000_000,
                    0 if value.id in selected else 1,
                    -subject_similarity(value),
                )
            return (
                0 if position >= 0 else 1,
                position if position >= 0 else 1_000_000_000,
                0 if value.id in selected else 1,
                -value.score,
            )

        ranked = sorted(
            items,
            key=suggestion_rank,
        )
        for item in ranked:
            if not item.published_content_path:
                continue
            if item.published_content_type != plan.suggestion_type:
                continue
            if (
                plan.suggestion_type == "person"
                and person_titles_in_answer
                and item.heading.casefold() not in person_titles_in_answer
            ):
                continue
            if (
                plan.suggestion_type == "event"
                and item.id != primary_event_id
                and event_title_positions.get(item.id, -1) < 0
            ):
                continue
            parts = item.published_content_path.strip("/").split("/")
            if not item.published_content_id:
                continue
            safe_slug = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
            if not safe_slug.fullmatch(item.published_content_id):
                continue
            if (
                item.published_content_type == "event"
                and len(parts) == 5
                and parts[0] == "periods"
                and parts[2] == "stages"
                and parts[4] == "events.json"
            ):
                period_slug, stage_slug = parts[1], parts[3]
                if not all(safe_slug.fullmatch(value) for value in (
                    period_slug, stage_slug
                )):
                    continue
                firestore_path = (
                    f"/periods/{period_slug}/stages/{stage_slug}/events/"
                    f"{item.published_content_id}"
                )
                app_path = (
                    f"/event/{period_slug}/{stage_slug}/"
                    f"{item.published_content_id}"
                )
            elif (
                item.published_content_type == "person"
                and len(parts) == 3
                and parts[0] == "persons"
                and parts[2] == f"{item.published_content_id}.json"
            ):
                period_slug = parts[1]
                if not safe_slug.fullmatch(period_slug):
                    continue
                firestore_path = (
                    f"/periods_person/{period_slug}/persons/"
                    f"{item.published_content_id}"
                )
                app_path = f"/person/{period_slug}/{item.published_content_id}"
            elif (
                item.published_content_type == "period"
                and item.published_content_path == "periods.json"
            ):
                firestore_path = f"/periods/{item.published_content_id}"
                app_path = f"/period-detail/{item.published_content_id}"
            else:
                continue
            if firestore_path in seen:
                continue
            if (
                item.published_content_type == "person"
                and item.heading.casefold() in seen
            ):
                continue
            seen.add(firestore_path)
            if item.published_content_type == "person":
                seen.add(item.heading.casefold())
            suggestions.append(ContentSuggestion(
                title=item.heading or item.published_content_id,
                firestore_path=firestore_path,
                app_path=app_path,
                content_type=item.published_content_type,
            ))
            if len(suggestions) >= content_limit:
                break
        return suggestions

    @staticmethod
    def _confidence(
        plan: Any,
        selection: Any,
        selected: list[EvidenceItem],
        verification: VerificationResult,
    ) -> tuple[float, dict[str, float]]:
        required = max(1, sum(1 for item in plan.requirements if item.required))
        verifier_missing = set(verification.missing_requirement_ids)
        covered = sum(
            1 for item in plan.requirements
            if (
                selection.requirement_evidence.get(item.id)
                and item.id not in verifier_missing
            )
        )
        coverage = min(1.0, covered / required)
        retrieval = sum(item.score for item in selected) / len(selected) if selected else 0.0
        verification_score = {
            "supported": 1.0,
            "corrected": 0.82,
            "not_run": 0.78,
            "insufficient": 0.25,
        }.get(verification.status, 0.25)
        confidence = max(0.0, min(1.0, 0.45 * coverage + 0.30 * retrieval + 0.25 * verification_score))
        if selection.conflicts:
            confidence = min(confidence, 0.60)
        if verification.status == "corrected":
            confidence = min(confidence, 0.80)
        if verification.status == "insufficient":
            confidence = min(confidence, 0.40)
        if coverage < 1.0:
            confidence = min(confidence, 0.65)
        return confidence, {
            "requirement_coverage": round(coverage, 4),
            "retrieval_quality": round(retrieval, 4),
            "semantic_verification": round(verification_score, 4),
        }

    def answer(self, request: ChatRequest, *, user_id: str = "") -> ChatResponse:
        started = time.perf_counter()
        prompts = self.prompt_provider()
        plan = self.planner.plan(request.question, request.messages, prompts)
        candidates, raw_diagnostics = self.tools.retrieve(
            plan,
            candidate_k=max(12, int(self.settings.ai_retrieval_candidate_k)),
        )
        selection = self.selector.select(
            plan,
            candidates,
            limit=max(10, int(self.settings.ai_retrieval_top_k) + 8),
        )
        recovery_attempted = False
        missing_ids = set(selection.missing_requirements)
        missing_requirements = tuple(
            item for item in plan.requirements
            if item.required and item.id in missing_ids
        )
        # One bounded recovery pass targets only requirements that the first
        # selection could not support. Avoid retry chains and do not retry a
        # selector outage/conflict.
        if candidates and missing_requirements and not selection.conflicts:
            recovery_attempted = True
            recovery_queries = tuple(dict.fromkeys(
                " ".join(filter(None, (plan.subject, item.question, plan.date_range))).strip()
                for item in missing_requirements
            ))
            recovery_plan = replace(
                plan,
                standalone_question=recovery_queries[0],
                requirements=missing_requirements,
                retrieval_queries=recovery_queries,
            )
            recovered, recovery_diagnostics = self.tools.retrieve(
                recovery_plan,
                candidate_k=max(8, int(self.settings.ai_retrieval_candidate_k) // 2),
            )
            merged = {item.id: item for item in candidates}
            for item in recovered:
                previous = merged.get(item.id)
                if previous is None or item.score > previous.score:
                    merged[item.id] = item
            candidates = list(merged.values())
            selection = self.selector.select(
                plan,
                candidates,
                limit=max(10, int(self.settings.ai_retrieval_top_k) + 8),
            )
            raw_diagnostics["candidate_count"] += int(
                recovery_diagnostics.get("candidate_count") or 0
            )
            raw_diagnostics["exact_match_count"] += int(
                recovery_diagnostics.get("exact_match_count") or 0
            )
            raw_diagnostics["temporal_filtered_count"] = int(
                raw_diagnostics.get("temporal_filtered_count") or 0
            ) + int(recovery_diagnostics.get("temporal_filtered_count") or 0)
            raw_diagnostics["channels"] = list(dict.fromkeys([
                *raw_diagnostics.get("channels", []),
                *recovery_diagnostics.get("channels", []),
                "single_gap_recovery",
            ]))
            raw_diagnostics["query_terms"] = list(dict.fromkeys([
                *raw_diagnostics.get("query_terms", []),
                *recovery_diagnostics.get("query_terms", []),
            ]))
        selected = [item for item in candidates if item.id in set(selection.evidence_ids)]

        if selected:
            generated = self.generator.generate(plan, candidates, selection, prompts)
        else:
            generated = GeneratedAnswer(
                "Xin lỗi, kho tri thức hiện chưa có đủ thông tin đáng tin cậy để trả lời câu hỏi này."
            )

        verification = VerificationResult("not_run", generated.answer)
        if generated.answer and selected and should_verify(plan, generated, selection.conflicts):
            verification = self.verifier.verify(
                plan, generated, selected, selection, selection.conflicts
            )
        answer = verification.answer or (
            "Xin lỗi, các nhận định tìm được chưa đủ căn cứ để tạo câu trả lời chính xác."
        )
        answer = remove_redundant_opening_heading(answer, plan.standalone_question)
        confidence, confidence_factors = self._confidence(
            plan, selection, selected, verification
        )
        citations = self._citations(candidates, selection.evidence_ids) if request.include_citations else []
        suggestion_candidates = list(candidates)
        published = getattr(self.repository, "published", None)
        suggest = getattr(published, "suggest", None)
        if callable(suggest) and plan.suggestion_type != "none":
            try:
                suggestion_records = suggest(
                    plan.standalone_question,
                    answer,
                    plan.suggestion_type,
                    limit=24,
                )
            except Exception:
                suggestion_records = []
            suggestion_by_id = {item.id: item for item in suggestion_candidates}
            for raw in suggestion_records:
                item = EvidenceItem.from_record(raw)
                if item is not None:
                    suggestion_by_id[item.id] = item
            suggestion_candidates = list(suggestion_by_id.values())
        content_suggestions = self._content_suggestions(
            plan, suggestion_candidates, selection.evidence_ids, answer
        )
        evidence_missing = [
            item.id for item in plan.requirements
            if not selection.requirement_evidence.get(item.id)
        ]
        answer_missing = set(verification.missing_requirement_ids)
        if verification.status == "not_run":
            answer_missing.update(
                item.id for item in plan.requirements
                if item.required and item.id not in generated.covered_requirement_ids
            )
        missing = list(dict.fromkeys([*evidence_missing, *answer_missing]))
        diagnostics = RetrievalDiagnostics(
            strategy="unified_planner_dual_store_model_selection",
            candidate_count=int(raw_diagnostics["candidate_count"]),
            selected_count=len(selected),
            intent=plan.mode,
            scope=plan.scope,
            date_range=plan.date_range,
            query_terms=raw_diagnostics["query_terms"],
            primary_entity=plan.subject,
            exact_match_count=int(raw_diagnostics["exact_match_count"]),
            temporal_filtered_count=int(
                raw_diagnostics.get("temporal_filtered_count") or 0
            ),
            recovery_attempted=recovery_attempted,
            channels=raw_diagnostics["channels"],
            confidence_factors=confidence_factors,
            original_question=request.question,
            standalone_question=plan.standalone_question,
            conversation_resolved=(plan.standalone_question != request.question),
            normalized_question=plan.standalone_question,
            query_ambiguous=bool(plan.ambiguity),
            ambiguity_notes=[plan.ambiguity] if plan.ambiguity else [],
            required_facets=[item.id for item in plan.requirements if item.required],
            covered_facets=[item.id for item in plan.requirements if selection.requirement_evidence.get(item.id)],
            missing_facets=missing,
            semantic_requirements={
                item.id: {
                    "label": item.question,
                    "answerType": item.answer_type,
                    "required": item.required,
                    "timeWindowIds": list(item.time_window_ids),
                }
                for item in plan.requirements
            },
            requirement_statuses={
                item.id: "supported" if selection.requirement_evidence.get(item.id) else "missing"
                for item in plan.requirements
            },
            requirement_evidence=selection.requirement_evidence,
            coverage_gate_outcome=("complete" if not missing else "partial"),
            claim_verification_status=verification.status,
            checked_claim_count=len(generated.claims) if verification.status != "not_run" else 0,
            unsupported_high_risk_claims=list(verification.unsupported_claims),
            coverage_gate_limitations=[
                *selection.conflicts,
                *(
                    [
                        "Đã loại "
                        f"{int(raw_diagnostics.get('temporal_filtered_count') or 0)} "
                        "đoạn lệch ngoài phạm vi thời gian."
                    ]
                    if raw_diagnostics.get("temporal_filtered_count")
                    else []
                ),
                *(verification.notes if verification.status != "supported" else ()),
                *(
                    f"Chưa có đủ bằng chứng cho: {item.question}"
                    for item in plan.requirements
                    if item.id in missing
                ),
            ],
            answer_structure=plan.output_style,
        )
        related_entities = list(dict.fromkeys(
            entity for item in selected for entity in item.related_entities
        ))[:20]
        latency_ms = int((time.perf_counter() - started) * 1000)
        self.query_logger({
            "question": request.question,
            "normalizedQuestion": plan.standalone_question,
            "userId": user_id,
            "answerable": bool(selected and answer),
            "intent": plan.mode,
            "scope": plan.scope,
            "retrievalScore": confidence,
            "latencyMs": latency_ms,
            "strategy": diagnostics.strategy,
            "selectedCount": len(selected),
            "missingRequirements": missing,
            "verificationStatus": verification.status,
        })
        return ChatResponse(
            answer=answer,
            citations=citations,
            confidence=confidence,
            related_entities=related_entities,
            content_suggestions=content_suggestions,
            retrieval=diagnostics,
        )
