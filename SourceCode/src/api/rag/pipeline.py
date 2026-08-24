"""Unified and intentionally short RAG orchestration."""

from __future__ import annotations

import time
from typing import Any, Callable

from ..models import ChatRequest, ChatResponse, Citation, RetrievalDiagnostics
from .contracts import EvidenceItem, GeneratedAnswer, VerificationResult
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
            retrieval=diagnostics,
        )
