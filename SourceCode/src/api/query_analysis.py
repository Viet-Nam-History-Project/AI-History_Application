import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

from .history_facets import (
    POLICY_OVERVIEW_FACETS,
    comparison_domain_for_question,
    comparison_facets_for_domain,
    evolution_facets_for_domain,
)


_LEADING_REQUESTS = re.compile(
    r"^(?:hãy\s+)?(?:nói|cho|giới\s+thiệu|trình\s+bày)\s+"
    r"(?:cho\s+)?(?:tôi|mình)\s+(?:biết\s+)?",
    re.IGNORECASE,
)
_SUBJECT_BOUNDARIES = re.compile(
    r"\s+(?:là\s+ai|là\s+gì|diễn\s+ra|gồm|có\s+kết\s+quả|"
    r"có\s+(?:ý\s+nghĩa|vai\s+trò|nguyên\s+nhân)|đã\s+|"
    r"như\s+thế\s+nào|xảy\s+ra|bắt\s+đầu|kết\s+thúc)\b",
    re.IGNORECASE,
)
_NAMED_OBJECT_PATTERNS = (
    re.compile(
        r"^(?:ai|những\s+ai|người\s+nào)\s+(?:đã\s+)?"
        r"(?:lãnh\s+đạo|chỉ\s+huy|tham\s+gia|ký|thành\s+lập)\s+"
        r"(?P<subject>.+?)(?:\s*[?.!]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:ý\s+nghĩa|kết\s+quả|nguyên\s+nhân|diễn\s+biến|"
        r"lực\s+lượng|lãnh\s+đạo|bối\s+cảnh)\s+(?:của\s+)?"
        r"(?P<subject>.+?)(?:\s+(?:là|gồm|như\s+thế\s+nào)\b|\s*[?.!]|$)",
        re.IGNORECASE,
    ),
)
_GENERIC_SUBJECTS = {
    "ai",
    "nhan vat",
    "nhan vat lich su",
    "su kien",
    "lich su viet nam",
    "nguoi lanh dao",
}

# Chỉ tự động sửa các mẫu có một cách hiểu gần như chắc chắn. Những từ viết tắt
# có nhiều nghĩa (ví dụ HCM) được đánh dấu mơ hồ thay vì âm thầm thay đổi.
_CONTEXTUAL_REPLACEMENTS = (
    (
        re.compile(r"\bcách\s+chính\s+sách\b", re.IGNORECASE),
        "các chính sách",
        "Sửa từ gõ nhầm theo cụm ngữ cảnh",
    ),
)
_SAFE_ABBREVIATIONS = (
    (re.compile(r"\bĐBP\b", re.IGNORECASE), "Điện Biên Phủ"),
    (re.compile(r"\bCMT\s*8\b", re.IGNORECASE), "Cách mạng tháng Tám"),
    (re.compile(r"\bCM\s+T(?:HÁNG\s*)?8\b", re.IGNORECASE), "Cách mạng tháng Tám"),
    (re.compile(r"\bVNDCCH\b", re.IGNORECASE), "Việt Nam Dân chủ Cộng hòa"),
    (re.compile(r"\bVNCH\b", re.IGNORECASE), "Việt Nam Cộng hòa"),
    (re.compile(r"\bXHCN\b", re.IGNORECASE), "xã hội chủ nghĩa"),
    (re.compile(r"\bK0\b", re.IGNORECASE), "không"),
    (re.compile(r"\bKO\b", re.IGNORECASE), "không"),
)
_AMBIGUOUS_ABBREVIATIONS = {
    "hcm": "HCM có thể là Hồ Chí Minh hoặc Thành phố Hồ Chí Minh",
    "tw": "TW có thể chỉ Trung ương hoặc một tên viết tắt khác",
    "tg": "TG có nhiều nghĩa tùy ngữ cảnh",
}
_DUPLICATE_WORD = re.compile(
    r"\b(?P<word>[0-9A-Za-zÀ-ỹĐđ]+)(?P<gap>\s+)(?P=word)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QuerySignals:
    primary_entity: str = ""
    exact_phrases: tuple[str, ...] = ()
    is_identity_query: bool = False


@dataclass(frozen=True)
class QueryCorrection:
    original: str
    replacement: str
    reason: str


@dataclass(frozen=True)
class QueryNormalization:
    original_question: str
    normalized_question: str
    corrections: tuple[QueryCorrection, ...] = ()
    ambiguous_terms: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return self.original_question != self.normalized_question


@dataclass(frozen=True)
class QueryPlan:
    intent: str = "history_lookup"
    subject: str = ""
    required_facets: tuple[str, ...] = ()
    optional_facets: tuple[str, ...] = ()
    answer_structure: str = "direct"
    inferred_scope: str = ""
    explicit_date_range: str = ""
    allow_inferred_date_recovery: bool = True
    comparison_subjects: tuple[str, ...] = ()
    comparison_date_ranges: tuple[tuple[int, int], ...] = ()
    comparison_domain: str = ""
    comparison_object_types: tuple[str, ...] = ()
    comparison_intent: str = ""
    explicit_facets: tuple[str, ...] = ()
    include_similarities: bool = True
    include_differences: bool = True
    include_explanation: bool = True
    include_judgement: bool = False
    evolution_intent: str = ""
    evolution_domain: str = ""
    evolution_subject_type: str = ""
    evolution_periods: tuple[tuple[int, int], ...] = ()
    evolution_period_labels: tuple[str, ...] = ()
    evolution_period_states: tuple[str, ...] = ()
    evolution_boundary_causes: tuple[str, ...] = ()
    evolution_boundary_years: tuple[int | None, ...] = ()
    evolution_boundary_actors: tuple[str, ...] = ()
    evolution_boundary_actions: tuple[str, ...] = ()
    evolution_boundary_excerpts: tuple[str, ...] = ()
    evolution_boundary_evidence_ids: tuple[str, ...] = ()
    evolution_period_evidence_ids: tuple[tuple[str, ...], ...] = ()
    evolution_subject_lifetime: tuple[int, int] = ()
    evolution_range_mismatch: bool = False
    evolution_range_resolution: str = ""


def _search_key(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.casefold().replace("đ", "d"))
    plain = "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )
    return re.sub(r"[^0-9a-z]+", " ", plain).strip()


def _clean_phrase(value: str) -> str:
    value = _LEADING_REQUESTS.sub("", value.strip(" \t\n\r.,?!:;\"'"))
    return re.sub(r"\s+", " ", value).strip()


def _clean_comparison_subject(value: str) -> str:
    """Remove the requested comparison dimension from an object name."""
    cleaned = _clean_phrase(value).strip("“”\"'")
    cleaned = re.sub(
        r"^(?:vai\s+trò\s+của|nguyên\s+nhân\s+thắng\s+lợi\s+của|"
        r"nguyên\s+nhân\s+của|kết\s+quả\s+của|hệ\s+quả\s+của|"
        r"ý\s+nghĩa\s+của|tác\s+động\s+của)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip(" \t\n\r.,?!:;“”\"'")


def _clean_evolution_subject(value: str) -> str:
    """Remove the F9 request/range while preserving the historical subject."""
    cleaned = _clean_phrase(value).strip("“”\"'")
    year = r"(?:1[0-9]{3}|20[0-9]{2})"
    cleaned = re.sub(
        rf"^(?:trong\s+)?(?:giai\s+đoạn\s+|thời\s+kỳ\s+)?"
        rf"từ\s+{year}\s+(?:đến|tới|-)\s+{year}\s*[:,.-]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s+(?:đã\s+)?(?:thay\s+đổi|chuyển\s+biến|phát\s+triển|"
        r"tiến\s+triển)\s+(?:như\s+thế\s+nào|ra\s+sao)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s+(?:qua|theo|trong)\s+(?:các|những)?\s*"
        r"(?:giai\s+đoạn|thời\s+kỳ|thời\s+gian)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        rf"\s+từ\s+{year}\s+(?:đến|tới|-)\s+{year}\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip(" \t\n\r.,?!:;“”\"'") or _clean_phrase(value)


def _replace_and_record(
    value: str,
    pattern: re.Pattern[str],
    replacement: str,
    reason: str,
    corrections: list[QueryCorrection],
) -> str:
    def replace(match: re.Match[str]) -> str:
        original = match.group(0)
        if original != replacement:
            corrections.append(QueryCorrection(original, replacement, reason))
        return replacement

    return pattern.sub(replace, value)


def _deduplicate_words(
    value: str,
    corrections: list[QueryCorrection],
) -> str:
    # Lặp tối đa vài vòng để xử lý "ai ai ai" nhưng luôn có giới hạn cứng.
    for _ in range(4):
        match = _DUPLICATE_WORD.search(value)
        if not match:
            break
        original = match.group(0)
        replacement = match.group("word")
        corrections.append(QueryCorrection(
            original,
            replacement,
            "Loại bỏ từ bị lặp liên tiếp",
        ))
        value = f"{value[:match.start()]}{replacement}{value[match.end():]}"
    return value


def _ambiguous_abbreviations(value: str) -> Iterable[str]:
    for token in re.findall(r"[0-9A-Za-zÀ-ỹĐđ]+", value):
        explanation = _AMBIGUOUS_ABBREVIATIONS.get(_search_key(token))
        if explanation:
            yield explanation


def normalize_query(question: str) -> QueryNormalization:
    """Apply only high-confidence Vietnamese query corrections without an LLM."""
    original = unicodedata.normalize("NFC", question).strip()
    normalized = re.sub(r"\s+", " ", original)
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    corrections: list[QueryCorrection] = []

    if normalized != original:
        corrections.append(QueryCorrection(
            original,
            normalized,
            "Chuẩn hóa khoảng trắng và dấu câu",
        ))

    normalized = _deduplicate_words(normalized, corrections)
    for pattern, replacement, reason in _CONTEXTUAL_REPLACEMENTS:
        normalized = _replace_and_record(
            normalized,
            pattern,
            replacement,
            reason,
            corrections,
        )
    for pattern, replacement in _SAFE_ABBREVIATIONS:
        normalized = _replace_and_record(
            normalized,
            pattern,
            replacement,
            "Mở rộng từ viết tắt phổ biến",
            corrections,
        )

    ambiguous_terms = tuple(dict.fromkeys(_ambiguous_abbreviations(normalized)))
    return QueryNormalization(
        original_question=original,
        normalized_question=normalized,
        corrections=tuple(corrections),
        ambiguous_terms=ambiguous_terms,
    )


def _is_specific_phrase(value: str) -> bool:
    key = _search_key(value)
    words = key.split()
    return 2 <= len(words) <= 14 and key not in _GENERIC_SUBJECTS


def analyze_query(question: str) -> QuerySignals:
    """Extract the main named subject without requiring an LLM call."""
    cleaned_question = _clean_phrase(question)
    identity_match = re.search(
        r"^(?P<subject>.+?)\s+là\s+ai(?:\s*[?.!]|$)",
        cleaned_question,
        re.IGNORECASE,
    )
    is_identity_query = identity_match is not None

    candidates: list[str] = []
    if identity_match:
        candidates.append(_clean_phrase(identity_match.group("subject")))

    for pattern in _NAMED_OBJECT_PATTERNS:
        named_object = pattern.search(cleaned_question)
        if named_object:
            candidates.append(_clean_phrase(named_object.group("subject")))

    boundary_match = _SUBJECT_BOUNDARIES.search(cleaned_question)
    if boundary_match:
        candidates.append(_clean_phrase(cleaned_question[:boundary_match.start()]))

    candidates.extend(
        _clean_phrase(match)
        for match in re.findall(r"[“”\"']([^\"'“”]{3,100})[“”\"']", question)
    )

    exact_phrases: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = _search_key(candidate)
        if not _is_specific_phrase(candidate) or key in seen:
            continue
        seen.add(key)
        exact_phrases.append(candidate)

    primary_entity = exact_phrases[0] if exact_phrases else ""
    return QuerySignals(
        primary_entity=primary_entity,
        exact_phrases=tuple(exact_phrases[:3]),
        is_identity_query=is_identity_query,
    )


def _comparison_intent(key: str) -> str:
    """Distinguish what the user wants to do with the two objects."""
    if any(marker in key for marker in (
        "nhan dinh", "co chinh xac khong", "co dung khong",
        "quyet dinh nhat", "quan trong nhat",
    )):
        return "judgement"
    if any(marker in key for marker in (
        "giong nhau den muc nao", "khac nhau den muc nao",
        "muc do giong", "muc do khac",
    )):
        return "extent_comparison"
    if any(marker in key for marker in (
        "hieu qua hon", "hieu qua nhat", "sau rong hon",
        "thanh cong hon", "tot hon",
    )):
        return "effectiveness_comparison"
    if any(marker in key for marker in (
        "tiep noi", "ke thua", "phat trien tu", "lien tuc va thay doi",
        "thay doi va tiep tuc",
    )):
        return "continuity_change"
    if any(marker in key for marker in (
        "hai tu lieu", "hai tai lieu", "hai nguon", "quan diem",
        "cach danh gia", "cach giai thich",
    )):
        return "interpretation_comparison"
    if "vai tro" in key:
        return "role_comparison"
    if "nguyen nhan" in key or "vi sao" in key:
        return "cause_comparison"
    if "y nghia" in key:
        return "significance_comparison"
    if any(marker in key for marker in (
        "he qua", "ket qua", "tac dong", "anh huong",
    )):
        return "consequence_comparison"
    if any(marker in key for marker in (
        "khac biet co ban", "khac biet lon nhat",
        "khac biet chu yeu", "diem khac nhau co ban",
    )):
        return "main_difference"
    if any(marker in key for marker in (
        "diem chung co ban", "diem giong chu yeu",
        "diem giong nhau co ban", "giong nhau chu yeu",
    )):
        return "main_similarity"
    return "similarities_differences"


def _explicit_comparison_facets(key: str, domain: str) -> tuple[str, ...]:
    """Extract criteria explicitly named by the user before model planning."""
    scale_facet = (
        "scale_investment"
        if domain == "economic_policy"
        else "scale_intensity"
    )
    marker_map = (
        (("thoi gian", "boi canh", "hoan canh"), "comparison_context"),
        (("nguoi to chuc", "chu the khoi xuong"), "organizer"),
        (("muc tieu",), "objectives_consequences"),
        (("am muu", "thu doan", "bien phap", "phuong thuc", "cach danh"), "strategy_methods"),
        (("luc luong", "quan doi tham gia"), "forces"),
        (("pham vi", "dia ban"), "geographic_scope"),
        (("quy mo", "von dau tu"), scale_facet),
        (("nong nghiep", "ruong dat", "don dien"), "agriculture"),
        (("cong nghiep", "khai mo"), "industry"),
        (("thuong nghiep", "thuong mai", "thi truong"), "commerce"),
        (("giao thong", "ha tang"), "transport"),
        (("thue khoa", "tai chinh"), "economic_taxation"),
        (("lanh dao", "chi huy"), "leadership"),
        (("dien bien", "cac dot", "tien trinh"), "progress"),
        (("thang loi tieu bieu", "tran danh tieu bieu"), "representative_events"),
        (("ket qua", "he qua"), "result"),
        (("y nghia", "tac dong lau dai"), "significance"),
        (("nguyen nhan",), "cause"),
        (("vai tro", "dong gop"), "role_contribution"),
        (("han che", "diem yeu"), "limitations"),
        (("noi dung", "dieu khoan"), "agreement_content"),
        (("cac ben tham gia",), "participants"),
        (("co che thuc hien", "muc do thuc hien"), "implementation_mechanism"),
        (("ke thua", "tiep noi", "thay doi"), "continuity_change"),
        (("quan diem", "goc nhin", "luan diem"), "source_perspective"),
    )
    return tuple(dict.fromkeys(
        facet
        for markers, facet in marker_map
        if any(marker in key for marker in markers)
    ))


def _comparison_answer_structure(intent: str) -> str:
    return {
        "main_similarity": "comparison_main_similarity",
        "main_difference": "comparison_main_difference",
        "extent_comparison": "comparison_extent_with_judgement",
        "cause_comparison": "comparison_focused_causes",
        "consequence_comparison": "comparison_focused_outcomes",
        "effectiveness_comparison": "comparison_effectiveness_with_criteria",
        "significance_comparison": "comparison_focused_significance",
        "continuity_change": "comparison_continuity_change",
        "role_comparison": "comparison_focused_roles",
        "interpretation_comparison": "comparison_source_perspectives",
        "judgement": "comparison_claim_evaluation",
        "focused_facets": "comparison_selected_facets",
    }.get(intent, "comparison_matrix_then_similarities_differences")


def _comparison_requirements(intent: str) -> tuple[bool, bool, bool, bool]:
    include_similarities = intent in {
        "similarities_differences", "main_similarity",
        "extent_comparison", "continuity_change",
    }
    include_differences = intent in {
        "similarities_differences", "main_difference",
        "extent_comparison", "continuity_change", "judgement",
    }
    include_explanation = intent not in {"main_similarity", "main_difference"}
    include_judgement = intent in {
        "extent_comparison", "effectiveness_comparison", "judgement",
    }
    return (
        include_similarities,
        include_differences,
        include_explanation,
        include_judgement,
    )


def _evolution_intent(key: str) -> str:
    """Classify the specific F9 continuity/change request."""
    if any(marker in key for marker in (
        "thay doi quan trong nhat", "thay doi nao co y nghia quyet dinh",
        "bien doi quyet dinh",
    )):
        return "decisive_change"
    if any(marker in key for marker in (
        "truoc va sau", "truoc sau", "ke tu truoc den sau",
    )):
        return "before_after"
    if any(marker in key for marker in (
        "ngan han va dai han", "truoc mat va lau dai",
    )):
        return "short_long_term"
    if any(marker in key for marker in (
        "nhanh hay cham", "toc do thay doi", "tang toc", "cham lai",
    )):
        return "acceleration_slowdown"
    if any(marker in key for marker in (
        "dao chieu", "quay lai", "chuyen nguoc", "tu mem deo sang dan ap",
    )):
        return "reversal"
    if any(marker in key for marker in (
        "ke thua va phat trien", "ke thua nhung", "phat trien tu",
    )):
        return "inheritance_development"
    if any(marker in key for marker in (
        "chia thanh nhung giai doan", "chia thanh cac giai doan",
        "co the chia", "phan ky",
    )):
        return "periodization"
    if any(marker in key for marker in (
        "buoc ngoat", "moc nao danh dau", "thay doi lon",
    )):
        return "turning_points"
    if any(marker in key for marker in (
        "thay doi den muc nao", "muc do thay doi", "thuc su thay doi",
    )):
        return "extent_of_change"
    if (
        ("vi sao" in key or "tai sao" in key or "nguyen nhan" in key)
        and ("thay doi" in key or "chuyen sang" in key)
    ):
        return "cause_of_change"
    if any(marker in key for marker in (
        "van duoc duy tri", "van giu nguyen", "tiep noi va thay doi",
        "thay doi va tiep noi", "yeu to tiep noi",
    )):
        return "continuity_and_change"
    return "evolution_over_time"


def _evolution_fallback_facets(
    evolution_intent: str,
    domain: str,
) -> tuple[str, ...]:
    intent_facets = {
        "evolution_over_time": (
            "historical_evolution", "continuity_change", "cause",
        ),
        "continuity_and_change": (
            "continuity_change", "historical_evolution", "cause",
        ),
        "turning_points": ("historical_evolution", "cause", "result"),
        "extent_of_change": (
            "continuity_change", "historical_evolution", "result",
        ),
        "cause_of_change": ("cause", "historical_evolution"),
        "before_after": (
            "historical_evolution", "continuity_change", "result",
        ),
        "short_long_term": ("result", "significance", "historical_evolution"),
        "acceleration_slowdown": ("historical_evolution", "cause"),
        "reversal": (
            "historical_evolution", "cause", "continuity_change",
        ),
        "inheritance_development": (
            "continuity_change", "historical_evolution", "result",
        ),
        "periodization": ("historical_evolution", "cause"),
        "decisive_change": (
            "historical_evolution", "result", "significance",
        ),
    }
    candidates = evolution_facets_for_domain(domain)
    domain_priority = {
        "political_administration": (
            "political_administrative",
            "governance_methods",
            "wartime_mobilization",
        ),
        "economic_policy": (
            "economic_taxation",
            "scale_investment",
            "social_transformation",
        ),
        "military_strategy": (
            "forces",
            "strategy_methods",
            "scale_intensity",
        ),
        "military_campaign": ("progress", "leadership", "result"),
        "movement_revolution": (
            "strategy_methods",
            "forces",
            "progress",
        ),
        "state_dynasty": (
            "organization_structure",
            "economic_taxation",
            "social_transformation",
        ),
        "multi_domain": (
            "political_administrative",
            "strategy_methods",
            "organization_structure",
        ),
    }.get(domain, ("progress", "result"))
    selected = (
        *intent_facets.get(
            evolution_intent,
            intent_facets["evolution_over_time"],
        ),
        *domain_priority,
    )
    return tuple(
        facet for facet in dict.fromkeys(selected)
        if facet in candidates
    )[:6]


def _fallback_comparison_facets(
    domain: str,
    intent: str,
    explicit_facets: tuple[str, ...],
) -> tuple[str, ...]:
    if explicit_facets:
        return explicit_facets
    focused = {
        "cause_comparison": ("cause", "comparison_context"),
        "consequence_comparison": ("result", "significance"),
        "effectiveness_comparison": (
            "objectives_consequences", "result", "limitations",
        ),
        "significance_comparison": ("significance", "result"),
        "continuity_change": ("continuity_change", "comparison_context", "result"),
        "role_comparison": ("role_contribution", "representative_events", "significance"),
        "interpretation_comparison": (
            "source_perspective", "comparison_context", "limitations",
        ),
        "judgement": ("objectives_consequences", "result", "significance"),
    }
    if intent in focused:
        return focused[intent]
    candidates = comparison_facets_for_domain(domain)
    return candidates[:7]


def build_query_plan(question: str) -> QueryPlan:
    """Build a retrieval plan before seeing candidates to avoid retrieval anchoring."""
    cleaned = _clean_phrase(question)
    key = _search_key(cleaned)
    years = [int(value) for value in re.findall(r"\b(?:1[0-9]{3}|20[0-9]{2})\b", question)]
    explicit_date_range = ""
    if years:
        explicit_date_range = (
            str(years[0]) if len(years) == 1
            else f"{min(years)}–{max(years)}"
        )

    asks_comparison = any(phrase in key for phrase in (
        "so sanh",
        "giong nhau va khac nhau",
        "giong nhau",
        "diem giong nhau",
        "diem chung",
        "khac biet",
        "diem khac nhau",
        "khac nhau nhu the nao",
        "hieu qua hon",
    ))
    if asks_comparison:
        comparison_subjects: tuple[str, ...] = ()
        comparison_date_ranges: tuple[tuple[int, int], ...] = ()
        comparison_domain = comparison_domain_for_question(cleaned)
        comparison_intent = _comparison_intent(key)
        explicit_facets = _explicit_comparison_facets(key, comparison_domain)
        intent_facet_scope = {
            "cause_comparison": {"cause"},
            "consequence_comparison": {"result", "significance"},
            "significance_comparison": {"significance"},
            "continuity_change": {"continuity_change"},
            "role_comparison": {"role_contribution"},
            "interpretation_comparison": {"source_perspective"},
        }
        if (
            explicit_facets
            and comparison_intent in intent_facet_scope
            and not set(explicit_facets).issubset(
                intent_facet_scope[comparison_intent]
            )
        ):
            comparison_intent = "focused_facets"
        if explicit_facets and comparison_intent == "similarities_differences":
            comparison_intent = "focused_facets"
        comparison_facets = _fallback_comparison_facets(
            comparison_domain,
            comparison_intent,
            explicit_facets,
        )
        (
            include_similarities,
            include_differences,
            include_explanation,
            include_judgement,
        ) = _comparison_requirements(comparison_intent)
        if (
            "hai cuoc khai thac thuoc dia" in key
            or (
                "khai thac thuoc dia lan thu nhat" in key
                and "khai thac thuoc dia lan thu hai" in key
            )
        ):
            comparison_subjects = (
                "Cuộc khai thác thuộc địa lần thứ nhất của Pháp",
                "Cuộc khai thác thuộc địa lần thứ hai của Pháp",
            )
            comparison_date_ranges = ((1897, 1918), (1919, 1930))
        elif (
            "chien tranh dac biet" in key
            and "chien tranh cuc bo" in key
        ):
            comparison_subjects = (
                "Chiến tranh đặc biệt",
                "Chiến tranh cục bộ",
            )
            comparison_date_ranges = ((1961, 1965), (1965, 1968))
        else:
            comparison_match = re.search(
                r"\bso\s+sánh(?:\s+giữa)?\s+(.+?)\s+(?:và|với)\s+(.+?)"
                r"(?:\s+về\b.+|[?.!]|$)",
                cleaned,
                re.IGNORECASE,
            )
            if comparison_match:
                comparison_subjects = tuple(
                    _clean_comparison_subject(value)
                    for value in comparison_match.groups()
                )
            else:
                between_match = re.search(
                    r"\bgiữa\s+(.+?)\s+(?:và|với)\s+(.+?)"
                    r"(?:\s+(?:là|ở|về)\b|[?.!]|$)",
                    cleaned,
                    re.IGNORECASE,
                )
                if between_match:
                    comparison_subjects = tuple(
                        _clean_comparison_subject(value)
                        for value in between_match.groups()
                    )

        return QueryPlan(
            intent="comparison",
            subject=cleaned,
            required_facets=comparison_facets,
            answer_structure=_comparison_answer_structure(comparison_intent),
            inferred_scope=(
                " ↔ ".join(comparison_subjects)
                if comparison_subjects
                else cleaned
            ),
            explicit_date_range=explicit_date_range,
            allow_inferred_date_recovery=False,
            comparison_subjects=comparison_subjects,
            comparison_date_ranges=comparison_date_ranges,
            comparison_domain=comparison_domain,
            comparison_object_types=(
                tuple(comparison_domain for _ in comparison_subjects)
            ),
            comparison_intent=comparison_intent,
            explicit_facets=explicit_facets,
            include_similarities=include_similarities,
            include_differences=include_differences,
            include_explanation=include_explanation,
            include_judgement=include_judgement,
        )

    is_colonial_policy = (
        "chinh sach" in key
        and ("thuoc dia" in key or "thuc dan phap" in key or "cua phap" in key)
    )
    asks_evolution = any(phrase in key for phrase in (
        "thay doi nhu the nao",
        "thay doi ra sao",
        "qua cac giai doan",
        "qua cac thoi ky",
        "theo thoi gian",
        "tien trien",
        "tiep noi va thay doi",
        "van duoc duy tri",
        "buoc ngoat",
        "muc do thay doi",
        "truoc va sau",
        "ke thua va phat trien",
        "chia thanh cac giai doan",
        "chia thanh nhung giai doan",
    )) or (
        ("vi sao" in key or "tai sao" in key or "nguyen nhan" in key)
        and ("thay doi" in key or "chuyen sang" in key)
    )

    if asks_evolution:
        evolution_intent = _evolution_intent(key)
        evolution_domain = comparison_domain_for_question(cleaned)
        evolution_subject = _clean_evolution_subject(cleaned)
        # "Chính sách thuộc địa" là chính sách cai trị nếu câu hỏi không giới
        # hạn riêng vào kinh tế.
        if is_colonial_policy and not any(marker in key for marker in (
            "kinh te", "khai thac thuoc dia lan", "nong nghiep",
            "cong nghiep", "thuong nghiep",
        )):
            evolution_domain = "political_administration"
        return QueryPlan(
            intent="historical_evolution",
            subject=evolution_subject,
            required_facets=_evolution_fallback_facets(
                evolution_intent,
                evolution_domain,
            ),
            optional_facets=(
                POLICY_OVERVIEW_FACETS
                if is_colonial_policy
                else ()
            ),
            answer_structure=f"evolution_{evolution_intent}",
            inferred_scope=(
                "Việt Nam" if is_colonial_policy else evolution_subject
            ),
            explicit_date_range=explicit_date_range,
            allow_inferred_date_recovery=False,
            evolution_intent=evolution_intent,
            evolution_domain=evolution_domain,
        )

    if is_colonial_policy:
        return QueryPlan(
            intent="overview",
            subject=cleaned,
            required_facets=POLICY_OVERVIEW_FACETS,
            optional_facets=("historical_evolution",),
            answer_structure="facets_first_then_timeline",
            inferred_scope="Việt Nam",
            explicit_date_range=explicit_date_range,
            allow_inferred_date_recovery=False,
        )

    asks_numbered_phases = any(phrase in key for phrase in (
        "qua nhung dot",
        "qua cac dot",
        "gom may dot",
        "gom nhung dot",
        "cac dot nao",
        "tung dot",
        "dien ra may dot",
        "dien ra qua nhung giai doan",
        "dien ra qua cac giai doan",
        "gom may giai doan",
        "gom nhung giai doan",
        "cac giai doan nao",
        "tung giai doan",
    ))
    if asks_numbered_phases:
        return QueryPlan(
            intent="event_phases",
            subject=cleaned,
            required_facets=("progress", "result"),
            optional_facets=("leadership",),
            answer_structure="phases_with_dates_then_result",
            explicit_date_range=explicit_date_range,
        )

    requested_facets: list[str] = []
    for markers, facet in (
        (("ket qua", "thang loi"), "result"),
        (("y nghia", "anh huong", "tac dong"), "significance"),
        (("nguyen nhan", "boi canh"), "cause"),
        (("lanh dao", "chi huy"), "leadership"),
        (("luc luong", "tham chien"), "forces"),
    ):
        if any(marker in key for marker in markers):
            requested_facets.append(facet)
    if requested_facets:
        subject = analyze_query(cleaned).primary_entity or cleaned
        answer_structure = (
            "result_then_significance"
            if requested_facets == ["result", "significance"]
            else "facets"
        )
        return QueryPlan(
            intent="event_outcome" if "result" in requested_facets else "history_lookup",
            subject=subject,
            required_facets=tuple(requested_facets),
            answer_structure=answer_structure,
            explicit_date_range=explicit_date_range,
        )

    return QueryPlan(
        subject=cleaned,
        explicit_date_range=explicit_date_range,
    )
