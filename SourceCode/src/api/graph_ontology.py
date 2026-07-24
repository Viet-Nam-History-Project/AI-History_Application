"""Canonical ontology for the history knowledge graph.

Neo4j relationship types are schema, not natural-language sentences.  Every
ingestion path must pass through this module so synonymous phrases do not
create thousands of one-off relationship types.
"""

from __future__ import annotations

import re
import unicodedata


CANONICAL_RELATIONSHIP_TYPES = frozenset(
    {
        # Document and application structure.
        "HAS_STAGE",
        "HAS_EVENT",
        "HAS_PAGE",
        "HAS_CHUNK",
        "MENTIONS",
        "SUPPORTED_BY",
        # Historical semantics.
        "PART_OF",
        "PRECEDES",
        "SUCCEEDS",
        "OCCURRED_AT",
        "OCCURRED_DURING",
        "PARTICIPATED_IN",
        "LED",
        "COMMANDED",
        "MEMBER_OF",
        "HELD_POSITION",
        "FOUNDED",
        "ALLY_OF",
        "OPPOSED",
        "DEFEATED",
        "AFFECTED",
        "CAUSED",
        "RESULTED_IN",
        "SIGNED",
        "SUPPORTED",
        "RELATED_TO",
    }
)


def ontology_prompt() -> str:
    """Return the canonical relationship vocabulary for extraction prompts."""
    return ", ".join(sorted(CANONICAL_RELATIONSHIP_TYPES))


def _fold(value: str) -> str:
    value = unicodedata.normalize("NFD", value or "")
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    value = value.replace("đ", "d").replace("Đ", "D").upper()
    return re.sub(r"[^A-Z0-9]+", "_", value).strip("_")


_EXACT_ALIASES = {
    "THUOC": "PART_OF",
    "LA_MOT_PHAN_CUA": "PART_OF",
    "TRUC_THUOC": "PART_OF",
    "DIEN_RA_TAI": "OCCURRED_AT",
    "XAY_RA_TAI": "OCCURRED_AT",
    "DIEN_RA_TRONG": "OCCURRED_DURING",
    "THAM_GIA": "PARTICIPATED_IN",
    "THAM_CHIEN": "PARTICIPATED_IN",
    "LANH_DAO": "LED",
    "CHI_HUY": "COMMANDED",
    "THANH_VIEN_CUA": "MEMBER_OF",
    "GIU_CHUC_VU": "HELD_POSITION",
    "THANH_LAP": "FOUNDED",
    "SANG_LAP": "FOUNDED",
    "DONG_MINH": "ALLY_OF",
    "DOI_DAU": "OPPOSED",
    "CHONG_LAI": "OPPOSED",
    "DANH_BAI": "DEFEATED",
    "ANH_HUONG_TOI": "AFFECTED",
    "ANH_HUONG_DEN": "AFFECTED",
    "GAY_RA": "CAUSED",
    "DAN_DEN": "RESULTED_IN",
    "KET_QUA_LA": "RESULTED_IN",
    "KY_KET": "SIGNED",
    "KY_HIEP_DINH": "SIGNED",
    "UNG_HO": "SUPPORTED",
    "HO_TRO": "SUPPORTED",
}


_PATTERN_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("HAS", "STAGE"), "HAS_STAGE"),
    (("HAS", "EVENT"), "HAS_EVENT"),
    (("HAS", "PAGE"), "HAS_PAGE"),
    (("HAS", "CHUNK"), "HAS_CHUNK"),
    (("MENTION",), "MENTIONS"),
    (("SUPPORTED", "BY"), "SUPPORTED_BY"),
    (("TRUOC",), "PRECEDES"),
    (("PRECEDE",), "PRECEDES"),
    (("SAU",), "SUCCEEDS"),
    (("SUCCEED",), "SUCCEEDS"),
    (("DIA_DIEM",), "OCCURRED_AT"),
    (("DIEN_RA_TAI",), "OCCURRED_AT"),
    (("XAY_RA_TAI",), "OCCURRED_AT"),
    (("DIEN_RA_TRONG",), "OCCURRED_DURING"),
    (("GIAI_DOAN",), "OCCURRED_DURING"),
    (("THAM_GIA",), "PARTICIPATED_IN"),
    (("THAM_CHIEN",), "PARTICIPATED_IN"),
    (("LANH_DAO",), "LED"),
    (("DAN_DAT",), "LED"),
    (("CHI_HUY",), "COMMANDED"),
    (("TU_LENH",), "COMMANDED"),
    (("THANH_VIEN",), "MEMBER_OF"),
    (("GIU_CHUC",), "HELD_POSITION"),
    (("DAM_NHIEM",), "HELD_POSITION"),
    (("THANH_LAP",), "FOUNDED"),
    (("SANG_LAP",), "FOUNDED"),
    (("DONG_MINH",), "ALLY_OF"),
    (("PHOI_HOP",), "ALLY_OF"),
    (("CHONG",), "OPPOSED"),
    (("DOI_DAU",), "OPPOSED"),
    (("XAM_LUOC",), "OPPOSED"),
    (("DANH_BAI",), "DEFEATED"),
    (("TIEU_DIET",), "DEFEATED"),
    (("CHIEN_THANG",), "DEFEATED"),
    (("ANH_HUONG",), "AFFECTED"),
    (("TAC_DONG",), "AFFECTED"),
    (("GAY_RA",), "CAUSED"),
    (("NGUYEN_NHAN",), "CAUSED"),
    (("DAN_DEN",), "RESULTED_IN"),
    (("KET_QUA",), "RESULTED_IN"),
    (("HE_QUA",), "RESULTED_IN"),
    (("KY", "HIEP_DINH"), "SIGNED"),
    (("KY_KET",), "SIGNED"),
    (("UNG_HO",), "SUPPORTED"),
    (("HO_TRO",), "SUPPORTED"),
    (("VIEN_TRO",), "SUPPORTED"),
    (("THUOC",), "PART_OF"),
    (("MOT_PHAN",), "PART_OF"),
)


def canonicalize_relationship_type(raw_type: str) -> str:
    """Map an extracted phrase or legacy Neo4j type to the fixed ontology."""
    folded = _fold(raw_type)
    if folded in CANONICAL_RELATIONSHIP_TYPES:
        return folded
    if folded in _EXACT_ALIASES:
        return _EXACT_ALIASES[folded]
    for tokens, canonical in _PATTERN_RULES:
        if all(token in folded for token in tokens):
            return canonical
    return "RELATED_TO"


def is_canonical_relationship_type(value: str) -> bool:
    return _fold(value) in CANONICAL_RELATIONSHIP_TYPES
