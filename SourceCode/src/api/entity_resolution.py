"""Conservative canonicalization and alias resolution for historical entities."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


def clean_entity_name(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(
        " \t\n\r.,;:!?()[]{}\"'"
    )


def normalize_search_key(value: str) -> str:
    decomposed = unicodedata.normalize(
        "NFD",
        clean_entity_name(value).casefold().replace("đ", "d"),
    )
    plain = "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )
    return re.sub(r"[^0-9a-z]+", " ", plain).strip()


# This registry intentionally contains only identities that are historically
# unambiguous. Ambiguous role phrases such as "quân Mỹ", "chính phủ Mỹ" and
# "Mỹ" must remain separate entities/types instead of being force-merged.
CURATED_ENTITY_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "PERSON": {
        "Hồ Chí Minh": (
            "Bác Hồ",
            "Chủ tịch Hồ Chí Minh",
            "Nguyễn Ái Quốc",
        ),
        "Võ Nguyên Giáp": (
            "Đại tướng Võ Nguyên Giáp",
            "Tướng Giáp",
        ),
        "Nguyễn Thị Định": (
            "Bà Nguyễn Thị Định",
            "Nữ tướng Nguyễn Thị Định",
        ),
    },
    "EVENT": {
        "Phong trào Đồng Khởi": (
            "Đồng Khởi",
            "Cuộc Đồng Khởi",
            "Cuộc đồng khởi",
            "Phong trào Đồng khởi",
            "Đồng Khởi Bến Tre",
        ),
        "Chiến dịch Điện Biên Phủ": (
            "Chiến thắng Điện Biên Phủ",
            "Điện Biên Phủ",
            "Điện Biên Phủ 1954",
        ),
        "Cách mạng tháng Tám": (
            "Cách mạng Tháng Tám năm 1945",
            "Cách mạng tháng Tám 1945",
        ),
        "Chiến dịch Hồ Chí Minh": (
            "Chiến dịch Hồ Chí Minh lịch sử",
        ),
        "Hiệp định Genève": (
            "Hiệp định Geneva",
            "Hiệp định Geneve",
            "Hiệp định Giơ-ne-vơ",
            "Hiệp định Giơnevơ",
            "Hiệp định Giơnơvơ",
            "Hiệp dịnh Giơnevơ",
            "Hiệp định Genève 1954",
            "Hiệp định Geneva 1954",
            "Hiệp định Giơnevơ 1954",
            "Hiệp định Giơnevơ về Đông Dương",
        ),
        "Hiệp định Paris": (
            "Hiệp định Pari",
            "Hiệp định Pa-ri",
            "Hiệp định Paris 1973",
        ),
    },
    "ORGANIZATION": {
        "Quân đội Nhân dân Việt Nam": (
            "Quân đội nhân dân Việt Nam",
            "QĐND Việt Nam",
            "QĐNDVN",
        ),
        "Mặt trận Dân tộc Giải phóng miền Nam Việt Nam": (
            "Mặt trận Giải phóng miền Nam Việt Nam",
            "Mặt trận Dân tộc Giải phóng miền Nam",
        ),
        "Chính phủ Cách mạng lâm thời Cộng hòa miền Nam Việt Nam": (
            "Chính phủ Cách mạng lâm thời miền Nam Việt Nam",
            "Chính phủ Cách mạng lâm thời",
        ),
        "Quân Giải phóng miền Nam Việt Nam": (
            "Quân Giải phóng miền Nam",
            "Quân Giải phóng",
        ),
    },
    "STATE": {
        "Việt Nam Dân chủ Cộng hòa": (
            "Nước Việt Nam Dân chủ Cộng hòa",
            "VNDCCH",
        ),
        "Việt Nam Cộng hòa": (
            "Chính quyền Việt Nam Cộng hòa",
            "VNCH",
        ),
        "Cộng hòa miền Nam Việt Nam": (
            "Cộng hoà miền Nam Việt Nam",
        ),
    },
}

# Các tên dưới đây có bản chất không mơ hồ trong ontology của ứng dụng. Khi
# model gán nhầm type, pipeline được phép sửa cả tên lẫn type. Không áp dụng
# cơ chế này cho tên vừa có thể là địa danh vừa có thể là sự kiện, chẳng hạn
# "Điện Biên Phủ".
_FORCED_CANONICAL_TYPES = {
    "Hiệp định Genève": "EVENT",
    "Hiệp định Paris": "EVENT",
}
_CURATED_ENTITY_YEAR_RANGES = {
    "Chiến dịch Điện Biên Phủ": (1954, 1954),
    "Cách mạng tháng Tám": (1945, 1945),
    "Chiến dịch Hồ Chí Minh": (1975, 1975),
    "Hiệp định Genève": (1954, 1954),
    "Hiệp định Paris": (1973, 1973),
}


_PERSON_TITLE_PATTERN = re.compile(
    r"^(?:chủ\s+tịch|đại\s+tướng|thượng\s+tướng|trung\s+tướng|"
    r"thiếu\s+tướng|tướng|đồng\s+chí|nữ\s+tướng|bà|ông)\s+",
    re.IGNORECASE,
)
_NON_ENTITY_KEYS = {
    "nhan dan",
    "nhan dan ta",
    "quan va dan ta",
    "chinh phu",
    "quan doi",
    "bo doi",
    "quan chung",
    "chien tranh",
    "cong san",
    "mien nam",
    "mien bac",
}
_QUANTITY_OR_DATE_PATTERN = re.compile(
    r"^\d[\d\s.,/-]*(?:nguoi|quan|binh\s+si|chien\s+si|nam|thang|ngay)?$",
    re.IGNORECASE,
)
_UNAMBIGUOUS_TYPE_PREFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "EVENT",
        (
            "hiep dinh ", "hiep nghi ", "hoi nghi ", "chien dich ", "tran ",
            "cuoc khoi nghia ", "khoi nghia ", "phong trao ",
            "cach mang ", "cuoc tong tien cong ", "tong tien cong ",
        ),
    ),
    (
        "DOCUMENT",
        (
            "sac lenh ", "nghi quyet ", "chi thi ", "tuyen ngon ",
            "bao cao ", "hien phap ", "dao luat ", "luat ",
        ),
    ),
    (
        "LOCATION",
        (
            # Các prefix có dấu được xử lý riêng bên dưới để không nhầm
            # "đảo" với "đạo", hoặc "xã" với "xà/xã hội".
        ),
    ),
    (
        "ORGANIZATION",
        (
            "dang ", "mat tran ", "quan doi ", "uy ban ", "chinh phu ",
            "bo tu lenh ", "bo chi huy ", "vien nghien cuu ",
        ),
    ),
)
_EXPLICIT_LOCATION_PREFIX = re.compile(
    r"^(?:tỉnh|thành\s+phố|huyện|sông|núi|đèo|đảo|quần\s+đảo|"
    r"vĩ\s+tuyến|cứ\s+điểm)\s+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EntityResolution:
    canonical_name: str
    entity_type: str
    aliases: tuple[str, ...]
    search_keys: tuple[str, ...]


def is_valid_entity_surface(name: str) -> bool:
    cleaned = clean_entity_name(name)
    key = normalize_search_key(name)
    if not key or key in _NON_ENTITY_KEYS:
        return False
    if _QUANTITY_OR_DATE_PATTERN.fullmatch(key):
        return False
    if re.fullmatch(r"(?:nam\s+)?(?:1[0-9]{3}|20[0-9]{2})", key):
        return False
    # Entity lịch sử phải có dấu hiệu tên riêng. Danh từ thường viết toàn
    # chữ thường như "cao su", "đường sắt", "địa chủ" là khái niệm/nội dung,
    # không phải node. Alias curated vẫn được giữ dù OCR làm mất hoa/thường.
    curated_keys = globals().get("CURATED_SEARCH_KEYS", set())
    infer_type = globals().get("infer_unambiguous_entity_type")
    if (
        key not in curated_keys
        and not any(character.isupper() for character in cleaned)
        and not (callable(infer_type) and infer_type(cleaned))
    ):
        return False
    return True


def infer_unambiguous_entity_type(name: str) -> str:
    """Infer only types whose naming convention is explicit in Vietnamese."""
    cleaned = clean_entity_name(name)
    if _EXPLICIT_LOCATION_PREFIX.search(cleaned):
        return "LOCATION"
    key = normalize_search_key(name)
    for entity_type, prefixes in _UNAMBIGUOUS_TYPE_PREFIXES:
        if any(key.startswith(prefix) for prefix in prefixes):
            return entity_type
    return ""


def _alias_index() -> dict[tuple[str, str], tuple[str, tuple[str, ...]]]:
    index: dict[tuple[str, str], tuple[str, tuple[str, ...]]] = {}
    for entity_type, entities in CURATED_ENTITY_ALIASES.items():
        for canonical_name, aliases in entities.items():
            all_names = (canonical_name, *aliases)
            for value in all_names:
                index[(entity_type, normalize_search_key(value))] = (
                    canonical_name,
                    aliases,
                )
    return index


ALIAS_INDEX = _alias_index()
CURATED_SEARCH_KEYS = {
    alias_key for (_entity_type, alias_key) in ALIAS_INDEX
}


def _forced_alias_index() -> dict[str, tuple[str, str, tuple[str, ...]]]:
    index: dict[str, tuple[str, str, tuple[str, ...]]] = {}
    for canonical_name, entity_type in _FORCED_CANONICAL_TYPES.items():
        aliases = CURATED_ENTITY_ALIASES[entity_type][canonical_name]
        for value in (canonical_name, *aliases):
            index[normalize_search_key(value)] = (
                canonical_name,
                entity_type,
                aliases,
            )
    return index


FORCED_ALIAS_INDEX = _forced_alias_index()


def expand_known_entity_aliases(name: str) -> tuple[str, ...]:
    """Return all curated spellings for an exact known identity."""
    key = normalize_search_key(name)
    forced = FORCED_ALIAS_INDEX.get(key)
    if forced:
        canonical_name, _entity_type, aliases = forced
        return tuple(dict.fromkeys((canonical_name, *aliases)))

    for (_entity_type, alias_key), (canonical_name, aliases) in ALIAS_INDEX.items():
        if alias_key == key:
            return tuple(dict.fromkeys((canonical_name, *aliases)))
    return ()


def forced_curated_identities() -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """Expose safe type overrides to the Neo4j audit/backfill path."""
    return tuple(
        (
            canonical_name,
            entity_type,
            tuple(CURATED_ENTITY_ALIASES[entity_type][canonical_name]),
        )
        for canonical_name, entity_type in _FORCED_CANONICAL_TYPES.items()
    )


def known_entity_year_range(name: str) -> tuple[int, int] | None:
    """Return a curated time anchor for unambiguous named identities."""
    aliases = expand_known_entity_aliases(name)
    if not aliases:
        return None
    canonical_name = aliases[0]
    return _CURATED_ENTITY_YEAR_RANGES.get(canonical_name)


def resolve_entity_name(name: str, entity_type: str) -> EntityResolution:
    """Resolve known aliases while avoiding speculative fuzzy merging."""
    surface_name = clean_entity_name(name)
    normalized_type = str(entity_type or "").upper()
    surface_key = normalize_search_key(surface_name)
    forced_lookup = FORCED_ALIAS_INDEX.get(surface_key)
    lookup = ALIAS_INDEX.get((normalized_type, surface_key))

    canonical_name = surface_name
    resolved_type = normalized_type
    curated_aliases: tuple[str, ...] = ()
    if forced_lookup:
        canonical_name, resolved_type, curated_aliases = forced_lookup
    elif lookup:
        canonical_name, curated_aliases = lookup
    elif normalized_type == "PERSON":
        without_title = clean_entity_name(
            _PERSON_TITLE_PATTERN.sub("", surface_name)
        )
        # A personal name must retain at least two words after title removal.
        if (
            without_title != surface_name
            and len(normalize_search_key(without_title).split()) >= 2
        ):
            canonical_name = without_title
            second_lookup = ALIAS_INDEX.get(
                (normalized_type, normalize_search_key(without_title))
            )
            if second_lookup:
                canonical_name, curated_aliases = second_lookup
    if not forced_lookup:
        inferred_type = infer_unambiguous_entity_type(canonical_name)
        if inferred_type:
            resolved_type = inferred_type

    unique_aliases: list[str] = []
    ordered_search_keys: list[str] = []
    seen_keys: set[str] = set()
    for alias in (canonical_name, surface_name, *curated_aliases):
        cleaned = clean_entity_name(alias)
        key = normalize_search_key(cleaned)
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        unique_aliases.append(cleaned)
        ordered_search_keys.append(key)

    return EntityResolution(
        canonical_name=canonical_name,
        entity_type=resolved_type,
        aliases=tuple(unique_aliases),
        search_keys=tuple(ordered_search_keys),
    )
