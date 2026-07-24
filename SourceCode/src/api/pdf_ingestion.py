import hashlib
import io
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import get_settings
from .entity_resolution import normalize_search_key
from .history_facets import infer_history_facets


@dataclass
class PdfChunk:
    chunk_id: str
    text: str
    page_start: int
    page_end: int
    content_hash: str
    sequence: int
    page_chunk_index: int
    char_count: int
    heading: str
    word_count: int = 0
    facets: tuple[str, ...] = ()
    year_start: int | None = None
    year_end: int | None = None


@dataclass
class PdfPage:
    page_id: str
    page_number: int
    char_count: int
    content_hash: str
    preview: str
    extraction_status: str
    chunk_count: int
    footnote_removed: bool = False
    removed_footnote_chars: int = 0
    running_header_removed: bool = False
    removed_running_header_chars: int = 0


@dataclass
class ParsedPdf:
    page_count: int
    text_page_count: int
    content_start_page: int
    content_end_page: int
    skipped_page_count: int
    boundary_detection: str
    footnote_page_count: int
    removed_footnote_chars: int
    running_header_page_count: int
    removed_running_header_chars: int
    chunks: list[PdfChunk]
    pages: list[PdfPage]
    duplicate_chunk_count: int
    ocr_applied: bool
    ocr_languages: str
    ocr_engine: str
    ocr_failed_pages: list[int]


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def infer_chunk_metadata(
    text: str,
) -> tuple[int, tuple[str, ...], int | None, int | None]:
    facets = infer_history_facets(text)
    years = [
        int(value)
        for value in re.findall(r"(?<!\d)(1[0-9]{3}|20[0-9]{2})(?!\d)", text)
    ]
    return (
        len(re.findall(r"\S+", text)),
        facets,
        min(years) if years else None,
        max(years) if years else None,
    )


def extract_heading(text: str) -> str:
    lines = [_normalize_line(line) for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return ""
    candidate = lines[0]
    if not 3 <= len(candidate) <= 120 or len(candidate.split()) > 18:
        return ""
    if candidate.endswith((".", ",", ";", "?", "!")):
        return ""

    letters = [character for character in candidate if character.isalpha()]
    uppercase_ratio = (
        sum(character.isupper() for character in letters) / len(letters)
        if letters
        else 0.0
    )
    has_section_marker = bool(re.match(
        r"^(?:chương|phần|mục|tiết|[IVXLC]+[.)]|[0-9]+(?:\.[0-9]+)*[.)])\s*",
        candidate,
        re.IGNORECASE,
    ))
    return candidate if has_section_marker or uppercase_ratio >= 0.65 else ""


def _clean_pages(raw_pages: list[str]) -> list[str]:
    normalized_pages = []
    line_frequency: Counter[str] = Counter()

    for text in raw_pages:
        lines = [_normalize_line(line) for line in text.splitlines()]
        lines = [line for line in lines if line]
        normalized_pages.append(lines)
        line_frequency.update(set(line for line in lines if len(line) <= 160))

    repeat_threshold = max(3, round(len(raw_pages) * 0.35))
    repeated_lines = {
        line for line, count in line_frequency.items()
        if count >= repeat_threshold
    }

    cleaned_pages = []
    for lines in normalized_pages:
        cleaned = [line for line in lines if line not in repeated_lines]
        cleaned_pages.append("\n".join(cleaned).strip())
    return cleaned_pages


_FOOTNOTE_START_PATTERN = re.compile(
    r"^\s*(?P<number>\d{1,2})\s*[.)]\s+\S"
)
_FOOTNOTE_CITATION_PATTERN = re.compile(
    r"(?:\bnxb\b|\bsdd\b|\btr\s+\d|\btrang\s+\d|\btap\s+\d|"
    r"\bquyen\s+\d|\bvan\s+kien\b|\bnien\s+giam\b|"
    r"\btong\s+cuc\s+thong\s+ke\b|\btai\s+lieu\b|"
    r"\bhttps?\b|\bwww\b|\b(?:18|19|20)\d{2}\b)"
)


def strip_page_footnotes(text: str) -> tuple[str, int]:
    """Remove a high-confidence numbered footnote block at a page bottom."""
    lines = [_normalize_line(line) for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        return text, 0

    starts: list[tuple[int, int]] = []
    for line_index, line in enumerate(lines):
        match = _FOOTNOTE_START_PATTERN.match(line)
        if match:
            starts.append((line_index, int(match.group("number"))))

    for start_position, (line_index, number) in enumerate(starts):
        if number != 1:
            continue
        body = "\n".join(lines[:line_index]).strip()
        footnote = "\n".join(lines[line_index:]).strip()
        if len(body) < 280 or not footnote:
            continue

        # Footnotes belong near the page bottom and should not consume most of
        # the page. These guards prevent ordinary numbered historical lists
        # from being mistaken for citations.
        line_ratio = line_index / len(lines)
        char_ratio = len(body) / max(1, len(body) + len(footnote))
        if line_ratio < 0.55 or char_ratio < 0.58:
            continue

        suffix_numbers = [
            candidate_number
            for _, candidate_number in starts[start_position:]
        ]
        sequential = (
            len(suffix_numbers) >= 2
            and suffix_numbers[0] == 1
            and all(
                current > previous
                for previous, current in zip(
                    suffix_numbers,
                    suffix_numbers[1:],
                    strict=False,
                )
            )
        )
        has_citation_signal = bool(
            _FOOTNOTE_CITATION_PATTERN.search(
                normalize_search_key(footnote)
            )
        )
        if not has_citation_signal:
            continue
        if not sequential and line_ratio < 0.68:
            continue

        return body, len(footnote)

    return text, 0


def _strip_page_footnotes(
    cleaned_pages: list[str],
    start_page: int = 1,
    end_page: int | None = None,
) -> tuple[list[str], dict[int, int]]:
    stripped_pages: list[str] = []
    removed_by_page: dict[int, int] = {}
    last_page = end_page if end_page is not None else len(cleaned_pages)
    for page_number, text in enumerate(cleaned_pages, start=1):
        if not start_page <= page_number <= last_page:
            stripped_pages.append(text)
            continue
        stripped, removed_chars = strip_page_footnotes(text)
        stripped_pages.append(stripped)
        if removed_chars > 0:
            removed_by_page[page_number] = removed_chars
    return stripped_pages, removed_by_page


_RUNNING_HEADER_END_PATTERN = re.compile(
    r"(?:\.{2,}|…)[,;:]?\s*$"
)
_GRAPHIC_RULE_PATTERN = re.compile(r"^[\s._—–-]{5,}$")
_BOOK_RUNNING_HEADER_PATTERN = re.compile(
    r"^lich\s+su\s+viet\s+nam\s+tap\s+\d+\b"
)


def _looks_like_chapter_word(value: str) -> bool:
    if value == "chuong":
        return True
    # Common OCR variants seen in the uploaded books: Chuomg, Chưcmg,
    # Chucmg. Requiring both the prefix and final "g" keeps this narrow.
    return (
        value.startswith("chu")
        and value.endswith("g")
        and 5 <= len(value) <= 7
    )


def strip_page_running_header(text: str) -> tuple[str, int]:
    """Remove a chapter/book running header while preserving real headings."""
    lines = [_normalize_line(line) for line in text.splitlines() if line.strip()]
    if len(lines) < 5:
        return text, 0

    first_line = lines[0]
    first_key = normalize_search_key(first_line)
    key_words = first_key.split()
    body = "\n".join(lines[1:]).strip()
    if len(body) < 280:
        return text, 0

    looks_like_chapter_header = (
        len(key_words) >= 4
        and _looks_like_chapter_word(key_words[0])
        and bool(_RUNNING_HEADER_END_PATTERN.search(first_line))
    )
    looks_like_book_header = bool(
        _BOOK_RUNNING_HEADER_PATTERN.match(first_key)
    )
    if not looks_like_chapter_header and not looks_like_book_header:
        return text, 0

    removed_lines = [first_line]
    remaining_lines = lines[1:]
    if remaining_lines and _GRAPHIC_RULE_PATTERN.fullmatch(
        remaining_lines[0]
    ):
        removed_lines.append(remaining_lines.pop(0))

    return "\n".join(remaining_lines).strip(), len("\n".join(removed_lines))


def _strip_page_running_headers(
    cleaned_pages: list[str],
    start_page: int = 1,
    end_page: int | None = None,
) -> tuple[list[str], dict[int, int]]:
    stripped_pages: list[str] = []
    removed_by_page: dict[int, int] = {}
    last_page = end_page if end_page is not None else len(cleaned_pages)
    for page_number, text in enumerate(cleaned_pages, start=1):
        if not start_page <= page_number <= last_page:
            stripped_pages.append(text)
            continue
        stripped, removed_chars = strip_page_running_header(text)
        stripped_pages.append(stripped)
        if removed_chars > 0:
            removed_by_page[page_number] = removed_chars
    return stripped_pages, removed_by_page


@dataclass(frozen=True)
class ContentPageRange:
    start_page: int
    end_page: int
    detection: str


_FIRST_CHAPTER_PATTERN = re.compile(
    r"^chuong\s+(?:i|1|mot|thu\s+nhat)(?:\b|$)"
)
_REFERENCES_PATTERN = re.compile(
    r"^(?:tai\s+lieu\s+tham\s+khao|thu\s+muc\s+tham\s+khao|"
    r"bibliography|references)(?:\b|$)"
)


def _leading_heading_keys(text: str, limit: int = 6) -> list[str]:
    """Return normalized lines near the top, where section headings belong."""
    keys: list[str] = []
    for line in text.splitlines():
        key = normalize_search_key(line)
        if not key:
            continue
        keys.append(key)
        if len(keys) >= limit:
            break
    if not keys:
        compact = normalize_search_key(text)
        if compact:
            keys.append(compact[:240])
    return keys


def detect_content_page_range(cleaned_pages: list[str]) -> ContentPageRange:
    """Find the historical body without sending front/back matter to OpenAI.

    The detector deliberately accepts only headings at the top of a page.
    This avoids matching "Chương I" or "Tài liệu tham khảo" inside a table of
    contents, author biography, footnote, or ordinary paragraph.
    """
    page_count = len(cleaned_pages)
    if page_count == 0:
        return ContentPageRange(1, 0, "empty")

    first_chapter_index: int | None = None
    references_index: int | None = None
    for page_index, text in enumerate(cleaned_pages):
        heading_keys = _leading_heading_keys(text)
        if (
            first_chapter_index is None
            and any(_FIRST_CHAPTER_PATTERN.match(key) for key in heading_keys[:3])
        ):
            first_chapter_index = page_index
        if (
            first_chapter_index is not None
            and page_index > first_chapter_index
            and any(_REFERENCES_PATTERN.match(key) for key in heading_keys[:3])
        ):
            references_index = page_index
            break

    start_index = first_chapter_index if first_chapter_index is not None else 0
    end_exclusive = references_index if references_index is not None else page_count

    # Never accept a range that would leave too little material. A false
    # positive must fall back to the whole document rather than silently lose
    # historical content.
    if end_exclusive - start_index < min(3, page_count):
        return ContentPageRange(1, page_count, "full_document")

    if first_chapter_index is not None and references_index is not None:
        detection = "chapter_1_to_references"
    elif first_chapter_index is not None:
        detection = "chapter_1_to_end"
    elif references_index is not None:
        detection = "start_to_references"
    else:
        detection = "full_document"

    return ContentPageRange(
        start_page=start_index + 1,
        end_page=end_exclusive,
        detection=detection,
    )


def _extract_pages(pdf_bytes: bytes) -> tuple[int, list[str]]:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise ValueError("PDF được mã hóa và không thể đọc để index.") from exc
        return len(reader.pages), [(page.extract_text() or "") for page in reader.pages]
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Không thể đọc cấu trúc PDF: {exc}") from exc


def _has_indexable_text(raw_pages: list[str]) -> bool:
    return any(len(re.sub(r"\s+", " ", page).strip()) >= 80 for page in raw_pages)


def _ocr_pdf_with_ocrmypdf(pdf_bytes: bytes) -> list[str]:
    settings = get_settings()
    executable = shutil.which("ocrmypdf")
    if not executable:
        raise ValueError(
            "PDF là bản scan nhưng máy chủ chưa có OCRmyPDF/Tesseract. "
            "Hãy cài ocrmypdf, tesseract-ocr và tesseract-ocr-vie."
        )

    with tempfile.TemporaryDirectory(prefix="history-pdf-ocr-") as temp_dir:
        input_path = Path(temp_dir) / "input.pdf"
        output_path = Path(temp_dir) / "output-ocr.pdf"
        input_path.write_bytes(pdf_bytes)
        command = [
            executable,
            "--skip-text",
            "--rotate-pages",
            "--output-type", "pdf",
            "--optimize", "0",
            "--jobs", str(max(1, settings.ai_ocr_jobs)),
            "--language", settings.ai_ocr_languages,
            "--tesseract-timeout", str(max(30, settings.ai_ocr_page_timeout_seconds)),
            str(input_path),
            str(output_path),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=max(300, settings.ai_ocr_process_timeout_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError(
                "OCR vượt quá thời gian cho phép. Hãy tăng "
                "AI_OCR_PROCESS_TIMEOUT_SECONDS hoặc OCR tài liệu trước khi tải lên."
            ) from exc

        if result.returncode != 0 or not output_path.exists():
            detail = (result.stderr or result.stdout or "OCR không tạo được PDF đầu ra.").strip()
            detail = re.sub(r"\s+", " ", detail)[-800:]
            raise ValueError(f"OCR PDF thất bại: {detail}")
        _, pages = _extract_pages(output_path.read_bytes())
        return pages


def _ocr_pages_with_poppler(
    pdf_bytes: bytes,
    page_count: int,
) -> tuple[list[str], list[int]]:
    """Fallback for malformed PDFs that Ghostscript cannot rasterize."""
    settings = get_settings()
    pdftoppm = shutil.which("pdftoppm")
    tesseract = shutil.which("tesseract")
    if not pdftoppm or not tesseract:
        raise ValueError(
            "OCR dự phòng cần Poppler (pdftoppm) và Tesseract nhưng máy chủ chưa cài đủ."
        )

    with tempfile.TemporaryDirectory(prefix="history-pdf-page-ocr-") as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "input.pdf"
        input_path.write_bytes(pdf_bytes)

        def extract_page(page_number: int) -> tuple[int, str, str]:
            output_prefix = temp_path / f"page-{page_number:06d}"
            image_path = output_prefix.with_suffix(".jpg")
            raster_command = [
                pdftoppm,
                "-f", str(page_number),
                "-l", str(page_number),
                "-r", str(max(150, settings.ai_ocr_fallback_dpi)),
                "-jpeg",
                "-gray",
                "-singlefile",
                str(input_path),
                str(output_prefix),
            ]
            try:
                raster = subprocess.run(
                    raster_command,
                    capture_output=True,
                    text=True,
                    timeout=max(30, settings.ai_ocr_page_timeout_seconds),
                    check=False,
                )
                if raster.returncode != 0 or not image_path.exists():
                    detail = (raster.stderr or raster.stdout or "pdftoppm không tạo được ảnh").strip()
                    return page_number, "", detail[-300:]

                ocr = subprocess.run(
                    [
                        tesseract,
                        str(image_path),
                        "stdout",
                        "-l", settings.ai_ocr_languages,
                        "--psm", "3",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=max(30, settings.ai_ocr_page_timeout_seconds),
                    check=False,
                )
                if ocr.returncode != 0:
                    detail = (ocr.stderr or "Tesseract không đọc được trang").strip()
                    return page_number, "", detail[-300:]
                return page_number, ocr.stdout or "", ""
            except subprocess.TimeoutExpired:
                return page_number, "", "quá thời gian OCR cho phép"
            finally:
                image_path.unlink(missing_ok=True)

        pages = ["" for _ in range(page_count)]
        failed_pages: list[int] = []
        worker_count = min(max(1, settings.ai_ocr_jobs), max(1, page_count))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(extract_page, page_number): page_number
                for page_number in range(1, page_count + 1)
            }
            for future in as_completed(futures):
                page_number, text, error = future.result()
                pages[page_number - 1] = text
                if error:
                    failed_pages.append(page_number)

        failed_pages.sort()
        if not _has_indexable_text(pages):
            failed_preview = ", ".join(map(str, failed_pages[:12])) or "tất cả"
            raise ValueError(
                "OCR dự phòng không trích xuất được văn bản. "
                f"Trang lỗi: {failed_preview}."
            )
        return pages, failed_pages


def parse_pdf(pdf_bytes: bytes, source_id: str) -> ParsedPdf:
    settings = get_settings()
    max_bytes = settings.ai_max_pdf_size_mb * 1024 * 1024
    if len(pdf_bytes) > max_bytes:
        raise ValueError(f"PDF vượt quá giới hạn {settings.ai_max_pdf_size_mb} MB.")
    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("Tệp tải lên không phải PDF hợp lệ.")

    page_count, raw_pages = _extract_pages(pdf_bytes)
    ocr_applied = False
    ocr_engine = ""
    ocr_failed_pages: list[int] = []
    if not _has_indexable_text(raw_pages):
        if not settings.ai_enable_pdf_ocr:
            raise ValueError(
                "PDF là bản scan và không có lớp văn bản. "
                "Hãy bật AI_ENABLE_PDF_OCR hoặc OCR tài liệu trước khi index."
            )
        try:
            raw_pages = _ocr_pdf_with_ocrmypdf(pdf_bytes)
            ocr_engine = "ocrmypdf"
        except ValueError as primary_error:
            try:
                raw_pages, ocr_failed_pages = _ocr_pages_with_poppler(pdf_bytes, page_count)
                ocr_engine = "poppler+tesseract"
            except ValueError as fallback_error:
                raise ValueError(
                    f"{primary_error} OCR dự phòng cũng thất bại: {fallback_error}"
                ) from fallback_error
        ocr_applied = True

    cleaned_pages = _clean_pages(raw_pages)
    content_range = detect_content_page_range(cleaned_pages)
    cleaned_pages, running_headers_by_page = _strip_page_running_headers(
        cleaned_pages,
        start_page=content_range.start_page,
        end_page=content_range.end_page,
    )
    cleaned_pages, footnotes_by_page = _strip_page_footnotes(
        cleaned_pages,
        start_page=content_range.start_page,
        end_page=content_range.end_page,
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.ai_chunk_size,
        chunk_overlap=settings.ai_chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " "],
        length_function=len,
    )

    chunks: list[PdfChunk] = []
    pages: list[PdfPage] = []
    ocr_failed_page_set = set(ocr_failed_pages)
    seen_hashes: set[str] = set()
    duplicate_count = 0

    for page_index, page_text in enumerate(cleaned_pages, start=1):
        removed_footnote_chars = footnotes_by_page.get(page_index, 0)
        removed_running_header_chars = running_headers_by_page.get(
            page_index,
            0,
        )
        if page_index < content_range.start_page:
            pages.append(PdfPage(
                page_id=f"{source_id}-p{page_index}",
                page_number=page_index,
                char_count=len(page_text),
                content_hash=hashlib.sha256(
                    page_text.casefold().encode("utf-8")
                ).hexdigest(),
                preview=re.sub(r"\s+", " ", page_text).strip()[:360],
                extraction_status="excluded_front_matter",
                chunk_count=0,
                footnote_removed=removed_footnote_chars > 0,
                removed_footnote_chars=removed_footnote_chars,
                running_header_removed=removed_running_header_chars > 0,
                removed_running_header_chars=removed_running_header_chars,
            ))
            continue
        if page_index > content_range.end_page:
            pages.append(PdfPage(
                page_id=f"{source_id}-p{page_index}",
                page_number=page_index,
                char_count=len(page_text),
                content_hash=hashlib.sha256(
                    page_text.casefold().encode("utf-8")
                ).hexdigest(),
                preview=re.sub(r"\s+", " ", page_text).strip()[:360],
                extraction_status="excluded_back_matter",
                chunk_count=0,
                footnote_removed=removed_footnote_chars > 0,
                removed_footnote_chars=removed_footnote_chars,
                running_header_removed=removed_running_header_chars > 0,
                removed_running_header_chars=removed_running_header_chars,
            ))
            continue
        if len(page_text) < 80:
            pages.append(PdfPage(
                page_id=f"{source_id}-p{page_index}",
                page_number=page_index,
                char_count=len(page_text),
                content_hash=hashlib.sha256(page_text.casefold().encode("utf-8")).hexdigest(),
                preview=page_text[:360],
                extraction_status=(
                    "ocr_failed" if page_index in ocr_failed_page_set else "empty"
                ),
                chunk_count=0,
                footnote_removed=removed_footnote_chars > 0,
                removed_footnote_chars=removed_footnote_chars,
                running_header_removed=removed_running_header_chars > 0,
                removed_running_header_chars=removed_running_header_chars,
            ))
            continue
        page_chunk_count = 0
        for part_index, text in enumerate(splitter.split_text(page_text)):
            normalized = re.sub(r"\s+", " ", text).strip()
            if len(normalized) < 80:
                continue
            content_hash = hashlib.sha256(normalized.casefold().encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes.add(content_hash)
            page_chunk_count += 1
            word_count, facets, year_start, year_end = infer_chunk_metadata(
                normalized
            )
            chunks.append(PdfChunk(
                chunk_id=f"{source_id}-p{page_index}-c{part_index + 1}",
                text=normalized,
                page_start=page_index,
                page_end=page_index,
                content_hash=content_hash,
                sequence=len(chunks) + 1,
                page_chunk_index=part_index + 1,
                char_count=len(normalized),
                heading=extract_heading(text),
                word_count=word_count,
                facets=facets,
                year_start=year_start,
                year_end=year_end,
            ))
        pages.append(PdfPage(
            page_id=f"{source_id}-p{page_index}",
            page_number=page_index,
            char_count=len(page_text),
            content_hash=hashlib.sha256(page_text.casefold().encode("utf-8")).hexdigest(),
            preview=re.sub(r"\s+", " ", page_text).strip()[:360],
            extraction_status=("ocr" if ocr_applied else "ready") if page_chunk_count else "empty",
            chunk_count=page_chunk_count,
            footnote_removed=removed_footnote_chars > 0,
            removed_footnote_chars=removed_footnote_chars,
            running_header_removed=removed_running_header_chars > 0,
            removed_running_header_chars=removed_running_header_chars,
        ))

    if not chunks:
        raise ValueError(
            "OCR đã chạy nhưng vẫn không trích xuất được văn bản đủ chất lượng để index."
            if ocr_applied
            else "Không trích xuất được văn bản từ PDF."
        )

    return ParsedPdf(
        page_count=page_count,
        text_page_count=sum(
            1
            for page_number, page in enumerate(cleaned_pages, start=1)
            if (
                content_range.start_page
                <= page_number
                <= content_range.end_page
                and len(page) >= 80
            )
        ),
        content_start_page=content_range.start_page,
        content_end_page=content_range.end_page,
        skipped_page_count=(
            content_range.start_page - 1
            + page_count - content_range.end_page
        ),
        boundary_detection=content_range.detection,
        footnote_page_count=len(footnotes_by_page),
        removed_footnote_chars=sum(footnotes_by_page.values()),
        running_header_page_count=len(running_headers_by_page),
        removed_running_header_chars=sum(
            running_headers_by_page.values()
        ),
        chunks=chunks,
        pages=pages,
        duplicate_chunk_count=duplicate_count,
        ocr_applied=ocr_applied,
        ocr_languages=settings.ai_ocr_languages if ocr_applied else "",
        ocr_engine=ocr_engine,
        ocr_failed_pages=ocr_failed_pages,
    )
