"""
Task 8 — PageIndex vectorless fallback.

Chức năng:
1. Đọc PAGEINDEX_API_KEY.
2. Upload tài liệu standardized lên PageIndex.
3. Cache document ID để tránh upload lại.
4. Search bằng PageIndex khi dense retrieval có confidence thấp.
5. Output SearchResult với retrieval_method="pageindex".
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


ROOT = Path(__file__).parent.parent

PAGEINDEX_API_KEY = os.getenv(
    "PAGEINDEX_API_KEY",
    "",
).strip()

PAGEINDEX_API_BASE = os.getenv(
    "PAGEINDEX_API_BASE",
    "https://api.pageindex.ai",
).rstrip("/")

PAGEINDEX_TIMEOUT = float(
    os.getenv(
        "PAGEINDEX_TIMEOUT",
        "30",
    )
)

STANDARDIZED_DIR = (
    ROOT
    / "data"
    / "standardized"
)

CACHE_PATH = (
    ROOT
    / "pageindex_doc_ids.json"
)

TEMP_PDF_DIR = (
    ROOT
    / "pageindex_pdfs"
)

_SUPPORTED_UPLOAD_SUFFIXES = {
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
}


def _headers() -> dict[str, str]:

    if not PAGEINDEX_API_KEY:
        raise RuntimeError(
            "PAGEINDEX_API_KEY "
            "is not configured in .env"
        )

    return {
        "api_key": PAGEINDEX_API_KEY
    }


def _load_cache() -> dict[
    str,
    dict[str, Any],
]:

    if not CACHE_PATH.exists():
        return {}

    try:

        data = json.loads(
            CACHE_PATH.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(data, dict):
            return data

        return {}

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return {}


def _save_cache(
    cache: dict[str, dict[str, Any]],
) -> None:

    CACHE_PATH.write_text(
        json.dumps(
            cache,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _fingerprint(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:

        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _find_unicode_font() -> str | None:

    candidates = [
        Path(
            "C:/Windows/Fonts/arial.ttf"
        ),
        Path(
            "C:/Windows/Fonts/segoeui.ttf"
        ),
        Path(
            "/usr/share/fonts/"
            "truetype/dejavu/"
            "DejaVuSans.ttf"
        ),
        Path(
            "/usr/share/fonts/"
            "truetype/dejavu/"
            "DejaVuSans.ttf"
        ),
        Path(
            "/Library/Fonts/"
            "Arial Unicode.ttf"
        ),
        Path(
            "/System/Library/Fonts/"
            "Supplemental/"
            "Arial Unicode.ttf"
        ),
    ]

    for candidate in candidates:

        if candidate.exists():
            return str(candidate)

    return None


def _markdown_to_pdf(
    source: Path,
) -> Path:
    """
    Chuyển Markdown sang PDF tạm
    để PageIndex upload.
    """

    from fpdf import FPDF

    TEMP_PDF_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    relative = (
        source.relative_to(
            STANDARDIZED_DIR
        )
        .as_posix()
        .replace("/", "__")
    )

    output = (
        TEMP_PDF_DIR
        / f"{Path(relative).stem}.pdf"
    )

    text = source.read_text(
        encoding="utf-8",
        errors="replace",
    )

    pdf = FPDF()

    pdf.set_auto_page_break(
        auto=True,
        margin=15,
    )

    pdf.add_page()

    font_path = (
        _find_unicode_font()
    )

    if font_path:

        pdf.add_font(
            "PageIndexUnicode",
            fname=font_path,
        )

        pdf.set_font(
            "PageIndexUnicode",
            size=11,
        )

    else:

        pdf.set_font(
            "Helvetica",
            size=11,
        )

        text = (
            text
            .encode(
                "latin-1",
                "replace",
            )
            .decode(
                "latin-1"
            )
        )

    for line in (
        text.splitlines()
        or [""]
    ):

        clean = re.sub(
            r"^\s{0,3}#{1,6}\s+",
            "",
            line,
        ).strip()

        if not clean:

            pdf.ln(4)

            continue

        pdf.multi_cell(
            0,
            6,
            clean,
        )

    pdf.output(
        str(output)
    )

    return output


def _iter_standardized_files() -> list[Path]:

    if not STANDARDIZED_DIR.exists():
        return []

    return sorted(
        path
        for path
        in STANDARDIZED_DIR.rglob("*")
        if (
            path.is_file()
            and not path.name.startswith(".")
        )
    )


def _prepare_upload_file(
    source: Path,
) -> Path:

    suffix = (
        source.suffix.lower()
    )

    if suffix == ".md":

        return _markdown_to_pdf(
            source
        )

    if (
        suffix
        in _SUPPORTED_UPLOAD_SUFFIXES
    ):

        return source

    raise ValueError(
        "Unsupported PageIndex "
        f"source format: {suffix}"
    )


def _upload_one(
    upload_path: Path,
) -> str:

    with upload_path.open(
        "rb"
    ) as handle:

        response = requests.post(
            f"{PAGEINDEX_API_BASE}/doc/",
            headers=_headers(),
            files={
                "file": (
                    upload_path.name,
                    handle,
                    "application/pdf",
                )
            },
            timeout=PAGEINDEX_TIMEOUT,
        )

    response.raise_for_status()

    payload = response.json()

    doc_id = (
        payload.get("doc_id")
        or payload.get("id")
    )

    if (
        not isinstance(doc_id, str)
        or not doc_id
    ):
        raise RuntimeError(
            "PageIndex upload response "
            f"missing doc_id: {payload}"
        )

    return doc_id


def _wait_until_ready(
    doc_id: str,
    timeout_seconds: float = 120.0,
) -> None:

    deadline = (
        time.monotonic()
        + timeout_seconds
    )

    while (
        time.monotonic()
        < deadline
    ):

        response = requests.get(
            (
                f"{PAGEINDEX_API_BASE}"
                f"/doc/{doc_id}/metadata"
            ),
            headers=_headers(),
            timeout=PAGEINDEX_TIMEOUT,
        )

        response.raise_for_status()

        payload = response.json()

        status = str(
            payload.get(
                "status",
                "",
            )
        ).lower()

        if status in {
            "completed",
            "ready",
            "success",
            "succeeded",
        }:
            return

        if status in {
            "failed",
            "error",
        }:
            raise RuntimeError(
                "PageIndex processing "
                f"failed for {doc_id}: "
                f"{payload}"
            )

        time.sleep(2.0)

    raise TimeoutError(
        "Timed out waiting for "
        f"PageIndex document {doc_id}"
    )


def upload_documents() -> None:
    """
    Upload toàn bộ dữ liệu standardized
    lên PageIndex.

    Cache document IDs để tránh upload lại.
    """

    files = (
        _iter_standardized_files()
    )

    if not files:

        print(
            "No standardized documents "
            f"found in {STANDARDIZED_DIR}"
        )

        return

    _headers()

    cache = _load_cache()

    changed = False

    for source in files:

        relative = (
            source
            .relative_to(
                STANDARDIZED_DIR
            )
            .as_posix()
        )

        fingerprint = (
            _fingerprint(source)
        )

        cached = cache.get(
            relative,
            {},
        )

        if (
            cached.get("sha256")
            == fingerprint
            and cached.get("doc_id")
        ):

            print(
                f"Cached: {relative} "
                f"-> {cached['doc_id']}"
            )

            continue

        try:

            upload_path = (
                _prepare_upload_file(
                    source
                )
            )

            doc_id = _upload_one(
                upload_path
            )

            cache[relative] = {
                "doc_id": doc_id,
                "sha256": fingerprint,
                "source": source.name,
                "title": source.stem,
                "doc_type": (
                    "legal"
                    if "legal"
                    in source.parts
                    else "news"
                ),
                "url": None,
            }

            changed = True

            print(
                f"Uploaded: {relative} "
                f"-> {doc_id}"
            )

        except Exception as error:

            print(
                "PageIndex upload failed "
                f"for {relative}: "
                f"{error}"
            )

    if changed:
        _save_cache(cache)


def _extract_json_array(
    text: str,
) -> list[dict]:
    """
    Parse JSON array kể cả khi
    model bọc trong markdown fence.
    """

    if not text:
        return []

    cleaned = text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.I,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    candidates = [cleaned]

    match = re.search(
        r"\[[\s\S]*\]",
        cleaned,
    )

    if match:
        candidates.append(
            match.group(0)
        )

    for candidate in candidates:

        try:

            parsed = json.loads(
                candidate
            )

            if isinstance(
                parsed,
                list,
            ):
                return [
                    item
                    for item in parsed
                    if isinstance(
                        item,
                        dict,
                    )
                ]

        except json.JSONDecodeError:
            pass

    return []


def _chat_for_evidence(
    query: str,
    doc_ids: list[str],
    top_k: int,
) -> dict:

    prompt = (
        "Bạn đang làm bước RETRIEVAL, "
        "không phải generation cuối. "
        f"Tìm tối đa {top_k} đoạn bằng "
        "chứng trực tiếp liên quan đến "
        "câu hỏi. "
        "Chỉ trả về JSON array hợp lệ, "
        "không thêm markdown hay giải thích. "
        "Mỗi phần tử có đúng các key: "
        "content, source, title, page. "
        "content phải là đoạn bằng chứng "
        "cụ thể từ tài liệu, không phải "
        "câu trả lời tổng hợp. "
        f"Câu hỏi: {query}"
    )

    response = requests.post(
        (
            f"{PAGEINDEX_API_BASE}"
            "/chat/completions"
        ),
        headers={
            **_headers(),
            "Content-Type": (
                "application/json"
            ),
        },
        json={
            "doc_id": doc_ids,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": False,
            "temperature": 0.0,
            "enable_citations": True,
        },
        timeout=max(
            PAGEINDEX_TIMEOUT,
            60.0,
        ),
    )

    response.raise_for_status()

    return response.json()


def _cache_metadata_by_name(
    cache: dict[
        str,
        dict[str, Any],
    ],
) -> dict[
    str,
    dict[str, Any],
]:

    mapping: dict[
        str,
        dict[str, Any],
    ] = {}

    for relative, item in cache.items():

        mapping[relative] = item

        source = item.get(
            "source"
        )

        if isinstance(
            source,
            str,
        ):

            mapping[source] = item

            mapping[
                Path(source).stem
            ] = item

    return mapping


def pageindex_search(
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Vectorless search bằng PageIndex.

    Output:
        retrieval_method = "pageindex"
    """

    if (
        top_k <= 0
        or not isinstance(query, str)
        or not query.strip()
    ):
        return []

    cache = _load_cache()

    if not cache:

        upload_documents()

        cache = _load_cache()

    if not cache:
        return []

    doc_ids = [
        str(item["doc_id"])
        for item
        in cache.values()
        if (
            isinstance(item, dict)
            and item.get("doc_id")
        )
    ]

    doc_ids = list(
        dict.fromkeys(
            doc_ids
        )
    )

    if not doc_ids:
        return []

    payload = _chat_for_evidence(
        query,
        doc_ids,
        top_k,
    )

    choices = (
        payload.get("choices")
        or []
    )

    content = ""

    if (
        choices
        and isinstance(
            choices[0],
            dict,
        )
    ):

        message = (
            choices[0].get(
                "message"
            )
            or {}
        )

        if isinstance(
            message,
            dict,
        ):
            content = str(
                message.get(
                    "content"
                )
                or ""
            )

    evidence = (
        _extract_json_array(
            content
        )
    )

    metadata_lookup = (
        _cache_metadata_by_name(
            cache
        )
    )

    results: list[dict] = []

    seen: set[str] = set()

    for rank, item in enumerate(
        evidence,
        start=1,
    ):

        snippet = str(
            item.get("content")
            or ""
        ).strip()

        if not snippet:
            continue

        source_name = str(
            item.get("source")
            or "pageindex"
        )

        title = str(
            item.get("title")
            or Path(
                source_name
            ).stem
            or "PageIndex source"
        )

        page = item.get(
            "page"
        )

        cached_meta = (
            metadata_lookup.get(
                source_name
            )
            or metadata_lookup.get(
                Path(
                    source_name
                ).stem
            )
            or {}
        )

        stable = hashlib.sha1(
            (
                f"{source_name}|"
                f"{page}|"
                f"{snippet}"
            ).encode(
                "utf-8"
            )
        ).hexdigest()[:16]

        item_id = (
            f"pageindex::{stable}"
        )

        if item_id in seen:
            continue

        seen.add(item_id)

        results.append(
            {
                "id": item_id,
                "content": snippet,
                "score": float(
                    1.0 / rank
                ),
                "metadata": {
                    "source": str(
                        cached_meta.get(
                            "source"
                        )
                        or source_name
                    ),
                    "title": str(
                        cached_meta.get(
                            "title"
                        )
                        or title
                    ),
                    "doc_type": str(
                        cached_meta.get(
                            "doc_type"
                        )
                        or "news"
                    ),
                    "url": (
                        cached_meta.get(
                            "url"
                        )
                    ),
                    "chunk_index": max(
                        rank - 1,
                        0,
                    ),
                },
                "retrieval_method": (
                    "pageindex"
                ),
            }
        )

        if len(results) >= top_k:
            break

    if (
        not results
        and content.strip()
    ):

        first_meta = next(
            iter(
                cache.values()
            )
        )

        results.append(
            {
                "id": (
                    "pageindex::"
                    "fallback-0"
                ),
                "content": (
                    content.strip()
                ),
                "score": 1.0,
                "metadata": {
                    "source": str(
                        first_meta.get(
                            "source"
                        )
                        or "pageindex"
                    ),
                    "title": str(
                        first_meta.get(
                            "title"
                        )
                        or "PageIndex result"
                    ),
                    "doc_type": str(
                        first_meta.get(
                            "doc_type"
                        )
                        or "news"
                    ),
                    "url": (
                        first_meta.get(
                            "url"
                        )
                    ),
                    "chunk_index": 0,
                },
                "retrieval_method": (
                    "pageindex"
                ),
            }
        )

    return results[:top_k]


if __name__ == "__main__":
    upload_documents()