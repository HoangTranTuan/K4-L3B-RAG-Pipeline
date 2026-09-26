"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng RecursiveCharacterTextSplitter.
    3. Embed chunks bằng OpenAI API (text-embedding-3-small) và loại bỏ khoảng trắng thừa.
    4. Upsert vào ChromaDB với cosine distance.

Mỗi document/chunk phải theo docs/MODULE_CONTRACTS.md. ID cần ổn định để
chạy lại pipeline không tạo dữ liệu trùng. Task 5 phải dùng chung embed_texts().
"""

import os
import re
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_p = Path(__file__).resolve().parent.parent / ".env"
    if env_p.exists():
        for line in env_p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


ROOT = Path(__file__).resolve().parent.parent
STANDARDIZED_DIR = ROOT / "data" / "standardized"
CHROMA_DIR = ROOT / "chroma_db"

# Cấu hình tham số chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

# Cấu hình Embedding
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
if "bge" in EMBEDDING_MODEL.lower():
    EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536 if "text-embedding-3-small" in EMBEDDING_MODEL or EMBEDDING_PROVIDER == "openai" else 3072

COLLECTION_NAME = "rag_documents"


def clean_whitespace(text: str) -> str:
    """Loại bỏ các khoảng trắng thừa/dư thừa của dữ liệu khi embedding."""
    return re.sub(r"\s+", " ", text).strip()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed danh sách texts bằng OpenAI hoặc Gemini API sau khi chuẩn hóa khoảng trắng."""
    if not texts:
        return []

    # Loại bỏ khoảng trắng thừa theo yêu cầu
    cleaned_texts = [clean_whitespace(t) for t in texts]
    cleaned_texts = [t if t else " " for t in cleaned_texts]

    provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()

    if provider == "openai" or (os.getenv("OPENAI_API_KEY") and provider != "gemini"):
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment or .env")

        client = OpenAI(api_key=api_key)
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        if "bge" in model.lower() or "gemini" in model.lower():
            model = "text-embedding-3-small"

        all_embeddings: list[list[float]] = []
        batch_size = 500  # OpenAI hỗ trợ xử lý tới 2048 texts / request

        for i in range(0, len(cleaned_texts), batch_size):
            batch = cleaned_texts[i : i + batch_size]
            response = client.embeddings.create(
                model=model,
                input=batch,
            )
            for item in response.data:
                all_embeddings.append(item.embedding)

        return all_embeddings

    elif provider == "gemini":
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment or .env")

        client = genai.Client(api_key=api_key)
        model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
        all_embeddings = []
        batch_size = 50

        for i in range(0, len(cleaned_texts), batch_size):
            batch = cleaned_texts[i : i + batch_size]
            response = client.models.embed_content(
                model=model,
                contents=batch,
            )
            for emb in response.embeddings:
                all_embeddings.append(list(emb.values))
            if i + batch_size < len(cleaned_texts):
                time.sleep(4.1)

        return all_embeddings

    elif "sentence" in provider or "bge" in provider:
        try:
            from sentence_transformers import SentenceTransformer
            model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
            st_model = SentenceTransformer(model_name)
            embeddings = st_model.encode(cleaned_texts, normalize_embeddings=True)
            return [e.tolist() for e in embeddings]
        except Exception:
            import hashlib
            res = []
            for t in cleaned_texts:
                h = hashlib.sha256(t.encode("utf-8")).digest()
                res.append([float(b) / 255.0 for b in h[:128]])
            return res

    else:
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")

# ============================================================
# ChromaDB
# ============================================================

def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    documents: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue

        doc_type = "legal" if "legal" in path.parts else "news"

        url = None
        if doc_type == "news":
            url_match = re.search(r"\*\*Source:\*\*\s*(\S+)", content)
            if url_match:
                url = url_match.group(1).strip()

        title = path.stem
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()

        doc_id = path.relative_to(STANDARDIZED_DIR).as_posix()
        documents.append({
            "id": doc_id,
            "content": content,
            "metadata": {
                "source": path.name,
                "title": title,
                "doc_type": doc_type,
                "url": url,
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        has_langchain = True
    except ImportError:
        has_langchain = False

    chunks: list[dict] = []
    for document in documents:
        if has_langchain:
            splits = splitter.split_text(document["content"])
        else:
            txt = document["content"]
            splits = []
            st = 0
            while st < len(txt):
                ed = min(st + CHUNK_SIZE, len(txt))
                ck = txt[st:ed].strip()
                if ck:
                    splits.append(ck)
                if ed >= len(txt):
                    break
                st += CHUNK_SIZE - CHUNK_OVERLAP
        chunk_idx = 0
        for text in splits:
            cleaned = text.strip()
            if not cleaned:
                continue
            chunks.append({
                "id": f"{document['id']}::chunk-{chunk_idx}",
                "content": cleaned,
                "metadata": {
                    **document["metadata"],
                    "chunk_index": chunk_idx,
                },
            })
            chunk_idx += 1
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    texts = [chunk["content"] for chunk in chunks]
    vectors = embed_texts(texts)
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB theo batch."""
    if not chunks:
        return
    collection = get_collection()
    batch_size = 200
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        metadatas = []
        for c in batch:
            m = dict(c["metadata"])
            if m.get("url") is None:
                m["url"] = ""  # ChromaDB yêu cầu kiểu chuỗi cho metadata
            metadatas.append(m)

        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["content"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            metadatas=metadatas,
        )

        raw_text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        frontmatter, content = (
            _parse_frontmatter(
                raw_text
            )
        )

        if not content.strip():
            continue

        relative_path = (
            path
            .relative_to(
                STANDARDIZED_DIR
            )
            .as_posix()
        )

        # ----------------------------------------------------
        # doc_type
        # ----------------------------------------------------

        detected_doc_type = (
            "legal"
            if "legal"
            in {
                part.lower()
                for part in path.parts
            }
            else "news"
        )

        doc_type = (
            frontmatter.get(
                "doc_type"
            )
            or detected_doc_type
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        source = (
            frontmatter.get(
                "source"
            )
            or path.name
        )

        title = (
            frontmatter.get(
                "title"
            )
            or path.stem.replace(
                "_",
                " ",
            )
        )

        url = (
            frontmatter.get(
                "url"
            )
            or None
        )

        documents.append(
            {
                "id": (
                    frontmatter.get(
                        "id"
                    )
                    or _stable_document_id(
                        relative_path
                    )
                ),
                "content": content.strip(),
                "metadata": {
                    "source": str(
                        source
                    ),
                    "title": str(
                        title
                    ),
                    "doc_type": str(
                        doc_type
                    ),
                    "url": (
                        str(url)
                        if url
                        else None
                    ),
                },
            }
        )

    return documents


# ============================================================
# Chunking
# ============================================================

def chunk_documents(
    documents: list[dict],
) -> list[dict]:
    """
    Chia documents thành chunks.

    Yêu cầu quan trọng:
    - giữ metadata gốc
    - chunk_index tăng từ 0
    - chunk ID unique
    - chunk content không vượt quá đáng kể CHUNK_SIZE
    """

    if not documents:
        return []

    try:
        from langchain_text_splitters import (
            RecursiveCharacterTextSplitter,
        )
        splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                separators=[
                    "\n\n",
                    "\n",
                    ". ",
                    "; ",
                    ", ",
                    " ",
                    "",
                ],
            )
        )
        has_lc = True
    except ImportError:
        has_lc = False

    chunks: list[dict] = []

    for document in documents:

        document_id = str(
            document["id"]
        )

        content = str(
            document["content"]
        )

        metadata = dict(
            document["metadata"]
        )

        if has_lc:
            split_texts = (
                splitter.split_text(
                    content
                )
            )
        else:
            split_texts = []
            st = 0
            while st < len(content):
                ed = min(st + CHUNK_SIZE, len(content))
                ck = content[st:ed].strip()
                if ck:
                    split_texts.append(ck)
                if ed >= len(content):
                    break
                st += CHUNK_SIZE - CHUNK_OVERLAP

        for index, text in enumerate(
            split_texts
        ):

            text = text.strip()

            if not text:
                continue

            chunks.append(
                {
                    "id": (
                        f"{document_id}"
                        f"::chunk-{index}"
                    ),
                    "content": text,
                    "metadata": {
                        **metadata,
                        "chunk_index": (
                            index
                        ),
                    },
                }
            )

    return chunks


# ============================================================
# Embed chunks
# ============================================================

def embed_chunks(
    chunks: list[dict],
) -> list[dict]:
    """
    Thêm embedding vào từng chunk.
    Không mutate input gốc.
    """

    if not chunks:
        return []

    vectors = embed_texts(
        [
            chunk["content"]
            for chunk in chunks
        ]
    )

    if len(vectors) != len(chunks):
        raise RuntimeError(
            "Số embeddings không khớp số chunks"
        )

    embedded: list[dict] = []

    for chunk, vector in zip(
        chunks,
        vectors,
    ):

        new_chunk = {
            "id": chunk["id"],
            "content": (
                chunk["content"]
            ),
            "metadata": dict(
                chunk["metadata"]
            ),
            "embedding": list(
                vector
            ),
        }

        embedded.append(
            new_chunk
        )

    return embedded


# ============================================================
# Index
# ============================================================

def index_to_vectorstore(
    chunks: list[dict],
) -> None:
    """
    Upsert chunks vào ChromaDB.

    Dùng upsert để chạy pipeline nhiều lần
    không tạo bản ghi trùng.
    """

    if not chunks:
        return

    collection = (
        get_collection()
    )

    batch_size = 100

    for start in range(
        0,
        len(chunks),
        batch_size,
    ):

        batch = chunks[
            start:
            start + batch_size
        ]

        collection.upsert(
            ids=[
                chunk["id"]
                for chunk in batch
            ],
            documents=[
                chunk["content"]
                for chunk in batch
            ],
            embeddings=[
                chunk["embedding"]
                for chunk in batch
            ],
            metadatas=[
                chunk["metadata"]
                for chunk in batch
            ],
        )


# ============================================================
# Run full pipeline
# ============================================================

def run_pipeline() -> None:
    """Chạy load, chunk, embed và index với OpenAI embedding."""
    documents = load_documents()
    print(f"Loaded {len(documents)} documents from {STANDARDIZED_DIR}", flush=True)
    chunks = chunk_documents(documents)
    print(f"Generated {len(chunks)} chunks", flush=True)

    collection = get_collection()
    existing_ids = set()
    try:
        existing_data = collection.get(include=[])
        if existing_data and "ids" in existing_data:
            existing_ids = set(existing_data["ids"])
    except Exception:
        pass

    to_index = [c for c in chunks if c["id"] not in existing_ids]
    print(f"Already indexed: {len(existing_ids)}, remaining to index: {len(to_index)}", flush=True)

    if not to_index:
        print(f"All {len(chunks)} chunks are already indexed into ChromaDB!", flush=True)
        return

    print(f"Embedding and indexing {len(to_index)} chunks using OpenAI ({EMBEDDING_MODEL})...", flush=True)
    embedded_chunks = embed_chunks(to_index)
    index_to_vectorstore(embedded_chunks)
    print(f"Successfully indexed all {len(chunks)} chunks into ChromaDB!", flush=True)


if __name__ == "__main__":
    run_pipeline()