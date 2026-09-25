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
from dotenv import load_dotenv

# Cấu hình UTF-8 cho console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

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

    else:
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")


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
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[dict] = []
    for document in documents:
        splits = splitter.split_text(document["content"])
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
