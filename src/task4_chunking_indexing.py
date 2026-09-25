"""
Task 4 — Chunking, embedding và indexing.

Pipeline:
1. Đọc Markdown trong data/standardized/
2. Chunk documents
3. Embed chunks
4. Upsert vào ChromaDB

Task 5 phải dùng chung embed_texts() và get_collection().
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


ROOT = Path(__file__).parent.parent

STANDARDIZED_DIR = ROOT / "data" / "standardized"
CHROMA_DIR = ROOT / "chroma_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

COLLECTION_NAME = "rag_documents"


# ============================================================
# Embedding
# ============================================================

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed danh sách texts.

    Ưu tiên provider từ .env:
        EMBEDDING_PROVIDER=openai
        EMBEDDING_PROVIDER=local

    Nếu dùng OpenAI:
        OPENAI_API_KEY=...
        EMBEDDING_MODEL=text-embedding-3-small

    Nếu không khai báo provider, mặc định thử OpenAI.
    """

    if not texts:
        return []

    provider = os.getenv(
        "EMBEDDING_PROVIDER",
        "openai",
    ).strip().lower()

    model_name = os.getenv(
        "EMBEDDING_MODEL",
        "",
    ).strip()

    # --------------------------------------------------------
    # OpenAI
    # --------------------------------------------------------

    if provider == "openai":

        from openai import OpenAI

        api_key = os.getenv(
            "OPENAI_API_KEY",
            "",
        ).strip()

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY chưa được cấu hình trong .env"
            )

        client = OpenAI(
            api_key=api_key,
        )

        model = (
            model_name
            or "text-embedding-3-small"
        )

        response = client.embeddings.create(
            model=model,
            input=texts,
        )

        return [
            item.embedding
            for item in response.data
        ]

    # --------------------------------------------------------
    # Local SentenceTransformer
    # --------------------------------------------------------

    if provider == "local":

        try:
            from sentence_transformers import (
                SentenceTransformer,
            )

        except ImportError as exc:
            raise RuntimeError(
                "Muốn dùng local embedding thì cần cài "
                "`sentence-transformers`."
            ) from exc

        model = SentenceTransformer(
            model_name
            or EMBEDDING_MODEL
        )

        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return vectors.tolist()

    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER: {provider}"
    )


# ============================================================
# ChromaDB
# ============================================================

def get_collection():
    """
    Mở persistent Chroma collection với cosine distance.
    """

    import chromadb

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
    )

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
        },
    )


# ============================================================
# Load standardized documents
# ============================================================

def _stable_document_id(
    relative_path: str,
) -> str:
    """
    Sinh ID ổn định từ đường dẫn file.
    """

    digest = hashlib.sha1(
        relative_path.encode("utf-8")
    ).hexdigest()[:12]

    return f"doc-{digest}"


def _parse_frontmatter(
    text: str,
) -> tuple[dict, str]:
    """
    Parse YAML frontmatter đơn giản nếu Markdown có dạng:

    ---
    title: ...
    source: ...
    url: ...
    doc_type: news
    ---

    Không phụ thuộc PyYAML.
    """

    if not text.startswith("---"):
        return {}, text

    lines = text.splitlines()

    if len(lines) < 3:
        return {}, text

    metadata: dict = {}

    end_index = None

    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            end_index = index
            break

        line = lines[index]

        if ":" not in line:
            continue

        key, value = line.split(
            ":",
            1,
        )

        metadata[
            key.strip()
        ] = value.strip().strip(
            "\"'"
        )

    if end_index is None:
        return {}, text

    content = "\n".join(
        lines[end_index + 1 :]
    ).strip()

    return metadata, content


def load_documents() -> list[dict]:
    """
    Đọc toàn bộ Markdown trong data/standardized/.

    Không cần hard-code URL 5 bài báo ở đây.
    URL nên được Task 2/Task 3 lưu trong metadata/frontmatter.
    """

    if not STANDARDIZED_DIR.exists():
        return []

    documents: list[dict] = []

    paths = sorted(
        STANDARDIZED_DIR.rglob("*.md")
    )

    for path in paths:

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

        split_texts = (
            splitter.split_text(
                content
            )
        )

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
    """
    Chạy full Task 4:
        Markdown
        -> documents
        -> chunks
        -> embeddings
        -> ChromaDB
    """

    documents = (
        load_documents()
    )

    print(
        f"Loaded {len(documents)} documents"
    )

    if not documents:
        print(
            "Không tìm thấy Markdown trong:",
            STANDARDIZED_DIR,
        )
        return

    chunks = chunk_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks"
    )

    embedded_chunks = (
        embed_chunks(
            chunks
        )
    )

    index_to_vectorstore(
        embedded_chunks
    )

    print(
        f"Indexed {len(embedded_chunks)} chunks"
    )


if __name__ == "__main__":
    run_pipeline()