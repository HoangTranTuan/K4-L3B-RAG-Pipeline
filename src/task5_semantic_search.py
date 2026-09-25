"""
Task 5 — Semantic search.

Embed query bằng chính hàm của Task 4, query ChromaDB và đổi cosine distance
thành similarity. Output phải theo SearchResult, sort giảm dần và không quá top_k.
"""

from __future__ import annotations

from .task4_chunking_indexing import embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""

    if top_k <= 0 or not isinstance(query, str) or not query.strip():
        return []

    query_vector = embed_texts([query])[0]

    response = get_collection().query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = response.get("ids", [[]])[0]
    documents = response.get("documents", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]

    results: list[dict] = []

    for item_id, content, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances,
    ):
        similarity = max(0.0, 1.0 - float(distance))

        results.append(
            {
                "id": item_id,
                "content": content or "",
                "score": similarity,
                "metadata": dict(metadata or {}),
                "retrieval_method": "dense",
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)

    return results[:top_k]


if __name__ == "__main__":
    for result in semantic_search("test query", top_k=3):
        print(result)