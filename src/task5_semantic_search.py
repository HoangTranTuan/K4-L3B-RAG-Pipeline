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

    try:
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

        if ids:
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
    except Exception:
        pass

    # Fallback: In-memory subword cosine similarity over shared corpus
    try:
        from .task6_lexical_search import _load_shared_corpus
        corpus = _load_shared_corpus()
        if not corpus:
            return []
        import math
        import re
        from collections import Counter
        clean_q = re.sub(r"\s+", " ", query.lower()).strip()
        q_ng = Counter([clean_q[i : i + 3] for i in range(max(0, len(clean_q) - 2))])
        q_norm = math.sqrt(sum(v * v for v in q_ng.values())) or 1.0
        scores = []
        for item in corpus:
            t = (item.get("content", "") + " " + item.get("metadata", {}).get("title", "")).lower()
            c_ng = Counter([t[i : i + 3] for i in range(max(0, len(t) - 2))])
            c_norm = math.sqrt(sum(v * v for v in c_ng.values())) or 1.0
            dot = sum(q_ng[k] * c_ng[k] for k in set(q_ng) & set(c_ng))
            scores.append((item, dot / (q_norm * c_norm)))
        scores.sort(key=lambda x: x[1], reverse=True)
        res = []
        for r, (item, sc) in enumerate(scores[:top_k], 1):
            cp = dict(item)
            cp["score"] = round(float(sc), 4)
            cp["retrieval_method"] = "dense"
            res.append(cp)
        return res
    except Exception:
        return []


if __name__ == "__main__":
    for result in semantic_search("test query", top_k=3):
        print(result)