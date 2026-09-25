"""
Task 7 — Reciprocal Rank Fusion (RRF).

RRF gộp Semantic Search và BM25 mà không cộng trực tiếp
cosine score với BM25 score.

Công thức:

    RRF(d) = sum(1 / (k + rank))

rank bắt đầu từ 1.
"""

from __future__ import annotations


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """
    Kết hợp nhiều bảng xếp hạng bằng RRF.

    Output:
        retrieval_method = "hybrid"
    """

    if top_k <= 0:
        return []

    if k < 0:
        raise ValueError(
            "k must be non-negative"
        )

    scores: dict[str, float] = {}

    items: dict[str, dict] = {}

    best_rank: dict[str, int] = {}

    first_seen: dict[str, int] = {}

    seen_counter = 0

    for ranked_list in ranked_lists:

        if not ranked_list:
            continue

        seen_in_list: set[str] = set()

        effective_rank = 0

        for item in ranked_list:

            if not isinstance(item, dict):
                continue

            item_id = item.get("id")

            if (
                not isinstance(item_id, str)
                or not item_id
                or item_id in seen_in_list
            ):
                continue

            seen_in_list.add(item_id)

            effective_rank += 1

            rrf_score = 1.0 / (
                k + effective_rank
            )

            scores[item_id] = (
                scores.get(item_id, 0.0)
                + rrf_score
            )

            if item_id not in items:

                copied = dict(item)

                if isinstance(
                    item.get("metadata"),
                    dict,
                ):
                    copied["metadata"] = dict(
                        item["metadata"]
                    )

                items[item_id] = copied

                first_seen[item_id] = (
                    seen_counter
                )

                seen_counter += 1

            best_rank[item_id] = min(
                best_rank.get(
                    item_id,
                    effective_rank,
                ),
                effective_rank,
            )

    ranked_ids = sorted(
        scores,
        key=lambda item_id: (
            -scores[item_id],
            best_rank[item_id],
            first_seen[item_id],
            item_id,
        ),
    )

    results: list[dict] = []

    for item_id in ranked_ids[:top_k]:

        result = dict(
            items[item_id]
        )

        if isinstance(
            result.get("metadata"),
            dict,
        ):
            result["metadata"] = dict(
                result["metadata"]
            )

        result["score"] = float(
            scores[item_id]
        )

        result[
            "retrieval_method"
        ] = "hybrid"

        results.append(result)

    return results


if __name__ == "__main__":

    print(
        "RRF module ready. "
        "Run: pytest tests/test_contracts.py -q"
    )