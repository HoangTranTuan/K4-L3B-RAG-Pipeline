"""
Task 9 — Retrieval pipeline hoàn chỉnh.

Luồng xử lý:
    1. Chạy semantic_search và lexical_search.
    2. Fuse hai danh sách bằng RRF đúng một lần.
    3. Lấy best cosine score gốc từ dense results.
    4. Nếu score dưới threshold, thử PageIndex fallback.
    5. Nếu fallback lỗi, trả hybrid results thay vì crash.

Lưu ý:
- Không dùng RRF score để quyết định fallback.
- Fallback phải dựa trên dense cosine score gốc.
- Không để lỗi PageIndex làm crash pipeline.
"""

from __future__ import annotations

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
    use_rerank: bool | None = None,
) -> list[dict]:
    if use_rerank is not None:
        use_reranking = use_rerank
    """
    Chạy retrieval pipeline hoàn chỉnh.

    Args:
        query:
            Câu hỏi của người dùng.

        top_k:
            Số kết quả cuối cùng tối đa.

        score_threshold:
            Ngưỡng confidence của dense retrieval.
            Nếu dense score tốt nhất thấp hơn ngưỡng này
            thì thử PageIndex fallback.

        use_reranking:
            True:
                Dense + BM25 -> RRF.

            False:
                Không dùng RRF, ưu tiên dense results.

    Returns:
        list[SearchResult]

        retrieval_method có thể là:
            - "hybrid"
            - "dense"
            - "pageindex"
    """

    # ----------------------------
    # Validate input
    # ----------------------------

    if (
        not isinstance(query, str)
        or not query.strip()
        or top_k <= 0
    ):
        return []

    query = query.strip()

    # Lấy nhiều candidate hơn số lượng cần trả về
    # để RRF có đủ dữ liệu để fuse.
    candidate_k = max(
        top_k * 2,
        top_k,
    )

    # ----------------------------
    # 1. Dense / Semantic Search
    # ----------------------------

    try:
        dense_results = semantic_search(
            query,
            top_k=candidate_k,
        )

    except Exception as error:
        print(
            "[Task9] Semantic search error:",
            error,
        )

        dense_results = []

    # ----------------------------
    # 2. BM25 / Lexical Search
    # ----------------------------

    try:
        sparse_results = lexical_search(
            query,
            top_k=candidate_k,
        )

    except Exception as error:
        print(
            "[Task9] Lexical search error:",
            error,
        )

        sparse_results = []

    # ----------------------------
    # 3. Hybrid retrieval
    # ----------------------------

    if use_reranking:

        try:
            hybrid_results = rerank_rrf(
                [
                    dense_results,
                    sparse_results,
                ],
                top_k=top_k,
            )

        except Exception as error:

            print(
                "[Task9] RRF error:",
                error,
            )

            # Nếu RRF lỗi thì fallback nhẹ:
            # ưu tiên dense, nếu dense rỗng thì dùng BM25.
            hybrid_results = (
                dense_results[:top_k]
                if dense_results
                else sparse_results[:top_k]
            )

    else:

        # Không sử dụng RRF.
        # Theo pipeline mặc định ưu tiên dense retrieval.
        hybrid_results = (
            dense_results[:top_k]
            if dense_results
            else sparse_results[:top_k]
        )

    # ----------------------------
    # 4. Dense confidence
    # ----------------------------

    # Quan trọng:
    # threshold phải so với dense score gốc,
    # KHÔNG dùng RRF score.
    best_dense_score = 0.0

    if dense_results:

        try:
            best_dense_score = max(
                float(
                    item.get(
                        "score",
                        0.0,
                    )
                )
                for item
                in dense_results
                if isinstance(
                    item,
                    dict,
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            best_dense_score = 0.0

    # ----------------------------
    # 5. PageIndex fallback
    # ----------------------------

    if best_dense_score < score_threshold:

        try:

            fallback_results = (
                pageindex_search(
                    query,
                    top_k=top_k,
                )
            )

            if fallback_results:

                return fallback_results[
                    :top_k
                ]

        except Exception as error:

            # PageIndex/provider lỗi
            # KHÔNG làm pipeline crash.
            print(
                "[Task9] "
                "PageIndex fallback error:",
                error,
            )

    # ----------------------------
    # 6. Trả kết quả retrieval
    # ----------------------------

    return hybrid_results[:top_k]


if __name__ == "__main__":

    query = (
        "Hộ kinh doanh có phải "
        "sử dụng hóa đơn điện tử không?"
    )

    results = retrieve(
        query,
        top_k=3,
    )

    for index, item in enumerate(
        results,
        start=1,
    ):

        print(
            f"\nResult {index}"
        )

        print(
            "Method:",
            item.get(
                "retrieval_method"
            ),
        )

        print(
            "Score:",
            item.get(
                "score"
            ),
        )

        print(
            "Source:",
            item.get(
                "metadata",
                {},
            ).get(
                "source"
            ),
        )

        print(
            "Content:",
            item.get(
                "content",
                "",
            )[:300],
        )