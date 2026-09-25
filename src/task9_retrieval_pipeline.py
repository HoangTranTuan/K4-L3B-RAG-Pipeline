"""
Task 9 — Retrieval pipeline hoàn chỉnh.

Luồng xử lý:
    1. Chạy semantic_search và lexical_search.
    2. Fuse hai danh sách bằng RRF đúng một lần.
    3. Lấy best cosine score gốc từ dense results.
    4. Nếu score dưới threshold, thử PageIndex fallback.
    5. Nếu fallback lỗi, trả hybrid results thay vì crash.

Không so sánh threshold với RRF score vì hai thang đo khác nhau.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.3"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Trả về hybrid hoặc pageindex SearchResult."""
    dense = semantic_search(query, top_k=top_k * 2)
    sparse = lexical_search(query, top_k=top_k * 2)

    if use_reranking:
        hybrid = rerank_rrf([dense, sparse], top_k=top_k)
    else:
        hybrid = dense[:top_k]

    best_dense_score = dense[0]["score"] if dense else 0.0
    if best_dense_score < score_threshold:
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback[:top_k]
        except Exception:
            pass

    return hybrid[:top_k]


if __name__ == "__main__":
    import json
    import sys
    from .contracts import validate_search_results

    print("=" * 60)
    print("TEST SUITE TASK 9: RETRIEVAL PIPELINE (VỚI MOCK DATA)")
    print("=" * 60)

    # 1. Định nghĩa mock data tuân thủ đầy đủ schema contracts.py
    mock_dense_results = [
        {
            "id": "legal-doc::chunk-0",
            "content": "Sinh viên đạt GPA từ 3.6 trở lên được xét học bổng khuyến khích học tập loại Xuất sắc.",
            "score": 0.88,
            "metadata": {
                "source": "quy_che_hoc_bong.md",
                "title": "Quy chế Học bổng & Học phí",
                "doc_type": "legal",
                "url": "https://vinuni.example.edu.vn/scholarship",
                "chunk_index": 0,
            },
            "retrieval_method": "dense",
        },
        {
            "id": "legal-doc::chunk-1",
            "content": "Sinh viên có chứng chỉ IELTS 7.5 trở lên được miễn một số tín chỉ tiếng Anh học thuật.",
            "score": 0.72,
            "metadata": {
                "source": "quy_che_dao_tao.md",
                "title": "Quy chế Đào tạo đại học",
                "doc_type": "legal",
                "url": "https://vinuni.example.edu.vn/academic",
                "chunk_index": 1,
            },
            "retrieval_method": "dense",
        },
    ]

    mock_sparse_results = [
        {
            "id": "news-doc::chunk-0",
            "content": "Hạn cuối nộp hồ sơ xin xét duyệt học bổng khuyến khích là ngày 15/10/2026.",
            "score": 4.25,
            "metadata": {
                "source": "thong_bao_hoc_bong_2026.md",
                "title": "Thông báo nộp hồ sơ học bổng kỳ Thu 2026",
                "doc_type": "news",
                "url": "https://vinuni.example.edu.vn/news/scholarship-2026",
                "chunk_index": 0,
            },
            "retrieval_method": "bm25",
        },
        {
            "id": "legal-doc::chunk-0",
            "content": "Sinh viên đạt GPA từ 3.6 trở lên được xét học bổng khuyến khích học tập loại Xuất sắc.",
            "score": 3.80,
            "metadata": {
                "source": "quy_che_hoc_bong.md",
                "title": "Quy chế Học bổng & Học phí",
                "doc_type": "legal",
                "url": "https://vinuni.example.edu.vn/scholarship",
                "chunk_index": 0,
            },
            "retrieval_method": "bm25",
        },
    ]

    mock_fallback_results = [
        {
            "id": "pageindex-doc::chunk-0",
            "content": "Kết quả fallback từ PageIndex: Quy định chi tiết về thủ tục cấp bảo lưu học tập.",
            "score": 0.95,
            "metadata": {
                "source": "pageindex_manual.md",
                "title": "Sổ tay sinh viên PageIndex",
                "doc_type": "legal",
                "url": "https://vinuni.example.edu.vn/handbook",
                "chunk_index": 0,
            },
            "retrieval_method": "pageindex",
        }
    ]

    current_module = sys.modules[__name__]

    # --- KỊCH BẢN 1: Dense score cao (>= threshold) -> Trả về Hybrid RRF ---
    print("\n[Kịch bản 1] Dense score cao (0.88 >= 0.3) -> Chạy Hybrid RRF, KHÔNG kích hoạt Fallback:")
    globals()["semantic_search"] = lambda q, top_k: mock_dense_results
    globals()["lexical_search"] = lambda q, top_k: mock_sparse_results

    # Mock RRF implementation để test độc lập nếu task 7 chưa hoàn thiện
    def mock_rrf_impl(ranked_lists, top_k, k=60):
        scores = {}
        items = {}
        for r_list in ranked_lists:
            for rank, item in enumerate(r_list, 1):
                i_id = item["id"]
                scores[i_id] = scores.get(i_id, 0.0) + 1.0 / (k + rank)
                items[i_id] = item
        ranked_ids = sorted(scores, key=lambda i: scores[i], reverse=True)
        out = []
        for i_id in ranked_ids[:top_k]:
            cp = items[i_id].copy()
            cp["score"] = scores[i_id]
            cp["retrieval_method"] = "hybrid"
            out.append(cp)
        return out

    globals()["rerank_rrf"] = mock_rrf_impl
    globals()["pageindex_search"] = lambda q, top_k: (_ for _ in ()).throw(AssertionError("Fallback không được gọi!"))

    res1 = retrieve("chính sách học bổng", top_k=2, score_threshold=0.3)
    validate_search_results(res1, top_k=2, expected_method="hybrid")
    print(f" -> Thành công! Nhận {len(res1)} kết quả hybrid.")
    for r in res1:
        print(f"    * [{r['retrieval_method'].upper()}] id={r['id']} (score={r['score']:.4f}): {r['content'][:60]}...")

    # --- KỊCH BẢN 2: Dense score thấp (< threshold) -> Kích hoạt PageIndex Fallback ---
    print("\n[Kịch bản 2] Dense score thấp (0.15 < 0.3) -> Kích hoạt PageIndex Fallback thành công:")
    low_dense = [dict(mock_dense_results[0], score=0.15)]
    globals()["semantic_search"] = lambda q, top_k: low_dense
    globals()["pageindex_search"] = lambda q, top_k: mock_fallback_results

    res2 = retrieve("thủ tục bảo lưu học tập", top_k=2, score_threshold=0.3)
    validate_search_results(res2, top_k=2, expected_method="pageindex")
    print(f" -> Thành công! Nhận {len(res2)} kết quả pageindex.")
    for r in res2:
        print(f"    * [{r['retrieval_method'].upper()}] id={r['id']} (score={r['score']:.4f}): {r['content'][:60]}...")

    # --- KỊCH BẢN 3: Dense score thấp nhưng Fallback gặp lỗi -> Trả hybrid an toàn, không crash ---
    print("\n[Kịch bản 3] Dense score thấp nhưng Fallback API lỗi -> Bắt ngoại lệ và trả kết quả hybrid an toàn:")
    def broken_fallback(q, top_k):
        raise ConnectionError("PageIndex API timeout or network failure")
    globals()["pageindex_search"] = broken_fallback

    res3 = retrieve("truy vấn khi fallback lỗi", top_k=2, score_threshold=0.3)
    validate_search_results(res3, top_k=2, expected_method="hybrid")
    print(f" -> Thành công! Pipeline không crash, an toàn trả về {len(res3)} kết quả hybrid.")

    print("\n" + "=" * 60)
    print("HOÀN THÀNH TẤT CẢ KỊCH BẢN KIỂM THỬ TASK 9!")
    print("=" * 60)
