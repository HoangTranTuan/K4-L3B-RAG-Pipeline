## Thông tin

- Họ và tên: Trần Tuấn Hoàng
- Mã học viên: 2A202602832
- Nhóm: BotVN
- Repository/branch:

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| **Retrieval Pipeline** | Lắp ráp module tìm kiếm Dense (Hải) và BM25 (Đại) thành pipeline hoàn chỉnh; tích hợp RRF reranking (`k=60`); cài đặt logic PageIndex fallback khi `best_dense_score < score_threshold`; bọc `try-except` an toàn chống crash khi provider lỗi. | `src/task9_retrieval_pipeline.py` | Done |
| **Generation & Citation** | Cài đặt Lost-in-the-middle reordering; format context kèm metadata trích dẫn; tích hợp LLM client đa provider (OpenRouter, Gemini, OpenAI, Mock); xây dựng cơ chế Safe Refusal chuẩn mực khi không có context hoặc LLM gặp sự cố. | `src/task10_generation.py` | Done |
| **Interactive UI & API** | Xây dựng Web UI dashboard và REST API server (`/api/query`, `/api/status`) trực quan hóa chi tiết từng bước truy xuất (Dense, BM25, RRF rank, Fallback), hiển thị câu trả lời Markdown kèm interactive citation badges tự scroll & highlight source card. | `app.py` | Done |
| **RAG Evaluation** | Chuẩn bị phương án và kịch bản đánh giá pipeline RAG với `golden_dataset.json` (đối sánh A/B giữa Dense-only và Hybrid RRF theo 4 chỉ số Ragas). | `group_project/evaluation/golden_dataset.json`, `reports/RESULT.md` | Partial |

## Quyết định kỹ thuật quan trọng

Mô tả tối đa hai quyết định mà bạn trực tiếp tham gia:

1. **Quyết định:** Sử dụng điểm Cosine Similarity gốc cao nhất của Dense Search (`best_dense_score`) thay vì điểm RRF để so sánh với `score_threshold` kích hoạt PageIndex fallback.  
   **Lý do/evidence:** Điểm RRF phụ thuộc vào số lượng retriever và thứ hạng (thường rất nhỏ, khoảng 0.01 - 0.03 do $1/(k + rank)$), không phản ánh độ tương đồng ngữ nghĩa thực tế của query. Dùng cosine gốc đo lường trực tiếp độ tin cậy ngữ nghĩa của corpus và tuân thủ đúng test contract `test_retrieve_uses_dense_score_for_fallback`.  
   **Trade-off:** Truy vấn chứa từ khóa hiếm hoặc từ viết tắt có thể có điểm dense thấp gây kích hoạt fallback dù BM25 tốt; tuy nhiên đã được bọc `try-except` để fallback an toàn về hybrid nếu API ngoài lỗi mạng/timeout.

2. **Quyết định:** Triển khai Lost-in-the-middle reordering kết hợp Safe Refusal đa tầng ở khâu Generation.  
   **Lý do/evidence:** LLM chú ý tốt nhất ở đầu và cuối ngữ cảnh (Liu et al.). Việc phân bổ chunk quan trọng nhất ra đầu (`chunks[::2]`) và đuôi (`chunks[1::2][::-1]`) giúp giảm tối đa ảo giác. Đồng thời, cơ chế Safe Refusal bắt buộc từ chối khi thiếu context hoặc khi mất kết nối LLM, đảm bảo tính trung thực (faithfulness).  
   **Trade-off:** Reordering làm thay đổi trật tự tuyến tính theo điểm số, đòi hỏi quản lý chính xác citation index và metadata; Safe Refusal nghiêm ngặt có thể từ chối các câu hỏi suy luận mở rộng nhưng đảm bảo an toàn thông tin.

## Kiểm thử và kết quả

- Test hoặc query tôi đã dùng:
  - Bộ test suite mock tích hợp trong `src/task9_retrieval_pipeline.py` và `src/task10_generation.py`: kiểm tra kịch bản Dense cao $\rightarrow$ Hybrid RRF, Dense thấp $\rightarrow$ PageIndex fallback, Fallback API lỗi $\rightarrow$ an toàn trả về hybrid, và kiểm tra cơ chế Safe Refusal.
  - Live query với OpenRouter API (`google/gemini-2.0-flash-001`) và kiểm thử giao diện Web UI / REST API (`/api/query`) tại `http://localhost:8501`.
- Kết quả trước/sau nếu có:
  - Trước: Module `retrieve` và `generate_with_citation` trả về `NotImplementedError`, chưa có luồng liên kết và giao diện tương tác.
  - Sau: Pipeline chạy thông suốt; 100% kết quả vượt qua validation schema `validate_search_results` và `validate_generation_result`.
- Lỗi đã phát hiện và cách xử lý:
  - Phát hiện trường hợp LLM API lỗi mạng hoặc hết quota gây crash app: Bọc `call_llm` trong khối `try-except`, tự động trả về `SAFE_REFUSAL` tuân thủ contract.
  - Xử lý giá trị `retrieval_source` trong `GenerationResult`: chuẩn hóa chỉ nhận `"hybrid"`, `"pageindex"`, hoặc `"none"`.

## Điều còn hạn chế

- Một hạn chế cụ thể của phần tôi làm: Do nhóm đang trong giai đoạn thu thập dữ liệu đầu vào, phần đánh giá Ragas (`group_project/evaluation/`) hiện mới dừng ở mức chuẩn bị framework và schema, chưa chạy benchmark đầy đủ trên corpus hoàn chỉnh.
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện: Sau khi nhóm nạp đủ dữ liệu chính thức, hoàn thiện `golden_dataset.json` (15+ Q&A), chạy đo lường 4 chỉ số RAG (Faithfulness, Relevance, Recall, Precision) để hoàn tất bảng số liệu A/B trong `reports/RESULT.md`.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 25/09/2026
- Tên thành viên: Trần Tuấn Hoàng
