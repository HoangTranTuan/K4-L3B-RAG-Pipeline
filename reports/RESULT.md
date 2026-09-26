# RAG evaluation results

## Run information

| Field | Value |
| --- | --- |
| Evaluation date | 2026-09-26 |
| Framework and version | Ragas 0.2.x / Python 3.10 Pipeline Benchmark |
| Evaluator model | google/gemini-2.0-flash |
| Generator model | google/gemini-2.0-flash-001 |
| Embedding model | sentence-transformers/bge-m3 |
| Corpus version/commit | Commit 4c89da1 (3 văn bản luật + 5 bài báo, 1.749 chunk đã chuẩn hóa) |
| Golden dataset size | 15 câu hỏi có đáp án chuẩn (Grounded Q&A) |
| top_k | 5 |
| Fallback threshold | Cosine similarity < 0.30 thì kích hoạt fallback |

## Configurations

- **Config A — dense-only:** Dùng dense embedding (`bge-m3`) tính cosine similarity trên ChromaDB, lấy top-5 chunk giống nhất đưa thẳng cho Generator.
- **Config B — hybrid + RRF:** Kết hợp Dense retrieval (ChromaDB) với BM25 (lexical), gộp kết quả bằng RRF (k=60) để ra top-5 chunk cuối cùng.

Cả hai config dùng chung golden dataset, generator, evaluator, prompt và top_k — chỉ khác nhau ở cách retrieval.

## Overall scores

| Metric | Config A | Config B | Delta B−A |
| --- | ---: | ---: | ---: |
| Faithfulness | 0.82 | 0.91 | +0.09 |
| Answer relevance | 0.80 | 0.88 | +0.08 |
| Context recall | 1.00 | 0.933 | -0.067 |
| Context precision | 0.75 | 0.89 | +0.14 |
| **Trung bình** | 0.842 | 0.903 | +0.061 |

## So sánh A/B

- **Config tốt hơn:** Config B (Hybrid + RRF) — điểm trung bình 0.903 so với 0.842 của Config A.
- **Vì sao:** Config A đạt Recall 100% vì Dense phủ khá rộng về mặt ngữ nghĩa, nhưng Precision lại thấp hơn hẳn (0.75 so với 0.89). Thêm BM25 giúp Config B bắt đúng các từ khóa cụ thể hơn — số hiệu thông tư, tên mẫu đơn 01/ĐKTĐ-HĐĐT, mốc doanh thu 1 tỷ đồng — nên Faithfulness tăng từ 0.82 lên 0.91 và Answer Relevance tăng từ 0.80 lên 0.88.
- **Đánh đổi:** Config B chậm hơn một chút vì phải chạy song song Dense và BM25 rồi tính thêm RRF (~0.507s/câu so với ~0.353s/câu của Config A). Mức chênh này không đáng lo so với phần cải thiện về độ chính xác.

## Các câu trả lời tệ nhất

| # | Câu hỏi | Config | Faithfulness | Relevance | Recall | Precision | Lỗi ở khâu | Nguyên nhân |
| --: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | Người nộp thuế có thể tiếp cận bộ tài liệu hướng dẫn của Cục Thuế bằng cách nào? | Config B | 0.60 | 0.65 | 0.00 | 0.40 | retrieval | Từ khóa ("người nộp thuế", "tài liệu hướng dẫn") trùng với các điều trong Nghị định 252, nên BM25 xếp các điều luật lên trên, đẩy chunk bài báo có thông tin "quét mã QR" ra khỏi top 5. |
| 2 | Vì sao thương mại điện tử làm tăng nguy cơ thất thu thuế? | Config A | 0.70 | 0.72 | 1.00 | 0.60 | generation | Context trả về có đủ số liệu tăng trưởng TMĐT và khó khăn quản lý, nhưng câu trả lời của model viết hơi chung chung, chưa nói rõ cơ chế trốn thuế ẩn danh. |
| 3 | Bộ tài liệu do Cục Thuế phát hành ngày 20/7/2026 tập trung vào những nội dung nào? | Config B | 0.75 | 0.78 | 1.00 | 0.70 | data | Chunk lấy từ bài báo bị dư khoảng trắng và ngắt dòng lộn xộn, khiến generator khó lấy đủ thông tin. |

## Đề xuất

| Ưu tiên | Việc cần làm | Bằng chứng | Kỳ vọng | Cách kiểm tra |
| --: | --- | --- | --- | --- |
| 1 | Thêm bước lọc metadata (tách luật / tin tức) trước khi retrieve | Câu 1 bị văn bản luật lấn át bài báo vì không tách loại tài liệu khi tìm kiếm | Recall của Config B có thể lên 100%, giảm nhiễu giữa luật và tin tức | Chạy lại `group_project/evaluate.py` với bộ lọc category, kiểm tra Recall đạt 15/15 |
| 2 | Dọn lại bước tiền xử lý văn bản (cleaning & normalization) | Câu 3 cho thấy dữ liệu markdown từ bài báo còn nhiều ký tự thừa, dòng trống | Context sạch hơn, kỳ vọng Precision tăng từ 0.89 lên >0.92, Faithfulness cũng tăng theo | Đo lại độ dài chunk trung bình và chạy lại Faithfulness |
| 3 | Thử Cross-Encoder Reranker thay vì chỉ dùng RRF | Câu hỏi dài có thể lợi hơn nếu chấm điểm trực tiếp giữa query và chunk | Answer Relevance và Context Precision có thể tăng thêm 5-10% | So sánh A/B giữa BM25+Dense+RRF và BM25+Dense+BGE-Reranker trên 15 câu golden dataset |

## Thử nghiệm thêm

| Thử nghiệm | So với | Chênh lệch metric | Chênh lệch latency/cost | Kết luận |
| --- | --- | ---: | ---: | --- |
| PageIndex Fallback (chuyển sang tìm không qua vector khi cosine score < 0.30) | Config A (Dense-only) | Recall giữ nguyên 100% trên câu hỏi out-of-domain | +0.12s khi fallback kích hoạt | Fallback chạy tốt khi câu hỏi không khớp vector store, giúp pipeline không bị crash |
