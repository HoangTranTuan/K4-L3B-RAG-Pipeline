# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-26 |
| Framework and version              | Ragas 0.2.x / Python 3.10 Pipeline Benchmark |
| Evaluator model                    | google/gemini-2.0-flash |
| Generator model                    | google/gemini-2.0-flash-001 |
| Embedding model                    | sentence-transformers/bge-m3 |
| Corpus version/commit              | Commit 4c89da1 (Corpus 3 legal docs + 5 news articles, 1,749 standardized chunks) |
| Golden dataset size                | 15 Grounded Q&A Cases |
| `top_k`                            | 5 |
| Fallback threshold and calibration | 0.30 (Cosine similarity on dense retrieval) |

## Configurations

- **Config A — dense-only:** Sử dụng mô hình dense embedding vector (`sentence-transformers/bge-m3`) tìm kiếm tương đồng cosine trên ChromaDB vector store, lấy top-5 chunks có độ tương đồng cao nhất chuyển trực tiếp tới Generator.
- **Config B — hybrid + RRF:** Kết hợp đồng thời Dense retrieval (ChromaDB) và Lexical retrieval (BM25 trên standardized corpus), sau đó chuẩn hóa và xếp hạng kết quả thông qua thuật toán Reciprocal Rank Fusion (RRF, $k=60$) để trích xuất top-5 chunks tối ưu nhất.

Hai config phải dùng cùng golden dataset, generator, evaluator, prompt và `top_k`; chỉ thay retrieval strategy.

## Overall scores

| Metric            | Config A | Config B | Delta B−A |
| ----------------- | -------: | -------: | --------: |
| Faithfulness      |     0.82 |     0.91 |     +0.09 |
| Answer relevance  |     0.80 |     0.88 |     +0.08 |
| Context recall    |     1.00 |    0.933 |    -0.067 |
| Context precision |     0.75 |     0.89 |     +0.14 |
| **Average**       |    0.842 |    0.903 |    +0.061 |

## A/B comparison

- Cấu hình tốt hơn: **Config B — Hybrid + RRF** đạt hiệu năng tổng thể vượt trội hơn (Average Score: 0.903 so với 0.842 của Config A).
- Evidence: Mặc dù Config A có Context Recall đạt 100% nhờ độ phủ ngữ nghĩa rộng của mô hình Dense, nhưng Context Precision lại thấp hơn đáng kể (0.75 so với 0.89). Config B kết hợp BM25 giúp loại bỏ các văn bản chung chung, đưa các điều khoản chính xác theo từ khóa (như số hiệu thông tư, tên mẫu đơn `01/ĐKTĐ-HĐĐT`, mốc doanh thu 1 tỷ đồng) lên vị trí đầu, giúp Faithfulness tăng từ 0.82 lên 0.91 và Answer Relevance tăng từ 0.80 lên 0.88.
- Trade-off về latency/cost: Config B chạy song song cả Dense Embedding lẫn BM25 Index và thêm bước tính điểm RRF nên có độ trễ trung bình cao hơn một chút (~0.507s/câu so với ~0.353s/câu của Config A). Tuy nhiên sự chênh lệch này là không đáng kể so với mức cải thiện lớn về độ chính xác và tính xác thực của câu trả lời.

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage             | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------------------- | ---------- |
|   1 | Người nộp thuế có thể tiếp cận bộ tài liệu hướng dẫn của Cục Thuế bằng cách nào? | Config B | 0.60 | 0.65 | 0.00 | 0.40 | retrieval | Trùng lặp từ khóa ("người nộp thuế", "tài liệu hướng dẫn") với các điều khoản dài trong Nghị định 252 khiến BM25 đẩy các điều khoản luật lên đầu và đẩy chunk bài báo chứa thông tin "quét mã QR" xuống ngoài top 5. |
|   2 | Vì sao thương mại điện tử làm tăng nguy cơ thất thu thuế? | Config A | 0.70 | 0.72 | 1.00 | 0.60 | generation | Ngữ cảnh trả về chứa số liệu tăng trưởng thương mại điện tử và các khó khăn quản lý, nhưng mô hình LLM tổng hợp diễn đạt hơi khái quát, chưa nêu bật trực diện các cơ chế trốn thuế ẩn danh. |
|   3 | Bộ tài liệu do Cục Thuế phát hành ngày 20/7/2026 tập trung vào những nội dung nào? | Config B | 0.75 | 0.78 | 1.00 | 0.70 | data | Chunk văn bản thu thập từ bài báo chứa nhiều khoảng trắng thừa và phân đoạn ngắt dòng rời rạc, làm giảm mật độ thông tin trích xuất của generator. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | Bổ sung Metadata Filtering (phân loại luật / tin tức) trước khi retrieve | Câu hỏi số 1 bị văn bản luật lấn át thông tin từ bài báo tin tức do không phân vùng tìm kiếm | Tăng Context Recall của Config B lên 100%, khắc phục hiện tượng nhiễu ngữ cảnh giữa các loại tài liệu khác nhau | Chạy lại `group_project/evaluate.py` với bộ lọc category và kiểm tra Recall đạt 15/15 |
|        2 | Chuẩn hóa tiền xử lý văn bản (Text Cleaning & Normalization) | Phân tích câu hỏi số 3 cho thấy dữ liệu markdown bài báo còn nhiều ký tự thừa và dòng trống | Cải thiện độ sạch của context, tăng Context Precision từ 0.89 lên >0.92 và tăng Faithfulness | Đánh giá lại độ dài chunk trung bình và chạy lại bộ đo Faithfulness |
|        3 | Tích hợp Cross-Encoder Reranker chuyên biệt thay vì thuần RRF | Các câu hỏi truy vấn dài có thể hưởng lợi từ việc chấm điểm tương đồng trực tiếp giữa query và toàn bộ văn bản chunk | Tăng Answer Relevance và Context Precision thêm 5-10% | Thực nghiệm A/B giữa BM25+Dense+RRF và BM25+Dense+BGE-Reranker trên 15 câu golden dataset |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| PageIndex Fallback (Vectorless Fallback khi Cosine Score < 0.30) | Config A (Dense-only) | Recall duy trì 100% trên out-of-domain queries | +0.12s khi kích hoạt fallback | Cơ chế fallback hoạt động hiệu quả khi gặp câu hỏi không khớp với vector store, đảm bảo pipeline không bị crash và trả lời an toàn |
