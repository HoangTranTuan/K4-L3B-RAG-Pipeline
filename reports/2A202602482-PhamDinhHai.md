# Individual contribution report

- **Họ và tên:** Phạm Đình Hải
- **Mã học viên:** 2A202602482
- **Nhóm:** Nhóm Dự Án RAG Pipeline — Day 8
- **Repository/branch:** `https://github.com/HoangTranTuan/K4-L3B-RAG-Pipeline` / nhánh `hai` (PR #1 đã merge vào `main`)

---

## 1. Phần việc đã thực hiện

| Module / Deliverable | Việc tôi trực tiếp làm | File / Commit / PR | Trạng thái |
|---|---|---|:---:|
| **Task 1: Thu thập văn bản pháp lý** | Tìm kiếm, chọn lọc và tải 3 văn bản quy định chính thức (định dạng DOCX) về đăng ký kinh doanh và thuế hộ kinh doanh (>1KB/file). | `data/landing/legal/`<br>Commit `67bb81e` | **Done** |
| **Task 2: Crawl bài viết tin tức** | Lập danh sách 5 URL bài viết từ Báo Chính Phủ; lập trình hàm `crawl_article` hỗ trợ Crawl4AI và fallback HTTP Requests + Regex; xử lý triệt để lỗi bảng mã UTF-8 trên Windows; xuất 5 file JSON đầy đủ metadata. | [`src/task2_crawl_news.py`](../src/task2_crawl_news.py)<br>`data/landing/news/*.json`<br>Commit `67bb81e` | **Done** |
| **Task 3: Chuẩn hóa sang Markdown** | Viết hàm `convert_legal_docs` dùng MarkItDown (mammoth) convert 3 file DOCX sang `.md` (>100k ký tự/file); viết `convert_news_articles` format lại header metadata; đảm bảo cấu trúc thư mục và chuẩn độ dài >200 ký tự. | [`src/task3_convert_markdown.py`](../src/task3_convert_markdown.py)<br>`data/standardized/`<br>Commit `67bb81e` | **Done** |
| **Task 4: Chunking, Embedding & Indexing** | Thiết kế hàm `chunk_documents` bằng Recursive Splitter (`chunk_size=500, overlap=50`) tạo 2,305 chunks; xây dựng `clean_whitespace` chuẩn hóa khoảng trắng trước khi embed; tích hợp OpenAI `text-embedding-3-small` (1536-dim) và nạp 2,305 chunks vào ChromaDB với cosine distance. | [`src/task4_chunking_indexing.py`](../src/task4_chunking_indexing.py)<br>`chroma_db/`<br>Commit `67bb81e` | **Done** |
| **Kế hoạch dự án & Git** | Khởi tạo bảng kế hoạch, checklist nghiệm thu [`TASKS.md`](../TASKS.md), bổ sung `.gitignore` bảo vệ database; tạo nhánh `hai`, commit và tạo PR #1 merge vào `main`. | [`TASKS.md`](../TASKS.md), [`.gitignore`](../.gitignore)<br>PR #1 (Merge `a3cd61b`) | **Done** |

---

## 2. Quyết định kỹ thuật quan trọng

### Quyết định 1: Xây dựng cơ chế HTTP Fallback và cấu hình UTF-8 cho Crawler
* **Lý do / Evidence:** Thư viện Crawl4AI phụ thuộc trình duyệt Playwright headless dễ phát sinh lỗi môi trường và tải nặng. Bên cạnh đó, môi trường Windows mặc định sử dụng bảng mã CP1252 dẫn đến lỗi `'charmap' codec can't encode character` khi in tiếng Việt có dấu ra console.
* **Giải pháp:** Cài đặt `sys.stdout.reconfigure(encoding="utf-8")` và bổ sung tầng fallback sử dụng `requests` kết hợp regex clean HTML chuyển đổi trực tiếp sang Markdown.
* **Trade-off:** Parser regex không xử lý được các trang render hoàn toàn bằng Single Page Application (SPA), tuy nhiên với các trang báo chính thống (`baochinhphu.vn`), HTML phản hồi từ server đã có đầy đủ nội dung bài viết, giúp tốc độ crawl nhanh gấp 5 lần và loại bỏ hoàn toàn rủi ro crash tiến trình do thiếu browser.

### Quyết định 2: Tích hợp OpenAI Embedding (`text-embedding-3-small`) và làm sạch khoảng trắng
* **Lý do / Evidence:** Khi thử nghiệm embedding 2,305 chunks với Gemini API Free Tier, hệ thống liên tục chạm giới hạn tần suất 15 Requests/phút (15 RPM) và trần ngày 1,000 units, dẫn đến lỗi `429 RESOURCE_EXHAUSTED` và tốn nhiều thời gian chờ đợi backoff.
* **Giải pháp:** Tích hợp bộ chuyển đổi sang OpenAI API (`text-embedding-3-small`, batch size 500 chunks/request), đồng thời áp dụng hàm `clean_whitespace(text)` loại bỏ mọi khoảng trắng/ký tự xuống dòng dư thừa trước khi tạo vector.
* **Trade-off:** Cần API Key OpenAI có credit trả phí thay vì miễn phí như Gemini, đổi lại thời gian xử lý toàn bộ 2,305 chunks giảm từ hơn 10 phút xuống còn dưới 30 giây, vector 1,536 chiều có chất lượng biểu diễn ngữ nghĩa cao và ổn định tuyệt đối.

---

## 3. Kiểm thử và kết quả

* **Các bài test đã thực thi và vượt qua:**
  - `pytest tests/test_acceptance.py -k test_corpus_has_required_legal_documents`: Đạt 3/3 tài liệu DOCX hợp lệ.
  - `pytest tests/test_acceptance.py -k test_corpus_has_required_news_with_metadata`: Đạt 5/5 bài viết JSON đầy đủ trường `url`, `title`, `date_crawled`, `content_markdown`.
  - `pytest tests/test_acceptance.py -k test_standardized_output_covers_both_source_types`: Toàn bộ 8 file Markdown có độ dài từ 6,600 đến 315,000 ký tự (vượt xa tiêu chuẩn $\ge 200$ ký tự).
  - `pytest tests/test_contracts.py -k "chunk or signature"`: Chữ ký hàm và tính toàn vẹn của chunk (độ dài $\le 550$, `chunk_index`, metadata duy nhất) đạt chuẩn 100%.
* **Lỗi phát hiện và xử lý trong quá trình làm:**
  - *Lỗi mạng DNS:* Máy tính kết nối Wi-Fi VinUni Guest bị chặn cổng 53 tới Google DNS 8.8.8.8 $\rightarrow$ chuyển đổi DNS sang Cloudflare `1.1.1.1` qua DHCP.
  - *Lỗi thiếu thư viện DOCX của MarkItDown:* Cài đặt bổ sung thư viện `mammoth` để kích hoạt bộ chuyển đổi DOCX sang Markdown.
  - *Lỗi Rate limit 429:* Chuyển đổi sang pipeline OpenAI embedding với batching tối ưu.

---

## 4. Điều còn hạn chế & Hướng phát triển

* **Hạn chế:** Các tài liệu pháp lý (Nghị định 168, 252, 254) có độ dài rất lớn. Việc phân đoạn văn bản bằng `RecursiveCharacterTextSplitter` thuần túy theo độ dài ký tự (500 ký tự) đôi khi có thể chia cắt giữa chừng một điều khoản luật hoặc một danh mục bảng biểu quy định.
* **Hướng cải tiến nếu có thêm thời gian:** Tôi sẽ xây dựng một bộ tách đoạn theo ngữ pháp văn bản quy phạm pháp luật (Rule-based Legal Chunker), nhận diện các cấp bậc "Chương $\rightarrow$ Mục $\rightarrow$ Điều $\rightarrow$ Khoản $\rightarrow$ Điểm" để mỗi chunk đại diện trọn vẹn cho một quy phạm pháp luật, từ đó nâng cao độ chính xác khi truy vấn ngữ nghĩa (Semantic Search).

---

## 5. Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc tôi trực tiếp thực hiện trong nhóm và có thể giải thích, demo chi tiết toàn bộ các module từ Task 1 đến Task 4.

- **Ngày:** 25/09/2026
- **Tên thành viên:** Phạm Đình Hải (2A202602482)
