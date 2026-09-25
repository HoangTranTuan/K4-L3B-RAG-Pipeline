"""
Task 2 — Crawl bài viết/thông báo.

Hướng dẫn:
    1. Điền tối thiểu 5 URL công khai vào ARTICLE_URLS.
    2. Crawl từng URL bằng Crawl4AI.
    3. Lưu mỗi bài thành một JSON trong data/landing/news/.
    4. Giữ đủ url, title, date_crawled và content_markdown.

Cài browser trước khi chạy:
    python -m playwright install chromium
    
-> Dùng Firecrawl or bất cứ công cụ nào bạn quen    
"""

import asyncio
import json
import sys
from pathlib import Path

# Cấu hình UTF-8 cho console Windows để in tiếng Việt không lỗi charmap
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://baochinhphu.vn/quy-dinh-moi-ve-dang-ky-ho-kinh-doanh-102250702150908133.htm?utm_",
    "https://baochinhphu.vn/go-vuong-cho-ho-kinh-doanh-khi-chuyen-tu-thue-khoan-sang-hoa-don-dien-tu-102250617190605619.htm",
    "https://baochinhphu.vn/ra-mat-bo-tai-lieu-va-giai-phap-ho-tro-tuan-thu-chinh-sach-thue-102260720175535264.htm?utm_",
    "https://baochinhphu.vn/thue-thuong-mai-dien-tu-bai-toan-quan-ly-va-chong-that-thu-102260811105246558.htm?utm_",
    "https://thanglong.baochinhphu.vn/loi-ich-thiet-thuc-khi-su-dung-hoa-don-dien-tu-khoi-tao-tu-may-tinh-tien-103250509112012314.htm",
]


async def crawl_article(url: str) -> dict:
    """Crawl nội dung bài viết từ URL và trả về dict theo format quy định."""
    from datetime import datetime

    # 1. Thử dùng Crawl4AI nếu môi trường đã cài đặt Playwright
    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            title = result.metadata.get("title") or "Unknown"
            content = result.markdown or ""
            if content.strip():
                return {
                    "url": url,
                    "title": title.strip(),
                    "date_crawled": datetime.now().isoformat(),
                    "content_markdown": content.strip(),
                }
    except Exception as error:
        print(f"Crawl4AI không khả dụng hoặc lỗi ({error}), chuyển sang fallback HTTP...")

    # 2. Fallback dùng requests + regex trích xuất nội dung clean markdown
    import re
    from html import unescape
    import requests

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    html_text = response.text

    # Lấy title
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.IGNORECASE | re.DOTALL)
    title = unescape(title_match.group(1)).strip() if title_match else url

    # Loại bỏ script, style, nav, footer, header
    cleaned = re.sub(r"<(script|style|nav|footer|header|noscript)[^>]*>.*?</\1>", " ", html_text, flags=re.IGNORECASE | re.DOTALL)
    # Chuyển đổi các thẻ heading, đoạn văn, danh sách
    cleaned = re.sub(r"<h[1-2][^>]*>(.*?)</h[1-2]>", r"\n\n## \1\n\n", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<h[3-6][^>]*>(.*?)</h[3-6]>", r"\n\n### \1\n\n", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<p[^>]*>(.*?)</p>", r"\n\n\1\n\n", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<li[^>]*>(.*?)</li>", r"\n- \1", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<br\s*/?>", r"\n", cleaned, flags=re.IGNORECASE)
    # Xoá các thẻ HTML còn lại
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = unescape(cleaned)

    # Chuẩn hoá khoảng trắng
    content_markdown = re.sub(r"[ \t]+", " ", cleaned)
    content_markdown = re.sub(r"\n{3,}", "\n\n", content_markdown).strip()

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content_markdown,
    }


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành một file JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for index, url in enumerate(ARTICLE_URLS, 1):
        try:
            article = await crawl_article(url)
            output = DATA_DIR / f"article_{index:02d}.json"
            output.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved: {output}")
        except Exception as error:
            print(f"Failed: {url} — {error}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
