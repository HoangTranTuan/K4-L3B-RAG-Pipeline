"""
app.py — RAG Pipeline Explorer & Chatbot UI.

Ứng dụng Web độc lập (Standalone Web Application) không phụ thuộc vào Streamlit,
sử dụng Python HTTP server kết hợp giao diện HTML5, Tailwind CSS và JavaScript hiện đại.
Bao gồm:
    1. Giao diện người dùng (User Input & Controls)
    2. Sơ đồ luồng hệ thống thời gian thực (System Flow Architecture Trace)
    3. Bảng xếp hạng chi tiết theo hàm Cosine Similarity, BM25 và RRF Fusion
    4. Cổng kiểm tra ngưỡng Cosine Gate (Fallback Decision Indicator)
    5. Output câu trả lời từ mô hình (LLM Answer) kèm trích dẫn (In-text Citations & Sources)
"""

import json
import os
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Nạp biến môi trường từ .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import types
    d = types.ModuleType("dotenv")
    d.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = d

    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

# Import các module cốt lõi từ src
import math
import re
from collections import Counter

from src.task10_generation import (
    SAFE_REFUSAL,
    call_llm,
    format_context,
    reorder_for_llm,
)
from src.task9_retrieval_pipeline import retrieve

# ============================================================
# Quản lý & nạp dữ liệu thực tế từ thư mục data/ (standardized)
# ============================================================
DATA_ROOT = Path(__file__).parent / "data"
STANDARDIZED_DIR = DATA_ROOT / "standardized"

_GLOBAL_CHUNKS_CACHE: list[dict] = []
_GLOBAL_BM25_INDEX = None
_GLOBAL_DENSE_INDEX = None


def tokenize_words(text: str) -> list[str]:
    """Tokenize nhẹ không phụ thuộc vào thư viện ngoài."""
    return re.findall(r"\w+", text.lower())


def load_real_corpus_chunks() -> list[dict]:
    """
    Đọc toàn bộ tài liệu Markdown thực tế trong data/standardized/ (hoặc data/)
    và phân đoạn (chunking) theo chuẩn size=500, overlap=50.
    """
    global _GLOBAL_CHUNKS_CACHE
    if _GLOBAL_CHUNKS_CACHE:
        return _GLOBAL_CHUNKS_CACHE

    # Thử gọi hàm load chuẩn của task 4 nếu có sẵn
    try:
        from src.task4_chunking_indexing import chunk_documents, load_documents
        docs = load_documents()
        if docs:
            c = chunk_documents(docs)
            if c:
                _GLOBAL_CHUNKS_CACHE = c
                return _GLOBAL_CHUNKS_CACHE
    except Exception:
        pass

    chunks: list[dict] = []
    target_dir = STANDARDIZED_DIR if STANDARDIZED_DIR.exists() else DATA_ROOT
    md_files = sorted(target_dir.rglob("*.md"))

    chunk_size = 500
    chunk_overlap = 50

    for path in md_files:
        try:
            content = path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            continue
        if not content:
            continue

        doc_type = "legal" if "legal" in path.parts else "news"
        title = path.stem.replace("_", " ")
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()

        source_url = ""
        url_match = re.search(r"\*\*Source:\*\*\s*(\S+)", content)
        if url_match:
            source_url = url_match.group(1).strip()

        start = 0
        chunk_idx = 0
        length = len(content)
        while start < length:
            end = min(start + chunk_size, length)
            chunk_text = content[start:end].strip()
            if chunk_text:
                rel_id = f"{path.name}::chunk-{chunk_idx}"
                chunks.append(
                    {
                        "id": rel_id,
                        "content": chunk_text,
                        "metadata": {
                            "source": path.name,
                            "title": title,
                            "doc_type": doc_type,
                            "url": source_url,
                            "chunk_index": chunk_idx,
                        },
                    }
                )
                chunk_idx += 1
            if end >= length:
                break
            start += chunk_size - chunk_overlap

    _GLOBAL_CHUNKS_CACHE = chunks
    return _GLOBAL_CHUNKS_CACHE


class RealCorpusBM25:
    """Bộ chỉ mục BM25Okapi thuần Python chạy trực tiếp trên các chunk thực tế trong data/."""
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.tokenized = [
            tokenize_words(c["content"] + " " + c["metadata"].get("title", ""))
            for c in chunks
        ]
        self.doc_len = [len(doc) for doc in self.tokenized]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 1.0
        self.doc_freqs = [Counter(doc) for doc in self.tokenized]
        self.doc_count = len(self.tokenized)

        df = Counter()
        for doc in self.tokenized:
            df.update(set(doc))
        self.idf = {
            t: math.log(1.0 + (self.doc_count - f + 0.5) / (f + 0.5))
            for t, f in df.items()
        }

    def search(self, query: str, top_k: int = 10, k1: float = 1.5, b: float = 0.75) -> list[dict]:
        q_tokens = tokenize_words(query)
        if not q_tokens:
            return []
        scores = [0.0] * self.doc_count
        for t in q_tokens:
            if t not in self.idf:
                continue
            term_idf = self.idf[t]
            for idx, freqs in enumerate(self.doc_freqs):
                tf = freqs.get(t, 0)
                if tf > 0:
                    denom = tf + k1 * (1.0 - b + b * self.doc_len[idx] / self.avgdl)
                    scores[idx] += term_idf * (tf * (k1 + 1.0)) / denom

        ranked = sorted(range(self.doc_count), key=lambda i: scores[i], reverse=True)
        results = []
        for rank, idx in enumerate(ranked[:top_k], 1):
            if scores[idx] > 0:
                c = dict(self.chunks[idx])
                c["score"] = round(scores[idx], 4)
                c["rank"] = rank
                c["retrieval_method"] = "bm25"
                results.append(c)
        return results


class RealCorpusDenseCosine:
    """Bộ tính điểm tương đồng Cosine ngữ nghĩa (Semantic Subword/N-gram Cosine) chuẩn [0.0, 1.0]."""
    def __init__(self, chunks: list[dict], n: int = 3):
        self.chunks = chunks
        self.n = n
        self.ngrams = []
        self.norms = []
        for c in chunks:
            text = (c["content"] + " " + c["metadata"].get("title", "")).lower()
            clean = re.sub(r"\s+", " ", text).strip()
            ng = Counter([clean[i : i + n] for i in range(max(0, len(clean) - n + 1))])
            norm = math.sqrt(sum(v * v for v in ng.values())) or 1.0
            self.ngrams.append(ng)
            self.norms.append(norm)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        clean_q = re.sub(r"\s+", " ", query.lower()).strip()
        q_ng = Counter([clean_q[i : i + self.n] for i in range(max(0, len(clean_q) - self.n + 1))])
        q_norm = math.sqrt(sum(v * v for v in q_ng.values())) or 1.0
        scores = []
        for idx, (c_ng, c_norm) in enumerate(zip(self.ngrams, self.norms)):
            common = set(q_ng.keys()) & set(c_ng.keys())
            dot = sum(q_ng[k] * c_ng[k] for k in common)
            cosine = dot / (q_norm * c_norm)
            scores.append((idx, cosine))
        scores.sort(key=lambda x: x[1], reverse=True)
        res = []
        for rank, (idx, score) in enumerate(scores[:top_k], 1):
            c = dict(self.chunks[idx])
            c["score"] = round(float(score), 4)
            c["rank"] = rank
            c["retrieval_method"] = "dense"
            res.append(c)
        return res


def get_search_indexes():
    global _GLOBAL_BM25_INDEX, _GLOBAL_DENSE_INDEX
    chunks = load_real_corpus_chunks()
    if _GLOBAL_BM25_INDEX is None:
        _GLOBAL_BM25_INDEX = RealCorpusBM25(chunks)
    if _GLOBAL_DENSE_INDEX is None:
        _GLOBAL_DENSE_INDEX = RealCorpusDenseCosine(chunks)
    return _GLOBAL_DENSE_INDEX, _GLOBAL_BM25_INDEX


def run_pipeline_trace(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.3,
    provider: str = "openrouter",
    model: str = "",
) -> dict:
    """
    Chạy toàn bộ pipeline truy xuất và ghi lại các bước trung gian (Trace) trên dữ liệu THỰC TẾ:
    1. Query Analysis
    2. Dense Semantic Search (Cosine)
    3. Lexical Search (BM25)
    4. RRF Fusion (Rank & Scores)
    5. Cosine Gate & Fallback Evaluation
    6. Lost-in-the-middle Reorder
    7. LLM Generation có trích dẫn nguồn
    """
    start_total = time.perf_counter()
    query_lower = query.lower().strip()

    dense_index, bm25_index = get_search_indexes()

    # --- BƯỚC 1 & 2: Thực thi Dense & BM25 trên dữ liệu thật ---
    t0_retrieval = time.perf_counter()
    dense_results = []
    sparse_results = []

    # 1. Dense Semantic Search
    try:
        from src.task5_semantic_search import semantic_search
        dense_results = semantic_search(query, top_k=top_k * 2)
    except Exception:
        dense_results = []

    if not dense_results:
        dense_results = dense_index.search(query, top_k=top_k * 2)

    dense_time_ms = round((time.perf_counter() - t0_retrieval) * 1000, 2)

    # 2. BM25 Lexical Search
    t0_bm25 = time.perf_counter()
    try:
        from src.task6_lexical_search import lexical_search
        sparse_results = lexical_search(query, top_k=top_k * 2)
    except Exception:
        sparse_results = []

    if not sparse_results:
        sparse_results = bm25_index.search(query, top_k=top_k * 2)

    lexical_time_ms = round((time.perf_counter() - t0_bm25) * 1000, 2)

    # Gán thứ hạng hiển thị
    for r, item in enumerate(dense_results, 1):
        item["rank"] = r
    for r, item in enumerate(sparse_results, 1):
        item["rank"] = r

    # --- BƯỚC 3: RRF Fusion ---
    t0_rrf = time.perf_counter()
    k_rrf = 60
    rrf_scores = {}
    item_map = {}

    for ranked_list in [dense_results, sparse_results]:
        for rank, item in enumerate(ranked_list, 1):
            i_id = item["id"]
            rrf_scores[i_id] = rrf_scores.get(i_id, 0.0) + 1.0 / (k_rrf + rank)
            item_map[i_id] = item

    ranked_ids = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)
    rrf_results = []
    for rank, i_id in enumerate(ranked_ids[:top_k], 1):
        cp = item_map[i_id].copy()
        cp["score"] = round(rrf_scores[i_id], 5)
        cp["rank"] = rank
        cp["retrieval_method"] = "hybrid"
        rrf_results.append(cp)

    rrf_time_ms = round((time.perf_counter() - t0_rrf) * 1000, 2)

    # --- BƯỚC 4: Gate Đa Điều Kiện (OR Logic) ---
    best_dense_score = dense_results[0]["score"] if dense_results else 0.0
    best_bm25_score = sparse_results[0]["score"] if sparse_results else 0.0

    cosine_passed = best_dense_score >= score_threshold
    bm25_exact_match = False
    bm25_match_reason = ""

    if sparse_results:
        top_sparse = sparse_results[0]
        doc_haystack = (top_sparse.get("content", "") + " " + top_sparse.get("metadata", {}).get("title", "")).lower()

        # Phát hiện từ khóa định danh trong câu hỏi (số văn bản, điều luật, con số)
        id_patterns = re.findall(
            r'(?:điều\s+\d+|khoản\s+\d+|nghị\s+định\s+[\w/]+|thông\s+tư\s+[\w/]+|luật\s+[\w/]+|\b\d+/\d+[\w/]*|\b\d+\s+triệu|\b\d+\s+tỷ|\b\d+\b)',
            query_lower,
        )
        matched_ids = [p.strip() for p in id_patterns if p.strip() in doc_haystack and len(p.strip()) > 1]

        stopwords = {
            "cho", "của", "các", "những", "được", "trong", "phải", "không", "nào", "là",
            "gì", "thế", "ở", "và", "với", "có", "như", "khi", "thì", "về", "ra", "sao"
        }
        content_words = [w for w in re.findall(r'\b\w+\b', query_lower) if len(w) > 2 and w not in stopwords]
        all_content_words_matched = (
            len(content_words) >= 2 and all(w in doc_haystack for w in content_words)
        )

        if matched_ids:
            bm25_exact_match = True
            bm25_match_reason = f"Khớp định danh '{matched_ids[0]}'"
        elif all_content_words_matched:
            bm25_exact_match = True
            bm25_match_reason = "Khớp tuyệt đối từ khóa chính"

    passed_gate = cosine_passed or bm25_exact_match
    decision = "hybrid" if passed_gate else "pageindex"

    comp_op = ">" if best_dense_score >= score_threshold else "<"
    compare_str = f"{best_dense_score:.3f} {comp_op} {score_threshold:.3f}"

    if cosine_passed and bm25_exact_match:
        gate_trigger = "both"
        gate_message = f"Cả 2 điều kiện đều đạt: Cosine ({compare_str}) & BM25 ({bm25_match_reason}) -> Dùng Hybrid RRF"
    elif cosine_passed:
        gate_trigger = "cosine"
        gate_message = f"Cosine vượt ngưỡng ({compare_str}) -> Dùng Hybrid RRF"
    elif bm25_exact_match:
        gate_trigger = "bm25_exact"
        gate_message = f"BM25 phát hiện từ khóa định danh khớp tuyệt đối ({bm25_match_reason}) -> Vượt cổng qua điều kiện OR"
    else:
        gate_trigger = "none"
        gate_message = f"Không đạt điều kiện OR: Cosine ({compare_str}) và BM25 không khớp định danh -> Kích hoạt Fallback"

    final_chunks = rrf_results
    fallback_success = False
    if not passed_gate:
        # Thử gọi PageIndex fallback thật
        try:
            from src.task8_pageindex_vectorless import pageindex_search
            fallback_res = pageindex_search(query, top_k=top_k)
            if fallback_res:
                final_chunks = fallback_res
                fallback_success = True
        except Exception:
            pass

    # --- BƯỚC 5: Lost-in-the-middle Reordering ---
    reordered_chunks = reorder_for_llm(final_chunks) if final_chunks else []

    # --- BƯỚC 6: LLM Generation ---
    t0_llm = time.perf_counter()
    answer_text = ""

    effective_provider = provider or os.getenv("LLM_PROVIDER", "openrouter")
    effective_model = model or os.getenv("LLM_MODEL", "google/gemini-2.0-flash-001")
    if "embedding" in effective_model.lower():
        effective_model = "google/gemini-2.0-flash-001"

    if not final_chunks:
        answer_text = SAFE_REFUSAL
    elif not passed_gate and not fallback_success:
        # Không vượt qua cổng và không có fallback hợp lệ -> Safe Refusal
        answer_text = SAFE_REFUSAL
    else:
        context = format_context(reordered_chunks)
        sys_prompt = (
            "Bạn là trợ lý pháp luật và thuế thông minh của VinUni. Trả lời câu hỏi ngắn gọn, chính xác "
            "dựa trên Context được cung cấp. Bắt buộc trích dẫn bằng mã [Document X] hoặc [X] "
            "ngay sau các khẳng định quan trọng."
        )
        user_msg = f"Context:\n{context}\n\nQuestion: {query}"

        if effective_provider in ("mock", "grounded"):
            # Chế độ trích xuất dẫn chứng trực tiếp từ tài liệu
            parts = []
            for i, c in enumerate(reordered_chunks[:top_k], 1):
                clean_text = c["content"].strip().replace("\n", " ")
                if len(clean_text) > 220:
                    clean_text = clean_text[:220] + "..."
                parts.append(f"{clean_text} [{i}]")
            answer_text = "\n\n".join(parts)
        else:
            try:
                os.environ["LLM_PROVIDER"] = effective_provider
                if effective_model:
                    os.environ["LLM_MODEL"] = effective_model
                answer_text = call_llm(sys_prompt, user_msg)
            except Exception as e:
                print(f"[app.py] LLM Error: {e}, fallback to verified grounded synthesis.")
                parts = []
                for i, c in enumerate(reordered_chunks[:top_k], 1):
                    clean_text = c["content"].strip().replace("\n", " ")
                    if len(clean_text) > 220:
                        clean_text = clean_text[:220] + "..."
                    parts.append(f"{clean_text} [{i}]")
                answer_text = "\n\n".join(parts)

        if not answer_text or not answer_text.strip():
            answer_text = SAFE_REFUSAL

    llm_time_ms = round((time.perf_counter() - t0_llm) * 1000, 2)
    total_time_ms = round((time.perf_counter() - start_total) * 1000, 2)

    return {
        "query": query,
        "timing": {
            "total_ms": total_time_ms,
            "dense_ms": dense_time_ms,
            "bm25_ms": lexical_time_ms,
            "rrf_ms": rrf_time_ms,
            "llm_ms": llm_time_ms,
        },
        "cosine_gate": {
            "best_dense_score": float(best_dense_score),
            "best_bm25_score": float(best_bm25_score),
            "score_threshold": float(score_threshold),
            "cosine_passed": bool(cosine_passed),
            "bm25_exact_match": bool(bm25_exact_match),
            "bm25_match_reason": bm25_match_reason,
            "compare_operator": comp_op,
            "compare_str": compare_str,
            "passed": bool(passed_gate),
            "decision": decision,
            "trigger": gate_trigger,
            "message": gate_message,
        },
        "dense_candidates": dense_results[:top_k],
        "sparse_candidates": sparse_results[:top_k],
        "rrf_candidates": rrf_results[:top_k],
        "generation": {
            "answer": answer_text,
            "retrieval_source": decision,
            "model": effective_model or "google/gemini-2.0-flash-001",
            "provider": effective_provider,
            "sources": final_chunks,
        },
    }


# Template HTML5 / Modern Responsive UI
HTML_PAGE = """<!DOCTYPE html>
<html lang="vi" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Pháp luật cho hộ kinh doanh — BotVN</title>
  <!-- Tailwind CSS CDN -->
  <script src="https://cdn.tailwindcss.com"></script>
  <!-- FontAwesome Icons -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" />
  <!-- Google Fonts: Inter & JetBrains Mono -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          fontFamily: {
            sans: ['Inter', 'sans-serif'],
            mono: ['JetBrains Mono', 'monospace'],
          },
          colors: {
            brand: {
              50: '#ecfdf5',
              100: '#d1fae5',
              500: '#10b981',
              600: '#059669',
              700: '#047857',
            },
            dark: {
              900: '#0b0f19',
              800: '#111827',
              700: '#1f2937',
              600: '#374151',
            }
          }
        }
      }
    }
  </script>
  <style>
    body {
      background-color: #0b0f19;
      color: #f3f4f6;
    }
    .glass-card {
      background: rgba(17, 24, 39, 0.75);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .flow-connector::after {
      content: '';
      position: absolute;
      right: -16px;
      top: 50%;
      transform: translateY(-50%);
      width: 16px;
      height: 2px;
      background: linear-gradient(90deg, #10b981, #06b6d4);
      z-index: 10;
    }
    .citation-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      background: rgba(16, 185, 129, 0.2);
      border: 1px solid rgba(16, 185, 129, 0.4);
      color: #34d399;
      font-weight: 600;
      font-size: 0.75rem;
      padding: 0.1rem 0.4rem;
      border-radius: 9999px;
      margin: 0 0.2rem;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .citation-badge:hover {
      background: #10b981;
      color: #064e3b;
      transform: scale(1.08);
    }
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
      width: 6px;
      height: 6px;
    }
    ::-webkit-scrollbar-track {
      background: #0b0f19;
    }
    ::-webkit-scrollbar-thumb {
      background: #1f2937;
      border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
      background: #374151;
    }
  </style>
</head>
<body class="min-h-screen flex flex-col font-sans selection:bg-brand-500 selection:text-white">

  <!-- Top Navigation Bar -->
  <header class="sticky top-0 z-50 glass-card border-b border-gray-800 px-6 py-3.5 flex items-center justify-between">
    <div class="flex items-center space-x-3">
      <div class="h-10 w-10 rounded-xl bg-gradient-to-tr from-emerald-500 to-cyan-500 flex items-center justify-center shadow-lg shadow-emerald-500/20">
        <i class="fa-solid fa-cube text-white text-lg"></i>
      </div>
      <div>
        <div class="flex items-center space-x-2">
          <h1 class="text-lg font-bold text-white tracking-tight">Pháp luật cho hộ kinh doanh</h1>
          <span class="px-2 py-0.5 text-xs font-semibold rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">Day 8 Live</span>
        </div>
        <p class="text-xs text-gray-400">BotVN</p>
      </div>
    </div>

    <!-- Right info & status -->
    <div class="flex items-center space-x-4">
      <div class="hidden md:flex items-center space-x-2 text-xs text-gray-400 bg-gray-900/80 px-3 py-1.5 rounded-lg border border-gray-800 font-mono">
        <span class="h-2 w-2 rounded-full bg-emerald-400 animate-pulse"></span>
        <span id="header-status">OpenRouter Ready</span>
      </div>
      <a href="https://openrouter.ai" target="_blank" class="text-xs text-gray-400 hover:text-white transition flex items-center space-x-1.5 px-3 py-1.5 rounded-lg hover:bg-gray-800">
        <i class="fa-solid fa-bolt text-amber-400"></i>
        <span>OpenRouter</span>
      </a>
    </div>
  </header>

  <!-- Main Workspace -->
  <main class="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6 space-y-6">

    <!-- 1. USER INPUT SECTION -->
    <section class="glass-card rounded-2xl p-5 shadow-2xl relative overflow-hidden">
      <div class="absolute -right-16 -top-16 w-64 h-64 bg-emerald-500/5 rounded-full blur-3xl pointer-events-none"></div>
      
      <div class="space-y-4">
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-3">
          <label for="query-input" class="text-sm font-semibold text-gray-200 flex items-center space-x-2">
            <i class="fa-solid fa-terminal text-emerald-400"></i>
            <span>Truy Vấn Quan Sát (User Query Input)</span>
          </label>
          
          <!-- Controls bar -->
          <div class="flex flex-wrap items-center gap-3 text-xs">
            <div class="flex items-center space-x-2 bg-gray-900/90 px-3 py-1.5 rounded-lg border border-gray-800">
              <span class="text-gray-400">Top K Chunks:</span>
              <span id="top-k-val" class="font-mono text-emerald-400 font-semibold">5</span>
              <input type="range" id="top-k-slider" min="3" max="10" value="5" class="w-20 accent-emerald-500 cursor-pointer">
            </div>

            <div class="flex items-center space-x-2 bg-gray-900/90 px-3 py-1.5 rounded-lg border border-gray-800">
              <span class="text-gray-400">Cosine Gate:</span>
              <span id="threshold-val" class="font-mono text-amber-400 font-semibold">0.30</span>
              <input type="range" id="threshold-slider" min="0.1" max="0.9" step="0.05" value="0.30" class="w-20 accent-amber-500 cursor-pointer">
            </div>

            <div class="flex items-center space-x-2 bg-gray-900/90 px-3 py-1.5 rounded-lg border border-gray-800">
              <i class="fa-solid fa-microchip text-cyan-400"></i>
              <select id="model-select" class="bg-transparent text-gray-200 focus:outline-none cursor-pointer">
                <option value="google/gemini-2.0-flash-001" class="bg-gray-900 text-white">Gemini 2.0 Flash (OpenRouter)</option>
                <option value="openai/gpt-4o-mini" class="bg-gray-900 text-white">GPT-4o Mini (OpenRouter)</option>
                <option value="deepseek/deepseek-chat" class="bg-gray-900 text-white">DeepSeek V3 (OpenRouter)</option>
                <option value="grounded" class="bg-gray-900 text-white">Trích xuất trực tiếp (No-LLM)</option>
              </select>
            </div>
          </div>
        </div>

        <!-- Big Search Bar -->
        <div class="relative flex items-center">
          <div class="absolute left-4 text-gray-400">
            <i class="fa-solid fa-magnifying-glass text-lg"></i>
          </div>
          <input
            type="text"
            id="query-input"
            value=""
            placeholder="Nhập câu hỏi tra cứu pháp luật & thuế (VD: Hóa đơn điện tử máy tính tiền, thuế thương mại điện tử, đăng ký kinh doanh)..."
            class="w-full pl-12 pr-36 py-4 bg-gray-900/90 border border-gray-700/80 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent text-sm md:text-base font-medium shadow-inner transition"
          />
          <button
            id="btn-run"
            onclick="executePipeline()"
            class="absolute right-2 px-5 py-2.5 bg-gradient-to-r from-emerald-500 to-cyan-500 hover:from-emerald-600 hover:to-cyan-600 text-white font-semibold text-sm rounded-lg shadow-lg shadow-emerald-500/25 flex items-center space-x-2 transition-all transform active:scale-95"
          >
            <span id="btn-text">Chạy Pipeline</span>
            <i id="btn-icon" class="fa-solid fa-play text-xs"></i>
          </button>
        </div>

        <!-- Quick Query Suggestions from real data -->
        <div class="flex flex-wrap items-center gap-2 pt-1 text-xs">
          <span class="text-gray-400 font-medium">Gợi ý câu hỏi thực tế:</span>
          <button type="button" onclick="setQuery('Hóa đơn điện tử khởi tạo từ máy tính tiền mang lại những lợi ích gì cho hộ kinh doanh?')" class="px-2.5 py-1 rounded-md bg-gray-800/80 hover:bg-emerald-500/20 text-gray-300 hover:text-emerald-300 border border-gray-700/60 transition">
            Hóa đơn điện tử máy tính tiền
          </button>
          <button type="button" onclick="setQuery('Từ ngày 1/7/2026, những văn bản nào tạo khung pháp lý mới cho hoạt động thương mại điện tử?')" class="px-2.5 py-1 rounded-md bg-gray-800/80 hover:bg-cyan-500/20 text-gray-300 hover:text-cyan-300 border border-gray-700/60 transition">
            Khung pháp lý TMĐT từ 1/7/2026
          </button>
          <button type="button" onclick="setQuery('Quy định về đăng ký doanh nghiệp theo Nghị định 168/2025/NĐ-CP?')" class="px-2.5 py-1 rounded-md bg-gray-800/80 hover:bg-purple-500/20 text-gray-300 hover:text-purple-300 border border-gray-700/60 transition">
            Nghị định 168/2025/NĐ-CP
          </button>
          <button type="button" onclick="setQuery('Thời tiết sao Hỏa hôm nay thế nào?')" class="px-2.5 py-1 rounded-md bg-gray-800/80 hover:bg-amber-500/20 text-gray-300 hover:text-amber-300 border border-gray-700/60 transition">
            Thời tiết sao Hỏa (Test Gate &lt; 0.3)
          </button>
        </div>
      </div>
    </section>

    <!-- 2. SYSTEM FLOW ARCHITECTURE (Sơ đồ luồng hệ thống trực quan) -->
    <section class="space-y-3">
      <div class="flex items-center justify-between">
        <h2 class="text-sm font-bold uppercase tracking-wider text-gray-400 flex items-center space-x-2">
          <i class="fa-solid fa-diagram-project text-cyan-400"></i>
          <span>Kiến Trúc Luồng Hệ Thống (Live Retrieval & Synthesis Flow)</span>
        </h2>
        <span id="flow-total-time" class="text-xs font-mono text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-1 rounded-full">
          Total Latency: -- ms
        </span>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-6 gap-3">
        <!-- Node 01: Query -->
        <div class="glass-card rounded-xl p-3.5 border-l-4 border-l-blue-500 flex flex-col justify-between space-y-2">
          <div class="flex items-center justify-between text-xs text-blue-400 font-semibold uppercase">
            <span>01 • Query</span>
            <i class="fa-solid fa-question text-blue-400"></i>
          </div>
          <div class="text-xs text-gray-400 font-medium" id="node-query">
            --
          </div>
          <div class="text-[11px] font-mono text-gray-500 flex items-center justify-between pt-1 border-t border-gray-800">
            <span>Corpus</span>
            <span class="text-gray-300 font-semibold" id="node-corpus-count">1,749 chunks (data/)</span>
          </div>
        </div>

        <!-- Node 02A: Semantic (Dense) -->
        <div class="glass-card rounded-xl p-3.5 border-l-4 border-l-emerald-500 flex flex-col justify-between space-y-2">
          <div class="flex items-center justify-between text-xs text-emerald-400 font-semibold uppercase">
            <span>02A • Dense</span>
            <i class="fa-solid fa-brain text-emerald-400"></i>
          </div>
          <div class="text-xs text-gray-300">
            <span class="font-medium text-emerald-300">Cosine Similarity</span>
            <div class="text-[11px] text-gray-400" id="node-dense-stats">Top candidates</div>
          </div>
          <div class="text-[11px] font-mono text-gray-500 flex items-center justify-between pt-1 border-t border-gray-800">
            <span>Time</span>
            <span class="text-emerald-400 font-semibold" id="node-dense-time">-- ms</span>
          </div>
        </div>

        <!-- Node 02B: Lexical (BM25) -->
        <div class="glass-card rounded-xl p-3.5 border-l-4 border-l-indigo-500 flex flex-col justify-between space-y-2">
          <div class="flex items-center justify-between text-xs text-indigo-400 font-semibold uppercase">
            <span>02B • Lexical</span>
            <i class="fa-solid fa-spell-check text-indigo-400"></i>
          </div>
          <div class="text-xs text-gray-300">
            <span class="font-medium text-indigo-300">BM25 Okapi</span>
            <div class="text-[11px] text-gray-400" id="node-bm25-stats">Top candidates</div>
          </div>
          <div class="text-[11px] font-mono text-gray-500 flex items-center justify-between pt-1 border-t border-gray-800">
            <span>Time</span>
            <span class="text-indigo-400 font-semibold" id="node-bm25-time">-- ms</span>
          </div>
        </div>

        <!-- Node 03: RRF Fusion -->
        <div class="glass-card rounded-xl p-3.5 border-l-4 border-l-purple-500 flex flex-col justify-between space-y-2">
          <div class="flex items-center justify-between text-xs text-purple-400 font-semibold uppercase">
            <span>03 • Fusion</span>
            <i class="fa-solid fa-code-merge text-purple-400"></i>
          </div>
          <div class="text-xs text-gray-300">
            <span class="font-medium text-purple-300">RRF Rank Fusion</span>
            <div class="text-[11px] text-gray-400">Hợp nhất k=60</div>
          </div>
          <div class="text-[11px] font-mono text-gray-500 flex items-center justify-between pt-1 border-t border-gray-800">
            <span>Time</span>
            <span class="text-purple-400 font-semibold" id="node-rrf-time">-- ms</span>
          </div>
        </div>

        <!-- Node 04: Gate (OR Condition) -->
        <div class="glass-card rounded-xl p-3.5 border-l-4 border-l-amber-500 flex flex-col justify-between space-y-2">
          <div class="flex items-center justify-between text-xs text-amber-400 font-semibold uppercase">
            <span>04 • Gate (OR)</span>
            <i class="fa-solid fa-shield-halved text-amber-400"></i>
          </div>
          <div class="text-xs text-gray-300">
            <span class="font-medium" id="node-gate-decision">Cosine HOẶC BM25</span>
            <div class="text-[11px] text-gray-400 font-mono" id="node-gate-compare">-- &gt; 0.300</div>
          </div>
          <div class="text-[11px] font-mono text-gray-500 flex items-center justify-between pt-1 border-t border-gray-800">
            <span>Route</span>
            <span class="text-amber-400 font-semibold" id="node-gate-badge">HYBRID</span>
          </div>
        </div>

        <!-- Node 05: LLM Synthesis -->
        <div class="glass-card rounded-xl p-3.5 border-l-4 border-l-teal-500 flex flex-col justify-between space-y-2">
          <div class="flex items-center justify-between text-xs text-teal-400 font-semibold uppercase">
            <span>05 • Generation</span>
            <i class="fa-solid fa-sparkles text-teal-400"></i>
          </div>
          <div class="text-xs text-gray-300">
            <span class="font-medium text-teal-300" id="node-llm-model">OpenRouter</span>
            <div class="text-[11px] text-gray-400">Trích dẫn citation</div>
          </div>
          <div class="text-[11px] font-mono text-gray-500 flex items-center justify-between pt-1 border-t border-gray-800">
            <span>LLM Time</span>
            <span class="text-teal-400 font-semibold" id="node-llm-time">-- ms</span>
          </div>
        </div>
      </div>
    </section>

    <!-- 3. GATE DECISION BANNER (Cosine >= Threshold OR BM25 Exact Match) -->
    <section id="gate-banner" class="glass-card rounded-xl p-4 border border-emerald-500/30 bg-emerald-950/20 flex flex-col md:flex-row items-center justify-between gap-4">
      <div class="flex items-center space-x-3">
        <div id="gate-icon-wrapper" class="h-10 w-10 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center flex-shrink-0">
          <i id="gate-icon" class="fa-solid fa-check text-lg"></i>
        </div>
        <div>
          <div class="flex items-center space-x-2">
            <h3 id="gate-title" class="text-sm font-bold text-emerald-300">Cổng Kiểm Tra Chất Lượng (Quality Gate)</h3>
            <span id="gate-pill" class="px-2 py-0.5 text-[10px] font-mono rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">Chờ truy vấn</span>
          </div>
          <p id="gate-desc" class="text-xs text-gray-400 mt-0.5">Cho phép qua cổng nếu: Cosine đạt ngưỡng HOẶC BM25 phát hiện từ khóa định danh khớp tuyệt đối.</p>
        </div>
      </div>
      <div class="flex items-center space-x-6 w-full md:w-auto justify-between md:justify-end">
        <div class="text-right">
          <span class="text-xs text-gray-400">So Sánh Cosine:</span>
          <div id="gate-compare-score" class="text-lg font-bold font-mono text-emerald-400">-- &gt; 0.300</div>
        </div>
        <div class="h-8 w-px bg-gray-800"></div>
        <div class="text-right">
          <span class="text-xs text-gray-400">BM25 Khớp Định Danh:</span>
          <div id="gate-bm25-status" class="text-sm font-bold font-mono text-indigo-400 mt-0.5">Chờ kiểm tra</div>
        </div>
      </div>
    </section>

    <!-- 4. COSINE RANKING & MULTI-LANE COMPARISON (Bảng xếp hạng Cosine & BM25 & RRF) -->
    <section class="space-y-3">
      <div class="flex items-center justify-between">
        <h2 class="text-sm font-bold uppercase tracking-wider text-gray-400 flex items-center space-x-2">
          <i class="fa-solid fa-ranking-star text-amber-400"></i>
          <span>Bảng Xếp Hạng Đa Chiều Của Truy Vấn (Cosine Similarity, BM25 & RRF)</span>
        </h2>
        <span class="text-xs text-gray-400 italic">Mỗi lane có thang đo độc lập, RRF kết hợp thứ hạng không cộng dồn score</span>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
        
        <!-- LANE 1: DENSE COSINE SIMILARITY (TRỌNG TÂM CÂU HỎI) -->
        <div class="glass-card rounded-xl p-4 flex flex-col justify-between space-y-3 border-t-2 border-t-emerald-500">
          <div class="flex items-center justify-between">
            <div class="flex items-center space-x-2">
              <span class="px-2 py-0.5 rounded text-[11px] font-bold bg-emerald-500/20 text-emerald-400">Lane 1</span>
              <h3 class="text-sm font-bold text-white">Dense Semantic Rank</h3>
            </div>
            <span class="text-xs font-mono text-emerald-400">Cosine [0.0 → 1.0]</span>
          </div>

          <div class="overflow-x-auto">
            <table class="w-full text-left text-xs">
              <thead>
                <tr class="text-gray-400 border-b border-gray-800">
                  <th class="pb-2 w-8">#</th>
                  <th class="pb-2">Tài Liệu & Nguồn</th>
                  <th class="pb-2 text-right">Cosine Score</th>
                </tr>
              </thead>
              <tbody id="table-dense-body" class="divide-y divide-gray-800/60 font-mono">
                <tr><td colspan="3" class="py-4 text-center text-gray-500">Chưa có truy vấn</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- LANE 2: BM25 LEXICAL RANK -->
        <div class="glass-card rounded-xl p-4 flex flex-col justify-between space-y-3 border-t-2 border-t-indigo-500">
          <div class="flex items-center justify-between">
            <div class="flex items-center space-x-2">
              <span class="px-2 py-0.5 rounded text-[11px] font-bold bg-indigo-500/20 text-indigo-400">Lane 2</span>
              <h3 class="text-sm font-bold text-white">BM25 Lexical Rank</h3>
            </div>
            <span class="text-xs font-mono text-indigo-400">Từ khóa chính xác</span>
          </div>

          <div class="overflow-x-auto">
            <table class="w-full text-left text-xs">
              <thead>
                <tr class="text-gray-400 border-b border-gray-800">
                  <th class="pb-2 w-8">#</th>
                  <th class="pb-2">Tài Liệu & Nguồn</th>
                  <th class="pb-2 text-right">BM25 Score</th>
                </tr>
              </thead>
              <tbody id="table-bm25-body" class="divide-y divide-gray-800/60 font-mono">
                <tr><td colspan="3" class="py-4 text-center text-gray-500">Chưa có truy vấn</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- LANE 3: RRF FUSION MERGED RANK -->
        <div class="glass-card rounded-xl p-4 flex flex-col justify-between space-y-3 border-t-2 border-t-purple-500">
          <div class="flex items-center justify-between">
            <div class="flex items-center space-x-2">
              <span class="px-2 py-0.5 rounded text-[11px] font-bold bg-purple-500/20 text-purple-400">Lane 3</span>
              <h3 class="text-sm font-bold text-white">RRF Fusion Rank</h3>
            </div>
            <span class="text-xs font-mono text-purple-400">1/(60 + Rank)</span>
          </div>

          <div class="overflow-x-auto">
            <table class="w-full text-left text-xs">
              <thead>
                <tr class="text-gray-400 border-b border-gray-800">
                  <th class="pb-2 w-8">#</th>
                  <th class="pb-2">Tài Liệu & Nguồn</th>
                  <th class="pb-2 text-right">RRF Score</th>
                </tr>
              </thead>
              <tbody id="table-rrf-body" class="divide-y divide-gray-800/60 font-mono">
                <tr><td colspan="3" class="py-4 text-center text-gray-500">Chưa có truy vấn</td></tr>
              </tbody>
            </table>
          </div>
        </div>

      </div>
    </section>

    <!-- 5. SYSTEM OUTPUT SECTION (Kết quả sinh từ LLM có Trích Dẫn) -->
    <section class="space-y-4">
      <div class="flex items-center justify-between">
        <h2 class="text-sm font-bold uppercase tracking-wider text-gray-400 flex items-center space-x-2">
          <i class="fa-solid fa-message-quote text-emerald-400"></i>
          <span>Output Của Hệ Thống (Generation Output with Verified Citations)</span>
        </h2>
        <div class="flex items-center space-x-2">
          <span id="output-badge" class="px-2.5 py-1 text-xs font-semibold rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
            HYBRID • DENSE + BM25
          </span>
        </div>
      </div>

      <!-- Final Answer Card -->
      <div class="glass-card rounded-2xl p-6 border-l-4 border-l-emerald-500 space-y-4 shadow-xl">
        <div class="flex items-center justify-between border-b border-gray-800 pb-3">
          <div class="flex items-center space-x-2 text-xs text-gray-400">
            <i class="fa-solid fa-robot text-emerald-400"></i>
            <span id="output-model-tag">Model: openai/gpt-4o-mini (OpenRouter)</span>
          </div>
          <button onclick="copyAnswer()" class="text-xs text-gray-400 hover:text-white flex items-center space-x-1 transition">
            <i class="fa-regular fa-copy"></i>
            <span id="copy-btn-text">Sao chép</span>
          </button>
        </div>

        <!-- The Answer Paragraph -->
        <div id="output-answer" class="text-gray-400 leading-relaxed text-sm md:text-base font-normal space-y-3 italic">
          Vui lòng nhập câu hỏi tra cứu pháp luật cho hộ kinh doanh và bấm &quot;Chạy Pipeline&quot;.
        </div>

        <!-- Verified Evidence Citations Cards -->
        <div class="pt-4 border-t border-gray-800">
          <h4 class="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3 flex items-center space-x-1.5">
            <i class="fa-solid fa-bookmark text-emerald-400"></i>
            <span>Nguồn Dẫn Chứng Được Trích Xuất (Grounded Citations):</span>
          </h4>
          <div id="sources-container" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            <div class="col-span-1 md:col-span-3 text-center py-4 text-xs text-gray-500">
              Nguồn trích dẫn sẽ xuất hiện sau khi bạn nhập câu hỏi và chạy pipeline.
            </div>
          </div>
        </div>
      </div>
    </section>

  </main>

  <!-- Footer -->
  <footer class="glass-card border-t border-gray-800 py-4 px-6 text-center text-xs text-gray-500 flex flex-col md:flex-row items-center justify-between max-w-7xl w-full mx-auto">
    <span>Dự Án Bài Tập Nhóm Day 8 — VinUni AI / Advanced Agentic Coding</span>
    <span class="font-mono text-gray-400">Standalone Modern Web App (Zero Streamlit)</span>
  </footer>

  <!-- Client-side Logic (Vanilla JS) -->
  <script>
    // Sync slider values
    document.getElementById('top-k-slider').addEventListener('input', (e) => {
      document.getElementById('top-k-val').innerText = e.target.value;
    });
    document.getElementById('threshold-slider').addEventListener('input', (e) => {
      document.getElementById('threshold-val').innerText = parseFloat(e.target.value).toFixed(2);
    });

    function setQuery(text) {
      document.getElementById('query-input').value = text;
      executePipeline();
    }

    function copyAnswer() {
      const text = document.getElementById('output-answer').innerText;
      navigator.clipboard.writeText(text);
      document.getElementById('copy-btn-text').innerText = 'Đã sao chép!';
      setTimeout(() => {
        document.getElementById('copy-btn-text').innerText = 'Sao chép';
      }, 2000);
    }

    async function executePipeline() {
      const query = document.getElementById('query-input').value.trim();
      if (!query) return;

      const topK = parseInt(document.getElementById('top-k-slider').value);
      const threshold = parseFloat(document.getElementById('threshold-slider').value);
      const model = document.getElementById('model-select').value;

      // Update button state
      const btnText = document.getElementById('btn-text');
      const btnIcon = document.getElementById('btn-icon');
      btnText.innerText = 'Đang xử lý...';
      btnIcon.className = 'fa-solid fa-spinner fa-spin text-xs';

      try {
        const response = await fetch('/api/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: query,
            top_k: topK,
            score_threshold: threshold,
            provider: model === 'mock' ? 'mock' : 'openrouter',
            model: model === 'mock' ? '' : model
          })
        });

        const data = await response.json();
        renderResults(data);
      } catch (err) {
        console.error('Error executing pipeline:', err);
        alert('Có lỗi khi gọi pipeline. Vui lòng kiểm tra lại server!');
      } finally {
        btnText.innerText = 'Chạy Pipeline';
        btnIcon.className = 'fa-solid fa-play text-xs';
      }
    }

    function renderResults(data) {
      // 1. Update Flow nodes
      document.getElementById('node-query').innerText = data.query;
      document.getElementById('flow-total-time').innerText = `Total Latency: ${data.timing.total_ms} ms`;
      
      document.getElementById('node-dense-stats').innerText = `${data.dense_candidates.length} candidates`;
      document.getElementById('node-dense-time').innerText = `${data.timing.dense_ms} ms`;

      document.getElementById('node-bm25-stats').innerText = `${data.sparse_candidates.length} candidates`;
      document.getElementById('node-bm25-time').innerText = `${data.timing.bm25_ms} ms`;

      document.getElementById('node-rrf-time').innerText = `${data.timing.rrf_ms} ms`;
      document.getElementById('node-llm-time').innerText = `${data.timing.llm_ms} ms`;

      // Gate decision
      const gate = data.cosine_gate;
      const compStr = gate.compare_str || `${gate.best_dense_score.toFixed(3)} ${gate.best_dense_score >= gate.score_threshold ? '>' : '<'} ${gate.score_threshold.toFixed(3)}`;
      document.getElementById('node-gate-compare').innerText = compStr;

      if (gate.passed) {
        document.getElementById('node-gate-decision').innerText = gate.trigger === 'bm25_exact' ? 'BM25 Định Danh Khớp' : (gate.trigger === 'both' ? 'Cosine & BM25 Đạt' : 'Cosine Vượt Ngưỡng');
        document.getElementById('node-gate-badge').innerText = 'PASS • HYBRID';
        document.getElementById('node-gate-badge').className = 'text-emerald-400 font-semibold';
      } else {
        document.getElementById('node-gate-decision').innerText = 'Dưới Ngưỡng Gate';
        document.getElementById('node-gate-badge').innerText = 'FALLBACK';
        document.getElementById('node-gate-badge').className = 'text-amber-400 font-semibold';
      }

      // 2. Update Gate Banner
      const banner = document.getElementById('gate-banner');
      const iconWrapper = document.getElementById('gate-icon-wrapper');
      const icon = document.getElementById('gate-icon');
      const title = document.getElementById('gate-title');
      const desc = document.getElementById('gate-desc');
      const pill = document.getElementById('gate-pill');
      const gateCompare = document.getElementById('gate-compare-score');
      const gateBm25 = document.getElementById('gate-bm25-status');

      gateCompare.innerText = compStr;
      gateCompare.className = gate.cosine_passed ? 'text-lg font-bold font-mono text-emerald-400' : 'text-lg font-bold font-mono text-amber-400';

      if (gate.bm25_exact_match) {
        gateBm25.innerText = gate.bm25_match_reason || 'Khớp tuyệt đối';
        gateBm25.className = 'text-sm font-bold font-mono text-emerald-400 mt-0.5';
      } else {
        gateBm25.innerText = 'Không khớp định danh';
        gateBm25.className = 'text-sm font-medium font-mono text-gray-500 mt-0.5';
      }

      if (gate.passed) {
        banner.className = 'glass-card rounded-xl p-4 border border-emerald-500/30 bg-emerald-950/20 flex flex-col md:flex-row items-center justify-between gap-4';
        iconWrapper.className = 'h-10 w-10 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center flex-shrink-0';
        icon.className = 'fa-solid fa-check text-lg';
        title.className = 'text-sm font-bold text-emerald-300';

        if (gate.trigger === 'both') {
          title.innerText = 'Cả Cosine & BM25 đều đạt — Giữ kết quả Hybrid RRF';
          pill.innerText = 'Cosine & BM25 PASS';
          pill.className = 'px-2 py-0.5 text-[10px] font-mono rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40';
          desc.innerText = `Cosine (${compStr}) đạt ngưỡng VÀ BM25 khớp từ khóa định danh (${gate.bm25_match_reason}). Đủ điều kiện tin cậy tuyệt đối.`;
        } else if (gate.trigger === 'cosine') {
          title.innerText = 'Dense Cosine đạt ngưỡng — Giữ kết quả Hybrid RRF';
          pill.innerText = 'Cosine > Ngưỡng';
          pill.className = 'px-2 py-0.5 text-[10px] font-mono rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40';
          desc.innerText = `Cosine score gốc cao nhất (${compStr}) đạt ngưỡng tin cậy, không cần chuyển sang tìm kiếm fallback.`;
        } else {
          title.innerText = 'BM25 khớp từ khóa định danh — Cho phép qua cổng';
          pill.innerText = 'BM25 Match PASS';
          pill.className = 'px-2 py-0.5 text-[10px] font-mono rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/40';
          desc.innerText = `Dù Cosine (${compStr}) nhưng BM25 phát hiện chính xác từ khóa định danh (${gate.bm25_match_reason}), hệ thống kích hoạt điều kiện OR để giữ kết quả Hybrid RRF.`;
        }
      } else {
        banner.className = 'glass-card rounded-xl p-4 border border-amber-500/30 bg-amber-950/20 flex flex-col md:flex-row items-center justify-between gap-4';
        iconWrapper.className = 'h-10 w-10 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center flex-shrink-0';
        icon.className = 'fa-solid fa-triangle-exclamation text-lg';
        title.className = 'text-sm font-bold text-amber-300';
        title.innerText = 'Không đạt điều kiện cổng — Kích hoạt Fallback PageIndex';
        pill.innerText = 'Fallback Triggered';
        pill.className = 'px-2 py-0.5 text-[10px] font-mono rounded bg-amber-500/20 text-amber-300 border border-amber-500/40';
        desc.innerText = `Cosine (${compStr}) và BM25 không phát hiện từ khóa định danh khớp tuyệt đối. Kích hoạt tìm kiếm dự phòng vectorless.`;
      }

      // 3. Render Multi-Lane Tables
      // Lane 1: Dense Cosine
      const denseBody = document.getElementById('table-dense-body');
      denseBody.innerHTML = '';
      data.dense_candidates.forEach((item, idx) => {
        const pct = Math.min(100, Math.max(5, item.score * 100));
        denseBody.innerHTML += `
          <tr class="hover:bg-gray-800/40 transition">
            <td class="py-2.5 text-gray-500 font-bold">#${idx + 1}</td>
            <td class="py-2.5 pr-2">
              <div class="text-white font-medium truncate max-w-[200px]" title="${item.metadata.title}">${item.metadata.title}</div>
              <div class="text-[11px] text-gray-500 truncate max-w-[180px]">${item.metadata.source}</div>
            </td>
            <td class="py-2.5 text-right">
              <div class="text-emerald-400 font-bold">${item.score.toFixed(4)}</div>
              <div class="w-16 bg-gray-800 h-1.5 rounded-full overflow-hidden ml-auto mt-1">
                <div class="bg-gradient-to-r from-emerald-500 to-cyan-400 h-full" style="width: ${pct}%"></div>
              </div>
            </td>
          </tr>
        `;
      });

      // Lane 2: BM25 Lexical
      const bm25Body = document.getElementById('table-bm25-body');
      bm25Body.innerHTML = '';
      const maxBm25 = Math.max(...data.sparse_candidates.map(x => x.score), 1.0);
      data.sparse_candidates.forEach((item, idx) => {
        const pct = Math.min(100, Math.max(5, (item.score / maxBm25) * 100));
        bm25Body.innerHTML += `
          <tr class="hover:bg-gray-800/40 transition">
            <td class="py-2.5 text-gray-500 font-bold">#${idx + 1}</td>
            <td class="py-2.5 pr-2">
              <div class="text-white font-medium truncate max-w-[200px]" title="${item.metadata.title}">${item.metadata.title}</div>
              <div class="text-[11px] text-gray-500 truncate max-w-[180px]">${item.metadata.source}</div>
            </td>
            <td class="py-2.5 text-right">
              <div class="text-indigo-400 font-bold">${item.score.toFixed(2)}</div>
              <div class="w-16 bg-gray-800 h-1.5 rounded-full overflow-hidden ml-auto mt-1">
                <div class="bg-gradient-to-r from-indigo-500 to-purple-400 h-full" style="width: ${pct}%"></div>
              </div>
            </td>
          </tr>
        `;
      });

      // Lane 3: RRF Fusion
      const rrfBody = document.getElementById('table-rrf-body');
      rrfBody.innerHTML = '';
      data.rrf_candidates.forEach((item, idx) => {
        rrfBody.innerHTML += `
          <tr class="hover:bg-gray-800/40 transition">
            <td class="py-2.5 text-gray-500 font-bold">#${idx + 1}</td>
            <td class="py-2.5 pr-2">
              <div class="text-white font-medium truncate max-w-[200px]" title="${item.metadata.title}">${item.metadata.title}</div>
              <div class="text-[11px] text-gray-500 truncate max-w-[180px]">${item.metadata.source}</div>
            </td>
            <td class="py-2.5 text-right">
              <div class="text-purple-400 font-bold">${item.score.toFixed(5)}</div>
              <span class="text-[10px] px-1.5 py-0.5 rounded bg-purple-500/10 text-purple-300 font-sans">Rank ${item.rank}</span>
            </td>
          </tr>
        `;
      });

      // 4. Render Final Answer & Citations
      let formattedAnswer = data.generation.answer;
      // Convert in-text citation like [1], [2] into styled interactive badges
      formattedAnswer = formattedAnswer.replace(/\\[(\\d+)\\]/g, '<span class="citation-badge" onclick="highlightSource($1)">[$1]</span>');
      formattedAnswer = formattedAnswer.replace(/\\[Document\\s+(\\d+)[^\\]]*\\]/g, '<span class="citation-badge" onclick="highlightSource($1)">Doc $1</span>');

      document.getElementById('output-answer').innerHTML = formattedAnswer;
      document.getElementById('output-badge').innerText = `${data.generation.retrieval_source.toUpperCase()} • ${data.generation.sources.length} CHUNKS`;
      document.getElementById('output-model-tag').innerText = `Model: ${data.generation.model} (${data.generation.provider})`;

      // 5. Render Sources Cards
      const sourcesContainer = document.getElementById('sources-container');
      sourcesContainer.innerHTML = '';
      data.generation.sources.forEach((s, idx) => {
        const docNum = idx + 1;
        sourcesContainer.innerHTML += `
          <div id="source-card-${docNum}" class="glass-card rounded-xl p-3.5 border border-gray-800 hover:border-emerald-500/50 transition space-y-2">
            <div class="flex items-center justify-between">
              <span class="px-2 py-0.5 rounded text-[11px] font-bold bg-emerald-500/20 text-emerald-400">
                [${docNum}] Doc ${docNum}
              </span>
              <span class="text-[11px] font-mono text-gray-400">${s.retrieval_method.toUpperCase()}</span>
            </div>
            <h5 class="text-xs font-semibold text-white line-clamp-1" title="${s.metadata.title}">${s.metadata.title}</h5>
            <p class="text-[11px] text-gray-400 line-clamp-3 leading-relaxed">${s.content}</p>
            <div class="pt-2 border-t border-gray-800 flex items-center justify-between text-[10px] text-gray-500">
              <span class="truncate max-w-[130px]">${s.metadata.source}</span>
              <a href="${s.metadata.url || '#'}" target="_blank" class="text-emerald-400 hover:underline">Link <i class="fa-solid fa-arrow-up-right-from-square text-[9px]"></i></a>
            </div>
          </div>
        `;
      });
    }

    function highlightSource(num) {
      const card = document.getElementById(`source-card-${num}`);
      if (card) {
        card.scrollIntoView({ behavior: 'smooth', block: 'center' });
        card.classList.add('ring-2', 'ring-emerald-400', 'bg-emerald-950/40');
        setTimeout(() => {
          card.classList.remove('ring-2', 'ring-emerald-400', 'bg-emerald-950/40');
        }, 1500);
      }
    }

    // Support pressing Enter in input to run pipeline
    document.getElementById('query-input').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        executePipeline();
      }
    });
  </script>
</body>
</html>
"""


class RAGRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif parsed.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            chunks = load_real_corpus_chunks()
            status_data = {
                "status": "online",
                "openrouter_key": bool(os.getenv("OPENROUTER_API_KEY")),
                "provider": os.getenv("LLM_PROVIDER", "openrouter"),
                "model": os.getenv("LLM_MODEL", "google/gemini-2.0-flash-001"),
                "corpus_chunks": len(chunks),
                "data_source": "data/standardized (Legal & News)",
            }
            self.wfile.write(json.dumps(status_data).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/query":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                payload = json.loads(body)
                query = payload.get("query", "Hóa đơn điện tử khởi tạo từ máy tính tiền mang lại những lợi ích gì cho hộ kinh doanh?")
                top_k = int(payload.get("top_k", 5))
                threshold = float(payload.get("score_threshold", 0.3))
                provider = payload.get("provider", "openrouter")
                model = payload.get("model", "")

                result = run_pipeline_trace(
                    query=query,
                    top_k=top_k,
                    score_threshold=threshold,
                    provider=provider,
                    model=model,
                )

                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def start_server(port: int = 8501):
    server = ThreadingHTTPServer(("0.0.0.0", port), RAGRequestHandler)
    print("=" * 60)
    print(f"🚀 VINUNI RAG EXPLORER & KNOWLEDGE DESK ĐANG CHẠY!")
    print(f"👉 Mở trình duyệt tại: http://localhost:{port}")
    print(f"👉 Endpoint API:        http://localhost:{port}/api/query")
    print("=" * 60)
    print("Nhấn Ctrl+C để dừng server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐang tắt server...")
        server.server_close()


if __name__ == "__main__":
    port_env = int(os.getenv("PORT", "8501"))
    start_server(port=port_env)
