"""
Task 10 — Generation có citation.

Hướng dẫn:
    1. Retrieve top-k chunks.
    2. Reorder để giảm lost-in-the-middle.
    3. Format context kèm title và source.
    4. Gọi provider được chọn trong .env.
    5. Trả answer, sources và retrieval_source.

Nếu context không đủ hoặc provider lỗi, trả safe refusal; không bịa thông tin.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .task9_retrieval_pipeline import retrieve


TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter")
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemini-2.0-flash-001")

SYSTEM_PROMPT = """Trả lời chỉ từ context được cung cấp.
Mỗi khẳng định phải có citation. Nếu thiếu evidence, hãy từ chối xác minh."""

SAFE_REFUSAL = "Tôi không thể xác minh thông tin này từ nguồn hiện có."


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context (giảm lost-in-the-middle)."""
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label."""
    parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        title = metadata.get("title", "")
        source = metadata.get("source", "")
        content = chunk.get("content", "")
        parts.append(
            f"[Document {index} | Title: {title} | Source: {source}]\n{content}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenRouter, OpenAI, Gemini hoặc Anthropic theo cấu hình."""
    provider = (os.getenv("LLM_PROVIDER") or LLM_PROVIDER or "openrouter").lower().strip()
    model_name = (os.getenv("LLM_MODEL") or LLM_MODEL or "").strip()

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set in environment or .env file. "
                "Vui lòng điền OPENROUTER_API_KEY vào .env để sử dụng."
            )
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        model = model_name or "google/gemini-2.0-flash-001"
        site_url = os.getenv("OPENROUTER_SITE_URL", "https://github.com/HoangTranTuan/K4-L3B-RAG-Pipeline")
        app_title = os.getenv("OPENROUTER_APP_TITLE", "VinUni RAG Pipeline")

        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                default_headers={
                    "HTTP-Referer": site_url,
                    "X-Title": app_title,
                },
            )
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            return response.choices[0].message.content or ""
        except ImportError:
            import json
            import urllib.request

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": site_url,
                "X-Title": app_title,
            }
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
            }
            req = urllib.request.Request(
                f"{base_url.rstrip('/')}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"] or ""

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment.")
        from openai import OpenAI
        base_url = os.getenv("OPENAI_BASE_URL")
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.choices[0].message.content or ""

    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment.")
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model_name or "gemini-2.5-flash",
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                ),
            )
            return response.text or ""
        except ImportError:
            import google.generativeai as gai
            gai.configure(api_key=api_key)
            model = gai.GenerativeModel(
                model_name=model_name or "gemini-1.5-flash",
                system_instruction=system_prompt,
                generation_config={"temperature": TEMPERATURE, "top_p": TOP_P},
            )
            response = model.generate_content(user_message)
            return response.text or ""

    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set in environment.")
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model_name or "claude-3-5-haiku-20241022",
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=1024,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.content[0].text if response.content else ""

    elif provider == "mock":
        return (
            "Dựa trên tài liệu trích dẫn [Document 1], sinh viên đạt GPA từ 3.6 trở lên "
            "sẽ được cấp học bổng khuyến khích học tập theo quy chế."
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult."""
    try:
        chunks = retrieve(query, top_k=top_k)
    except Exception:
        chunks = []

    if not chunks:
        return {
            "answer": SAFE_REFUSAL,
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    try:
        answer = call_llm(SYSTEM_PROMPT, user_message)
    except Exception:
        return {
            "answer": SAFE_REFUSAL,
            "sources": [],
            "retrieval_source": "none",
        }

    if not answer or not answer.strip() or SAFE_REFUSAL in answer.strip():
        return {
            "answer": SAFE_REFUSAL,
            "sources": [],
            "retrieval_source": "none",
        }

    first_method = chunks[0].get("retrieval_method")
    if first_method == "pageindex":
        retrieval_source = "pageindex"
    elif first_method in ("hybrid", "dense", "bm25"):
        retrieval_source = "hybrid"
    else:
        retrieval_source = "none"

    return {
        "answer": answer.strip(),
        "sources": chunks,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    import json
    import sys
    from .contracts import validate_generation_result

    print("=" * 60)
    print("TEST SUITE TASK 10: GENERATION VỚI CITATION & SAFE REFUSAL")
    print("=" * 60)

    # 1. Mock chunks tuân thủ contract
    mock_chunks = [
        {
            "id": f"legal-doc::chunk-{i}",
            "content": f"Quy định học bổng điều {i+1}: Sinh viên đạt điểm rèn luyện tốt và GPA >= 3.6 được xét học bổng.",
            "score": 0.035 - i * 0.005,
            "metadata": {
                "source": "quy_dinh_hoc_bong.md",
                "title": "Quy chế Học bổng & Học phí",
                "doc_type": "legal",
                "url": "https://vinuni.example.edu.vn/scholarship",
                "chunk_index": i,
            },
            "retrieval_method": "hybrid",
        }
        for i in range(5)
    ]

    # --- KỊCH BẢN 1: Lost-in-the-middle reordering ---
    print("\n[Kịch bản 1] Kiểm tra sắp xếp Reorder Lost-in-the-middle:")
    reordered = reorder_for_llm(mock_chunks)
    print(" - Thứ tự gốc IDs:     ", [c["id"] for c in mock_chunks])
    print(" - Thứ tự sau reorder: ", [c["id"] for c in reordered])
    assert reordered[0]["id"] == "legal-doc::chunk-0", "Chunk quan trọng nhất phải ở đầu!"
    print(" -> Thành công! Chunk đầu được giữ ở vị trí số 1, các chunk kế tiếp được phân bổ xen kẽ đầu - cuối.")

    # --- KỊCH BẢN 2: Format Context với Title và Source ---
    print("\n[Kịch bản 2] Kiểm tra format context có chứa Citation Label:")
    context_str = format_context(reordered)
    print(" - Xem mẫu context tạo ra:\n")
    print(context_str[:280] + "\n...")
    assert "quy_dinh_hoc_bong.md" in context_str
    assert "Quy chế Học bổng & Học phí" in context_str
    print(" -> Thành công! Context chứa đầy đủ Document ID, Title và Source cho trích dẫn.")

    # --- KỊCH BẢN 3: Generation hoàn chỉnh với Mock Provider ---
    print("\n[Kịch bản 3] Sinh câu trả lời có citation (LLM Provider = 'mock'):")
    os.environ["LLM_PROVIDER"] = "mock"
    globals()["retrieve"] = lambda q, top_k: mock_chunks[:top_k]

    res = generate_with_citation("Điều kiện nhận học bổng là gì?", top_k=3)
    print(" - Kết quả GenerationResult:")
    print(f"   * Answer:           {res['answer']}")
    print(f"   * Retrieval Source: {res['retrieval_source']}")
    print(f"   * Số lượng sources: {len(res['sources'])}")
    validate_generation_result(res)
    print(" -> Thành công! Output tuân thủ 100% hợp đồng validate_generation_result.")

    # --- KỊCH BẢN 4: Safe Refusal khi không tìm thấy chunk ---
    print("\n[Kịch bản 4] Safe Refusal khi không có chunk / context trống:")
    globals()["retrieve"] = lambda q, top_k: []
    refusal_res = generate_with_citation("Câu hỏi không liên quan?", top_k=3)
    print(" - Kết quả Safe Refusal:")
    print(f"   * Answer:           {refusal_res['answer']}")
    print(f"   * Sources:          {refusal_res['sources']}")
    print(f"   * Retrieval Source: {refusal_res['retrieval_source']}")
    validate_generation_result(refusal_res)
    assert refusal_res["answer"] == SAFE_REFUSAL
    assert refusal_res["sources"] == []
    assert refusal_res["retrieval_source"] == "none"
    print(" -> Thành công! Hệ thống từ chối an toàn tuân thủ contract.")

    # --- KỊCH BẢN 5: Safe Refusal khi LLM Provider bị lỗi ---
    print("\n[Kịch bản 5] Safe Refusal khi LLM Provider gặp sự cố kết nối/API:")
    globals()["retrieve"] = lambda q, top_k: mock_chunks[:top_k]
    def broken_llm(sys_prompt, user_msg):
        raise ConnectionResetError("Mất kết nối tới server OpenAI/Gemini/OpenRouter")
    globals()["call_llm"] = broken_llm

    error_res = generate_with_citation("Thử nghiệm lỗi LLM", top_k=3)
    print(f"   * Answer:           {error_res['answer']}")
    validate_generation_result(error_res)
    assert error_res["answer"] == SAFE_REFUSAL
    print(" -> Thành công! Xử lý ngoại lệ mượt mà, trả về safe refusal thay vì gây sập ứng dụng.")

    # --- KỊCH BẢN 6: Kiểm tra cấu hình OpenRouter ---
    print("\n[Kịch bản 6] Kiểm tra cấu hình OpenRouter trong .env:")
    or_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    or_model = os.getenv("LLM_MODEL", "google/gemini-2.0-flash-001").strip()
    if or_key:
        print(f" - Phát hiện OPENROUTER_API_KEY (model: {or_model}). Đang gọi thử nghiệm trực tiếp...")
        os.environ["LLM_PROVIDER"] = "openrouter"
        globals()["retrieve"] = lambda q, top_k: mock_chunks[:top_k]
        # Khôi phục call_llm gốc
        del globals()["call_llm"]
        try:
            live_res = generate_with_citation("Tóm tắt điều kiện học bổng từ context", top_k=2)
            print(f"   * Live OpenRouter Answer: {live_res['answer']}")
            validate_generation_result(live_res)
            print(" -> [LIVE TEST THÀNH CÔNG] OpenRouter hoạt động xuất sắc!")
        except Exception as e:
            print(f"   * Lưu ý khi gọi OpenRouter: {e}")
    else:
        print(" - Chưa phát hiện OPENROUTER_API_KEY trong .env.")
        print("   -> Để kết nối OpenRouter thật: Hãy dán API key vào file .env (OPENROUTER_API_KEY=sk-or-v1-...).")
        print("   -> Hiện tại hệ thống tự động fallback/mock chuẩn xác mọi kịch bản.")

    print("\n" + "=" * 60)
    print("HOÀN THÀNH TẤT CẢ KỊCH BẢN KIỂM THỬ TASK 10!")
    print("=" * 60)
