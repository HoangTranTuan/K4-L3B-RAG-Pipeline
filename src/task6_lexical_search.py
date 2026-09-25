"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5/Task 4. BM25 phù hợp với từ khóa chính xác,
mã tài liệu và tên riêng. Output tuân theo SearchResult và được sort theo score
giảm dần.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable


CORPUS: list[dict] = []

_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Tokenize nhẹ cho cả tiếng Việt và tiếng Anh."""

    if not isinstance(text, str):
        return []

    return _TOKEN_PATTERN.findall(text.casefold())


class _FallbackBM25Okapi:
    """
    BM25 fallback nếu môi trường chưa cài rank_bm25.
    """

    def __init__(
        self,
        tokenized_corpus: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.corpus = tokenized_corpus
        self.k1 = k1
        self.b = b

        self.doc_len = [len(doc) for doc in tokenized_corpus]

        self.avgdl = (
            sum(self.doc_len) / len(self.doc_len)
            if self.doc_len
            else 0.0
        )

        self.doc_freqs = [
            Counter(doc)
            for doc in tokenized_corpus
        ]

        document_frequency: Counter[str] = Counter()

        for doc in tokenized_corpus:
            document_frequency.update(set(doc))

        n_docs = len(tokenized_corpus)

        self.idf: dict[str, float] = {}

        for token, freq in document_frequency.items():
            self.idf[token] = math.log(
                1.0
                + (n_docs - freq + 0.5)
                / (freq + 0.5)
            )

    def get_scores(
        self,
        query_tokens: Iterable[str],
    ) -> list[float]:

        scores = [0.0] * len(self.corpus)

        if not self.corpus or self.avgdl <= 0:
            return scores

        for token in query_tokens:

            idf = self.idf.get(token)

            if idf is None:
                continue

            for index, frequencies in enumerate(
                self.doc_freqs
            ):
                tf = frequencies.get(token, 0)

                if tf == 0:
                    continue

                denominator = (
                    tf
                    + self.k1
                    * (
                        1.0
                        - self.b
                        + self.b
                        * self.doc_len[index]
                        / self.avgdl
                    )
                )

                scores[index] += (
                    idf
                    * (tf * (self.k1 + 1.0))
                    / denominator
                )

        return scores


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ corpus chunks."""

    tokenized = [
        _tokenize(item.get("content", ""))
        for item in corpus
    ]

    try:
        from rank_bm25 import BM25Okapi

        return BM25Okapi(tokenized)

    except ImportError:
        return _FallbackBM25Okapi(tokenized)


def _load_shared_corpus() -> list[dict]:
    """
    Load cùng chunks của Task 4.

    CORPUS vẫn được giữ global để tests có thể
    monkeypatch dữ liệu giả.
    """

    global CORPUS

    if CORPUS:
        return CORPUS

    try:
        from .task4_chunking_indexing import (
            chunk_documents,
            load_documents,
        )

        documents = load_documents()

        CORPUS = chunk_documents(documents)

    except (
        FileNotFoundError,
        NotImplementedError,
    ):
        CORPUS = []

    return CORPUS


def lexical_search(
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """
    Search BM25 trên cùng corpus chunks.
    """

    if (
        top_k <= 0
        or not isinstance(query, str)
        or not query.strip()
    ):
        return []

    corpus = _load_shared_corpus()

    if not corpus:
        return []

    query_tokens = _tokenize(query)

    if not query_tokens:
        return []

    bm25 = build_bm25_index(corpus)

    raw_scores = bm25.get_scores(query_tokens)

    scores = [
        float(score)
        for score in raw_scores
    ]

    indices = sorted(
        range(len(scores)),
        key=lambda i: (-scores[i], i),
    )

    results: list[dict] = []

    seen_ids: set[str] = set()

    for index in indices:

        score = scores[index]

        if score <= 0:
            continue

        item = corpus[index]

        item_id = item.get("id")

        if (
            not isinstance(item_id, str)
            or not item_id
            or item_id in seen_ids
        ):
            continue

        results.append(
            {
                "id": item_id,
                "content": item.get(
                    "content",
                    "",
                ),
                "score": score,
                "metadata": dict(
                    item.get(
                        "metadata",
                        {},
                    )
                ),
                "retrieval_method": "bm25",
            }
        )

        seen_ids.add(item_id)

        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":

    for result in lexical_search(
        "test query",
        top_k=3,
    ):
        print(result)