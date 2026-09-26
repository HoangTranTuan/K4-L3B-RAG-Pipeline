import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import generate_with_citation as generate_answer

DATASET_PATH = (
    ROOT
    / "group_project"
    / "evaluation"
    / "golden_dataset.json"
)


def normalize(text: str) -> str:
    return " ".join(
        str(text)
        .lower()
        .strip()
        .split()
    )


def context_hit(
    retrieved: list[dict],
    expected_context: str,
    source_title: str = "",
) -> bool:
    """
    Kiểm tra expected_context có xuất hiện
    trong một trong các chunks retrieve được không.
    """
    expected = normalize(expected_context)
    if not expected:
        return False

    expected_tokens = set(expected.split())

    for item in retrieved:
        content = normalize(item.get("content", ""))
        if expected in content or content in expected:
            return True
        c_tokens = set(content.split())
        overlap = len(expected_tokens & c_tokens) / len(expected_tokens)
        if overlap >= 0.40:
            return True
        if source_title and normalize(source_title) in normalize(item.get("metadata", {}).get("title", "")):
            if overlap >= 0.20:
                return True

    return False


def answer_keyword_score(
    generated_answer: str,
    expected_answer: str,
) -> float:
    """
    Metric đơn giản để kiểm tra mức overlap
    giữa expected answer và generated answer.

    Đây không phải semantic evaluation,
    nhưng đủ để có baseline.
    """

    generated_tokens = set(
        normalize(
            generated_answer
        ).split()
    )

    expected_tokens = set(
        normalize(
            expected_answer
        ).split()
    )

    if not expected_tokens:
        return 0.0

    overlap = (
        generated_tokens
        & expected_tokens
    )

    return (
        len(overlap)
        / len(expected_tokens)
    )


def evaluate() -> None:

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        dataset = json.load(
            handle
        )

    total = len(dataset)

    if total == 0:
        print(
            "golden_dataset.json đang rỗng."
        )
        return

    retrieval_hits = 0
    answer_scores = []
    latencies = []

    print(
        f"Evaluating {total} questions...\n"
    )

    for index, sample in enumerate(
        dataset,
        start=1,
    ):

        question = sample[
            "question"
        ]

        expected_answer = sample[
            "expected_answer"
        ]

        expected_context = sample[
            "expected_context"
        ]

        start = time.perf_counter()

        retrieved = retrieve(
            question,
            top_k=5,
        )

        result = generate_answer(
            question,
            retrieved,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        latencies.append(
            elapsed
        )

        # Nếu generate_answer trả dict
        if isinstance(
            result,
            dict,
        ):
            answer = str(
                result.get(
                    "answer",
                    "",
                )
            )

        else:
            answer = str(
                result
            )

        hit = context_hit(
            retrieved,
            expected_context,
            sample.get("source_title", ""),
        )

        if hit:
            retrieval_hits += 1

        score = answer_keyword_score(
            answer,
            expected_answer,
        )

        answer_scores.append(
            score
        )

        print(
            f"[{index}/{total}] "
            f"{question}"
        )

        print(
            f"  Retrieval hit: {hit}"
        )

        print(
            f"  Answer score: {score:.3f}"
        )

        print(
            f"  Latency: {elapsed:.3f}s"
        )

    retrieval_accuracy = (
        retrieval_hits / total
    )

    average_answer_score = (
        sum(answer_scores)
        / len(answer_scores)
    )

    average_latency = (
        sum(latencies)
        / len(latencies)
    )

    print(
        "\n============================"
    )

    print(
        "EVALUATION RESULT"
    )

    print(
        "============================"
    )

    print(
        f"Questions: {total}"
    )

    print(
        "Retrieval accuracy: "
        f"{retrieval_accuracy:.2%}"
    )

    print(
        "Average answer score: "
        f"{average_answer_score:.3f}"
    )

    print(
        "Average latency: "
        f"{average_latency:.3f}s"
    )


if __name__ == "__main__":
    evaluate()