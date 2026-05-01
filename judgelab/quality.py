from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Sequence


Record = Dict[str, Any]


def normalize_label(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def label_distribution(records: Sequence[Record], label_column: str) -> Dict[str, int]:
    counter = Counter()
    for record in records:
        label = normalize_label(record.get(label_column)) or "未标注"
        counter[label] += 1
    return dict(counter)


def missing_label_rows(records: Sequence[Record], label_column: str) -> List[int]:
    return [index for index, record in enumerate(records) if not normalize_label(record.get(label_column))]


def low_confidence_rows(records: Sequence[Record], confidence_column: str, threshold: float = 0.7) -> List[int]:
    rows: List[int] = []
    for index, record in enumerate(records):
        value = record.get(confidence_column)
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            continue
        if confidence < threshold:
            rows.append(index)
    return rows


def cohen_kappa(labels_a: Sequence[Any], labels_b: Sequence[Any]) -> float:
    if len(labels_a) != len(labels_b):
        raise ValueError("两组标签长度必须一致")
    total = len(labels_a)
    if total == 0:
        return 0.0

    normalized_a = [normalize_label(value) for value in labels_a]
    normalized_b = [normalize_label(value) for value in labels_b]
    observed = sum(1 for a, b in zip(normalized_a, normalized_b) if a == b) / total

    count_a = Counter(normalized_a)
    count_b = Counter(normalized_b)
    expected = sum((count_a[label] / total) * (count_b[label] / total) for label in set(count_a) | set(count_b))
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def quality_summary(
    records: Sequence[Record],
    label_column: str,
    confidence_column: str | None = None,
    threshold: float = 0.7,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total": len(records),
        "label_distribution": label_distribution(records, label_column),
        "missing_label_count": len(missing_label_rows(records, label_column)),
    }
    if confidence_column:
        summary["low_confidence_count"] = len(low_confidence_rows(records, confidence_column, threshold))
    return summary

