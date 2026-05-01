from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple


Record = Dict[str, Any]


def stratified_train_val_test_split(
    records: Sequence[Record],
    label_column: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[Record]]:
    validate_ratios(train_ratio, val_ratio, test_ratio)
    groups: Dict[str, List[Record]] = defaultdict(list)
    for record in records:
        label = str(record.get(label_column, "")).strip()
        if not label:
            continue
        groups[label].append(record)

    rng = random.Random(seed)
    result = {"train": [], "val": [], "test": []}
    for label in sorted(groups.keys()):
        group = list(groups[label])
        rng.shuffle(group)
        train_count, val_count, test_count = split_counts(len(group), train_ratio, val_ratio, test_ratio)
        result["train"].extend(group[:train_count])
        result["val"].extend(group[train_count : train_count + val_count])
        result["test"].extend(group[train_count + val_count : train_count + val_count + test_count])

    for part in result.values():
        rng.shuffle(part)
    return result


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = [train_ratio, val_ratio, test_ratio]
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("划分比例不能小于 0")
    total = sum(ratios)
    if abs(total - 1.0) > 1e-6:
        raise ValueError("训练集、验证集、测试集比例之和必须等于 1")


def split_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    raw = [total * train_ratio, total * val_ratio, total * test_ratio]
    counts = [int(value) for value in raw]
    remaining = total - sum(counts)
    order = sorted(range(3), key=lambda index: raw[index] - counts[index], reverse=True)
    for index in order[:remaining]:
        counts[index] += 1

    if total >= 3:
        for index in range(3):
            if counts[index] == 0:
                donor = max(range(3), key=lambda i: counts[i])
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[index] += 1
    return counts[0], counts[1], counts[2]

