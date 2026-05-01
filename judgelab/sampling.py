from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List, Sequence


Record = Dict[str, Any]


def random_sample(records: Sequence[Record], sample_size: int, seed: int = 42) -> List[Record]:
    if sample_size < 0:
        raise ValueError("sample_size 不能小于 0")
    if sample_size >= len(records):
        return list(records)
    rng = random.Random(seed)
    indexes = rng.sample(range(len(records)), sample_size)
    return [records[index] for index in sorted(indexes)]


def stratified_sample(records: Sequence[Record], strata_column: str, sample_size: int, seed: int = 42) -> List[Record]:
    if sample_size < 0:
        raise ValueError("sample_size 不能小于 0")
    if not strata_column:
        raise ValueError("strata_column 不能为空")
    if sample_size >= len(records):
        return list(records)

    groups: Dict[str, List[Record]] = defaultdict(list)
    for record in records:
        groups[str(record.get(strata_column, "") or "未填写")].append(record)

    rng = random.Random(seed)
    allocations = proportional_allocations({key: len(value) for key, value in groups.items()}, sample_size)
    sampled: List[Record] = []
    for group_key in sorted(groups.keys()):
        group_records = groups[group_key]
        count = min(allocations.get(group_key, 0), len(group_records))
        sampled.extend(rng.sample(group_records, count))

    if len(sampled) < sample_size:
        selected_ids = {id(record) for record in sampled}
        remaining = [record for record in records if id(record) not in selected_ids]
        sampled.extend(rng.sample(remaining, min(sample_size - len(sampled), len(remaining))))

    return sampled


def proportional_allocations(group_sizes: Dict[str, int], total: int) -> Dict[str, int]:
    if total <= 0 or not group_sizes:
        return {key: 0 for key in group_sizes}

    population = sum(group_sizes.values())
    raw = {key: (size / population) * total for key, size in group_sizes.items()}
    allocations = {key: int(value) for key, value in raw.items()}

    for key, size in group_sizes.items():
        if size > 0 and allocations[key] == 0 and sum(allocations.values()) < total:
            allocations[key] = 1

    remaining = total - sum(allocations.values())
    if remaining > 0:
        by_remainder = sorted(raw.keys(), key=lambda key: raw[key] - int(raw[key]), reverse=True)
        for key in by_remainder:
            if remaining <= 0:
                break
            if allocations[key] < group_sizes[key]:
                allocations[key] += 1
                remaining -= 1

    while sum(allocations.values()) > total:
        reducible = sorted(allocations.keys(), key=lambda key: allocations[key], reverse=True)
        for key in reducible:
            if sum(allocations.values()) <= total:
                break
            if allocations[key] > 0:
                allocations[key] -= 1

    return allocations

