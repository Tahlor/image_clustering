from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from .models import PairEdge


class UnionFind:
    def __init__(self, values: list[int]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def connected_groups(indices: list[int], edges: list[PairEdge]) -> list[list[int]]:
    union_find = UnionFind(indices)
    for edge in edges:
        union_find.union(edge.a, edge.b)
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in indices:
        grouped[union_find.find(index)].append(index)
    return [sorted(values) for values in grouped.values()]


def select_anchor(group: list[int], edges: list[PairEdge]) -> int:
    score = {index: 0.0 for index in group}
    for edge in edges:
        if edge.a in score and edge.b in score:
            quality = (
                edge.registration_b_to_a.inlier_ratio
                + edge.registration_b_to_a.overlap_fraction
                - edge.registration_b_to_a.stable_residual
            )
            score[edge.a] += quality
            score[edge.b] += quality
    return max(group, key=lambda index: (score[index], -index))


def transforms_to_anchor(
    group: list[int], anchor: int, edges: list[PairEdge]
) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    adjacency: dict[int, list[tuple[int, np.ndarray, str]]] = defaultdict(list)
    for edge in edges:
        if edge.a not in group or edge.b not in group:
            continue
        matrix_b_to_a = edge.registration_b_to_a.matrix
        if matrix_b_to_a is None:
            continue
        adjacency[edge.a].append(
            (edge.b, matrix_b_to_a, edge.registration_b_to_a.model or "homography")
        )
        adjacency[edge.b].append(
            (
                edge.a,
                np.linalg.inv(matrix_b_to_a),
                edge.registration_b_to_a.model or "homography",
            )
        )

    transforms = {anchor: np.eye(3, dtype=np.float64)}
    models = {anchor: "identity"}
    queue = deque([anchor])
    while queue:
        current = queue.popleft()
        for neighbor, neighbor_to_current, model in adjacency[current]:
            if neighbor in transforms:
                continue
            transforms[neighbor] = transforms[current] @ neighbor_to_current
            models[neighbor] = model
            queue.append(neighbor)
    return transforms, models
