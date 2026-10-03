from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix, vstack

from .data import CuttingStockInstance
from .patterns import Pattern, validate_pattern


@dataclass(frozen=True)
class Arc:
    tail: int
    head: int
    item_index: int | None

    @property
    def is_loss(self) -> bool:
        return self.item_index is None


@dataclass(frozen=True)
class ArcFlowSolution:
    objective: int
    patterns: tuple[Pattern, ...]
    arcs: tuple[Arc, ...]
    flow: np.ndarray
    message: str

    @property
    def rolls_used(self) -> int:
        return self.objective

    @property
    def produced(self) -> tuple[int, ...]:
        if not self.patterns:
            return ()
        return tuple(
            int(sum(pattern[i] for pattern in self.patterns))
            for i in range(len(self.patterns[0]))
        )


def build_arc_flow_graph(instance: CuttingStockInstance) -> tuple[Arc, ...]:
    """Build the pseudo-polynomial capacity graph used by the arc-flow model.

    Nodes represent used stock length 0..W. Item arcs consume an item length and
    unit loss arcs represent trim waste. A feedback arc from W to 0 closes each
    source-to-sink path into one circulation cycle; its flow counts stock rolls.
    """
    stock_length = instance.stock_length
    arcs: list[Arc] = []
    for used in range(stock_length):
        for item_index, length in enumerate(instance.item_lengths):
            if used + length <= stock_length:
                arcs.append(Arc(used, used + length, item_index))
        arcs.append(Arc(used, used + 1, None))
    arcs.append(Arc(stock_length, 0, None))
    return tuple(arcs)


def _decompose_patterns(
    instance: CuttingStockInstance,
    arcs: tuple[Arc, ...],
    flow: np.ndarray,
    feedback_index: int,
) -> tuple[Pattern, ...]:
    integer_flow = np.rint(flow).astype(int)
    rolls = int(integer_flow[feedback_index])
    forward_indices_by_tail: dict[int, list[int]] = {
        node: [] for node in range(instance.stock_length)
    }
    for idx, arc in enumerate(arcs[:feedback_index]):
        forward_indices_by_tail[arc.tail].append(idx)

    for indices in forward_indices_by_tail.values():
        indices.sort(key=lambda idx: arcs[idx].is_loss)

    patterns: list[Pattern] = []
    residual = integer_flow.copy()
    for _ in range(rolls):
        node = 0
        counts = [0] * instance.n_items
        while node < instance.stock_length:
            candidates = [
                idx
                for idx in forward_indices_by_tail[node]
                if residual[idx] > 0
            ]
            if not candidates:
                raise RuntimeError(
                    "arc-flow solution could not be decomposed into stock-roll paths"
                )
            idx = candidates[0]
            residual[idx] -= 1
            arc = arcs[idx]
            if arc.item_index is not None:
                counts[arc.item_index] += 1
            node = arc.head
        pattern = tuple(counts)
        if any(pattern):
            validate_pattern(instance, pattern)
        patterns.append(pattern)

    if np.any(residual[:feedback_index] != 0):
        raise RuntimeError(
            "arc-flow path decomposition left residual forward flow"
        )
    return tuple(patterns)


def solve_arc_flow(instance: CuttingStockInstance) -> ArcFlowSolution:
    """Solve one-dimensional cutting stock as an exact integer arc-flow model.

    The formulation is pseudo-polynomial in stock_length and therefore best
    suited to integer capacities of moderate size. It avoids explicit pattern
    enumeration and provides an exact benchmark against column generation.
    """
    arcs = build_arc_flow_graph(instance)
    feedback_index = len(arcs) - 1
    n_arcs = len(arcs)
    n_nodes = instance.stock_length + 1

    balance = lil_matrix((n_nodes, n_arcs), dtype=float)
    demand = lil_matrix((instance.n_items, n_arcs), dtype=float)

    for j, arc in enumerate(arcs):
        balance[arc.tail, j] += 1.0
        balance[arc.head, j] -= 1.0
        if j != feedback_index and arc.item_index is not None:
            demand[arc.item_index, j] = 1.0

    matrix = vstack([balance.tocsr(), demand.tocsr()], format="csr")
    lower = np.concatenate(
        [
            np.zeros(n_nodes, dtype=float),
            np.asarray(instance.demands, dtype=float),
        ]
    )
    upper = np.concatenate(
        [
            np.zeros(n_nodes, dtype=float),
            np.full(instance.n_items, np.inf),
        ]
    )
    objective = np.zeros(n_arcs, dtype=float)
    objective[feedback_index] = 1.0

    result = milp(
        c=objective,
        integrality=np.ones(n_arcs, dtype=int),
        bounds=Bounds(
            np.zeros(n_arcs),
            np.full(n_arcs, np.inf),
        ),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"disp": False},
    )
    if result.x is None or result.fun is None or result.status != 0:
        raise RuntimeError(f"arc-flow MILP failed: {result.message}")

    rounded_objective = int(round(float(result.fun)))
    patterns = _decompose_patterns(
        instance,
        arcs,
        np.asarray(result.x, dtype=float),
        feedback_index,
    )
    if len(patterns) != rounded_objective:
        raise RuntimeError(
            "arc-flow path count does not match objective"
        )

    return ArcFlowSolution(
        objective=rounded_objective,
        patterns=patterns,
        arcs=arcs,
        flow=np.asarray(result.x, dtype=float),
        message=str(result.message),
    )
