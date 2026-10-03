import pytest

from cutting_stock.arc_flow import build_arc_flow_graph, solve_arc_flow
from cutting_stock.data import CuttingStockInstance, demo_instance
from cutting_stock.master import solve_integer_master
from cutting_stock.patterns import enumerate_patterns, used_length


def test_arc_flow_matches_full_integer_master_on_demo():
    instance = demo_instance()
    arc_flow = solve_arc_flow(instance)
    full = solve_integer_master(instance, enumerate_patterns(instance))
    assert arc_flow.objective == 40
    assert arc_flow.objective == pytest.approx(full.objective)
    assert all(produced >= demand for produced, demand in zip(arc_flow.produced, instance.demands))


def test_arc_flow_resolves_mixed_pattern_instance():
    instance = CuttingStockInstance(
        stock_length=10,
        item_lengths=(6, 4),
        demands=(2, 2),
    )
    solution = solve_arc_flow(instance)
    assert solution.objective == 2
    assert solution.produced == (2, 2)
    assert len(solution.patterns) == 2
    assert all(
        used_length(instance, pattern) <= instance.stock_length
        for pattern in solution.patterns
    )


def test_arc_flow_graph_contains_feedback_and_loss_arcs():
    instance = CuttingStockInstance(
        stock_length=8,
        item_lengths=(3, 5),
        demands=(1, 1),
    )
    arcs = build_arc_flow_graph(instance)
    assert arcs[-1].tail == instance.stock_length
    assert arcs[-1].head == 0
    assert any(
        arc.item_index is None and arc.tail == 0 and arc.head == 1
        for arc in arcs[:-1]
    )
