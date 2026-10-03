from __future__ import annotations

from .arc_flow import solve_arc_flow
from .column_generation import run_column_generation
from .data import demo_instance
from .master import solve_integer_master, solve_master_lp
from .patterns import enumerate_patterns, waste


def _selected(patterns, usage, instance):
    rows = []
    for pattern, amount in zip(patterns, usage):
        if amount > 1e-8:
            rows.append(
                {
                    "pattern": list(pattern),
                    "rolls": int(round(float(amount))),
                    "waste_per_roll": waste(instance, pattern),
                }
            )
    return rows


def run_demo(tolerance: float = 1e-9, max_iterations: int = 100) -> dict:
    instance = demo_instance()
    cg = run_column_generation(instance, tolerance=tolerance, max_iterations=max_iterations)
    restricted_integer = solve_integer_master(instance, list(cg.patterns))

    all_patterns = enumerate_patterns(instance)
    full_lp = solve_master_lp(instance, all_patterns)
    full_integer = solve_integer_master(instance, all_patterns)
    arc_flow = solve_arc_flow(instance)

    return {
        "instance": {
            "stock_length": instance.stock_length,
            "item_lengths": list(instance.item_lengths),
            "demands": list(instance.demands),
        },
        "column_generation": {
            "converged": cg.converged,
            "iterations": len(cg.records),
            "initial_columns": len(cg.patterns) - cg.generated_columns,
            "generated_columns": cg.generated_columns,
            "final_columns": len(cg.patterns),
            "lp_objective": cg.lp_solution.objective,
            "final_reduced_cost": cg.records[-1].reduced_cost,
        },
        "verification": {
            "enumerated_patterns": len(all_patterns),
            "full_lp_objective": full_lp.objective,
            "lp_objective_difference": abs(cg.lp_solution.objective - full_lp.objective),
            "restricted_integer_objective": restricted_integer.objective,
            "full_integer_objective": full_integer.objective,
            "restricted_integer_gap": restricted_integer.objective - full_integer.objective,
            "arc_flow_integer_objective": arc_flow.objective,
            "arc_flow_matches_full_integer": abs(arc_flow.objective - full_integer.objective) < 1e-8,
        },
        "arc_flow": {
            "rolls_used": arc_flow.rolls_used,
            "produced": list(arc_flow.produced),
            "patterns": [list(pattern) for pattern in arc_flow.patterns],
        },
        "selected_restricted_integer_patterns": _selected(
            list(cg.patterns), restricted_integer.usage, instance
        ),
        "iterations": [
            {
                "iteration": r.iteration,
                "master_objective": r.master_objective,
                "pricing_dual_value": r.pricing_dual_value,
                "reduced_cost": r.reduced_cost,
                "pattern": list(r.generated_pattern),
                "added": r.added,
            }
            for r in cg.records
        ],
    }
