import json
import subprocess
import sys

from cutting_stock.experiment import run_demo


def test_demo_verification_metrics():
    result = run_demo()
    assert result["column_generation"]["converged"]
    assert result["verification"]["lp_objective_difference"] < 1e-8
    assert result["verification"]["restricted_integer_objective"] == 40.0
    assert result["verification"]["full_integer_objective"] == 40.0
    assert result["verification"]["arc_flow_integer_objective"] == 40
    assert result["verification"]["arc_flow_matches_full_integer"]


def test_cli_outputs_json():
    completed = subprocess.run(
        [sys.executable, "-m", "cutting_stock", "--max-iterations", "100"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["column_generation"]["converged"] is True
    assert payload["verification"]["arc_flow_matches_full_integer"] is True
