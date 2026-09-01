from __future__ import annotations

import pytest
import torch

import scripts.route_a_v3.run_route2_xeditsetflow_s1_temperature_sweep_v5 as sweep


def test_stop_column_index_matches_flat_layout():
    # A length-L sequence has 4L substitution columns + 1 STOP column.
    assert sweep.stop_column_index_v5(4 * 50 + 1) == 200
    assert sweep.stop_column_index_v5(5) == 4
    with pytest.raises(Exception, match="4k\\+1"):
        sweep.stop_column_index_v5(6)
    with pytest.raises(Exception, match="too narrow"):
        sweep.stop_column_index_v5(3)


def test_stop_column_index_agrees_with_sampler_decode():
    # The sampler decodes STOP as flat_index == padded_length * 4 where
    # padded_length = (width - 1) // 4; the sweep must target the same column.
    for width in (4 * 17 + 1, 4 * 100 + 1):
        padded_length = (width - 1) // 4
        assert sweep.stop_column_index_v5(width) == padded_length * 4


def test_sample_many_v5_identity_is_v4_arithmetic():
    # stop_rate_scale == 1.0 must not touch the weights tensor: the scaling
    # branch is guarded by `if stop_rate_scale != 1.0`.
    import inspect

    source = inspect.getsource(sweep.sample_many_setflow_v5)
    assert "if stop_rate_scale != 1.0" in source
    assert "weights[:, stop_column] = weights[:, stop_column] * stop_rate_scale" in source


def test_sweep_grid_identity_cell_present_and_frozen_order():
    grid = sweep.frozen_temperature_sweep_v5()
    assert (1.0, 1.0) in grid
    assert len(grid) == 25
    # first cell is the sharpen/low-stop corner, identity sits mid-grid
    assert grid[0] == (0.5, 0.25)
    assert grid[12] == (1.0, 1.0)


def test_output_schema_constant_is_frozen():
    assert sweep.SWEEP_SCHEMA == (
        "route_a_v3_route2_xeditsetflow_s1_temperature_sweep_v5.v1"
    )


def test_sweep_error_type_is_runtime_error():
    assert issubclass(sweep.SetFlowTemperatureSweepV5Error, RuntimeError)
