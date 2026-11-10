"""Tests for the known-nuisance synthetic calendar-phase negative control."""

from __future__ import annotations

import numpy as np
import pandas as pd

from calendar_shift_audit.negative_control import run_known_nuisance_control


def test_known_nuisance_control_shrinks_with_sample_size() -> None:
    res = run_known_nuisance_control(
        training_sizes=(150, 1500),
        replicates=16,
        seed=20261008,
    )
    df = pd.DataFrame(res["summary"]).set_index(["training_n", "model", "metric"])

    for n in (150, 1500):
        assert df.loc[(n, "pooled_shared", "mean_nonzero_shift_tv"), "mean"] == 0.0
        assert df.loc[(n, "shift_average", "mean_nonzero_shift_tv"), "mean"] == 0.0

    tv_small = df.loc[(150, "absolute_table", "mean_nonzero_shift_tv"), "mean"]
    tv_large = df.loc[(1500, "absolute_table", "mean_nonzero_shift_tv"), "mean"]
    assert tv_small > tv_large > 0.0

    loss_small = df.loc[(150, "absolute_table", "excess_expected_log_loss"), "mean"]
    loss_large = df.loc[(1500, "absolute_table", "excess_expected_log_loss"), "mean"]
    loss_pooled = df.loc[(150, "pooled_shared", "excess_expected_log_loss"), "mean"]
    assert loss_small > loss_large > 0.0
    assert loss_small > loss_pooled
    assert np.isfinite(res["population_entropy_nats"])
