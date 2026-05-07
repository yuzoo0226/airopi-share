"""Comprehensive tests for aggregate_stats_fast.py.

Verifies that the optimized implementations produce numerically identical
results to reference implementations, and tests all public/internal functions.

Run:
    uv run pytest tests/scripts/test_aggregate_stats_fast.py -v
"""
from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


from scripts.aggregate_stats_fast import (  # noqa: E402
    _approx_transform_mean_std,
    _assert_type_and_shape,
    _combine_action_quantiles,
    _combine_action_quantiles_arm_head_relative_gripper_base,
    _mixture_quantiles as fast_mixture_quantiles,
    _normalize,
    _std_normal_cdf,
    _unnormalize,
    aggregate_feature_stats,
    aggregate_stats,
    combine_action_stats,
    combine_action_stats_arm_head_relative_gripper_base,
    compute_gripper_stats_with_action_transform,
    compute_gripper_stats_with_transform,
    load_episodes_stats,
    load_gripper_data_from_multiple_dirs as fast_load_multiple,
    load_gripper_data_from_parquet_dir as fast_load_single,
    numpy_dict_to_lists,
    pad_hsr_action_stats,
    pad_hsr_state_stats,
)
from scripts.aggregate_stats_simple import (  # noqa: E402
    load_gripper_data_from_parquet_dir as _ref_load_gripper_data_from_parquet_dir,
)


# ---------------------------------------------------------------------------
# Reference implementations (for comparison tests)
# ---------------------------------------------------------------------------


def _ref_std_normal_cdf(x):
    x_arr = np.asarray(x, dtype=float)
    v_erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + v_erf(x_arr / math.sqrt(2.0)))


def _ref_mixture_quantiles(per_episode, feature_key, q_levels=(0.01, 0.99)):
    comps = []
    for ep_stats in per_episode:
        if feature_key not in ep_stats:
            continue
        s = ep_stats[feature_key]
        if not all(k in s for k in ("mean", "std", "min", "max", "count")):
            continue
        comps.append(s)
    if not comps:
        return {q: [] for q in q_levels}
    mean_arr = np.stack([c["mean"] for c in comps])
    std_arr = np.stack([c["std"] for c in comps])
    min_arr = np.stack([c["min"] for c in comps])
    max_arr = np.stack([c["max"] for c in comps])
    count_arr = np.stack([c["count"] for c in comps])
    if count_arr.ndim > 1:
        count_arr = count_arr.reshape(count_arr.shape[0], -1)[:, 0]
    total_counts = count_arr.sum()
    weights = count_arr / (total_counts if total_counts > 0 else 1)
    dim = mean_arr.shape[1]
    results = {q: [0.0] * dim for q in q_levels}
    global_min = np.min(min_arr, axis=0)
    global_max = np.max(max_arr, axis=0)
    for d in range(dim):
        if not np.isfinite(global_min[d]) or not np.isfinite(global_max[d]):
            for q in q_levels:
                results[q][d] = float("nan")
            continue
        if global_min[d] >= global_max[d]:
            val = float(global_min[d])
            for q in q_levels:
                results[q][d] = val
            continue
        mu_j = mean_arr[:, d]
        sigma_j = std_arr[:, d]
        lo_j = min_arr[:, d]
        hi_j = max_arr[:, d]
        point_mask = (sigma_j <= 1e-12) | (hi_j <= lo_j)
        cont_mask = ~point_mask
        mu_c = mu_j[cont_mask]
        sig_c = sigma_j[cont_mask]
        lo_c = lo_j[cont_mask]
        hi_c = hi_j[cont_mask]
        w_point = weights[point_mask]
        w_cont = weights[cont_mask]
        if mu_c.size > 0:
            a_vals = (lo_c - mu_c) / sig_c
            b_vals = (hi_c - mu_c) / sig_c
            Phi_a = _ref_std_normal_cdf(a_vals)
            Phi_b = _ref_std_normal_cdf(b_vals)
            denom = np.clip(Phi_b - Phi_a, 1e-12, None)

        def cdf_scalar(x):
            total = float(np.sum(w_point * (x >= mu_j[point_mask]))) if w_point.size else 0.0
            if mu_c.size:
                z = (x - mu_c) / sig_c
                Phi_z = _ref_std_normal_cdf(z)
                p = np.clip((Phi_z - Phi_a) / denom, 0.0, 1.0)
                p = np.where(x <= lo_c, 0.0, p)
                p = np.where(x >= hi_c, 1.0, p)
                total += float(np.sum(w_cont * p))
            return total

        for q in q_levels:
            left, right = global_min[d], global_max[d]
            for _ in range(50):
                mid = 0.5 * (left + right)
                if cdf_scalar(mid) < q:
                    left = mid
                else:
                    right = mid
            results[q][d] = float(0.5 * (left + right))
    return results


# ---------------------------------------------------------------------------
# Fixtures / Helpers
# ---------------------------------------------------------------------------


def _make_parquet_episode(
    path: Path,
    n_rows: int,
    state_dim: int,
    action_dim: int,
    seed: int,
    state_column: str = "observation.state",
    action_column: str = "action.relative",
):
    rng = np.random.default_rng(seed)
    state_data = rng.standard_normal((n_rows, state_dim)).tolist()
    action_data = rng.standard_normal((n_rows, action_dim)).tolist()
    table = pa.table(
        {
            state_column: state_data,
            action_column: action_data,
            "episode_index": [seed] * n_rows,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))


def _make_episodes_stats(n_episodes: int, dims: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    stats_list = []
    for _ in range(n_episodes):
        mean = rng.standard_normal(dims)
        std = np.abs(rng.standard_normal(dims)) + 0.01
        mn = mean - 3 * std
        mx = mean + 3 * std
        count = np.array([rng.integers(50, 500)])
        stats_list.append(
            {
                "observation.state": {
                    "mean": mean,
                    "std": std,
                    "min": mn,
                    "max": mx,
                    "count": count,
                },
                "action.relative": {
                    "mean": rng.standard_normal(dims + 3),
                    "std": np.abs(rng.standard_normal(dims + 3)) + 0.01,
                    "min": rng.standard_normal(dims + 3) - 3,
                    "max": rng.standard_normal(dims + 3) + 3,
                    "count": count.copy(),
                },
            }
        )
    return stats_list


def _write_episodes_stats_jsonl(path: Path, stats_list: list[dict]):
    with open(path, "w") as f:
        for i, stats in enumerate(stats_list):
            serialized = {}
            for fkey, sval in stats.items():
                serialized[fkey] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in sval.items()
                }
            obj = {"episode_index": i, "stats": serialized}
            f.write(json.dumps(obj) + "\n")


# ---------------------------------------------------------------------------
# Tests: _std_normal_cdf
# ---------------------------------------------------------------------------


class TestStdNormalCdf:
    def test_at_zero(self):
        assert _std_normal_cdf(np.array([0.0]))[0] == pytest.approx(0.5)

    def test_symmetry(self):
        vals = np.array([-2.0, -1.0, 1.0, 2.0])
        result = _std_normal_cdf(vals)
        np.testing.assert_allclose(result[0] + result[3], 1.0, atol=1e-14)
        np.testing.assert_allclose(result[1] + result[2], 1.0, atol=1e-14)

    def test_monotonic(self):
        xs = np.linspace(-3.0, 3.0, 100)
        result = _std_normal_cdf(xs)
        assert np.all(np.diff(result) >= 0)

    def test_known_values(self):
        # Phi(1) ≈ 0.8413, Phi(-1) ≈ 0.1587
        result = _std_normal_cdf(np.array([1.0, -1.0]))
        assert result[0] == pytest.approx(0.8413, abs=1e-4)
        assert result[1] == pytest.approx(0.1587, abs=1e-4)

    def test_extreme_values(self):
        result = _std_normal_cdf(np.array([-10.0, 10.0]))
        assert result[0] == pytest.approx(0.0, abs=1e-10)
        assert result[1] == pytest.approx(1.0, abs=1e-10)

    def test_scalar_input(self):
        result = _std_normal_cdf(0.0)
        assert float(result) == pytest.approx(0.5)

    def test_matches_reference(self):
        xs = np.linspace(-4.0, 4.0, 50)
        np.testing.assert_allclose(_std_normal_cdf(xs), _ref_std_normal_cdf(xs), atol=1e-14)


# ---------------------------------------------------------------------------
# Tests: _normalize / _unnormalize
# ---------------------------------------------------------------------------


class TestNormalizeUnnormalize:
    def test_roundtrip(self):
        x, lo, hi = 3.0, 1.0, 5.0
        assert _unnormalize(_normalize(x, lo, hi), lo, hi) == pytest.approx(x)

    def test_normalize_known(self):
        assert _normalize(3.0, 0.0, 10.0) == pytest.approx(0.3)

    def test_unnormalize_known(self):
        assert _unnormalize(0.5, 2.0, 8.0) == pytest.approx(5.0)

    def test_boundary_values(self):
        assert _normalize(0.0, 0.0, 10.0) == pytest.approx(0.0)
        assert _normalize(10.0, 0.0, 10.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests: _approx_transform_mean_std
# ---------------------------------------------------------------------------


class TestApproxTransformMeanStd:
    def test_identity(self):
        m, s = _approx_transform_mean_std(5.0, 1.0, lambda x: x)
        assert m == pytest.approx(5.0)
        assert s == pytest.approx(1.0)

    def test_zero_std(self):
        m, s = _approx_transform_mean_std(3.0, 0.0, lambda x: x * 2)
        assert m == pytest.approx(6.0)
        assert s == 0.0

    def test_linear_transform(self):
        m, s = _approx_transform_mean_std(5.0, 2.0, lambda x: 2 * x + 1)
        assert m == pytest.approx(11.0)
        assert s == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Tests: _assert_type_and_shape
# ---------------------------------------------------------------------------


class TestAssertTypeAndShape:
    def test_valid_input(self):
        stats = [
            {
                "feat": {
                    "mean": np.array([1.0, 2.0]),
                    "std": np.array([0.1, 0.2]),
                    "min": np.array([0.5, 1.5]),
                    "max": np.array([1.5, 2.5]),
                    "count": np.array([100]),
                }
            }
        ]
        _assert_type_and_shape(stats)  # should not raise

    def test_non_ndarray_raises(self):
        stats = [{"feat": {"mean": [1.0, 2.0]}}]
        with pytest.raises(ValueError, match="numpy array"):
            _assert_type_and_shape(stats)

    def test_zero_dim_raises(self):
        stats = [{"feat": {"mean": np.array(1.0)}}]
        with pytest.raises(ValueError, match="Number of dimensions"):
            _assert_type_and_shape(stats)

    def test_wrong_count_shape_raises(self):
        stats = [{"feat": {"count": np.array([1, 2])}}]
        with pytest.raises(ValueError, match="Shape of 'count'"):
            _assert_type_and_shape(stats)


# ---------------------------------------------------------------------------
# Tests: aggregate_feature_stats
# ---------------------------------------------------------------------------


class TestAggregateFeatureStats:
    def test_single_episode(self):
        stats = [
            {
                "mean": np.array([1.0, 2.0]),
                "std": np.array([0.5, 0.3]),
                "min": np.array([0.0, 1.0]),
                "max": np.array([2.0, 3.0]),
                "count": np.array([100]),
            }
        ]
        result = aggregate_feature_stats(stats)
        np.testing.assert_allclose(result["mean"], [1.0, 2.0])
        np.testing.assert_allclose(result["std"], [0.5, 0.3])
        np.testing.assert_allclose(result["min"], [0.0, 1.0])
        np.testing.assert_allclose(result["max"], [2.0, 3.0])

    def test_equal_weight_episodes(self):
        stats = [
            {
                "mean": np.array([0.0]),
                "std": np.array([1.0]),
                "min": np.array([-3.0]),
                "max": np.array([3.0]),
                "count": np.array([100]),
            },
            {
                "mean": np.array([2.0]),
                "std": np.array([1.0]),
                "min": np.array([-1.0]),
                "max": np.array([5.0]),
                "count": np.array([100]),
            },
        ]
        result = aggregate_feature_stats(stats)
        # Weighted mean: (0*100 + 2*100) / 200 = 1.0
        np.testing.assert_allclose(result["mean"], [1.0])
        np.testing.assert_allclose(result["min"], [-3.0])
        np.testing.assert_allclose(result["max"], [5.0])

    def test_unequal_weight_episodes(self):
        stats = [
            {
                "mean": np.array([0.0]),
                "std": np.array([0.0]),
                "min": np.array([0.0]),
                "max": np.array([0.0]),
                "count": np.array([300]),
            },
            {
                "mean": np.array([3.0]),
                "std": np.array([0.0]),
                "min": np.array([3.0]),
                "max": np.array([3.0]),
                "count": np.array([100]),
            },
        ]
        result = aggregate_feature_stats(stats)
        # Weighted mean: (0*300 + 3*100) / 400 = 0.75
        np.testing.assert_allclose(result["mean"], [0.75])


# ---------------------------------------------------------------------------
# Tests: aggregate_stats
# ---------------------------------------------------------------------------


class TestAggregateStats:
    def test_multiple_features(self):
        stats_list = [
            {
                "observation.state": {
                    "mean": np.array([1.0]),
                    "std": np.array([0.5]),
                    "min": np.array([0.0]),
                    "max": np.array([2.0]),
                    "count": np.array([100]),
                },
                "action.relative": {
                    "mean": np.array([0.5]),
                    "std": np.array([0.2]),
                    "min": np.array([0.0]),
                    "max": np.array([1.0]),
                    "count": np.array([100]),
                },
            }
        ]
        result = aggregate_stats(stats_list)
        assert "observation.state" in result
        assert "action.relative" in result

    def test_partial_features(self):
        """Episodes with different subsets of feature keys."""
        stats_list = [
            {
                "feat_a": {
                    "mean": np.array([1.0]),
                    "std": np.array([0.1]),
                    "min": np.array([0.5]),
                    "max": np.array([1.5]),
                    "count": np.array([50]),
                },
            },
            {
                "feat_a": {
                    "mean": np.array([2.0]),
                    "std": np.array([0.2]),
                    "min": np.array([1.5]),
                    "max": np.array([2.5]),
                    "count": np.array([50]),
                },
                "feat_b": {
                    "mean": np.array([0.0]),
                    "std": np.array([1.0]),
                    "min": np.array([-2.0]),
                    "max": np.array([2.0]),
                    "count": np.array([50]),
                },
            },
        ]
        result = aggregate_stats(stats_list)
        assert "feat_a" in result
        assert "feat_b" in result
        # feat_a aggregated from 2 episodes
        np.testing.assert_allclose(result["feat_a"]["mean"], [1.5])


# ---------------------------------------------------------------------------
# Tests: pad_hsr_state_stats / pad_hsr_action_stats
# ---------------------------------------------------------------------------


class TestPadHsrStateStats:
    def test_aligned_ids(self):
        stat = {
            "mean": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
            "std": np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]),
            "min": np.array([-1.0] * 8),
            "max": np.array([1.0] * 8),
            "count": np.array([100]),
        }
        result = pad_hsr_state_stats(stat, adapt_to_pi=True)
        # aligned_ids = [0, 1, 2, 3, 4, 6, 11, 12]
        assert result["mean"][0] == pytest.approx(0.1)
        assert result["mean"][6] == pytest.approx(0.6)  # src_i=5 -> dst_i=6
        assert result["mean"][11] == pytest.approx(0.7)  # src_i=6 -> dst_i=11
        assert result["mean"][12] == pytest.approx(0.8)  # src_i=7 -> dst_i=12
        assert result["mean"][5] == 0.0  # not in aligned_ids

    def test_output_length(self):
        stat = {
            "mean": np.array([1.0] * 8),
            "std": np.array([0.1] * 8),
            "min": np.array([0.0] * 8),
            "max": np.array([2.0] * 8),
            "count": np.array([10]),
        }
        result = pad_hsr_state_stats(stat, adapt_to_pi=True)
        assert len(result["mean"]) == 32

    def test_q01_q99_are_none(self):
        stat = {
            "mean": np.array([1.0] * 8),
            "std": np.array([0.1] * 8),
            "count": np.array([10]),
        }
        result = pad_hsr_state_stats(stat, adapt_to_pi=True)
        assert result["q01"] is None
        assert result["q99"] is None


class TestPadHsrActionStats:
    def test_aligned_ids(self):
        stat = {
            "mean": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]),
            "std": np.array([1.0] * 11),
            "min": np.array([-1.0] * 11),
            "max": np.array([1.0] * 11),
            "count": np.array([100]),
        }
        result = pad_hsr_action_stats(stat, adapt_to_pi=True)
        # aligned_ids = [0, 1, 2, 3, 4, 6, 11, 12, 13, 14, 15]
        assert result["mean"][0] == pytest.approx(0.1)
        assert result["mean"][6] == pytest.approx(0.6)
        assert result["mean"][15] == pytest.approx(1.1)
        assert result["mean"][5] == 0.0  # not in aligned_ids

    def test_output_length(self):
        stat = {
            "mean": np.array([1.0] * 11),
            "std": np.array([0.1] * 11),
            "min": np.array([0.0] * 11),
            "max": np.array([2.0] * 11),
            "count": np.array([10]),
        }
        result = pad_hsr_action_stats(stat, adapt_to_pi=True)
        assert len(result["mean"]) == 32


# ---------------------------------------------------------------------------
# Tests: combine_action_stats
# ---------------------------------------------------------------------------


class TestCombineActionStats:
    def test_none_state_diff(self):
        relative = {"mean": np.array([1.0, 2.0, 3.0])}
        result = combine_action_stats(None, relative)
        assert result is relative

    def test_same_dim(self):
        state_diff = {"mean": np.array([1.0, 2.0]), "count": np.array([100])}
        relative = {"mean": np.array([3.0, 4.0]), "count": np.array([100])}
        result = combine_action_stats(state_diff, relative)
        assert result is state_diff

    def test_concatenation(self):
        state_diff = {
            "mean": np.array([1.0, 2.0]),
            "std": np.array([0.1, 0.2]),
            "min": np.array([0.0, 1.0]),
            "max": np.array([2.0, 3.0]),
            "count": np.array([100]),
        }
        relative = {
            "mean": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "std": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            "min": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            "max": np.array([2.0, 3.0, 4.0, 5.0, 6.0]),
            "count": np.array([100]),
        }
        result = combine_action_stats(state_diff, relative, base_dim=3)
        np.testing.assert_allclose(result["mean"], [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_dim_mismatch_raises(self):
        state_diff = {"mean": np.array([1.0, 2.0])}
        relative = {"mean": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])}
        with pytest.raises(ValueError, match="Cannot combine"):
            combine_action_stats(state_diff, relative, base_dim=3)


# ---------------------------------------------------------------------------
# Tests: combine_action_stats_arm_head_relative_gripper_base
# ---------------------------------------------------------------------------


class TestCombineActionStatsArmHeadRelativeGripperBase:
    def test_none_state_diff(self):
        relative = {"mean": np.array([1.0] * 11)}
        result = combine_action_stats_arm_head_relative_gripper_base(None, relative)
        assert result is relative

    def test_proper_combining(self):
        state_diff = {
            "mean": np.arange(8, dtype=float),  # arm(5) + gripper(1) + head(2)
            "std": np.ones(8),
            "min": np.zeros(8),
            "max": np.ones(8) * 10,
            "count": np.array([100]),
        }
        relative = {
            "mean": np.arange(11, dtype=float),  # state_diff(8) + base(3)
            "std": np.ones(11),
            "min": np.zeros(11),
            "max": np.ones(11) * 10,
            "count": np.array([100]),
        }
        result = combine_action_stats_arm_head_relative_gripper_base(state_diff, relative)
        # arm from state_diff[:5], gripper from relative[5:6], head from state_diff[6:8], base from relative[-3:]
        expected_mean = np.array([0, 1, 2, 3, 4, 5.0, 6, 7, 8, 9, 10])
        np.testing.assert_allclose(result["mean"], expected_mean)

    def test_dim_mismatch_raises(self):
        state_diff = {"mean": np.array([1.0] * 5)}
        relative = {"mean": np.array([1.0] * 11)}
        with pytest.raises(ValueError, match="Cannot combine"):
            combine_action_stats_arm_head_relative_gripper_base(state_diff, relative)


# ---------------------------------------------------------------------------
# Tests: load_episodes_stats
# ---------------------------------------------------------------------------


class TestLoadEpisodesStats:
    def test_valid_jsonl(self, tmp_path):
        stats_list = _make_episodes_stats(3, 4, seed=42)
        path = tmp_path / "episodes_stats.jsonl"
        _write_episodes_stats_jsonl(path, stats_list)

        loaded = load_episodes_stats(str(path))
        assert len(loaded) == 3
        assert "observation.state" in loaded[0]
        assert isinstance(loaded[0]["observation.state"]["mean"], np.ndarray)

    def test_null_stats_skipped(self, tmp_path):
        path = tmp_path / "episodes_stats.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"episode_index": 0, "stats": None}) + "\n")
            f.write(
                json.dumps(
                    {
                        "episode_index": 1,
                        "stats": {
                            "feat": {
                                "mean": [1.0],
                                "std": [0.1],
                                "min": [0.5],
                                "max": [1.5],
                                "count": [10],
                            }
                        },
                    }
                )
                + "\n"
            )
        loaded = load_episodes_stats(str(path))
        assert len(loaded) == 1

    def test_null_feature_skipped(self, tmp_path):
        path = tmp_path / "episodes_stats.jsonl"
        with open(path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "episode_index": 0,
                        "stats": {
                            "feat_a": None,
                            "feat_b": {
                                "mean": [1.0],
                                "std": [0.1],
                                "min": [0.5],
                                "max": [1.5],
                                "count": [10],
                            },
                        },
                    }
                )
                + "\n"
            )
        loaded = load_episodes_stats(str(path))
        assert len(loaded) == 1
        assert "feat_a" not in loaded[0]
        assert "feat_b" in loaded[0]

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "episodes_stats.jsonl"
        path.write_text("")
        with pytest.raises(ValueError, match="No stats found"):
            load_episodes_stats(str(path))

    def test_all_null_raises(self, tmp_path):
        path = tmp_path / "episodes_stats.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"episode_index": 0, "stats": None}) + "\n")
            f.write(json.dumps({"episode_index": 1, "stats": None}) + "\n")
        with pytest.raises(ValueError, match="No stats found"):
            load_episodes_stats(str(path))


# ---------------------------------------------------------------------------
# Tests: compute_gripper_stats_with_transform / compute_gripper_stats_with_action_transform
# ---------------------------------------------------------------------------


class TestComputeGripperStats:
    def test_empty_input(self):
        result = compute_gripper_stats_with_transform([], convert_gripper=False)
        assert result["count"] == [0]
        assert result["mean"] == [0.0]

    def test_no_conversion(self):
        data = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])]
        result = compute_gripper_stats_with_transform(data, convert_gripper=False)
        all_vals = np.concatenate(data)
        assert result["mean"][0] == pytest.approx(float(np.mean(all_vals)))
        assert result["std"][0] == pytest.approx(float(np.std(all_vals, ddof=0)))
        assert result["min"][0] == pytest.approx(1.0)
        assert result["max"][0] == pytest.approx(5.0)
        assert result["count"] == [5]

    def test_empty_episodes_skipped(self):
        data = [np.array([]), np.array([1.0, 2.0])]
        result = compute_gripper_stats_with_transform(data, convert_gripper=False)
        assert result["count"] == [2]
        assert result["mean"][0] == pytest.approx(1.5)


class TestComputeGripperStatsWithActionTransform:
    def test_empty_input(self):
        result = compute_gripper_stats_with_action_transform([], convert_gripper=False)
        assert result["count"] == [0]

    def test_no_conversion(self):
        data = [np.array([10.0, 20.0])]
        result = compute_gripper_stats_with_action_transform(data, convert_gripper=False)
        assert result["mean"][0] == pytest.approx(15.0)
        assert result["min"][0] == pytest.approx(10.0)
        assert result["max"][0] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Tests: numpy_dict_to_lists
# ---------------------------------------------------------------------------


class TestNumpyDictToLists:
    def test_conversion(self):
        d = {
            "feat": {
                "mean": np.array([1.0, 2.0]),
                "std": np.array([0.1, 0.2]),
                "count": np.array([100]),
            }
        }
        result = numpy_dict_to_lists(d)
        assert result["feat"]["mean"] == [1.0, 2.0]
        assert result["feat"]["count"] == [100]
        assert isinstance(result["feat"]["mean"], list)

    def test_non_ndarray_passthrough(self):
        d = {"feat": {"label": "test", "count": np.array([5])}}
        result = numpy_dict_to_lists(d)
        assert result["feat"]["label"] == "test"
        assert result["feat"]["count"] == [5]


# ---------------------------------------------------------------------------
# Tests: _combine_action_quantiles
# ---------------------------------------------------------------------------


class TestCombineActionQuantiles:
    def test_none_state_qs(self):
        rel = {0.01: [1.0, 2.0, 3.0], 0.99: [4.0, 5.0, 6.0]}
        result = _combine_action_quantiles(None, rel)
        assert result is rel

    def test_empty_state_qs(self):
        state = {0.01: [], 0.99: []}
        rel = {0.01: [1.0], 0.99: [2.0]}
        result = _combine_action_quantiles(state, rel)
        assert result is rel

    def test_same_dim(self):
        state = {0.01: [1.0, 2.0], 0.99: [3.0, 4.0]}
        rel = {0.01: [5.0, 6.0], 0.99: [7.0, 8.0]}
        result = _combine_action_quantiles(state, rel)
        assert result is state

    def test_concatenation(self):
        state = {0.01: [1.0, 2.0], 0.99: [3.0, 4.0]}
        rel = {0.01: [1.0, 2.0, 10.0, 20.0, 30.0], 0.99: [3.0, 4.0, 40.0, 50.0, 60.0]}
        result = _combine_action_quantiles(state, rel, base_dim=3)
        assert result[0.01] == [1.0, 2.0, 10.0, 20.0, 30.0]
        assert result[0.99] == [3.0, 4.0, 40.0, 50.0, 60.0]

    def test_dim_mismatch_raises(self):
        state = {0.01: [1.0], 0.99: [2.0]}
        rel = {0.01: [1.0, 2.0, 3.0, 4.0, 5.0], 0.99: [6.0, 7.0, 8.0, 9.0, 10.0]}
        with pytest.raises(ValueError, match="Cannot combine"):
            _combine_action_quantiles(state, rel, base_dim=3)


# ---------------------------------------------------------------------------
# Tests: _combine_action_quantiles_arm_head_relative_gripper_base
# ---------------------------------------------------------------------------


class TestCombineActionQuantilesArmHeadRelativeGripperBase:
    def test_none_state_qs(self):
        rel = {0.01: [1.0] * 11, 0.99: [2.0] * 11}
        result = _combine_action_quantiles_arm_head_relative_gripper_base(None, rel)
        assert result is rel

    def test_proper_combining(self):
        state = {0.01: list(range(8)), 0.99: list(range(10, 18))}
        rel = {0.01: list(range(11)), 0.99: list(range(10, 21))}
        result = _combine_action_quantiles_arm_head_relative_gripper_base(state, rel)
        # arm=state[:5], gripper=rel[5:6], head=state[6:8], base=rel[-3:]
        assert result[0.01] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert result[0.99] == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    def test_dim_mismatch_raises(self):
        state = {0.01: [1.0] * 5, 0.99: [2.0] * 5}
        rel = {0.01: [1.0] * 11, 0.99: [2.0] * 11}
        with pytest.raises(ValueError, match="Cannot combine"):
            _combine_action_quantiles_arm_head_relative_gripper_base(state, rel)


# ---------------------------------------------------------------------------
# Tests: load_gripper_data_from_parquet_dir
# ---------------------------------------------------------------------------


class TestLoadGripperDataFromParquetDir:
    def test_matches_reference(self, tmp_path):
        d = tmp_path / "chunk-000"
        d.mkdir()
        for ep in range(5):
            _make_parquet_episode(
                d / f"episode_{ep:06d}.parquet",
                n_rows=20,
                state_dim=8,
                action_dim=11,
                seed=ep,
            )
        fast_s, fast_a = fast_load_single(str(d))
        ref_s, ref_a = _ref_load_gripper_data_from_parquet_dir(str(d))
        assert len(fast_s) == len(ref_s)
        assert len(fast_a) == len(ref_a)
        for fs, rs in zip(fast_s, ref_s):
            np.testing.assert_allclose(fs, rs, atol=1e-10)
        for fa, ra in zip(fast_a, ref_a):
            np.testing.assert_allclose(fa, ra, atol=1e-10)

    def test_nonexistent_dir_raises(self):
        with pytest.raises(ValueError, match="ディレクトリが存在しません"):
            fast_load_single("/nonexistent/path")

    def test_empty_dir_raises(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(ValueError, match="parquetファイルが見つかりません"):
            fast_load_single(str(d))

    def test_missing_state_column(self, tmp_path):
        d = tmp_path / "chunk-000"
        d.mkdir()
        rng = np.random.default_rng(99)
        n_rows = 10
        table = pa.table(
            {
                "action.relative": rng.standard_normal((n_rows, 11)).tolist(),
                "episode_index": [0] * n_rows,
            }
        )
        pq.write_table(table, str(d / "episode_000000.parquet"))
        state_data, action_data = fast_load_single(str(d))
        assert len(state_data) == 0
        assert len(action_data) == 1
        assert len(action_data[0]) == n_rows

    def test_missing_action_column(self, tmp_path):
        d = tmp_path / "chunk-000"
        d.mkdir()
        rng = np.random.default_rng(99)
        n_rows = 10
        table = pa.table(
            {
                "observation.state": rng.standard_normal((n_rows, 8)).tolist(),
                "episode_index": [0] * n_rows,
            }
        )
        pq.write_table(table, str(d / "episode_000000.parquet"))
        state_data, action_data = fast_load_single(str(d))
        assert len(state_data) == 1
        assert len(action_data) == 0

    def test_custom_column_names(self, tmp_path):
        d = tmp_path / "chunk-000"
        d.mkdir()
        _make_parquet_episode(
            d / "episode_000000.parquet",
            n_rows=10,
            state_dim=8,
            action_dim=11,
            seed=0,
            state_column="my.state",
            action_column="my.action",
        )
        s, a = fast_load_single(str(d), state_column="my.state", action_column="my.action")
        assert len(s) == 1
        assert len(a) == 1

    def test_short_dim_column(self, tmp_path):
        """Columns with fewer than 6 elements should be skipped (no index 5)."""
        d = tmp_path / "chunk-000"
        d.mkdir()
        rng = np.random.default_rng(42)
        table = pa.table(
            {
                "observation.state": rng.standard_normal((5, 3)).tolist(),
                "action.relative": rng.standard_normal((5, 3)).tolist(),
                "episode_index": [0] * 5,
            }
        )
        pq.write_table(table, str(d / "episode_000000.parquet"))
        state_data, action_data = fast_load_single(str(d))
        assert len(state_data) == 0
        assert len(action_data) == 0


# ---------------------------------------------------------------------------
# Tests: load_gripper_data_from_multiple_dirs
# ---------------------------------------------------------------------------


class TestLoadGripperDataFromMultipleDirs:
    def test_matches_reference(self, tmp_path):
        dirs = []
        for chunk_idx in range(3):
            d = tmp_path / f"chunk-{chunk_idx:03d}"
            d.mkdir()
            for ep in range(4):
                seed = chunk_idx * 100 + ep
                _make_parquet_episode(
                    d / f"episode_{ep:06d}.parquet",
                    n_rows=15,
                    state_dim=8,
                    action_dim=11,
                    seed=seed,
                )
            dirs.append(str(d))
        fast_s, fast_a = fast_load_multiple(dirs)
        ref_s, ref_a = [], []
        for d in dirs:
            s, a = _ref_load_gripper_data_from_parquet_dir(d)
            ref_s.extend(s)
            ref_a.extend(a)
        assert len(fast_s) == len(ref_s)
        assert len(fast_a) == len(ref_a)
        for fs, rs in zip(fast_s, ref_s):
            np.testing.assert_allclose(fs, rs, atol=1e-10)
        for fa, ra in zip(fast_a, ref_a):
            np.testing.assert_allclose(fa, ra, atol=1e-10)

    def test_single_dir(self, tmp_path):
        d = tmp_path / "chunk-000"
        d.mkdir()
        _make_parquet_episode(
            d / "episode_000000.parquet",
            n_rows=10,
            state_dim=8,
            action_dim=11,
            seed=0,
        )
        fast_s, fast_a = fast_load_multiple([str(d)])
        single_s, single_a = fast_load_single(str(d))
        assert len(fast_s) == len(single_s)
        for fs, ss in zip(fast_s, single_s):
            np.testing.assert_allclose(fs, ss)

    def test_empty_dirs_list(self, tmp_path):
        """No directories means no data."""
        fast_s, fast_a = fast_load_multiple([])
        assert fast_s == []
        assert fast_a == []


# ---------------------------------------------------------------------------
# Tests: _mixture_quantiles
# ---------------------------------------------------------------------------


class TestMixtureQuantiles:
    def test_matches_reference(self):
        stats_list = _make_episodes_stats(n_episodes=10, dims=8, seed=42)
        ref = _ref_mixture_quantiles(stats_list, "observation.state")
        fast = fast_mixture_quantiles(stats_list, "observation.state")
        for q in (0.01, 0.99):
            np.testing.assert_allclose(fast[q], ref[q], atol=1e-10, err_msg=f"q={q}")

    def test_action_key(self):
        stats_list = _make_episodes_stats(n_episodes=8, dims=8, seed=123)
        ref = _ref_mixture_quantiles(stats_list, "action.relative")
        fast = fast_mixture_quantiles(stats_list, "action.relative")
        for q in (0.01, 0.99):
            np.testing.assert_allclose(fast[q], ref[q], atol=1e-10, err_msg=f"q={q}")

    def test_edge_constant_dimension(self):
        stats_list = []
        for _ in range(5):
            mean = np.array([1.0, 2.0, 3.0])
            std = np.array([0.5, 0.0, 0.3])
            mn = np.array([0.0, 2.0, 2.0])
            mx = np.array([2.0, 2.0, 4.0])
            stats_list.append(
                {
                    "feat": {
                        "mean": mean,
                        "std": std,
                        "min": mn,
                        "max": mx,
                        "count": np.array([100]),
                    }
                }
            )
        ref = _ref_mixture_quantiles(stats_list, "feat")
        fast = fast_mixture_quantiles(stats_list, "feat")
        for q in (0.01, 0.99):
            np.testing.assert_allclose(fast[q], ref[q], atol=1e-10)
        assert fast[0.01][1] == 2.0
        assert fast[0.99][1] == 2.0

    def test_edge_single_episode(self):
        stats_list = [
            {
                "feat": {
                    "mean": np.array([0.5, -0.5]),
                    "std": np.array([1.0, 0.2]),
                    "min": np.array([-2.0, -1.0]),
                    "max": np.array([3.0, 0.0]),
                    "count": np.array([200]),
                }
            }
        ]
        ref = _ref_mixture_quantiles(stats_list, "feat")
        fast = fast_mixture_quantiles(stats_list, "feat")
        for q in (0.01, 0.99):
            np.testing.assert_allclose(fast[q], ref[q], atol=1e-10)

    def test_edge_tiny_std(self):
        stats_list = [
            {
                "feat": {
                    "mean": np.array([1.0]),
                    "std": np.array([1e-14]),
                    "min": np.array([0.999]),
                    "max": np.array([1.001]),
                    "count": np.array([100]),
                }
            }
        ]
        ref = _ref_mixture_quantiles(stats_list, "feat")
        fast = fast_mixture_quantiles(stats_list, "feat")
        for q in (0.01, 0.99):
            np.testing.assert_allclose(fast[q], ref[q], atol=1e-10)

    def test_missing_feature_key(self):
        stats_list = [
            {
                "other": {
                    "mean": np.array([1.0]),
                    "std": np.array([0.1]),
                    "min": np.array([0.5]),
                    "max": np.array([1.5]),
                    "count": np.array([10]),
                }
            }
        ]
        result = fast_mixture_quantiles(stats_list, "nonexistent")
        assert result == {0.01: [], 0.99: []}

    def test_custom_q_levels(self):
        stats_list = _make_episodes_stats(n_episodes=5, dims=4, seed=99)
        q_levels = (0.05, 0.25, 0.5, 0.75, 0.95)
        ref = _ref_mixture_quantiles(stats_list, "observation.state", q_levels=q_levels)
        fast = fast_mixture_quantiles(stats_list, "observation.state", q_levels=q_levels)
        for q in q_levels:
            np.testing.assert_allclose(fast[q], ref[q], atol=1e-10, err_msg=f"q={q}")

    def test_unequal_counts(self):
        """Episodes with very different counts should weight correctly."""
        stats_list = [
            {
                "feat": {
                    "mean": np.array([0.0]),
                    "std": np.array([1.0]),
                    "min": np.array([-3.0]),
                    "max": np.array([3.0]),
                    "count": np.array([10000]),
                }
            },
            {
                "feat": {
                    "mean": np.array([100.0]),
                    "std": np.array([1.0]),
                    "min": np.array([97.0]),
                    "max": np.array([103.0]),
                    "count": np.array([1]),
                }
            },
        ]
        ref = _ref_mixture_quantiles(stats_list, "feat")
        fast = fast_mixture_quantiles(stats_list, "feat")
        for q in (0.01, 0.99):
            np.testing.assert_allclose(fast[q], ref[q], atol=1e-10, err_msg=f"q={q}")

    def test_large_number_of_episodes(self):
        stats_list = _make_episodes_stats(n_episodes=100, dims=4, seed=7)
        ref = _ref_mixture_quantiles(stats_list, "observation.state")
        fast = fast_mixture_quantiles(stats_list, "observation.state")
        for q in (0.01, 0.99):
            np.testing.assert_allclose(fast[q], ref[q], atol=1e-10, err_msg=f"q={q}")

    def test_nonfinite_dimension(self):
        stats_list = [
            {
                "feat": {
                    "mean": np.array([1.0, float("inf")]),
                    "std": np.array([0.5, 1.0]),
                    "min": np.array([0.0, float("-inf")]),
                    "max": np.array([2.0, float("inf")]),
                    "count": np.array([100]),
                }
            }
        ]
        result = fast_mixture_quantiles(stats_list, "feat")
        # Non-finite dimension should be nan
        assert math.isnan(result[0.01][1])
        assert math.isnan(result[0.99][1])
        # Finite dimension should be valid
        assert math.isfinite(result[0.01][0])
        assert math.isfinite(result[0.99][0])

    def test_incomplete_stats_skipped(self):
        """Episodes missing required keys should be skipped."""
        stats_list = [
            {
                "feat": {
                    "mean": np.array([1.0]),
                    # missing std, min, max, count
                }
            },
            {
                "feat": {
                    "mean": np.array([2.0]),
                    "std": np.array([0.5]),
                    "min": np.array([1.0]),
                    "max": np.array([3.0]),
                    "count": np.array([100]),
                }
            },
        ]
        result = fast_mixture_quantiles(stats_list, "feat")
        # Only second episode should be used
        assert len(result[0.01]) == 1
        assert len(result[0.99]) == 1


# ---------------------------------------------------------------------------
# Tests: End-to-end JSON equivalence
#
# Runs both pipelines (reference scalar-loop vs fast vectorised) in-process
# without subprocess, so no jax/flax dependency is needed.
# ---------------------------------------------------------------------------


def _run_pipeline(stats_list, mixture_quantiles_fn, action_mode="relative"):
    """Run the full aggregation pipeline and return the final JSON-like dict.

    This mirrors the main() logic shared by both aggregate_stats_simple.py
    and aggregate_stats_fast.py.
    """
    agg = aggregate_stats(stats_list)
    agg_lists = numpy_dict_to_lists(agg)

    padded_state = pad_hsr_state_stats(agg["observation.state"], adapt_to_pi=True)

    if action_mode == "relative":
        combined_action_stats = agg["action.relative"]
    else:
        raise ValueError(f"Unsupported action_mode for test: {action_mode}")

    padded_actions = pad_hsr_action_stats(combined_action_stats, adapt_to_pi=True)

    state_qs = mixture_quantiles_fn(stats_list, "observation.state")
    action_qs = mixture_quantiles_fn(stats_list, "action.relative")

    state_aligned_ids = [0, 1, 2, 3, 4, 6, 11, 12]
    action_aligned_ids = [0, 1, 2, 3, 4, 6, 11, 12, 13, 14, 15]

    if state_qs[0.01]:
        padded_state["q01"] = [0.0] * len(padded_state.get("mean", []))
        padded_state["q99"] = [0.0] * len(padded_state.get("mean", []))
        for src_i, dst_i in enumerate(state_aligned_ids):
            if dst_i < len(padded_state["q01"]):
                padded_state["q01"][dst_i] = float(state_qs[0.01][src_i])
                padded_state["q99"][dst_i] = float(state_qs[0.99][src_i])

    if action_qs[0.01]:
        padded_actions["q01"] = [0.0] * len(padded_actions.get("mean", []))
        padded_actions["q99"] = [0.0] * len(padded_actions.get("mean", []))
        for src_i, dst_i in enumerate(action_aligned_ids):
            if dst_i < len(padded_actions["q01"]):
                padded_actions["q01"][dst_i] = float(action_qs[0.01][src_i])
                padded_actions["q99"][dst_i] = float(action_qs[0.99][src_i])

    agg_lists["state"] = padded_state
    agg_lists["actions"] = padded_actions

    def _finalize(d):
        return {k: d.get(k) for k in ("mean", "std", "q01", "q99")}

    return {
        "norm_stats": {
            "state": _finalize(agg_lists["state"]),
            "actions": _finalize(agg_lists["actions"]),
        }
    }


class TestEndToEndJsonEquivalence:
    def test_json_output_matches(self, tmp_path):
        """Full pipeline: reference (scalar-loop) vs fast (vectorised) produce identical output."""
        n_episodes = 6
        state_dim = 8
        stats_list = _make_episodes_stats(n_episodes=n_episodes, dims=state_dim, seed=777)

        # Write and re-load through JSONL to match real usage (list -> JSON -> np.ndarray)
        jsonl_path = tmp_path / "episodes_stats.jsonl"
        _write_episodes_stats_jsonl(jsonl_path, stats_list)
        loaded_stats = load_episodes_stats(str(jsonl_path))

        simple_data = _run_pipeline(loaded_stats, _ref_mixture_quantiles)
        fast_data = _run_pipeline(loaded_stats, fast_mixture_quantiles)

        assert set(simple_data.keys()) == set(fast_data.keys())
        assert set(simple_data["norm_stats"].keys()) == set(fast_data["norm_stats"].keys())

        for section in ("state", "actions"):
            for stat_key in ("mean", "std", "q01", "q99"):
                s_val = simple_data["norm_stats"][section][stat_key]
                f_val = fast_data["norm_stats"][section][stat_key]
                if s_val is None and f_val is None:
                    continue
                np.testing.assert_allclose(
                    f_val,
                    s_val,
                    atol=1e-10,
                    err_msg=f"Mismatch in norm_stats.{section}.{stat_key}",
                )

    def test_json_output_with_many_episodes(self, tmp_path):
        """Same pipeline test with more episodes for robustness."""
        stats_list = _make_episodes_stats(n_episodes=50, dims=8, seed=12345)
        jsonl_path = tmp_path / "episodes_stats.jsonl"
        _write_episodes_stats_jsonl(jsonl_path, stats_list)
        loaded_stats = load_episodes_stats(str(jsonl_path))

        simple_data = _run_pipeline(loaded_stats, _ref_mixture_quantiles)
        fast_data = _run_pipeline(loaded_stats, fast_mixture_quantiles)

        for section in ("state", "actions"):
            for stat_key in ("mean", "std", "q01", "q99"):
                s_val = simple_data["norm_stats"][section][stat_key]
                f_val = fast_data["norm_stats"][section][stat_key]
                if s_val is None and f_val is None:
                    continue
                np.testing.assert_allclose(
                    f_val,
                    s_val,
                    atol=1e-10,
                    err_msg=f"Mismatch in norm_stats.{section}.{stat_key}",
                )
