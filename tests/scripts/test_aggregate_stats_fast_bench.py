"""Benchmark tests for aggregate_stats_fast.py vs reference (simple) implementations.

Run:
    python -m pytest tests/scripts/test_aggregate_stats_fast_bench.py -v -s
"""
from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import math  # noqa: E402

import pandas as pd  # noqa: E402

from scripts.aggregate_stats_fast import (  # noqa: E402
    _mixture_quantiles as fast_mixture_quantiles,
    _std_normal_cdf,
    load_gripper_data_from_multiple_dirs as fast_load_multiple,
    load_gripper_data_from_parquet_dir as fast_load_single,
)


# ---------------------------------------------------------------------------
# Reference implementations (simple/slow)
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


def _ref_load_gripper_data_from_parquet_dir(
    parquet_dir,
    state_column="observation.state",
    action_column="action.relative",
):
    import glob as _glob

    parquet_dir_path = Path(parquet_dir)
    parquet_files = sorted(_glob.glob(str(parquet_dir_path / "episode_*.parquet")))
    state_data, action_data = [], []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        s_vals, a_vals = [], []
        if state_column in df.columns:
            for _, row in df.iterrows():
                sd = row[state_column]
                if isinstance(sd, (list, np.ndarray)) and len(sd) > 5:
                    s_vals.append(float(sd[5]))
        if action_column in df.columns:
            for _, row in df.iterrows():
                ad = row[action_column]
                if isinstance(ad, (list, np.ndarray)) and len(ad) > 5:
                    a_vals.append(float(ad[5]))
        if s_vals:
            state_data.append(np.array(s_vals))
        if a_vals:
            action_data.append(np.array(a_vals))
    return state_data, action_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_parquet_episode(path, n_rows, state_dim, action_dim, seed):
    rng = np.random.default_rng(seed)
    table = pa.table(
        {
            "observation.state": rng.standard_normal((n_rows, state_dim)).tolist(),
            "action.relative": rng.standard_normal((n_rows, action_dim)).tolist(),
            "episode_index": [seed] * n_rows,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))


def _make_episodes_stats(n_episodes, dims, seed):
    rng = np.random.default_rng(seed)
    stats_list = []
    for _ in range(n_episodes):
        mean = rng.standard_normal(dims)
        std = np.abs(rng.standard_normal(dims)) + 0.01
        stats_list.append(
            {
                "observation.state": {
                    "mean": mean,
                    "std": std,
                    "min": mean - 3 * std,
                    "max": mean + 3 * std,
                    "count": np.array([rng.integers(50, 500)]),
                }
            }
        )
    return stats_list


def _timeit(fn, repeats=3):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


class TestBenchmarkMixtureQuantiles:
    """Compare _mixture_quantiles: fast (vectorised) vs reference (scalar loop)."""

    def test_bench_small(self):
        """10 episodes, 8 dims."""
        stats = _make_episodes_stats(10, 8, seed=1)
        t_ref = _timeit(lambda: _ref_mixture_quantiles(stats, "observation.state"))
        t_fast = _timeit(lambda: fast_mixture_quantiles(stats, "observation.state"))
        speedup = t_ref / t_fast if t_fast > 0 else float("inf")
        print(f"\n[mixture_quantiles small] ref={t_ref:.4f}s  fast={t_fast:.4f}s  speedup={speedup:.1f}x")

    def test_bench_medium(self):
        """50 episodes, 32 dims."""
        stats = _make_episodes_stats(50, 32, seed=2)
        t_ref = _timeit(lambda: _ref_mixture_quantiles(stats, "observation.state"))
        t_fast = _timeit(lambda: fast_mixture_quantiles(stats, "observation.state"))
        speedup = t_ref / t_fast if t_fast > 0 else float("inf")
        print(f"\n[mixture_quantiles medium] ref={t_ref:.4f}s  fast={t_fast:.4f}s  speedup={speedup:.1f}x")

    def test_bench_large(self):
        """200 episodes, 64 dims."""
        stats = _make_episodes_stats(200, 64, seed=3)
        t_ref = _timeit(lambda: _ref_mixture_quantiles(stats, "observation.state"))
        t_fast = _timeit(lambda: fast_mixture_quantiles(stats, "observation.state"))
        speedup = t_ref / t_fast if t_fast > 0 else float("inf")
        print(f"\n[mixture_quantiles large] ref={t_ref:.4f}s  fast={t_fast:.4f}s  speedup={speedup:.1f}x")


class TestBenchmarkParquetLoader:
    """Compare parquet loading: fast (PyArrow column projection) vs reference (pandas iterrows)."""

    def test_bench_single_dir(self, tmp_path):
        """20 episodes x 100 rows."""
        d = tmp_path / "chunk-000"
        d.mkdir()
        for ep in range(20):
            _make_parquet_episode(d / f"episode_{ep:06d}.parquet", 100, 8, 11, seed=ep)

        t_ref = _timeit(lambda: _ref_load_gripper_data_from_parquet_dir(str(d)))
        t_fast = _timeit(lambda: fast_load_single(str(d)))
        speedup = t_ref / t_fast if t_fast > 0 else float("inf")
        print(f"\n[parquet_load single] ref={t_ref:.4f}s  fast={t_fast:.4f}s  speedup={speedup:.1f}x")

    def test_bench_multiple_dirs(self, tmp_path):
        """4 dirs x 10 episodes x 100 rows."""
        dirs = []
        for ci in range(4):
            d = tmp_path / f"chunk-{ci:03d}"
            d.mkdir()
            for ep in range(10):
                _make_parquet_episode(d / f"episode_{ep:06d}.parquet", 100, 8, 11, seed=ci * 100 + ep)
            dirs.append(str(d))

        t_ref = _timeit(lambda: [_ref_load_gripper_data_from_parquet_dir(d) for d in dirs])
        t_fast = _timeit(lambda: fast_load_multiple(dirs))
        speedup = t_ref / t_fast if t_fast > 0 else float("inf")
        print(f"\n[parquet_load multi] ref={t_ref:.4f}s  fast={t_fast:.4f}s  speedup={speedup:.1f}x")

    def test_bench_large_episodes(self, tmp_path):
        """5 episodes x 1000 rows (larger per-episode data)."""
        d = tmp_path / "chunk-000"
        d.mkdir()
        for ep in range(5):
            _make_parquet_episode(d / f"episode_{ep:06d}.parquet", 1000, 8, 11, seed=ep)

        t_ref = _timeit(lambda: _ref_load_gripper_data_from_parquet_dir(str(d)))
        t_fast = _timeit(lambda: fast_load_single(str(d)))
        speedup = t_ref / t_fast if t_fast > 0 else float("inf")
        print(f"\n[parquet_load large_ep] ref={t_ref:.4f}s  fast={t_fast:.4f}s  speedup={speedup:.1f}x")
