"""Tier-3 测试：Purged/Embargoed CV(无泄漏) + 数据质量监控。"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from stats import purged_cv as pcv
from analysis import data_quality as dq


# ---------------------------------------------------------------------------
# Purged K-Fold
# ---------------------------------------------------------------------------
def test_purged_splits_cover_all_test():
    splits = pcv.purged_kfold_indices(100, horizon=5, n_splits=5, embargo=0.02)
    assert len(splits) == 5
    all_test = np.concatenate([t for _, t in splits])
    assert sorted(all_test.tolist()) == list(range(100))  # 测试折无重叠且覆盖全集


def test_purged_no_leakage():
    n, h = 200, 10
    for train, test in pcv.purged_kfold_indices(n, horizon=h, n_splits=6, embargo=0.03):
        assert pcv.verify_no_leakage(train, test, horizon=h)
        # 训练与测试不相交
        assert len(np.intersect1d(train, test)) == 0


def test_purge_removes_more_with_larger_horizon():
    s_small = pcv.purged_kfold_indices(300, horizon=2, n_splits=5)
    s_big = pcv.purged_kfold_indices(300, horizon=40, n_splits=5)
    # horizon 越大，purge 掉的训练样本越多 → 训练集越小
    train_small = np.mean([len(tr) for tr, _ in s_small])
    train_big = np.mean([len(tr) for tr, _ in s_big])
    assert train_big < train_small


def test_purged_cv_ic_runs():
    rng = np.random.RandomState(0)
    idx = pd.date_range("2016-01-01", periods=800, freq="B")
    cols = [f"T{i}" for i in range(8)]
    prices = pd.DataFrame({c: np.cumprod(1 + rng.randn(800) * 0.01 + 0.0003) * 100 for c in cols}, index=idx)
    score = prices.pct_change(21)  # 简单动量作因子
    res = pcv.purged_cv_ic(score, prices, horizon=21, n_splits=5, embargo=0.02)
    assert res["n_folds"] >= 2
    assert "mean_oos_ic" in res and "t_across_folds" in res


# ---------------------------------------------------------------------------
# 数据质量
# ---------------------------------------------------------------------------
def test_data_health_fresh_vs_stale():
    today = dt.date(2026, 6, 7)
    idx_fresh = pd.bdate_range(end="2026-06-05", periods=300)
    idx_stale = pd.bdate_range(end="2025-01-01", periods=300)
    prices = pd.DataFrame({
        "FRESH": pd.Series(np.linspace(100, 120, 300), index=idx_fresh),
        "STALE": pd.Series(np.linspace(50, 60, 300), index=idx_stale),
    })
    res = dq.data_health(prices, today=today)
    tbl = res["table"].set_index("标的")
    assert "✅" in tbl.loc["FRESH", "状态"]
    assert "🔴" in tbl.loc["STALE", "状态"]
    assert res["n_fresh"] >= 1 and res["n_stale"] >= 1


def test_data_health_flags_jump():
    today = dt.date(2026, 6, 7)
    idx = pd.bdate_range(end="2026-06-05", periods=200)
    v = np.linspace(100, 110, 200).copy()
    v[100] = v[99] * 2.0  # +100% 异常跳空
    res = dq.data_health(pd.DataFrame({"JUMP": pd.Series(v, index=idx)}), today=today)
    row = res["table"].iloc[0]
    assert row["异常跳空"] >= 1
    assert "⚠️" in row["状态"]


def test_data_health_empty_series():
    today = dt.date(2026, 6, 7)
    idx = pd.bdate_range(end="2026-06-05", periods=10)
    res = dq.data_health(pd.DataFrame({"EMPTY": pd.Series([np.nan] * 10, index=idx)}), today=today)
    assert "无数据" in res["table"].iloc[0]["状态"]
