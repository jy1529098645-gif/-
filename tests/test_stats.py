"""Phase 3 验收：反过拟合基建（合成数据，不联网）。

- block bootstrap：CI 覆盖已知真值；自相关序列的 CI 比逐点重采样更宽。
- walk-forward：训练严格早于测试、无泄漏、OOS 不重叠。
- deflated Sharpe：尝试次数越多 DSR 越低；PSR 在已知情形给出预期值。
"""
import numpy as np
import pandas as pd

from stats.bootstrap import block_bootstrap_ci, sharpe
from stats.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)
from stats.walkforward import walk_forward_splits


# ---------------------------------------------------------------------------
# block bootstrap
# ---------------------------------------------------------------------------
def test_bootstrap_ci_covers_true_mean():
    rng = np.random.default_rng(0)
    true_mean = 0.001
    x = rng.normal(true_mean, 0.01, size=3000)
    point, lo, hi = block_bootstrap_ci(x, np.mean, block_size=21, n=1000)
    assert lo < point < hi
    assert lo < true_mean < hi          # 95% CI 应覆盖真值
    assert hi - lo > 0


def test_bootstrap_reproducible():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.01, size=1000)
    a = block_bootstrap_ci(x, np.mean, block_size=10, n=500, seed=7)
    b = block_bootstrap_ci(x, np.mean, block_size=10, n=500, seed=7)
    assert a == b


def test_bootstrap_block_wider_under_autocorrelation():
    """强自相关序列：大块长的 CI 应比块长=1（≈逐点）更宽（诚实地反映有效样本更少）。"""
    rng = np.random.default_rng(2)
    n = 4000
    eps = rng.normal(0, 1, size=n)
    x = np.zeros(n)
    phi = 0.9                      # 强正自相关 AR(1)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    _, lo1, hi1 = block_bootstrap_ci(x, np.mean, block_size=1, n=1000, seed=3)
    _, lo50, hi50 = block_bootstrap_ci(x, np.mean, block_size=100, n=1000, seed=3)
    assert (hi50 - lo50) > (hi1 - lo1)


def test_sharpe_helper():
    rng = np.random.default_rng(4)
    x = rng.normal(0.0005, 0.01, size=2520)
    s = sharpe(x)
    assert 0.3 < s < 1.5            # 大致合理范围（年化）


# ---------------------------------------------------------------------------
# walk-forward
# ---------------------------------------------------------------------------
def test_walk_forward_no_leakage_and_nonoverlap():
    idx = pd.bdate_range("2000-01-01", periods=1000)
    splits = walk_forward_splits(idx, train=250, test=125, step=125)
    assert len(splits) >= 5
    prev_test_end = -1
    for sp in splits:
        # 训练严格早于测试，无泄漏
        assert sp.train_end == sp.test_start
        assert sp.train_start < sp.train_end < sp.test_end
        # 滚动定长窗口
        assert sp.train_end - sp.train_start == 250
        assert sp.test_end - sp.test_start == 125
        # OOS 段不重叠且递增
        assert sp.test_start >= prev_test_end
        prev_test_end = sp.test_end
        # 标签切片正确
        tr, te = sp.slice(idx)
        assert tr.max() < te.min()


def test_walk_forward_anchored_expands():
    splits = walk_forward_splits(1000, train=250, test=125, anchored=True)
    assert all(sp.train_start == 0 for sp in splits)
    # 扩张窗口：后面的训练段更长
    assert splits[-1].train_end > splits[0].train_end


# ---------------------------------------------------------------------------
# deflated Sharpe
# ---------------------------------------------------------------------------
def test_psr_known_values():
    # sr == benchmark → 概率 0.5
    assert abs(probabilistic_sharpe_ratio(0.1, 0.1, 1000) - 0.5) < 1e-9
    # sr 远高于 benchmark 且样本大 → 概率→1
    assert probabilistic_sharpe_ratio(0.2, 0.0, 2000) > 0.99
    # sr 低于 benchmark → 概率 < 0.5
    assert probabilistic_sharpe_ratio(0.0, 0.1, 1000) < 0.5


def test_expected_max_sharpe_increases_with_trials():
    e1 = expected_max_sharpe(0.1, 10)
    e2 = expected_max_sharpe(0.1, 1000)
    assert 0 < e1 < e2              # 尝试越多，期望最大夏普越高
    assert expected_max_sharpe(0.1, 1) == 0.0


def test_deflated_sharpe_penalizes_multiple_trials():
    """同一夏普，从更多尝试里选出 → DSR 更低（更可能是运气）。"""
    sr, std, n_obs = 0.12, 0.1, 1500   # 日度口径夏普
    dsr_few = deflated_sharpe_ratio(sr, std, n_trials=2, n_obs=n_obs)
    dsr_many = deflated_sharpe_ratio(sr, std, n_trials=500, n_obs=n_obs)
    assert dsr_few > dsr_many
    assert 0.0 <= dsr_many <= dsr_few <= 1.0
