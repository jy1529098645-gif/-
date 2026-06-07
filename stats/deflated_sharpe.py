"""Deflated Sharpe Ratio（López de Prado, 2014）。

当你从 N 次尝试（多个因子/参数）里挑出最优夏普时，最优值天然被运气抬高。
DSR 按尝试次数对夏普打折，扣掉「靠运气挑出来的部分」，治多重检验/数据窥探。

筛选了多个候选时**必须**调用本模块；否则报告须显式声明
「未做多重检验校正，下列显著性不可信」。

记号：所有夏普为**同一周期口径**（如都用日度收益、未年化），n_obs 为收益样本数。
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

_EULER = 0.5772156649015329  # Euler–Mascheroni 常数


def probabilistic_sharpe_ratio(
    sr: float,
    sr_benchmark: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """概率夏普 PSR：观测夏普 sr 真正超过基准 sr_benchmark 的概率（考虑偏度/峰度）。

    PSR = Φ( (sr - sr_benchmark) * sqrt(n_obs - 1) /
             sqrt(1 - skew*sr + (kurtosis - 1)/4 * sr^2) )
    kurtosis 为「全峰度」（正态=3）。
    """
    if n_obs < 2:
        raise ValueError("n_obs 必须 ≥ 2")
    denom = np.sqrt(1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2)
    if denom == 0 or not np.isfinite(denom):
        raise ValueError("PSR 分母非法（检查 skew/kurtosis/sr）")
    z = (sr - sr_benchmark) * np.sqrt(n_obs - 1) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(sr_trials_std: float, n_trials: int) -> float:
    """N 次独立尝试下、零真实信号时夏普「期望最大值」（即应被扣掉的基准 SR0）。

    SR0 = sr_trials_std * [ (1-γ)·Z(1 - 1/N) + γ·Z(1 - 1/(N·e)) ]
    其中 Z 为标准正态分位函数，γ 为 Euler–Mascheroni 常数。
    """
    if n_trials < 1:
        raise ValueError("n_trials 必须 ≥ 1")
    if n_trials == 1:
        return 0.0
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sr_trials_std * ((1.0 - _EULER) * z1 + _EULER * z2))


def deflated_sharpe_ratio(
    sr: float,
    sr_trials_std: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe：在「期望最大夏普」基准下的 PSR。

    参数
    ----
    sr : 入选（通常是最优）策略的夏普（与下面同口径）
    sr_trials_std : 各次尝试夏普估计值的标准差（衡量搜索的广度/方差）
    n_trials : 尝试次数 N（独立或近似独立的候选数）
    n_obs : 收益样本数
    skew, kurtosis : 收益的偏度与全峰度（正态峰度=3）

    返回
    ----
    DSR ∈ [0,1]：扣除多重检验后，该夏普仍显著为正的概率。
    经验阈值：DSR > 0.95 才算稳健。
    """
    sr0 = expected_max_sharpe(sr_trials_std, n_trials)
    return probabilistic_sharpe_ratio(sr, sr0, n_obs, skew=skew, kurtosis=kurtosis)
