"""财报日历因子评估（补充规格 B / Phase F1）。

只测对**未来**收益的领先预测力，不看同期相关：
- earnings_event_study：财报前后 [-pre,+post] 的平均累计收益结构（含分位锥），可按"上次是否超预期"分组。
- earnings_drift_ic：PEAD 领先 IC——surprise(%) 与公布后 1/5/21/63 日远期收益的事件级 Spearman 相关。
  附**假财报日对照组**（随机日期 → IC≈0）作健全性检查。

多重检验：F1 验证的假设条数记在 HYPOTHESES_TESTED，报告须随之做折扣/标注。
铁律：池化七票、基于事件、PIT（财报盘后公布→反应日取公布日的下一个交易日）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from factors.fundamentals import _reported

# F1 当前累计验证的可证伪假设条数（用于多重检验校正/标注）
HYPOTHESES_TESTED = 2  # H1: PEAD(surprise→drift)；H2: 财报前后收益结构


def _reaction_pos(index: pd.DatetimeIndex, date: pd.Timestamp) -> int:
    """财报公布日的"反应日"= 严格晚于公布日的第一个交易日（盘后公布，次日反应）。"""
    return int(np.searchsorted(index.values.astype("datetime64[ns]"),
                               np.datetime64(date, "ns"), side="right"))


def earnings_event_study(
    prices: dict[str, pd.Series],
    edates: dict[str, pd.DataFrame],
    pre: int = 10,
    post: int = 20,
    by_beat: bool = False,
) -> dict:
    """池化七票，对齐财报反应日，统计 [-pre,+post] 的平均累计收益（CAR）+ 分位锥。"""
    offsets = np.arange(-pre, post + 1)
    paths, beat_flags = [], []
    for t, price in prices.items():
        price = price.dropna()
        p = price.to_numpy(dtype=float)
        idx = price.index
        rep = _reported(edates[t])
        for d, row in rep.iterrows():
            r0 = _reaction_pos(idx, d)
            lo, hi = r0 - 1 - pre, r0 - 1 + post  # 以反应日前一收盘为基准
            if lo < 0 or hi >= len(p):
                continue
            base = p[r0 - 1]
            path = p[lo : hi + 1] / base - 1.0  # 相对基准的累计收益
            if path.shape[0] == len(offsets):
                paths.append(path)
                beat_flags.append(bool(row["Surprise(%)"] > 0))

    M = np.array(paths)
    res = {
        "offsets": offsets,
        "n_events": int(M.shape[0]),
        "mean_car": M.mean(axis=0),
        "p10": np.percentile(M, 10, axis=0),
        "p90": np.percentile(M, 90, axis=0),
        "pre": pre, "post": post,
    }
    if by_beat:
        bf = np.array(beat_flags)
        res["mean_car_beat"] = M[bf].mean(axis=0) if bf.any() else None
        res["mean_car_miss"] = M[~bf].mean(axis=0) if (~bf).any() else None
        res["n_beat"] = int(bf.sum())
        res["n_miss"] = int((~bf).sum())
    return res


def _drift_events(prices, edates, horizons, fake_seed=None, demean=True):
    """收集事件级 (surprise, 公布后 h 日收益)。fake_seed 非 None → 随机假日期对照。

    demean=True：在**每只票内**对远期收益去均值，剔除个股基线漂移（高增长股既常超预期又长期
    大涨会制造虚假长周期相关）；这样假对照才真正 IC≈0、真实 IC 才是纯事件层面 PEAD。
    """
    rng = np.random.default_rng(fake_seed) if fake_seed is not None else None
    surprises, fwd = [], {h: [] for h in horizons}
    maxh = max(horizons)
    for t, price in prices.items():
        price = price.dropna()
        p = price.to_numpy(dtype=float)
        idx = price.index
        rep = _reported(edates[t])
        sv = rep["Surprise(%)"].to_numpy(dtype=float)
        if rng is not None:
            r0s = rng.integers(1, len(p) - maxh - 1, size=len(sv))  # 随机假财报日
        else:
            r0s = np.array([_reaction_pos(idx, d) for d in rep.index])

        s_t, fwd_t = [], {h: [] for h in horizons}
        for s, r0 in zip(sv, r0s):
            if r0 < 1 or r0 + maxh >= len(p):
                continue
            s_t.append(s)
            for h in horizons:
                fwd_t[h].append(p[r0 + h] / p[r0] - 1.0)
        if not s_t:
            continue
        surprises.extend(s_t)
        for h in horizons:
            arr = np.array(fwd_t[h])
            if demean and arr.size:
                arr = arr - arr.mean()   # 票内去均值，剔除个股基线漂移
            fwd[h].extend(arr)
    return np.array(surprises), {h: np.array(v) for h, v in fwd.items()}


def earnings_drift_ic(
    prices: dict[str, pd.Series],
    edates: dict[str, pd.DataFrame],
    horizons: tuple[int, ...] = (1, 5, 21, 63),
    n_control: int = 100,
    seed: int = 0,
) -> dict:
    """PEAD 领先 IC：surprise(%) 与公布后 h 日远期收益的事件级 Spearman 相关。

    对照：跑 n_control 次**随机假财报日**，取平均假 IC（应≈0），并用假 IC 经验分布给真实 IC
    算双侧置换 p 值（real 落在假分布之外才显著）。单次随机抽样噪声大，必须蒙特卡洛。
    """
    s_real, fwd_real = _drift_events(prices, edates, horizons)
    real_ic = {h: (spearmanr(s_real, fwd_real[h]).statistic if len(s_real) > 3 else np.nan)
               for h in horizons}

    fake_ic = {h: [] for h in horizons}
    for k in range(n_control):
        s_f, fwd_f = _drift_events(prices, edates, horizons, fake_seed=seed + k)
        for h in horizons:
            if len(s_f) > 3:
                fake_ic[h].append(spearmanr(s_f, fwd_f[h]).statistic)

    rows = []
    for h in horizons:
        fa = np.array(fake_ic[h])
        mean_fake = float(fa.mean()) if fa.size else np.nan
        # 双侧置换 p：随机对照里 |IC| ≥ |真实IC| 的比例
        pval = float((np.abs(fa) >= abs(real_ic[h])).mean()) if fa.size else np.nan
        rows.append({
            "horizon": h,
            "ic_real": float(real_ic[h]),
            "ic_fake_mean": mean_fake,
            "ic_fake_p5": float(np.percentile(fa, 5)) if fa.size else np.nan,
            "ic_fake_p95": float(np.percentile(fa, 95)) if fa.size else np.nan,
            "perm_pvalue": pval,
            "significant": bool(pval < 0.05),
        })
    tab = pd.DataFrame(rows)
    return {
        "ic_table": tab,
        "n_events": int(len(s_real)),
        "n_control": n_control,
        "hypotheses_tested": HYPOTHESES_TESTED,
        "note": (
            f"PEAD 领先 IC，基于 {len(s_real)} 个财报事件（池化七票），{n_control} 次随机假财报日对照。"
            f"假对照平均 IC 应≈0；真实 IC 的显著性用置换 p 值（按 {HYPOTHESES_TESTED} 条假设做 FDR 后才算数）。"
            "单因子 IC 0.03–0.05 即可用。"
        ),
    }
