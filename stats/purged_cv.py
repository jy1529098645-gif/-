"""Purged & Embargoed 交叉验证（López de Prado, Advances in Financial ML, ch.7）。

普通 K 折/walk-forward 在金融里会泄漏：远期收益标签 [t, t+h] 跨越多天，训练样本的标签会和
测试段重叠 → 信息泄漏 → OOS 虚高。Purged CV 把与测试段标签重叠的训练样本**剔除(purge)**，
并在测试段后加一段**禁运(embargo)**缓冲，给出真正无泄漏的样本外评估。

铁律：这是反过拟合基建（和 block bootstrap / deflated Sharpe / walk-forward 同类），
只做诚实评估，不产生买卖信号。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def purged_kfold_indices(n: int, horizon: int, n_splits: int = 5,
                         embargo: float = 0.01) -> list[tuple[np.ndarray, np.ndarray]]:
    """生成 purged + embargoed K 折的 (train_idx, test_idx)。

    horizon：标签跨度（远期收益天数）——决定 purge 宽度。
    embargo：测试段后额外禁运比例（× n）。
    每个测试折是连续块；训练集剔除 [test_start−horizon, test_end+horizon+embargo] 内的样本。"""
    if n_splits < 2 or n < n_splits:
        raise ValueError("n_splits≥2 且 n≥n_splits")
    indices = np.arange(n)
    folds = np.array_split(indices, n_splits)
    emb = int(n * embargo)
    out = []
    for fold in folds:
        if len(fold) == 0:
            continue
        t0, t1 = int(fold[0]), int(fold[-1])
        test = indices[t0:t1 + 1]
        mask = np.ones(n, dtype=bool)
        lo = max(0, t0 - horizon)            # purge 左：标签会延伸进测试段的训练样本
        hi = min(n, t1 + horizon + emb + 1)  # purge 右 + embargo
        mask[lo:hi] = False
        train = indices[mask]
        out.append((train, test))
    return out


def verify_no_leakage(train: np.ndarray, test: np.ndarray, horizon: int) -> bool:
    """校验：训练样本的标签窗口 [j, j+horizon] 与测试段 [min,max] 无重叠（含右侧 features 泄漏）。"""
    if len(test) == 0 or len(train) == 0:
        return True
    t0, t1 = int(test.min()), int(test.max())
    for j in train:
        if (j + horizon >= t0) and (j <= t1 + horizon):  # 标签重叠 或 features 落在测试标签内
            return False
    return True


def purged_cv_ic(score: pd.DataFrame, prices: pd.DataFrame, horizon: int = 21,
                 n_splits: int = 6, embargo: float = 0.02) -> dict:
    """对横截面因子做 purged CV 的**无泄漏 OOS IC**评估。

    在每个 purged 测试折内，按调仓日算截面 Spearman IC（score vs 未来 horizon 日收益），
    汇总各折 OOS IC、跨折均值、跨折 t 统计量。返回 {fold_ic, mean_oos_ic, t_across_folds, n_folds, note}。"""
    from scipy.stats import spearmanr

    px = prices.dropna(how="all").ffill()
    fwd = px.shift(-horizon) / px - 1.0
    # 在调仓日(每 horizon 取一次，标签天然不重叠)上评估
    rebal = px.index[::horizon]
    sc = score.reindex(rebal)
    fw = fwd.reindex(rebal)
    valid_rows = [i for i in range(len(rebal)) if sc.iloc[i].notna().sum() >= 5 and fw.iloc[i].notna().sum() >= 5]
    if len(valid_rows) < n_splits * 2:
        return {"fold_ic": [], "mean_oos_ic": float("nan"), "t_across_folds": float("nan"),
                "n_folds": 0, "note": "有效调仓日不足，无法做 purged CV。"}

    m = len(valid_rows)
    # horizon 已=调仓步长 → 相邻行标签不重叠，purge 宽度取 1 行 + embargo
    splits = purged_kfold_indices(m, horizon=1, n_splits=n_splits, embargo=embargo)
    fold_ic = []
    for _, test in splits:
        ics = []
        for pos in test:
            ridx = valid_rows[pos]
            s_row = sc.iloc[ridx].dropna(); f_row = fw.iloc[ridx].dropna()
            common = s_row.index.intersection(f_row.index)
            if len(common) >= 5:
                r = spearmanr(s_row[common], f_row[common]).statistic
                if r == r:
                    ics.append(r)
        if ics:
            fold_ic.append(float(np.mean(ics)))

    arr = np.array(fold_ic)
    mean_ic = float(arr.mean()) if arr.size else float("nan")
    t = float(arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr)))) if arr.size >= 2 and arr.std(ddof=1) > 0 else float("nan")
    return {"fold_ic": fold_ic, "mean_oos_ic": mean_ic, "t_across_folds": t,
            "n_folds": len(fold_ic),
            "note": ("Purged+Embargo CV 的无泄漏 OOS IC：剔除标签重叠+禁运后各折的截面 IC。"
                     "跨折均值>0 且 |t|>2 才算稳健；这是比普通 walk-forward 更严的反泄漏评估。")}
