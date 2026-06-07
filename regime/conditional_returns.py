"""建仓概率引擎（核心）。

回答「当前点位该不该建仓/加仓」。**粗分桶，不做精确情景匹配。**

三条铁律在此体现：
- 校准而非预测：每个桶输出经验分布（胜率/中位/分位/途中最大浮亏），绝不输出单一概率。
- 永远对比无条件基准：每个桶都给「条件中位 − 无条件中位」的差值。
- 反过拟合：N 按**独立事件**计（非天数）、CI 用 block bootstrap（重叠样本必须用块）。

状态变量（最多 3 个，每个粗切）：
1. in_drawdown：当前是否处于距前高 >阈值（默认 10%）的回撤中（二元）。
2. valuation_tercile：估值高/中/低三分位。
   ⚠️ 真口径应为盈利收益率/CAPE（Phase 6）。当前用「价格相对长期均线的偏离」作**价格代理**，已标注。
3. credit_trend：信用利差走阔 or 收窄（二元，来自 FRED 信用利差）。

DO NOT：禁止输出「建仓概率 73%」这种光秃秃的数字。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from stats.bootstrap import block_bootstrap_ci

_CFG = config.load_config()


# ---------------------------------------------------------------------------
# 远期收益 / 路径浮亏
# ---------------------------------------------------------------------------
def _forward_return(price: pd.Series, h: int) -> pd.Series:
    """从 t 往后 h 个交易日的简单收益：P_{t+h}/P_t - 1。"""
    return price.shift(-h) / price - 1.0


def _path_min_return(price: pd.Series, h: int) -> pd.Series:
    """从 t 进场后、未来 h 日内的**途中最大浮亏**（最差的 P_{t+k}/P_t - 1, k=1..h）。"""
    p = price.to_numpy(dtype=float)
    n = len(p)
    out = np.full(n, np.nan)
    for t in range(n):
        end = min(t + h, n - 1)
        if end <= t:
            continue
        window = p[t + 1 : end + 1]
        if window.size == 0 or not np.isfinite(p[t]) or p[t] == 0:
            continue
        out[t] = np.nanmin(window) / p[t] - 1.0
    return pd.Series(out, index=price.index)


def _count_independent_events(mask: pd.Series) -> int:
    """独立事件数 = 连续 True 段（spell）的个数，对重叠/聚类去重。

    连续多天处于同一状态算「一次」进入该状态的事件，而非多次。
    """
    m = mask.fillna(False).to_numpy(dtype=bool)
    if m.size == 0:
        return 0
    # 段的起点 = True 且（首位 或 前一位为 False）
    starts = m & np.concatenate(([True], ~m[:-1]))
    return int(starts.sum())


# ---------------------------------------------------------------------------
# 状态变量构造
# ---------------------------------------------------------------------------
def build_state(
    prices: pd.Series,
    macro: pd.DataFrame,
    drawdown_threshold: float | None = None,
    valuation_ma: int = 1000,
    credit_ma: int = 126,
) -> pd.DataFrame:
    """从价格 + 宏观构造 3 个粗状态变量，对齐到价格交易日。

    返回 DataFrame，列：in_drawdown(bool)、valuation_tercile({'low','mid','high'})、
    credit_trend({'widening','narrowing'})，外加诊断用的连续值列（前缀 _）。
    """
    dd_thr = drawdown_threshold if drawdown_threshold is not None else _CFG["regime"]["drawdown_threshold"]
    price = prices.dropna()

    # 1) 回撤（二元）
    running_max = price.cummax()
    drawdown = price / running_max - 1.0
    in_drawdown = drawdown < -dd_thr

    # 2) 估值三分位（⚠️ 价格代理：价格相对长期均线偏离；真口径=盈利收益率/CAPE，Phase 6）
    long_ma = price.rolling(valuation_ma, min_periods=valuation_ma // 2).mean()
    val_dev = price / long_ma - 1.0  # >0 偏贵
    q1, q2 = val_dev.quantile([1 / 3, 2 / 3])
    valuation_tercile = pd.Series(
        np.where(val_dev <= q1, "low", np.where(val_dev >= q2, "high", "mid")),
        index=price.index,
    ).where(val_dev.notna())

    # 3) 信用趋势（二元）：信用利差 > 其中期均线 → 走阔
    credit = macro["credit_spread"].reindex(price.index).ffill()
    credit_ma_s = credit.rolling(credit_ma, min_periods=credit_ma // 2).mean()
    credit_trend = pd.Series(
        np.where(credit > credit_ma_s, "widening", "narrowing"), index=price.index
    ).where(credit_ma_s.notna())

    state = pd.DataFrame(
        {
            "in_drawdown": in_drawdown,
            "valuation_tercile": valuation_tercile,
            "credit_trend": credit_trend,
            "_drawdown": drawdown,
            "_val_dev": val_dev,
            "_credit": credit,
            "_yield_curve": macro.get("yield_curve", pd.Series(index=price.index, dtype=float)).reindex(price.index).ffill(),
        }
    )
    return state


# ---------------------------------------------------------------------------
# 条件远期收益（核心）
# ---------------------------------------------------------------------------
def _bucket_stats(
    fwd: pd.Series,
    path_min: pd.Series,
    mask: pd.Series,
    baseline_median: float,
    block_size: int,
    n_boot: int,
    horizon: int = 1,
) -> dict | None:
    """单桶 × 单 horizon 的经验分布统计。N 不足返回 None。

    n_independent：重叠窗口的**有效独立样本** ≈ 命中天数 / horizon。长周期(如 252)下
    重叠严重，名义 N 大但独立窗口很少——据此给 low_power 标记，避免 CI 假性显著。
    """
    sel = fwd[mask].dropna()
    if sel.shape[0] < 20:  # 样本过少不报
        return None
    n_events = _count_independent_events(mask.reindex(fwd.index))
    n_independent = max(1, int(sel.shape[0] // max(1, horizon)))  # 非重叠窗口数
    arr = sel.to_numpy()
    med = float(np.median(arr))
    # 块长理想≈horizon 以吸收重叠，但须留足够多块（≥~5）否则 bootstrap 退化、CI 假塌缩。
    eff_block = max(1, min(block_size, len(arr) // 5))
    try:
        _, ci_lo, ci_hi = block_bootstrap_ci(arr, np.median, block_size=eff_block, n=n_boot)
    except Exception:  # noqa: BLE001
        ci_lo = ci_hi = float("nan")
    pm = path_min[mask].dropna().to_numpy()
    return {
        "n_events": n_events,
        "n_independent": n_independent,
        "low_power": bool(n_independent < 8),  # 独立窗口太少 → 显著性不可信
        "n_days": int(sel.shape[0]),
        "win_rate": float((arr > 0).mean()),
        "median": med,
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "median_path_drawdown": float(np.median(pm)) if pm.size else float("nan"),
        "baseline_median": baseline_median,
        "excess_median": med - baseline_median,
        "ci_low": ci_lo,
        "ci_high": ci_hi,
    }


def conditional_forward_returns(
    prices: pd.DataFrame | pd.Series,
    macro: pd.DataFrame,
    asset: str = "SPY",
    horizons: tuple[int, ...] | None = None,
    groupings: list[list[str]] | None = None,
    n_boot: int = 1000,
) -> pd.DataFrame:
    """对每个状态桶，统计往后各 horizon 的远期收益经验分布 + N + CI + 基准差值。

    参数
    ----
    prices : 价格面板（取 asset 列）或单列 Series
    macro : load_macro() 输出
    asset : 标的（默认 SPY）
    horizons : 远期天数，默认取 config.regime.horizons
    groupings : 要分桶的变量组合列表；默认 = 三个变量各自的边际 + 回撤×估值联合
    n_boot : bootstrap 次数

    返回
    ----
    长表 DataFrame，每行一个 (grouping, bucket, horizon)：含 n_events/win_rate/median/
    分位/途中浮亏/baseline_median/excess_median/ci_low/ci_high。
    并含 grouping='__baseline__' 的无条件基准行。
    """
    horizons = horizons or tuple(_CFG["regime"]["horizons"])
    if groupings is None:
        groupings = [
            ["in_drawdown"],
            ["valuation_tercile"],
            ["credit_trend"],
            ["in_drawdown", "valuation_tercile"],
        ]

    price = (prices[asset] if isinstance(prices, pd.DataFrame) else prices).dropna()
    state = build_state(price, macro)
    state = state.reindex(price.index)

    block_cfg = int(_CFG["stats"]["bootstrap"]["block_size"])
    rows = []

    for h in horizons:
        fwd = _forward_return(price, h)
        path_min = _path_min_return(price, h)
        base = fwd.dropna()
        baseline_median = float(np.median(base.to_numpy())) if base.shape[0] else float("nan")
        block_size = max(block_cfg, h)  # 块至少覆盖 horizon，吸收重叠

        # 无条件基准行
        rows.append(
            {
                "grouping": "__baseline__",
                "bucket": "unconditional",
                "horizon": h,
                "n_events": int(base.shape[0]),
                "n_independent": max(1, int(base.shape[0] // max(1, h))),
                "low_power": False,
                "n_days": int(base.shape[0]),
                "win_rate": float((base.to_numpy() > 0).mean()),
                "median": baseline_median,
                "p10": float(np.percentile(base, 10)),
                "p25": float(np.percentile(base, 25)),
                "p75": float(np.percentile(base, 75)),
                "p90": float(np.percentile(base, 90)),
                "median_path_drawdown": float(np.median(_path_min_return(price, h).dropna())),
                "baseline_median": baseline_median,
                "excess_median": 0.0,
                "ci_low": float("nan"),
                "ci_high": float("nan"),
            }
        )

        for grouping in groupings:
            sub = state[grouping].dropna()
            if sub.empty:
                continue
            # 笛卡尔分桶：按 grouping 各列的取值组合
            combos = sub.drop_duplicates().itertuples(index=False)
            for combo in combos:
                vals = dict(zip(grouping, combo))
                mask = pd.Series(True, index=price.index)
                for k, v in vals.items():
                    mask &= state[k] == v
                stats = _bucket_stats(
                    fwd, path_min, mask, baseline_median, block_size, n_boot, horizon=h
                )
                if stats is None:
                    continue
                label = " & ".join(f"{k}={v}" for k, v in vals.items())
                rows.append(
                    {"grouping": "+".join(grouping), "bucket": label, "horizon": h, **stats}
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 当前指纹 vs 历史指纹（供人判断，不自动给结论）
# ---------------------------------------------------------------------------
def current_fingerprint(
    prices: pd.DataFrame | pd.Series,
    macro: pd.DataFrame,
    asset: str = "SPY",
    drawdown_threshold: float = 0.10,
    top_k: int = 8,
) -> pd.DataFrame:
    """把历次大跌当时的利率曲线/信用利差/估值分位，与「今天」并排列出。

    DO NOT：不自动判断「当前更像哪次历史下跌」并据此给买卖结论（实时不可解）。
    本面板仅供人对照。
    """
    price = (prices[asset] if isinstance(prices, pd.DataFrame) else prices).dropna()
    state = build_state(price, macro).reindex(price.index)

    drawdown = state["_drawdown"]
    # 历史大跌「谷底」：在 >阈值 的回撤段内，取每段回撤最深的那天
    in_dd = drawdown < -drawdown_threshold
    seg_id = (in_dd != in_dd.shift()).cumsum()
    troughs = []
    for _, idx in drawdown[in_dd].groupby(seg_id[in_dd]).groups.items():
        seg = drawdown.loc[idx]
        troughs.append(seg.idxmin())
    troughs = sorted(troughs, key=lambda d: drawdown.loc[d])[:top_k]  # 最深的 top_k

    def _row(d, when):
        return {
            "when": when,
            "date": d,
            "drawdown": round(float(drawdown.loc[d]), 4),
            "val_dev": round(float(state["_val_dev"].loc[d]), 4) if pd.notna(state["_val_dev"].loc[d]) else np.nan,
            "valuation_tercile": state["valuation_tercile"].loc[d],
            "credit_spread": round(float(state["_credit"].loc[d]), 3) if pd.notna(state["_credit"].loc[d]) else np.nan,
            "credit_trend": state["credit_trend"].loc[d],
            "yield_curve": round(float(state["_yield_curve"].loc[d]), 3) if pd.notna(state["_yield_curve"].loc[d]) else np.nan,
        }

    today = price.index[-1]
    rows = [_row(today, "TODAY")]
    rows += [_row(d, f"trough_{i+1}") for i, d in enumerate(troughs)]
    out = pd.DataFrame(rows).set_index("when")
    out.attrs["disclaimer"] = (
        "仅供人对照历史指纹，不代表「当前更像某次下跌」；原因只能事后归类，实时不可解。"
    )
    return out


def format_bucket_verdict(row: pd.Series) -> str:
    """按规格书强制模板，把一行桶统计渲染成措辞。"""
    h = int(row["horizon"])
    verdict = "弱倾斜，非信号" if abs(row["excess_median"]) < 0.02 else "明显倾斜，仍需结合 N 与 CI 判断"
    return (
        f"状态=[{row['bucket']}]：{h} 日远期收益中位 {row['median']*100:+.1f}%，"
        f"10 分位 {row['p10']*100:+.1f}%，相对无条件基准 {row['excess_median']*100:+.1f} 个点，"
        f"基于 N≈{int(row['n_events'])} 个独立事件"
        f"（95% CI 中位: [{row['ci_low']*100:+.1f}%, {row['ci_high']*100:+.1f}%]）。结论：{verdict}。"
    )
