"""进出场规则池化评估（补充规格 A，S2）。

铁律（本文件专属）：
1. 池化统计，不做单票结论：七只票的交易事件合并成一个样本池统计；单票仅展示并标注 N。
2. 折算有效独立样本数 N_eff：七姐妹高相关，名义 N 须按相关性折算，所有 CI 基于 N_eff。
3. 入场×出场成对评估；参数选择交给 walk-forward（S3）。
4. 永远对比基准：与"随机进场 + 持有相同平均天数"的基准并排，给超额 + CI。
禁止输出"最佳入场点 / 目标价"。
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

import config
from backtest import exits as ex

_CFG = config.load_config()


# ---------------------------------------------------------------------------
# 有效独立样本数
# ---------------------------------------------------------------------------
def effective_n(returns: pd.DataFrame, n_trades: int) -> dict:
    """按标的间相关性把名义 N 折算成 N_eff。

    k 只标的、平均成对相关 rho_bar →
        k_eff = k / (1 + (k-1)*rho_bar)，  N_eff = N * k_eff / k。
    rho_bar 越高，N_eff 越小（七姐妹常 ~2 倍独立证据，而非 7 倍）。
    """
    k = returns.shape[1]
    corr = returns.corr()
    off = corr.where(~np.eye(k, dtype=bool)).stack()
    rho_bar = float(off.mean()) if not off.empty else 0.0
    rho_bar = max(0.0, rho_bar)  # 负相关不"增益"证据，地板取 0
    k_eff = k / (1.0 + (k - 1) * rho_bar) if k > 1 else 1.0
    n_eff = n_trades * k_eff / k if k > 0 else n_trades
    return {"k": k, "rho_bar": rho_bar, "k_eff": k_eff, "n_eff": n_eff}


# ---------------------------------------------------------------------------
# 随机进场基准
# ---------------------------------------------------------------------------
def _random_baseline_returns(
    prices: dict[str, pd.Series], avg_hold: int, n_per_ticker: int, seed: int = 0
) -> np.ndarray:
    """随机进场 + 持有 avg_hold 个交易日的收益分布（无条件基准）。"""
    rng = np.random.default_rng(seed)
    out = []
    for s in prices.values():
        p = s.dropna().to_numpy(dtype=float)
        hi = len(p) - avg_hold - 1
        if hi <= 1:
            continue
        starts = rng.integers(0, hi, size=n_per_ticker)
        out.extend(p[starts + avg_hold] / p[starts] - 1.0)
    return np.asarray(out)


def _bootstrap_excess_ci(
    rule_rets: np.ndarray, base_rets: np.ndarray, n_eff: float, n_boot: int, seed: int = 0
) -> tuple[float, float, float]:
    """超额（rule 中位 − baseline 中位）的 bootstrap CI，并按 N_eff 加宽。

    朴素 bootstrap 把每笔当独立 → 高估证据；用 sqrt(N/N_eff) 放大半宽以诚实反映相关性。
    """
    rng = np.random.default_rng(seed)
    n = len(rule_rets)
    obs = float(np.median(rule_rets) - np.median(base_rets))
    samples = np.empty(n_boot)
    for b in range(n_boot):
        rb = rng.choice(rule_rets, size=n, replace=True)
        bb = rng.choice(base_rets, size=len(base_rets), replace=True)
        samples[b] = np.median(rb) - np.median(bb)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    half = (hi - lo) / 2.0
    factor = float(np.sqrt(max(1.0, n / max(1e-9, n_eff))))  # N_eff 折算放大
    return obs, obs - half * factor, obs + half * factor


# ---------------------------------------------------------------------------
# 池化统计辅助
# ---------------------------------------------------------------------------
def _longest_losing_streak(trades: pd.DataFrame) -> int:
    r = trades.sort_values("entry_date")["return"].to_numpy()
    best = cur = 0
    for x in r:
        cur = cur + 1 if x <= 0 else 0
        best = max(best, cur)
    return int(best)


# ---------------------------------------------------------------------------
# 主评估
# ---------------------------------------------------------------------------
def evaluate_rule(
    entry_fn: Callable[[pd.Series], pd.Series],
    exit_spec: dict,
    tickers: list[str] | None = None,
    start: str = "2010-01-01",
    end: str | None = None,
    rule_name: str = "rule",
    n_boot: int = 1000,
    make_plots: bool = False,
    seed: int = 0,
    condition_fn: Callable[[str, pd.Series], pd.Series] | None = None,
) -> dict:
    """在多只票上逐笔回测"入场→出场"，合并成池统计（N_eff + bootstrap CI + 基准对比）。

    entry_fn：price(Series) -> 布尔入场信号。
    condition_fn：可选 (ticker, price) -> 布尔 Series，与入场 AND（分条件池化评估，如"仅财报后超预期窗口"）。
    返回 dict：pooled（池化统计）、per_ticker（单票展示，含各自 N）、baseline、n_eff、plots。
    """
    from data import loader

    tickers = tickers or loader.load_universe(_CFG["single_name"]["universe"])
    # 容错加载：宽池里个别票（已私有化/退市/网络）取不到时跳过，不让整池回测崩。
    prices = {}
    for t in tickers:
        try:
            prices[t] = loader.load_prices([t], start, end)[t].dropna()
        except Exception:  # noqa: BLE001
            continue

    per_ticker, all_trades = {}, []
    for t, p in prices.items():
        entries = entry_fn(p).reindex(p.index).fillna(False).astype(bool)
        if condition_fn is not None:
            cond = condition_fn(t, p).reindex(p.index).fillna(False).astype(bool)
            entries = entries & cond
        pf = ex.run_trades(p, entries, exit_spec)
        tr = ex.extract_trades(pf, p)
        tr["ticker"] = t
        per_ticker[t] = {
            "n_trades": len(tr),
            "win_rate": float((tr["return"] > 0).mean()) if len(tr) else float("nan"),
            "median_return": float(tr["return"].median()) if len(tr) else float("nan"),
        }
        all_trades.append(tr)

    pool = pd.concat(all_trades, ignore_index=True)
    if pool.empty:
        raise ValueError("规则在所有标的上都没有产生交易")

    rule_rets = pool["return"].dropna().to_numpy()
    avg_hold = int(np.nanmedian(pool["duration_bars"].dropna())) or 21

    # N_eff（基于日收益相关性）
    rets_df = pd.DataFrame({t: p.pct_change() for t, p in prices.items()}).dropna(how="all")
    neff = effective_n(rets_df, len(rule_rets))

    # 随机基准 + 超额 CI
    base_rets = _random_baseline_returns(prices, avg_hold, n_per_ticker=2000, seed=seed)
    excess, ci_lo, ci_hi = _bootstrap_excess_ci(rule_rets, base_rets, neff["n_eff"], n_boot, seed)

    worst5 = float(np.percentile(rule_rets, 5))
    pooled = {
        "n_trades": int(len(rule_rets)),
        "n_eff": neff["n_eff"],
        "k_tickers": neff["k"],
        "rho_bar": neff["rho_bar"],
        "win_rate": float((rule_rets > 0).mean()),
        "median_return": float(np.median(rule_rets)),
        "p5_return": worst5,
        "p25_return": float(np.percentile(rule_rets, 25)),
        "p75_return": float(np.percentile(rule_rets, 75)),
        "avg_hold_bars": avg_hold,
        "median_mae": float(pool["mae"].median()),
        "p5_mae": float(np.nanpercentile(pool["mae"], 5)),
        "worst_5pct_mean_return": float(rule_rets[rule_rets <= worst5].mean()),
        "longest_losing_streak": _longest_losing_streak(pool),
        "baseline_median": float(np.median(base_rets)),
        "excess_median": excess,
        "excess_ci_low": ci_lo,
        "excess_ci_high": ci_hi,
        "excess_significant": bool(ci_lo > 0 or ci_hi < 0),
    }

    result = {
        "rule_name": rule_name,
        "exit_spec": exit_spec,
        "pooled": pooled,
        "per_ticker": per_ticker,
        "trades": pool,
        "baseline": {"median": float(np.median(base_rets)), "avg_hold_bars": avg_hold, "n": len(base_rets)},
        "sample": f"{start}~{end or 'today'}",
    }

    if make_plots:
        from reports import plots

        result["plots"] = {
            "return_hist": plots.return_hist(pool, rule_name, n_eff=neff["n_eff"], baseline_median=pooled["baseline_median"], title=rule_name),
            "mae_hist": plots.mae_hist(pool, rule_name, n_eff=neff["n_eff"], title=rule_name),
            "correlation_heatmap": plots.correlation_heatmap(rets_df, rule_name, avg_corr=neff["rho_bar"]),
        }
    return result


def earnings_condition(kind: str = "post_beat", window: int = 20) -> Callable[[str, pd.Series], pd.Series]:
    """构造财报日历条件门（F3 的免费子集：把已验证的 PEAD/财报结构接成 A 的进出场条件）。

    kind:
      post_beat        → 上次超预期 且 处于财报后 window 天内（顺 PEAD 漂移）
      post_earnings    → 处于财报后 window 天内（不限超预期方向）
      pre_earnings     → 处于下次财报前 window 天内
      away_from_earnings → 距下次财报 > window 天（避开财报不确定性）
    返回 (ticker, price) -> 布尔 Series，供 evaluate_rule 的 condition_fn 使用。
    """
    from data import loader
    from factors import fundamentals as fu

    _cache: dict[str, pd.DataFrame] = {}

    def _cond(ticker: str, price: pd.Series) -> pd.Series:
        if ticker not in _cache:
            _cache[ticker] = loader.load_earnings_dates(ticker, limit=80)
        ed = _cache[ticker]
        idx = price.index
        if kind == "post_beat":
            return (fu.post_earnings_window(idx, ed, window).fillna(False).astype(bool)
                    & fu.last_beat(idx, ed).fillna(False).astype(bool))
        if kind == "post_earnings":
            return fu.post_earnings_window(idx, ed, window).fillna(False).astype(bool)
        if kind == "pre_earnings":
            return fu.pre_earnings_window(idx, ed, window).fillna(False).astype(bool)
        if kind == "away_from_earnings":
            d = fu.days_to_next_earnings(idx, ed)
            return (d > window).where(d.notna()).fillna(False).astype(bool)
        raise ValueError(f"未知 earnings_condition kind '{kind}'")

    return _cond


def regime_condition(kind: str = "up_trend") -> Callable[[str, pd.Series], pd.Series]:
    """构造市场状态条件门（U2：免费可观测状态作 A 进出场条件）。

    kind: up_trend / down_trend / low_vol / high_vol / in_drawdown / near_high
    返回 (ticker, price) -> 布尔 Series。
    """
    from regime import observables as ob

    def _cond(ticker: str, price: pd.Series) -> pd.Series:
        idx = price.index
        if kind in ("up_trend", "down_trend"):
            s = ob.trend_state(price)
        elif kind in ("low_vol", "high_vol", "mid_vol"):
            s = ob.vol_state(price)
        elif kind in ("in_drawdown", "near_high"):
            s = ob.drawdown_state(price)
        else:
            raise ValueError(f"未知 regime_condition kind '{kind}'")
        return (s == kind).reindex(idx).fillna(False).astype(bool)

    return _cond


def combine_conditions(*conds: Callable) -> Callable[[str, pd.Series], pd.Series]:
    """把多个条件门 AND 起来（如 财报后超预期 且 上升趋势）。"""
    conds = [c for c in conds if c is not None]

    def _cond(ticker: str, price: pd.Series) -> pd.Series:
        out = pd.Series(True, index=price.index)
        for c in conds:
            out = out & c(ticker, price).reindex(price.index).fillna(False).astype(bool)
        return out

    return _cond


def format_rule_verdict(result: dict) -> str:
    """按规格书强制模板渲染池化结论。"""
    p = result["pooled"]
    strength = "弱" if abs(p["excess_median"]) < 0.02 else ("中等" if abs(p["excess_median"]) < 0.05 else "较强")
    return (
        f"入场=[{result['rule_name']}] × 出场=[{result['exit_spec']}]："
        f"合并 {p['k_tickers']} 票共 {p['n_trades']} 笔（N_eff≈{p['n_eff']:.1f}），"
        f"胜率 {p['win_rate']:.0%}，每笔收益中位 {p['median_return']:+.1%}、5 分位 {p['p5_return']:+.1%}，"
        f"MAE 中位 {p['median_mae']:+.1%}，相对随机基准超额约 {p['excess_median']*100:+.1f} 点"
        f"（95% CI [{p['excess_ci_low']*100:+.1f}, {p['excess_ci_high']*100:+.1f}]）。"
        f"结论：{strength}优势{'（显著）' if p['excess_significant'] else '（不显著，非择时圣杯）'}。"
    )
