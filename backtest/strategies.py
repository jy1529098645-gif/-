"""回测策略（vectorbt，含 fees + slippage）。

铁律：所有回测结果走 stats/ 出**置信区间**，不报点估计。

三个建仓策略对比（同一预算 C、同一持有窗口，差别只在「怎么把钱投进去」）：
- lump_sum   ：day0 一次性全投。
- dca        ：在 deploy 窗口内等额分 n 批定投。
- average_down：先投一批，价格每跌 dip 就补一批；deploy 窗口末把剩余一次性投完。
  未投入的现金留在组合里（0 收益）——这正是三者的真实代价/收益差异。

对比口径：滚动多个起点，每个窗口算各策略的「资本回报 = 期末价值/预算 − 1」，
汇总成经验分布并用 block bootstrap 给置信区间；并给 dca/avg_down 相对 lump_sum 的配对差值 CI。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from stats.bootstrap import block_bootstrap_ci

_CFG = config.load_config()
_FEES = float(_CFG["costs"]["fees"])
_SLIP = float(_CFG["costs"]["slippage"])

STRATEGIES = ["lump_sum", "dca", "average_down"]


# ---------------------------------------------------------------------------
# 单窗口内的「投入计划」（value 口径，单位=现金）
# ---------------------------------------------------------------------------
def _schedule_lump_sum(close: pd.Series, budget: float, **_) -> pd.Series:
    s = pd.Series(np.nan, index=close.index)
    s.iloc[0] = budget
    return s


def _schedule_dca(close: pd.Series, budget: float, deploy: int, n_dca: int, **_) -> pd.Series:
    s = pd.Series(np.nan, index=close.index)
    tranche = budget / n_dca
    step = max(1, deploy // n_dca)
    for i in range(n_dca):
        pos = min(i * step, len(close) - 1)
        s.iloc[pos] = (s.iloc[pos] if pd.notna(s.iloc[pos]) else 0.0) + tranche
    return s


def _schedule_average_down(
    close: pd.Series, budget: float, deploy: int, n_dca: int, dip: float, **_
) -> pd.Series:
    s = pd.Series(np.nan, index=close.index)
    tranche = budget / n_dca
    remaining = n_dca
    px = close.to_numpy(dtype=float)
    last_buy = px[0]
    s.iloc[0] = tranche
    remaining -= 1
    deploy_end = min(deploy, len(close) - 1)
    for t in range(1, deploy_end + 1):
        if remaining <= 0:
            break
        if px[t] <= last_buy * (1 - dip):
            s.iloc[t] = (s.iloc[t] if pd.notna(s.iloc[t]) else 0.0) + tranche
            last_buy = px[t]
            remaining -= 1
    if remaining > 0:  # 窗口末把没投完的一次性投完，保证总投入 = budget
        pos = deploy_end
        s.iloc[pos] = (s.iloc[pos] if pd.notna(s.iloc[pos]) else 0.0) + tranche * remaining
    return s


_BUILDERS = {
    "lump_sum": _schedule_lump_sum,
    "dca": _schedule_dca,
    "average_down": _schedule_average_down,
}


def _simulate_window(
    close: pd.Series, budget: float, deploy: int, n_dca: int, dip: float
) -> dict[str, float]:
    """对单个价格窗口模拟三策略，返回各自的资本回报（期末价值/budget − 1）。"""
    import vectorbt as vbt

    close = close.dropna()
    if close.shape[0] < 5:
        return {k: np.nan for k in STRATEGIES}

    size = pd.DataFrame(
        {
            k: _BUILDERS[k](close, budget=budget, deploy=deploy, n_dca=n_dca, dip=dip)
            for k in STRATEGIES
        }
    )
    close_df = pd.concat({k: close for k in STRATEGIES}, axis=1)

    pf = vbt.Portfolio.from_orders(
        close_df,
        size=size,
        size_type="value",
        direction="longonly",
        fees=_FEES,
        slippage=_SLIP,
        init_cash=budget,
        freq="1D",
    )
    final_value = pf.value().iloc[-1]
    return {k: float(final_value[k] / budget - 1.0) for k in STRATEGIES}


def _summary(outcomes: np.ndarray, block_size: int, n_boot: int) -> dict:
    """一组资本回报的分布摘要 + block bootstrap 中位数 CI。"""
    o = outcomes[~np.isnan(outcomes)]
    point, lo, hi = block_bootstrap_ci(o, np.median, block_size=min(block_size, max(1, len(o) // 5)), n=n_boot)
    return {
        "n_windows": int(len(o)),
        "median": float(np.median(o)),
        "mean": float(np.mean(o)),
        "p10": float(np.percentile(o, 10)),
        "p90": float(np.percentile(o, 90)),
        "win_rate_vs0": float((o > 0).mean()),
        "median_ci_low": lo,
        "median_ci_high": hi,
    }


def compare_entry_strategies(
    prices: pd.DataFrame | pd.Series,
    asset: str = "SPY",
    budget: float = 10000.0,
    hold: int = 504,
    deploy: int = 252,
    n_dca: int = 12,
    dip: float = 0.05,
    start_step: int = 63,
    n_boot: int = 1000,
) -> dict:
    """滚动多窗口对比 lump_sum / dca / average_down，结果带置信区间。

    返回 dict：
      - 'per_strategy'：各策略资本回报分布摘要 + 中位数 CI
      - 'vs_lump_sum'：dca/avg_down 相对 lump_sum 的配对差值（中位差 + block bootstrap CI）
      - 'window_returns'：每窗口每策略的资本回报 DataFrame（原料）
      - 'note'：样本与口径说明
    """
    price = (prices[asset] if isinstance(prices, pd.DataFrame) else prices).dropna()
    n = price.shape[0]
    starts = list(range(0, n - hold, start_step))
    if not starts:
        raise ValueError(f"价格长度 {n} 不足以容纳 hold={hold}")

    rows = []
    for st in starts:
        window = price.iloc[st : st + hold]
        out = _simulate_window(window, budget, deploy, n_dca, dip)
        out["start_date"] = price.index[st]
        rows.append(out)
    wr = pd.DataFrame(rows).set_index("start_date")

    # 重叠窗口 → 块至少覆盖一个 hold 跨越的窗口数
    block = max(2, hold // start_step)

    per_strategy = {k: _summary(wr[k].to_numpy(), block, n_boot) for k in STRATEGIES}

    vs_lump = {}
    base = wr["lump_sum"].to_numpy()
    for k in ("dca", "average_down"):
        diff = (wr[k].to_numpy() - base)
        diff = diff[~np.isnan(diff)]
        point, lo, hi = block_bootstrap_ci(
            diff, np.median, block_size=min(block, max(1, len(diff) // 5)), n=n_boot
        )
        vs_lump[k] = {
            "median_diff": float(np.median(diff)),
            "ci_low": lo,
            "ci_high": hi,
            "beats_lump_rate": float((diff > 0).mean()),
            "significant": bool(lo > 0 or hi < 0),  # CI 不跨 0 才算显著
        }

    note = (
        f"标的 {asset}，预算 {budget:.0f}，持有 {hold} 交易日、deploy {deploy} 日 "
        f"({n_dca} 批，average_down 跌 {dip:.0%} 补一批)，"
        f"{len(starts)} 个滚动起点（步长 {start_step}）。"
        f"样本期 {price.index[0].date()}~{price.index[-1].date()}。"
        "结论以中位数 + 95% block bootstrap CI 表示，不报点估计。"
    )
    return {"per_strategy": per_strategy, "vs_lump_sum": vs_lump, "window_returns": wr, "note": note}


# ---------------------------------------------------------------------------
# 因子分位组合回测（基础版）
# ---------------------------------------------------------------------------
def factor_quantile_backtest(
    factor_values: pd.DataFrame,
    prices: pd.DataFrame,
    quantiles: int = 5,
    rebalance: int = 21,
    long_short: bool = True,
    n_boot: int = 1000,
) -> dict:
    """基于因子分位的纯多/多空组合回测：每 rebalance 日取最高分位（多空则减最低分位），等权。

    结果给年化收益、夏普及其 block bootstrap CI（不报点估计）。
    """
    import vectorbt as vbt

    fac = factor_values.reindex(prices.index)
    rb = prices.index[::rebalance]

    # 目标权重面板：在调仓日按分位赋权，其余日前向填充
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for dt in rb:
        row = fac.loc[dt].dropna()
        if row.shape[0] < quantiles:
            continue
        ranks = row.rank(pct=True)
        top = ranks[ranks > 1 - 1 / quantiles].index
        weights.loc[dt, top] = 1.0 / max(1, len(top))
        if long_short:
            bot = ranks[ranks <= 1 / quantiles].index
            weights.loc[dt, bot] = -1.0 / max(1, len(bot))
    weights = weights.replace(0.0, np.nan).ffill().fillna(0.0)

    pf = vbt.Portfolio.from_orders(
        prices,
        size=weights,
        size_type="targetpercent",
        direction="both" if long_short else "longonly",
        fees=_FEES,
        slippage=_SLIP,
        cash_sharing=True,
        group_by=True,
        init_cash=10000.0,
        freq="1D",
    )
    rets = pf.returns().to_numpy().ravel()
    rets = rets[~np.isnan(rets)]
    from stats.bootstrap import sharpe

    point, lo, hi = block_bootstrap_ci(rets, sharpe, block_size=21, n=n_boot)
    return {
        "sharpe": point,
        "sharpe_ci_low": lo,
        "sharpe_ci_high": hi,
        "ann_return": float(np.nanmean(rets) * 252),
        "n_days": int(len(rets)),
        "note": "夏普带 block bootstrap 95% CI；CI 跨 0 则不显著。",
    }


# ---------------------------------------------------------------------------
# 策略 vs 持有 绩效对照（年化/波动/夏普/最大回撤）—— 单票一键
# ---------------------------------------------------------------------------
def _perf(value: "pd.Series") -> dict:
    """从净值序列算 CAGR / 年化波动 / 夏普(rf=0) / 最大回撤。"""
    v = value.dropna()
    n = len(v)
    if n < 60:
        return {"cagr": float("nan"), "vol": float("nan"), "sharpe": float("nan"), "maxdd": float("nan")}
    cagr = (v.iloc[-1] / v.iloc[0]) ** (252 / n) - 1
    ret = v.pct_change().dropna()
    vol = float(ret.std() * np.sqrt(252))
    sharpe = float(cagr / vol) if vol > 0 else float("nan")
    maxdd = float((v / v.cummax() - 1).min())
    return {"cagr": float(cagr), "vol": vol, "sharpe": sharpe, "maxdd": maxdd}


def strategy_vs_hold(ticker: str, start: str = "2014-01-01", end: str | None = None,
                     dip_lookback: int = 20, dip_pct: float = 0.15,
                     trailing: float = 0.30, time_stop: int = 504) -> dict:
    """工具定稿策略(深跌买入×让利润奔跑) vs 闭眼持有 的年化绩效对照。

    返回 {ticker, strategy{cagr,vol,sharpe,maxdd,in_market}, hold{...}, equity(DataFrame 归一净值)}。
    诚实口径：策略大部分时间空仓(有现金拖累)，对照同期买入持有。非预测。
    """
    from data import loader
    from factors import signals as sg
    from backtest import exits as ex

    p = loader.load_prices([ticker], start, end)[ticker].dropna()
    entries = sg.dip_from_high(p, lookback=dip_lookback, pct=dip_pct)
    pf = ex.run_trades(p, entries, {"trailing_stop": trailing, "time_stop": int(time_stop)}, init_cash=10000.0)
    sval = pf.value().dropna()
    try:
        in_market = float((pf.asset_value() > 0).mean())
    except Exception:  # noqa: BLE001
        in_market = float("nan")

    strat = _perf(sval); strat["in_market"] = in_market
    hold = _perf(p)
    equity = pd.DataFrame({
        "策略": sval / sval.iloc[0],
        "持有": (p / p.iloc[0]).reindex(sval.index),
    })
    return {"ticker": ticker, "strategy": strat, "hold": hold, "equity": equity,
            "sample": f"{p.index[0].date()}~{p.index[-1].date()}"}
