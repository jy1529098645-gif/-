"""建仓作战室（升级模块）——把「最近几个月该在哪建仓/补仓」做成**校准式**决策支持。

铁律(贯穿)：校准而非预测 / 永远对比无条件基准 / 反过拟合优先。
本模块**绝不**输出单一目标价或单一概率，只输出：条件价位带的**经验分布** + 盈亏比 + 期望值
+ N(独立事件) + block bootstrap CI + 相对无条件基准的超额。未来事件只列**客观日程**与
**历史反应分布**，不预测财报好坏（市场提前消化 = 量化财报前 drift 窗口，而非"已 price in"的判断)。

四块：
  1. entry_zones        —— 距前高各回撤带 → 对应价位 + 历史远期收益分布/盈亏比/期望值/CI/超额。
  2. earnings_reaction_stats / upcoming_events —— 未来日程(财报/期权到期) + 历史财报前后 drift。
  3. ladder_plan_backtest —— 阶梯式分批建仓(在各回撤带补仓)历史回测 vs lump/DCA，带 CI。
  4. （整合页在 app.py: page_cockpit）

DO NOT：不给"最佳买点 $X / 上涨概率 73%"；价位带是**区间 + 分布**，不是点。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from regime.conditional_returns import (
    _count_independent_events,
    _forward_return,
    _path_min_return,
)
from stats.bootstrap import block_bootstrap_ci

_CFG = config.load_config()
_FEES = float(_CFG["costs"]["fees"])
_SLIP = float(_CFG["costs"]["slippage"])

# 默认回撤带阈值（距前高的深度），相邻两档构成一个价位带；最后一段为 ">最深档"。
DEFAULT_BANDS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
_MIN_N = 20  # 单带样本(天)下限，低于此不报分布，只保留价位（诚实标注样本不足）


# ===========================================================================
# 块 1：条件价位带 + 盈亏比 + 期望值
# ===========================================================================
def _trailing_high(price: pd.Series, lookback: int = 252) -> pd.Series:
    """滚动 lookback 日的前高（默认约 1 年，贴合"最近几个月"语义）。"""
    return price.rolling(lookback, min_periods=lookback // 2).max()


def _zone_stats(arr: np.ndarray, pmin: np.ndarray, baseline_median: float,
                block_size: int, n_boot: int) -> dict:
    """单价位带的分布 + 盈亏比 + 期望值 + 中位数 block bootstrap CI。"""
    med = float(np.median(arr))
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    p_win = float(len(wins) / len(arr))
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0  # ≤0
    expectancy = p_win * avg_win + (1 - p_win) * avg_loss
    winloss_ratio = float(avg_win / abs(avg_loss)) if avg_loss < 0 else float("inf")
    med_mae = float(np.median(pmin)) if pmin.size else float("nan")
    # 盈亏比(决策口径)：中位远期收益 / 中位途中浮亏(要忍受的回撤) —— 校准"赔率"
    reward_risk = float(med / abs(med_mae)) if med_mae and med_mae < 0 else float("nan")
    eff_block = max(1, min(block_size, len(arr) // 5))
    try:
        _, ci_lo, ci_hi = block_bootstrap_ci(arr, np.median, block_size=eff_block, n=n_boot)
    except Exception:  # noqa: BLE001
        ci_lo = ci_hi = float("nan")
    return {
        "win_rate": p_win,
        "median": med,
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "median_mae": med_mae,
        "expectancy": expectancy,
        "winloss_ratio": winloss_ratio,
        "reward_risk": reward_risk,
        "baseline_median": baseline_median,
        "excess_median": med - baseline_median,
        "ci_low": ci_lo,
        "ci_high": ci_hi,
    }


def entry_zones(
    prices: pd.DataFrame | pd.Series,
    asset: str = "SPY",
    horizon: int = 252,
    bands: tuple[float, ...] = DEFAULT_BANDS,
    lookback_high: int = 252,
    n_boot: int = 400,
) -> pd.DataFrame:
    """距前高各回撤价位带 → 对应价位 + 历史 horizon 日远期收益分布/盈亏比/期望值/CI/超额。

    每一行 = 一个价位带 [depth_lo, depth_hi)，含：
      price_high/price_low(该带价位区间，按**当前前高**折算)、n_events(独立事件)、n_days、
      win_rate、median/p10..p90、avg_win/avg_loss、median_mae、expectancy(期望值)、
      winloss_ratio(均盈/均亏)、reward_risk(中位收益/中位浮亏)、excess_median(超无条件基准)、CI。
    并标记 is_current(今日所处带)。is_current 之下的带 = 尚未到达的"补仓候选"价位。
    """
    price = (prices[asset] if isinstance(prices, pd.DataFrame) else prices).dropna()
    high = _trailing_high(price, lookback_high)
    dd = (price / high - 1.0)  # ≤0

    fwd = _forward_return(price, horizon)
    pmin = _path_min_return(price, horizon)
    base = fwd.dropna()
    baseline_median = float(np.median(base.to_numpy())) if base.shape[0] else float("nan")
    block_cfg = int(_CFG["stats"]["bootstrap"]["block_size"])
    block_size = max(block_cfg, horizon)

    cur_high = float(high.dropna().iloc[-1])
    cur_dd = float(dd.dropna().iloc[-1])
    cur_price = float(price.iloc[-1])

    edges = list(bands) + [1.0]  # 末档兜底到 -100%
    rows = []
    for i in range(len(bands)):
        lo_d, hi_d = edges[i], edges[i + 1]
        # 价位带：dd ∈ (-hi_d, -lo_d]
        mask = (dd <= -lo_d) & (dd > -hi_d)
        sel = fwd[mask].dropna()
        n_days = int(sel.shape[0])
        n_events = _count_independent_events(mask.reindex(fwd.index))
        price_high = cur_high * (1 - lo_d)
        price_low = cur_high * (1 - hi_d) if hi_d < 1.0 else 0.0
        label = (f"距前高 {lo_d:.0%}–{hi_d:.0%}" if hi_d < 1.0 else f"距前高 >{lo_d:.0%}")
        is_current = bool(-hi_d < cur_dd <= -lo_d)
        row = {
            "zone": label,
            "depth_lo": lo_d,
            "depth_hi": hi_d,
            "price_high": price_high,
            "price_low": price_low,
            "n_days": n_days,
            "n_events": n_events,
            "is_current": is_current,
            "enough": n_days >= _MIN_N,
        }
        if n_days >= _MIN_N:
            pm = pmin[mask].dropna().to_numpy()
            row.update(_zone_stats(sel.to_numpy(), pm, baseline_median, block_size, n_boot))
        else:
            for k in ("win_rate", "median", "p10", "p25", "p75", "p90", "avg_win", "avg_loss",
                      "median_mae", "expectancy", "winloss_ratio", "reward_risk",
                      "excess_median", "ci_low", "ci_high"):
                row[k] = float("nan")
            row["baseline_median"] = baseline_median
        rows.append(row)

    out = pd.DataFrame(rows)
    out.attrs.update(
        asset=asset, horizon=horizon, current_price=cur_price, current_high=cur_high,
        current_drawdown=cur_dd, baseline_median=baseline_median,
        sample_start=str(price.index[0].date()), sample_end=str(price.index[-1].date()),
        disclaimer=("价位带是区间+经验分布，非目标价；盈亏比/期望值基于历史 N 个独立事件，"
                    "未来非平稳，仅校准预期，不预测点位。"),
    )
    return out


def format_zone_verdict(row: pd.Series, horizon: int) -> str:
    """把一行价位带渲染成强制措辞模板（分布+盈亏比+期望值+N+CI+基准超额）。"""
    if not bool(row.get("enough", False)):
        return (f"价位带[{row['zone']}]：样本不足(N_days={int(row['n_days'])})，不下分布结论；"
                f"价位约 {row['price_low']:.1f}–{row['price_high']:.1f}。")
    h_m = horizon / 21
    rr = row["reward_risk"]
    rr_s = f"{rr:.2f}" if rr == rr else "—"
    sig = "" if (row["ci_low"] != row["ci_low"]) else (
        "（CI 不跨 0，倾斜较稳）" if (row["ci_low"] > 0 or row["ci_high"] < 0) else "（CI 跨 0，证据弱）")
    verdict = "弱倾斜，非信号" if abs(row["excess_median"]) < 0.02 else "明显倾斜，仍需结合 N 与 CI"
    return (
        f"价位带[{row['zone']}]≈ {row['price_low']:.1f}–{row['price_high']:.1f}："
        f"{h_m:.0f} 个月远期收益中位 {row['median']*100:+.1f}%，10 分位 {row['p10']*100:+.1f}%，"
        f"盈亏比(中位收益/中位浮亏) {rr_s}，期望值 {row['expectancy']*100:+.1f}%，"
        f"相对无条件基准 {row['excess_median']*100:+.1f} 个点，基于 N≈{int(row['n_events'])} 个独立事件"
        f"（中位 95% CI [{row['ci_low']*100:+.1f}%, {row['ci_high']*100:+.1f}%]）{sig}。结论：{verdict}。"
    )


# ===========================================================================
# 块 2：未来日程事件 + 提前消化窗口（历史反应分布，不预测好坏）
# ===========================================================================
def earnings_reaction_stats(price: pd.Series, edates: pd.DataFrame,
                            pre: int = 10, post: int = 20) -> dict:
    """单票财报历史反应分布（客观）：财报日典型波动、财报前 drift(提前消化)、财报后漂移(分超预期)。

    全部是**历史经验分布**，不预测下次财报结果。"""
    from factors.fundamentals import _reported
    price = price.dropna()
    p = price.to_numpy(dtype=float)
    idx = price.index
    rep = _reported(edates)
    day_moves, pre_drifts, post_beat, post_miss = [], [], [], []
    for d, r in rep.iterrows():
        r0 = int(np.searchsorted(idx.values.astype("datetime64[ns]"),
                                 np.datetime64(d, "ns"), side="right"))  # 反应日(次个交易日)
        if r0 < pre + 1 or r0 + post >= len(p):
            continue
        day_moves.append(p[r0] / p[r0 - 1] - 1.0)
        pre_drifts.append(p[r0 - 1] / p[r0 - 1 - pre] - 1.0)   # 财报前 pre 日的 run-up(提前消化)
        pd_ = p[r0 + post] / p[r0] - 1.0                        # 财报后 post 日漂移
        (post_beat if r0 <= len(p) and r["Surprise(%)"] > 0 else post_miss).append(pd_)

    def _agg(x):
        a = np.asarray(x, dtype=float)
        if a.size == 0:
            return {"n": 0, "median": float("nan"), "p10": float("nan"), "p90": float("nan")}
        return {"n": int(a.size), "median": float(np.median(a)),
                "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90))}

    dm = np.abs(np.asarray(day_moves, dtype=float))
    return {
        "pre": pre, "post": post,
        "n_events": int(len(day_moves)),
        "day_abs_move": {"n": int(dm.size),
                         "median": float(np.median(dm)) if dm.size else float("nan"),
                         "p90": float(np.percentile(dm, 90)) if dm.size else float("nan")},
        "pre_drift": _agg(pre_drifts),
        "post_beat": _agg(post_beat),
        "post_miss": _agg(post_miss),
        "note": ("客观历史分布：不预测下次财报好坏。'财报前 drift'量化市场提前消化的幅度；"
                 "'财报后漂移'按上次是否超预期分组(PEAD)。"),
    }


def _third_friday(year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    # 第一个周五的日号，再加 14 天 = 第三个周五
    offset = (4 - first.dayofweek) % 7
    return first + pd.Timedelta(days=offset + 14)


def upcoming_events(price: pd.Series, edates: pd.DataFrame, n_opex: int = 3) -> pd.DataFrame:
    """未来客观日程：已排期但未公布的财报日 + 接下来 n_opex 个月度期权到期(第三个周五)。

    全部是**日历事实**，非预测。距今天数以最后一个交易日为基准(PIT)。"""
    today = price.dropna().index[-1]
    rows = []
    # 未来财报(Reported EPS 为空 = 未公布；或日期晚于今天)
    future = edates[(edates.index > today)]
    if "Reported EPS" in edates.columns:
        future = edates[(edates.index > today) | (edates["Reported EPS"].isna() & (edates.index >= today))]
    for d in sorted(set(future.index)):
        if d <= today:
            continue
        rows.append({"event": "财报(已排期)", "date": pd.Timestamp(d).date(),
                     "days_ahead": int((pd.Timestamp(d) - today).days)})
    # 月度期权到期
    y, m = today.year, today.month
    added = 0
    while added < n_opex:
        opex = _third_friday(y, m)
        if opex > today:
            rows.append({"event": "月度期权到期", "date": opex.date(),
                         "days_ahead": int((opex - today).days)})
            added += 1
        m += 1
        if m > 12:
            m = 1; y += 1
    out = pd.DataFrame(rows).sort_values("days_ahead").reset_index(drop=True)
    out.attrs["as_of"] = str(today.date())
    return out


# ===========================================================================
# 块 3：阶梯式建仓布局回测（在各回撤带补仓） vs lump / DCA
# ===========================================================================
def _ladder_schedule(close: pd.Series, budget: float, bands: tuple[float, ...]) -> pd.Series:
    """阶梯计划：day0 投 1 档，之后每次距窗口内前高首次跌破一档阈值再补 1 档；
    窗口末把剩余档一次性投完(保证总投入=budget，与 lump/DCA 同口径)。"""
    s = pd.Series(np.nan, index=close.index)
    n_tr = len(bands) + 1
    tr = budget / n_tr
    px = close.to_numpy(dtype=float)
    s.iloc[0] = tr
    remaining = n_tr - 1
    run_high = px[0]
    fired = [False] * len(bands)
    for t in range(1, len(px)):
        if px[t] > run_high:
            run_high = px[t]
        dd = px[t] / run_high - 1.0
        for i, b in enumerate(bands):
            if not fired[i] and dd <= -b and remaining > 0:
                fired[i] = True
                s.iloc[t] = (s.iloc[t] if pd.notna(s.iloc[t]) else 0.0) + tr
                remaining -= 1
    if remaining > 0:
        pos = len(px) - 1
        s.iloc[pos] = (s.iloc[pos] if pd.notna(s.iloc[pos]) else 0.0) + tr * remaining
    return s


def _simulate_ladder_window(close: pd.Series, budget: float, bands: tuple[float, ...],
                            deploy: int, n_dca: int) -> dict[str, float]:
    """单窗口模拟 lump_sum / dca / ladder：返回各策略的资本回报、建仓期最深浮亏、到位时间(天)。

    扁平 key：'{strat}'(资本回报)、'{strat}_mdd'(净值最大回撤)、'{strat}_deploy'(投满预算用了几天)。
    最深浮亏是"建仓+持有"全程净值相对自身峰值的最深回撤——这是分批真正想压低的痛感指标。"""
    import vectorbt as vbt
    from backtest.strategies import _schedule_dca, _schedule_lump_sum

    close = close.dropna()
    cols = ["lump_sum", "dca", "ladder"]
    out: dict[str, float] = {}
    if close.shape[0] < 5:
        for k in cols:
            out[k] = out[f"{k}_mdd"] = out[f"{k}_deploy"] = np.nan
        return out
    sched = {
        "lump_sum": _schedule_lump_sum(close, budget=budget),
        "dca": _schedule_dca(close, budget=budget, deploy=deploy, n_dca=n_dca),
        "ladder": _ladder_schedule(close, budget, bands),
    }
    size = pd.DataFrame(sched)
    close_df = pd.concat({k: close for k in cols}, axis=1)
    pf = vbt.Portfolio.from_orders(
        close_df, size=size, size_type="value", direction="longonly",
        fees=_FEES, slippage=_SLIP, init_cash=budget, freq="1D",
    )
    val = pf.value()
    fv = val.iloc[-1]
    for k in cols:
        out[k] = float(fv[k] / budget - 1.0)
        v = val[k].to_numpy(dtype=float)
        out[f"{k}_mdd"] = float(np.min(v / np.maximum.accumulate(v) - 1.0))
        # 到位时间：现金投入累计达预算 99% 的第一天序号
        invested = sched[k].fillna(0.0).cumsum().to_numpy()
        hit = np.where(invested >= budget * 0.99)[0]
        out[f"{k}_deploy"] = float(hit[0]) if hit.size else float(len(v))
    return out


def ladder_plan_backtest(
    prices: pd.DataFrame | pd.Series,
    asset: str = "SPY",
    budget: float = 10000.0,
    bands: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25),
    hold: int = 504,
    deploy: int = 252,
    n_dca: int = 12,
    start_step: int = 63,
    n_boot: int = 600,
) -> dict:
    """阶梯式分批建仓(各回撤带补仓)历史回测 vs lump_sum / DCA，滚动多窗口 + block bootstrap CI。

    返回与 backtest.strategies.compare_entry_strategies 同形：
      per_strategy / vs_lump_sum / window_returns / note（可直接喂前端 strategy_compare 图）。
    """
    from stats.bootstrap import block_bootstrap_ci

    price = (prices[asset] if isinstance(prices, pd.DataFrame) else prices).dropna()
    n = price.shape[0]
    starts = list(range(0, n - hold, start_step))
    if not starts:
        raise ValueError(f"价格长度 {n} 不足以容纳 hold={hold}")

    cols = ["lump_sum", "dca", "ladder"]
    rows = []
    for st in starts:
        out = _simulate_ladder_window(price.iloc[st:st + hold], budget, bands, deploy, n_dca)
        out["start_date"] = price.index[st]
        rows.append(out)
    wr = pd.DataFrame(rows).set_index("start_date")
    block = max(2, hold // start_step)

    def _summ(k):
        o = wr[k].to_numpy(); o = o[~np.isnan(o)]
        mdd = wr[f"{k}_mdd"].to_numpy(); mdd = mdd[~np.isnan(mdd)]
        dep = wr[f"{k}_deploy"].to_numpy(); dep = dep[~np.isnan(dep)]
        pt, lo, hi = block_bootstrap_ci(o, np.median, block_size=min(block, max(1, len(o) // 5)), n=n_boot)
        return {"n_windows": int(len(o)), "median": float(np.median(o)), "mean": float(np.mean(o)),
                "p10": float(np.percentile(o, 10)), "p90": float(np.percentile(o, 90)),
                "p5": float(np.percentile(o, 5)),
                "win_rate_vs0": float((o > 0).mean()), "median_ci_low": lo, "median_ci_high": hi,
                "mdd_median": float(np.median(mdd)) if mdd.size else float("nan"),
                "mdd_worst": float(np.percentile(mdd, 5)) if mdd.size else float("nan"),
                "deploy_days_median": float(np.median(dep)) if dep.size else float("nan")}

    per_strategy = {k: _summ(k) for k in cols}
    base = wr["lump_sum"].to_numpy()
    vs_lump = {}
    for k in ("dca", "ladder"):
        diff = (wr[k].to_numpy() - base)
        diff = diff[~np.isnan(diff)]
        pt, lo, hi = block_bootstrap_ci(diff, np.median, block_size=min(block, max(1, len(diff) // 5)), n=n_boot)
        # 回撤改善：阶梯/DCA 相对 lump 的最深浮亏差（正=更浅、更不痛）
        mdd_diff = (wr[f"{k}_mdd"].to_numpy() - wr["lump_sum_mdd"].to_numpy())
        mdd_diff = mdd_diff[~np.isnan(mdd_diff)]
        vs_lump[k] = {"median_diff": float(np.median(diff)), "ci_low": lo, "ci_high": hi,
                      "beats_lump_rate": float((diff > 0).mean()), "significant": bool(lo > 0 or hi < 0),
                      "mdd_improve_median": float(np.median(mdd_diff)) if mdd_diff.size else float("nan")}

    verdict = _ladder_verdict(asset, per_strategy, vs_lump)

    note = (f"标的 {asset}：阶梯在距前高 {'/'.join(f'{b:.0%}' for b in bands)} 各补一档"
            f"（共 {len(bands)+1} 档，未触发的窗口末补齐，总投入=预算 {budget:.0f}）；"
            f"持有 {hold} 交易日、DCA {n_dca} 批；{len(starts)} 个滚动起点(步长 {start_step})。"
            f"样本 {price.index[0].date()}~{price.index[-1].date()}。"
            "中位数 + 95% block bootstrap CI；CI 跨 0 不算显著。")
    return {"per_strategy": per_strategy, "vs_lump_sum": vs_lump, "window_returns": wr,
            "note": note, "verdict": verdict, "budget": budget, "hold": hold}


def _ladder_verdict(asset: str, per: dict, vs_lump: dict) -> str:
    """把数字翻成一句可执行的白话：这只票该一次性还是分批，代价/好处各多少。"""
    lump, lad = per["lump_sum"], per["ladder"]
    ret_cost = lad["median"] - lump["median"]          # 阶梯相对一次性的收益差(通常为负)
    mdd_gain = vs_lump["ladder"]["mdd_improve_median"]  # 浮亏改善(正=更浅)
    beat = vs_lump["ladder"]["beats_lump_rate"]
    if mdd_gain == mdd_gain and mdd_gain > 0.02 and ret_cost > -0.03:
        return (f"📌 {asset}：**值得分批**。越跌越补把建仓期最深浮亏从 {lump['mdd_median']:+.0%} "
                f"减到 {lad['mdd_median']:+.0%}（少痛 {mdd_gain:+.0%}），代价只是中位回报少 {abs(ret_cost):.0%}。"
                f"适合：怕买在高点、想拿得稳的人。")
    if ret_cost < -0.05:
        return (f"📌 {asset}：**倾向一次性**。该票长期向上，分批的现金拖累让中位回报少了 {abs(ret_cost):.0%}，"
                f"而浮亏只少 {max(mdd_gain,0):+.0%}——等跌的机会成本 > 抗跌收益。除非你强烈想压低短期回撤。")
    return (f"📌 {asset}：**两者接近**。分批 vs 一次性中位回报差 {ret_cost:+.0%}、最深浮亏差 {mdd_gain:+.0%}，"
            f"历史上分批跑赢一次性的概率 {beat:.0%}。按你的心理承受力选即可。")
