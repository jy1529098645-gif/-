"""建仓作战室（升级模块）——把「最近几个月该在哪建仓/补仓」做成**校准式**决策支持。

铁律(贯穿)：校准而非预测 / 永远对比无条件基准 / 反过拟合优先。
输出：条件价位带的**经验分布** + 盈亏比 + 期望值 + N(独立事件) + block bootstrap CI +
相对无条件基准的超额。未来事件只列**客观日程**与**历史反应分布**，不预测财报好坏。

五块：
  1. entry_zones        —— 距前高各回撤带 → 对应价位 + 历史远期收益分布/盈亏比/期望值/CI/超额。
  1b. best_entry_zone   —— 从各档中**按 CI 下界(保守地板超额)排名选出最佳入场区 + 历史常驻价**。
  2. earnings_reaction_stats / upcoming_events —— 未来日程(财报/期权到期) + 历史财报前后 drift。
  3. ladder_plan_backtest —— 阶梯式分批建仓(在各回撤带补仓)历史回测 vs lump/DCA，带 CI。
  4. （整合页在 app.py: page_cockpit）

关于"最佳入场点"(2026 升级，用户明确要求)：best_entry_zone **会**给一个历史常驻价，但它是
**校准式参考**不是预测：(a) 它是历史 reward/risk 最优档的**价位区间中值**，区间一并给出，
历史常驻价只是区内代表；(b) 永远附 N / CI / 置信分层；CI 跨 0 → 降级"低置信"，开口深档 → 历史常驻价
改报**触发价**且强制低置信；没有任何档正超额 → **不硬给点**，转防守("观望/轻仓")；(c) 个股
附幸存者偏差提醒。仍**不给**"上涨概率 73%"这类单一概率。历史常驻价是「若到达就分批行动」的区间参考，
非"会涨到/必反弹"。
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
from stats.deflated_sharpe import deflated_sharpe_ratio

_CFG = config.load_config()
_FEES = float(_CFG["costs"]["fees"])
_SLIP = float(_CFG["costs"]["slippage"])

# 公共 API（显式导出，便于核对部署完整性；缺任一即构建不完整）
__all__ = [
    "entry_zones", "format_zone_verdict", "best_entry_zone", "best_entry_across_horizons",
    "format_best_entry", "earnings_reaction_stats", "upcoming_events", "ladder_plan_backtest",
    "entry_miss_risk", "format_miss_risk", "entry_confluence",
]

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
    # per-观测 夏普(均值/标准差)——供 deflated Sharpe 多重检验折扣用
    sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    sharpe = float(np.mean(arr) / sd) if sd > 0 else 0.0
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
        "sharpe": sharpe,
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
        # 有效独立窗口：远期窗口重叠，n_days//horizon 才是真·独立样本数（防 CI 假性变窄）
        n_independent = max(1, n_days // max(1, horizon))
        # 历史常驻价：历史命中的**回撤深度中位**投影到当前前高（"当前价位"口径，反映档内聚集位置，
        #       必落在 [price_low, price_high] 内；比几何中点更贴历史分布）。
        hit_dd = dd[mask].dropna()
        if hit_dd.shape[0]:
            median_dd = float(hit_dd.median())
            anchor_median = cur_high * (1 + median_dd)
            yr = hit_dd.index.year.value_counts()
            regime_year_frac = float(yr.iloc[0] / hit_dd.shape[0])
        else:
            median_dd = float("nan"); anchor_median = float("nan"); regime_year_frac = float("nan")
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
            "anchor_median": anchor_median,
            "n_days": n_days,
            "n_events": n_events,
            "n_independent": n_independent,
            "regime_year_frac": regime_year_frac,
            "is_current": is_current,
            "enough": n_days >= _MIN_N,
        }
        if n_days >= _MIN_N:
            pm = pmin[mask].dropna().to_numpy()
            row.update(_zone_stats(sel.to_numpy(), pm, baseline_median, block_size, n_boot))
        else:
            for k in ("win_rate", "median", "p10", "p25", "p75", "p90", "avg_win", "avg_loss",
                      "median_mae", "expectancy", "winloss_ratio", "reward_risk", "sharpe",
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
# 块 1b：最佳入场区 + 历史常驻价（按历史 reward/risk 排名，诚实标注置信度）
# ===========================================================================
# 比 DEFAULT_BANDS 更深，覆盖"资本投降档"（深跌/急跌往往才是历史最优入场区）。
DEEP_BANDS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
_LOW_POWER_EVENTS = 8  # 独立事件 <8 → 低功率，"最佳"降级为"历史最高倾斜(低置信)"


def best_entry_zone(
    prices: pd.DataFrame | pd.Series,
    asset: str = "SPY",
    horizon: int = 252,
    bands: tuple[float, ...] = DEEP_BANDS,
    lookback_high: int = 252,
    n_boot: int = 400,
    single_name: bool = True,
) -> dict:
    """从条件价位带中**排名选出最佳入场区**，并给出区内历史常驻价（校准式，非预测）。

    排名口径：以每档中位远期收益的 **block bootstrap CI 下界** 为主排序键（保守地板超额），
    且「稳健」要求 ci_low > 基准中位（即**超额**的 CI 下界>0，强势股每档绝对收益 CI 都>0 会让
    ci_low>0 形同虚设）。但选出后必须再过三道**反过度自信**关：
      1. **多重检验折扣**：选档=「N 选 1」。用各合格档的 per-观测夏普做 deflated Sharpe，
         得 DSR(从 N 档里选最优后这档夏普为真的概率)。DSR<0.95 → 不得标"稳健"。
      2. **有效独立样本**：远期窗口重叠，真·独立样本 = n_days//horizon。有效N<5 → 强制低置信。
      3. **幸存者偏差 / regime 聚集**：个股深档(>30%回撤)永不"稳健"；命中日 >70% 挤在单一年份
         → 标 regime_clustered 降置信。
    历史常驻价 = 历史上落在该档的价格**中位**(anchor_median，比几何中点更贴历史分布)；开口档报触发价。
    无任何正超额档 → 防守裁决，绝不硬凑买点。single_name=True 附幸存者偏差提醒。
    """
    zones = entry_zones(prices, asset=asset, horizon=horizon, bands=bands,
                        lookback_high=lookback_high, n_boot=n_boot)
    cur_price = zones.attrs.get("current_price", float("nan"))
    cur_dd = zones.attrs.get("current_drawdown", float("nan"))
    elig = zones[(zones["enough"]) & (zones["excess_median"] > 0)].copy()

    base = {
        "asset": asset, "horizon": horizon, "current_price": cur_price,
        "current_drawdown": cur_dd, "current_high": zones.attrs.get("current_high", float("nan")),
        "sample_start": zones.attrs.get("sample_start"), "sample_end": zones.attrs.get("sample_end"),
        "zones": zones,
    }

    if elig.empty:
        return {**base, "has_zone": False, "tier": "防守",
                "verdict": ("当前**没有任何回撤档历史上跑赢无条件基准**——这只票/资产上"
                            "「越跌越买」没有历史优势(可能是趋势破坏/价值陷阱)。不给最佳入场点，"
                            "建议观望或仅极轻仓试探、严格止损。"),
                "caveats": ["无正超额档：硬给买点=骗自己。"]}

    base_med = float(zones.attrs.get("baseline_median", 0.0))
    # 选档原则（修正"按 max 超额→选到深档少样本噪声"）：
    #   ① 优先**样本充足**档(独立窗口≥5)，避免把 N独立=2~3 的深档当"最佳"；
    #   ② 一律按 **ci_low(保守下界)** 排名，而非点估计超额——宽CI的少样本深档 ci_low 自然低、被惩罚；
    #   ③ robust = ci_low>基准(超额显著)；否则取 ci_low 最高(最不坏)者，并标低置信。
    _N_FLOOR = 5
    adequate = elig[elig["n_independent"] >= _N_FLOOR]
    pool0 = adequate if not adequate.empty else elig
    robust = pool0[pool0["ci_low"] > base_med]
    if not robust.empty:
        pool, confident = robust, True
    else:
        pool, confident = pool0, False
    bounded = pool[pool["depth_hi"] < 1.0]
    use = bounded if not bounded.empty else pool
    best = use.loc[use["ci_low"].idxmax()]   # 始终按保守下界择优

    # —— 反关 1：多重检验折扣（从 len(elig) 个合格档里选最优）——
    sharpes = elig["sharpe"].dropna().to_numpy()
    n_trials = max(1, len(sharpes))
    sr_std = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
    n_ind = int(best.get("n_independent", max(1, int(best["n_days"]) // max(1, horizon))))
    try:
        dsr = deflated_sharpe_ratio(sr=float(best["sharpe"]),
                                    sr_trials_std=sr_std if sr_std > 0 else 1e-6,
                                    n_trials=n_trials, n_obs=max(2, n_ind))
    except Exception:  # noqa: BLE001
        dsr = float("nan")
    dsr_ok = bool(dsr == dsr and dsr >= 0.95)

    open_ended = bool(best["depth_hi"] >= 1.0) or float(best["price_low"]) <= 0
    # —— 反关 2：有效独立样本 ——
    low_power = (n_ind < 5) or (int(best["n_events"]) < _LOW_POWER_EVENTS)
    # —— 反关 3：幸存者偏差 / regime 聚集 ——
    deep_single = bool(single_name and float(best["depth_hi"]) > 0.30)
    ryf = float(best.get("regime_year_frac", float("nan")))
    regime_clustered = bool(ryf == ryf and ryf > 0.70)

    # 任一反关不过 → 强制降置信
    if open_ended or not dsr_ok or low_power or deep_single or regime_clustered:
        confident = False

    p_low, p_high = float(best["price_low"]), float(best["price_high"])
    anchor_median = float(best.get("anchor_median", float("nan")))
    if open_ended:
        anchor = p_high                  # 触发价：「跌到此价以下进入深跌档」
        price_band = [None, p_high]
    else:
        anchor = anchor_median if anchor_median == anchor_median else (p_low + p_high) / 2.0
        price_band = [p_low, p_high]
    dist_pct = (anchor / cur_price - 1.0) if cur_price == cur_price and cur_price else float("nan")

    if confident:
        tier = "稳健最佳入场区"
    elif open_ended:
        tier = "深跌触发区(开口·低置信)"
    elif not dsr_ok:
        tier = "最佳入场区(多重检验后存疑)"
    else:
        tier = "历史最高倾斜档(低置信)"

    caveats = []
    if not dsr_ok:
        caveats.append(f"从 {n_trials} 个回撤档里选最优，deflated Sharpe={dsr:.2f}(<0.95)——"
                       f"这个'最佳'有相当概率是多重检验下的运气。")
    if low_power:
        caveats.append(f"有效独立窗口仅 ~{n_ind} 个(远期重叠后)，长周期样本不足，CI 可能偏窄。")
    if regime_clustered:
        caveats.append(f"该档 {ryf:.0%} 的历史样本挤在单一年份，是特定 regime 而非稳定规律。")
    if open_ended:
        caveats.append("开口深跌档无价格下限，历史常驻价以触发价计、非区间中值。")
    if deep_single:
        caveats.append("个股深档(>30%)幸存者偏差最重，已封顶为低置信。")
    if single_name:
        caveats.append("个股深跌存在幸存者偏差：历史回升样本天然偏多，下一个深跌可能是不回头的价值陷阱。")
    caveats.append("历史常驻价=历史落在该档价格的中位，是「**若到达就分批行动**」的参考"
                   + ("触发价" if open_ended else "价") + "，不是预测/不是保证会到。")

    return {
        **base, "has_zone": True, "confident": confident, "low_power": low_power,
        "open_ended": open_ended, "dsr": dsr, "dsr_ok": dsr_ok, "n_trials": n_trials,
        "n_independent": n_ind, "regime_clustered": regime_clustered,
        "tier": tier, "zone_label": best["zone"],
        "price_band": price_band, "anchor_price": anchor, "anchor_distance": dist_pct,
        "median_fwd": float(best["median"]), "excess_median": float(best["excess_median"]),
        "reward_risk": float(best["reward_risk"]), "win_rate": float(best["win_rate"]),
        "expectancy": float(best["expectancy"]),
        "ci": [float(best["ci_low"]), float(best["ci_high"])],
        "n_events": int(best["n_events"]), "n_days": int(best["n_days"]),
        "is_current": bool(best["is_current"]), "caveats": caveats,
    }


def best_entry_across_horizons(
    prices: pd.DataFrame | pd.Series,
    asset: str = "SPY",
    horizons: tuple[int, ...] = (21, 63, 126, 252),
    single_name: bool = True,
    n_boot: int = 400,
) -> dict:
    """跨多个持有期跑 best_entry_zone，选出**置信度最高**的 (horizon, 入场区)。

    动机：长持有期(h=252)在单票上有效独立窗口常只剩 ~1 个 → 必为低置信，把短周期里
    本可"稳健"的入场点埋没。本函数扫 21/63/126/252，按下列优先级择优：
      ① 有入场区且 confident=True 优先；② DSR 高者优先；③ 超额的 CI 下界高者优先。
    返回**胜出的那个 bez**(含其 horizon) + `horizon_scan`(各周期摘要,供 UI 展示比较)。
    全周期都防守 → 返回防守裁决。
    """
    results = []
    for h in horizons:
        try:
            bez = best_entry_zone(prices, asset=asset, horizon=h, single_name=single_name, n_boot=n_boot)
        except Exception:  # noqa: BLE001
            continue
        results.append(bez)
    if not results:
        return {"asset": asset, "has_zone": False, "tier": "防守",
                "verdict": "各持有期均无法计算入场区（样本不足）。", "caveats": [], "horizon_scan": []}

    def _score(b):
        if not b.get("has_zone"):
            return (-1.0, -1.0, -1.0, -1.0)
        conf = 1.0 if b.get("confident") else 0.0
        # 样本充足优先：独立窗口≥5 才进"够样本"档——杜绝跨周期选到长周期 N独立=1 的开口深档噪声
        n_ind = b.get("n_independent", 0) or 0
        adequate = 1.0 if n_ind >= 5 else 0.0
        dsr = b.get("dsr", 0.0)
        dsr = dsr if dsr == dsr else 0.0
        ci_low = b.get("ci", [float("nan")])[0]
        ci_low = ci_low if ci_low == ci_low else -1.0
        return (conf, adequate, dsr, ci_low)

    ranked = sorted(results, key=_score, reverse=True)
    best = dict(ranked[0])
    best["horizon_scan"] = [
        {"horizon": r.get("horizon"), "tier": r.get("tier", "防守"),
         "has_zone": bool(r.get("has_zone")), "confident": bool(r.get("confident")),
         "dsr": r.get("dsr"), "anchor": r.get("anchor_price"),
         "excess": r.get("excess_median"), "n_independent": r.get("n_independent"),
         "zone_label": r.get("zone_label")}
        for r in sorted(results, key=lambda x: x.get("horizon", 0))
    ]
    return best


def format_best_entry(bez: dict) -> str:
    """把最佳入场区裁决渲染成一句话（含历史常驻价/区间/超额/盈亏比/N/CI/置信标注）。"""
    if not bez.get("has_zone"):
        return f"🛡️ {bez['verdict']}"
    h_m = bez["horizon"] / 21
    band = bez["price_band"]
    rr = bez["reward_risk"]
    rr_s = f"{rr:.2f}" if rr == rr else "—"
    ci = bez["ci"]
    dist = bez.get("anchor_distance", float("nan"))
    dist_s = (f"，距现价 {dist:+.1%}" if dist == dist else "")
    badge = {"稳健最佳入场区": "✅", "最佳入场区(样本偏少)": "🟡"}.get(bez["tier"], "🟠")
    if band[0] is None:  # 开口深跌档：历史常驻价以触发价计，区间为"≤ 触发价"
        anchor_label = "触发价"
        band_s = f"≤ {band[1]:.1f}{dist_s}"
    else:
        anchor_label = "历史常驻价"
        band_s = f"区间 {band[0]:.1f}–{band[1]:.1f}{dist_s}"
    head = (f"{badge} 最佳入场区[{bez['zone_label']}·持有约{h_m:.0f}个月]　"
            f"{anchor_label} ≈ **{bez['anchor_price']:.1f}**（{band_s}）")
    dsr = bez.get("dsr", float("nan"))
    dsr_s = f"、DSR {dsr:.2f}" if dsr == dsr else ""
    body = (f"｜历史 {h_m:.0f} 个月远期中位 {bez['median_fwd']*100:+.1f}%、"
            f"超无条件基准 {bez['excess_median']*100:+.1f} 点、盈亏比 {rr_s}、胜率 {bez['win_rate']:.0%}、"
            f"有效独立窗口≈{bez.get('n_independent','?')}(名义 N={bez['n_events']})、"
            f"CI[{ci[0]*100:+.1f}%,{ci[1]*100:+.1f}%]{dsr_s}｜{bez['tier']}")
    tail = "　⚠️ " + " ".join(bez.get("caveats", []))
    return head + body + tail


# ===========================================================================
# 块 1c：踏空风险（"没等到入场点"的机会成本）—— 校准而非预测
# ===========================================================================
def entry_miss_risk(price: pd.Series, bez: dict, dd_tol: float = 0.04,
                    lookback_high: int = 252) -> dict:
    """量化"死等推荐入场区却踏空"的风险：从今天这种回撤状态出发，历史上推荐档在 horizon 内
    **被触及的频率** vs **价格直接走高没回该档的频率**，以及没触及时的**平均踏空涨幅(机会成本)**。

    口径：条件门=历史上回撤(距 lookback_high 高点) 接近今日(±dd_tol) 的日子；
    "触及"=未来 horizon 日内最低价跌到入场带上沿(最先可买处)；无前视(条件只用当日及之前)。
    返回 {available, reach_prob, miss_prob, missed_oppcost(中位), fwd_median, gap, n, horizon, already_in}。
    """
    if not bez or not bez.get("has_zone"):
        return {"available": False, "reason": "no_zone"}
    px = price.dropna()
    cur = float(px.iloc[-1])
    band = bez.get("price_band") or [None, None]
    edge = band[1] if band[1] is not None else bez.get("anchor_price")  # 入场带上沿=最先触及处
    if edge is None or edge != edge or cur <= 0:
        return {"available": False, "reason": "no_edge"}
    gap = edge / cur - 1.0                      # 需从现价再跌多少才到带(通常<0)
    if gap >= -0.005:                           # 现价已在/低于入场带 → 无需等、无踏空风险
        return {"available": True, "already_in": True, "gap": gap,
                "horizon": int(bez.get("horizon", 63))}
    h = int(bez.get("horizon", 63))
    high = _trailing_high(px, lookback_high)
    dd = (px / high - 1.0)
    dd_now = float(dd.iloc[-1])
    p = px.to_numpy()
    ddv = dd.to_numpy()
    mask = (ddv >= dd_now - dd_tol) & (ddv <= dd_now + dd_tol)
    reached, fwd_missed, fwd_all = [], [], []
    n_total = len(p)
    for i in np.where(mask)[0]:
        if i + h >= n_total or i + 1 > n_total - 1:
            continue
        win = p[i + 1:i + h + 1]
        path_min_ret = win.min() / p[i] - 1.0
        fwd = p[i + h] / p[i] - 1.0
        hit = bool(path_min_ret <= gap)        # 跌到入场带上沿
        reached.append(hit); fwd_all.append(fwd)
        if not hit:
            fwd_missed.append(fwd)
    n = len(reached)
    if n < 20:
        return {"available": False, "reason": "low_sample", "n": n}
    reach_prob = float(np.mean(reached))
    return {
        "available": True, "already_in": False, "gap": gap, "horizon": h, "dd_now": dd_now,
        "n": n, "reach_prob": reach_prob, "miss_prob": 1.0 - reach_prob,
        "missed_oppcost": float(np.median(fwd_missed)) if fwd_missed else float("nan"),
        "fwd_median": float(np.median(fwd_all)),
    }


def format_miss_risk(mr: dict) -> str | None:
    """把踏空风险量化成一句裁决 + Plan B（先建底仓/突破追小仓 的两头不踏空口径）。"""
    if not mr or not mr.get("available"):
        return None
    if mr.get("already_in"):
        return "✅ **踏空风险**：现价已在/低于推荐入场区——可直接分批，无需死等、无踏空风险。"
    rp, mp, oc = mr["reach_prob"], mr["miss_prob"], mr.get("missed_oppcost", float("nan"))
    hm = max(1, round(mr["horizon"] / 21))
    oc_s = f"{oc:+.0%}" if oc == oc else "—"
    n = mr["n"]
    long_h = mr["horizon"] >= 189            # ≥~9个月：长窗里"迟早会跌到"是常态，到达率天然偏高
    tail = "(N=%d·校准非预测%s%s)" % (
        n, "·样本偏少低置信" if n < 40 else "",
        "·长持有窗到达率天然偏高、机会成本含特定regime" if long_h else "")
    if mp >= 0.6:
        return (f"⚠️ **踏空风险高**：历史上从今天这状态，推荐档 ~{hm}个月内**仅 {rp:.0%} 情形会到达**；"
                f"{mp:.0%} 情形价格直接走高、踏空中位 {oc_s}。→ **别全押等深档：现在先建 1/3 底仓，"
                f"深档来了再加；价格突破/站上 MA50 也追小仓**。{tail}")
    if mp >= 0.4:
        return (f"🟡 **踏空风险中**：推荐档 ~{hm}个月内到达概率 {rp:.0%}；没到的 {mp:.0%} 情形踏空中位 {oc_s}。"
                f"→ **分两半：一半现在/突破确认就进，一半挂深档等**。{tail}")
    # 到达率高：短窗=性价比高值得等；长窗=别为等一年踏空，重点仍是先建底仓
    if long_h:
        return (f"🟡 **要等很久**：推荐档虽 ~{hm}个月内 {rp:.0%} 会到达，但那是**长达 {hm} 个月的等待**——"
                f"空仓干等的时间成本高。→ **先建 1/3~1/2 底仓在场，深档来了再加**。{tail}")
    return (f"🟢 **踏空风险低**：历史上从今天这状态，推荐档 ~{hm}个月内 {rp:.0%} 会到达，等的性价比较高；"
            f"仍建议留小底仓防小概率踏空。{tail}")


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


# ===========================================================================
# 块 1c：入场位技术共振（合理入场位 = 统计正边际 × 技术支撑共振）
# ===========================================================================
def _tech_supports(ohlcv: pd.DataFrame, lookback: int = 252) -> list[dict]:
    """当前价**附近/下方**的技术支撑位清单（合理入场位的'技术'依据）。

    收录：MA50/MA200、Volume Profile POC/价值区下沿、近20/60/120日前低、
    52周区间斐波那契回撤(38.2/50/61.8%)。每个支撑标 {label, price, kind}。
    只是客观技术位，非预测——用于和'统计正边际档'求共振。
    """
    from analysis.volume_profile import volume_profile
    px = ohlcv["close"].dropna()
    cur = float(px.iloc[-1])
    out: list[dict] = []

    def _add(label, lvl, kind):
        if lvl == lvl and lvl > 0:
            out.append({"label": label, "price": float(lvl), "kind": kind,
                        "dist_pct": float(lvl / cur - 1.0)})

    ma50 = px.rolling(50, min_periods=25).mean().iloc[-1]
    ma200 = px.rolling(200, min_periods=100).mean().iloc[-1]
    _add("MA50", ma50, "均线")
    _add("MA200", ma200, "均线")
    try:
        vp = volume_profile(ohlcv, lookback=lookback)
        _add("POC换手密集价", vp["poc"], "筹码")
        _add("价值区下沿", vp["value_area"][0], "筹码")
    except Exception:  # noqa: BLE001
        pass
    for w, nm in ((20, "近20日低"), (60, "近60日低"), (120, "近120日低")):
        _add(nm, px.rolling(w, min_periods=max(5, w // 2)).min().iloc[-1], "前低")
    seg = px.tail(lookback)
    hi, lo = float(seg.max()), float(seg.min())
    if hi > lo:
        for f, nm in ((0.382, "Fib38.2%"), (0.5, "Fib50%"), (0.618, "Fib61.8%")):
            _add(nm, hi - (hi - lo) * f, "斐波那契")
    return out


def entry_confluence(ohlcv: pd.DataFrame, asset: str = "SPY", best_entry: dict | None = None,
                     horizon: int = 252, tol: float = 0.025, lookback: int = 252,
                     warn_red: bool = False, warn_amber: bool = False, warn_label: str = "",
                     rsi_window: int = 14, vix_pctile: float | None = None) -> dict:
    """**合理入场位**：把'统计正边际档'(best_entry_zone) 与'技术支撑共振'融合。

    逻辑：① 取统计最佳入场区的锚定价/区间(回撤桶里 CI 下界最优、已过反过拟合关)；
         ② 找当前价附近/下方的技术支撑(均线/POC/前低/Fib)；
         ③ 数锚定价 ±tol 内有几个技术支撑 → **共振度**(confluence)；共振越多=该价位越'合理'(多重技术认同)；
         ④ **飞刀防护**：跌破200线且均线下行 → 标 falling_knife，建议等企稳别接；
         ⑤ 评级 = 统计置信 × 技术共振，给 强/中/弱-观望。
    返回 dict：anchor/band/confluence/confirms[]/grade/falling_knife/at_support_now/supports[]/note。
    校准非预测：入场区是'若到达就分批行动'的参考，共振高≠保证反弹。
    """
    px = ohlcv["close"].dropna()
    cur = float(px.iloc[-1])
    ma200_s = px.rolling(200, min_periods=100).mean()
    ma200 = float(ma200_s.iloc[-1]) if ma200_s.notna().iloc[-1] else float("nan")
    slope_dn = bool(ma200_s.notna().iloc[-1] and ma200_s.notna().iloc[-21]
                    and ma200_s.iloc[-1] < ma200_s.iloc[-21])
    falling_knife = bool(ma200 == ma200 and cur < ma200 and slope_dn)
    # 临近撤离线：价just在200线上方(撤离阈值附近)——趋势薄弱，与 exit_warning 的黄灯口径一致
    dist_ma200 = (cur / ma200 - 1.0) if (ma200 == ma200 and ma200) else float("nan")
    near_ma200 = bool(dist_ma200 == dist_ma200 and 0 <= dist_ma200 < 0.04)

    supports = _tech_supports(ohlcv, lookback=lookback)
    # 现价**下方**最近的技术支撑 = 可执行的"回踩分批区"（按距现价由近到远）
    supports_below = sorted([s for s in supports if s["dist_pct"] < -0.005],
                            key=lambda s: -s["dist_pct"])
    # 现价正落在支撑共振区？(≥2 技术位 ±tol 聚集) → 现在就能分批
    near_now = [s for s in supports if abs(s["price"] / cur - 1.0) <= tol]
    at_support_now = bool(len(near_now) >= 2)

    # —— 优质回踩（趋势 + 恐慌折价）：回测验证(scripts/vix_rsi_signals.py)的更准入场 ——
    # 趋势健康 × RSI回踩<40 × VIX分位>70% → 历史胜率/收益均高于裸"趋势健康"(两段样本外均胜)。
    from factors import signals as _sg
    _rsi_s = _sg.rsi(px, int(rsi_window))
    rsi_now = float(_rsi_s.iloc[-1]) if _rsi_s.notna().iloc[-1] else float("nan")
    trend_up = bool(ma200 == ma200 and cur > ma200)
    # 关键：fear_pullback 必须服从 regime——飞刀/离场红灯时强制为 False，否则前端会同时弹
    # "🔴暂不建仓"评级 和 "✨优质回踩·可加码"绿条，自相矛盾（市场宽度恶化触发的红灯尤其会撞上）。
    fear_pullback = bool(trend_up and not falling_knife and not warn_red
                         and rsi_now == rsi_now and rsi_now < 40
                         and vix_pctile is not None and vix_pctile == vix_pctile and vix_pctile > 0.70)

    # 统计'最佳档'降级为参考：单票常 N独立=1、且趋势股最佳档在高点(锚定价>现价)，是噪声非买点。
    if best_entry is None:
        best_entry = best_entry_zone(px, asset=asset, horizon=horizon)
    stat_ok = bool(best_entry.get("has_zone"))
    stat_conf = bool(best_entry.get("confident"))
    anchor = best_entry.get("anchor_price", float("nan")) if stat_ok else float("nan")
    band = best_entry.get("price_band", [None, None]) if stat_ok else [None, None]
    n_ind = best_entry.get("n_independent", None)
    zlabel = best_entry.get("zone_label", "")
    anchor_below = bool(anchor == anchor and anchor and anchor <= cur * 1.005)
    # 只有"真置信 + 锚定价在现价附近/下方 + 样本足"时，统计档才值得当参考；否则明确标噪声
    anchor_actionable = bool(stat_conf and anchor_below and (n_ind is None or n_ind >= 5))
    if not stat_ok:
        stat_note = "单票回撤桶无正超额档——不硬给统计买点。"
    elif anchor_actionable:
        stat_note = f"（统计参考：稳健档「{zlabel}」≈{anchor:.1f}，可作分批位之一）"
    else:
        _why = ("锚定价在现价上方(趋势股'最佳档'常在高点)" if (anchor == anchor and not anchor_below)
                else f"N独立={n_ind}样本太少" if (n_ind is not None and n_ind < 5) else "置信不足")
        stat_note = f"⚠️ 单票统计'最佳档「{zlabel}」'{_why}、属噪声，**别当买点**（仅列以示诚实）。"

    # 锚定价附近技术共振（保留给"统计可执行"时用；不再作为主评级依据）
    confirms = [s for s in supports if anchor == anchor and anchor and abs(s["price"] / anchor - 1.0) <= tol]
    confluence = len(confirms)

    # —— 主评级：先答"现在能不能碰" → 再答"在哪买"。统计档已降级。——
    # 红/黄灯区别(已回测验证)：红灯(破位/宽度恶化=确认趋势坏)→暂停建仓；黄灯(波动/临近线/高位拉伸)→
    # 远期收益并不变差、只是 MAE 略深 → **降为小仓/分批**而非全停(数据不支持暂停)。
    if falling_knife:
        grade, gtag = "观望（破位接飞刀）", "🔴"
        note = ("⚠️ 飞刀防护：价在200线下方且均线下行——历史上'接飞刀'胜率差。别在此抢反弹，"
                "等**站回200线 / 构筑双底**确认企稳，或只极轻仓试探+严止损。")
    elif warn_red:
        grade, gtag = "离场红灯·暂不建新仓", "🔴"
        note = (f"⚠️ 当前是**离场红灯**（{warn_label or '破位/宽度恶化'}）——确认趋势坏了，别建新仓。"
                "入场和离场同一 regime：等站回200线、宽度转健康再说（与下方撤离口径一致）。")
    elif fear_pullback:
        grade, gtag = "优质回踩（趋势+恐慌折价）·分批进", "🟢"
        note = (f"✅ **优质回踩**：趋势健康(>200线) + RSI回踩{rsi_now:.0f}(<40) + VIX处历史高分位(恐慌折价)。"
                "回测验证(含两段样本外)：这种'趋势内逢恐慌回踩'进场**历史胜率/收益均高于普通健康进场**"
                "(约+3~5pt胜率)——是相对更准的入场窗口，**在支撑处分批进**（恐慌期波动大，小步分批、别一次性梭哈）。")
    elif at_support_now:
        _names = "、".join(s["label"] for s in near_now)
        grade, gtag = "现价即在支撑共振区·可分批", "🟢"
        note = (f"现价正落在 **{len(near_now)} 个技术支撑**（{_names}）——可在此分批。"
                "提醒：技术共振**不提高远期收益**，只让进场后浮亏更浅、更拿得住；入场对长期收益影响本就小。")
    elif near_ma200:
        grade, gtag = "临近撤离线·只小仓试探", "🟡"
        note = (f"现价just在200线上方（距撤离线 {dist_ma200:+.1%}），趋势薄弱——只小仓试探，"
                "**跌破200线即按撤离口径**，别重仓押反弹。")
    elif supports_below:
        _b = "、".join(f"{s['label']} {s['price']:.1f}({s['dist_pct']:+.1%})" for s in supports_below[:3])
        grade, gtag = "趋势健康·等回踩支撑分批", "🟡"
        note = (f"趋势健康，但现价不在支撑上。**可执行回踩分批区（现价下方最近支撑）**：{_b}。"
                "回踩到这些位置再分批、进场更不易被套；想追则小仓。别为'完美买点'空等——入场对长期收益影响小。")
    else:
        grade, gtag = "趋势健康但离支撑远·小仓或等回踩", "⚪"
        note = ("趋势健康，但现价下方近处无明显技术支撑——小仓参与或耐心等回踩，别空等'完美买点'。")

    # 黄灯(非红/非飞刀)：不改买/不买结论，只附"小仓"提示(回测：黄灯远期不更差、仅MAE略深)
    if warn_amber and not (falling_knife or warn_red):
        note += (f"　⚠️ 当前有离场黄灯（{warn_label or '波动/高位/临近线'}）——回测显示远期收益并不更差、"
                 "只是进场后浮亏略深，故**建议小仓/分批降暴露**，不必完全暂停。")

    return {
        "asset": asset, "current_price": cur, "ma200": ma200, "dist_ma200": dist_ma200,
        "falling_knife": falling_knife, "near_ma200": near_ma200,
        "rsi": rsi_now, "vix_pctile": vix_pctile, "fear_pullback": fear_pullback,
        "warn_red": bool(warn_red), "warn_amber": bool(warn_amber),
        "stat_confident": stat_conf, "stat_has_zone": stat_ok, "anchor_actionable": anchor_actionable,
        "anchor": float(anchor) if anchor == anchor else None, "band": band, "stat_note": stat_note,
        "confluence": confluence, "confirms": confirms, "supports": supports,
        "supports_below": supports_below, "at_support_now": at_support_now, "supports_near_now": near_now,
        "grade": grade, "grade_tag": gtag, "note": note,
        "tier": best_entry.get("tier", ""), "stat_excess": best_entry.get("excess_median", float("nan")),
    }
