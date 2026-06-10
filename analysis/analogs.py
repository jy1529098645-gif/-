"""历史相似案例查看器——把"当前状态桶"的真实历史实例摊给用户看（可审计、透明）。

引擎给的是该状态的远期收益分布；本模块进一步列出**构成这个分布的具体历史日期**：
当时价格、往后 horizon 的实现收益、途中最大浮亏、距今多久。让结论可点开核对，而非黑箱。

铁律：是历史样本陈列，不是"现在更像哪一次"的预测；原因只能事后归类，实时不可解。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def historical_analogs(price: pd.Series, macro: pd.DataFrame, asset: str,
                       horizon: int = 63, top_k: int = 12,
                       use_valuation: bool = True, use_vol: bool = True) -> dict:
    """找出历史上与"今天"同状态的日期，列出其往后 horizon 的实现收益。

    状态 = 回撤/近高位（必用）× 估值三分位（可选）× 波动三分位（可选）。
    返回 {current: {...}, cases: DataFrame, summary: str}。"""
    from regime import conditional_returns as cr
    from regime import observables as ob

    price = price.dropna()
    state = cr.build_state(price, macro).reindex(price.index)
    dd_state = ob.drawdown_state(price)            # in_drawdown / near_high
    vol_st = ob.vol_state(price) if use_vol else None

    # 估值分位用**扩张窗口(point-in-time)**重算，杜绝前视：每天只用截至当天的历史定三分位
    # （build_state 的 valuation_tercile 是全样本分位，会把未来信息漏进历史标签）。
    val_tercile = None
    if use_valuation:
        vd = state["_val_dev"]
        q1 = vd.expanding(min_periods=252).quantile(1 / 3)
        q2 = vd.expanding(min_periods=252).quantile(2 / 3)
        val_tercile = pd.Series(
            np.where(vd <= q1, "low", np.where(vd >= q2, "high", "mid")),
            index=vd.index).where(vd.notna() & q1.notna())

    today = price.index[-1]
    cur_dd = str(dd_state.iloc[-1])
    cur_val = str(val_tercile.iloc[-1]) if (use_valuation and val_tercile is not None) else None
    cur_vol = str(vol_st.iloc[-1]) if (use_vol and vol_st is not None) else None

    mask = (dd_state == cur_dd)
    if use_valuation and cur_val is not None and cur_val != "nan":
        mask &= (val_tercile == cur_val)
    if use_vol and cur_vol is not None and cur_vol != "nan":
        mask &= (vol_st == cur_vol)

    p = price.to_numpy(dtype=float)
    idx = price.index
    rows = []
    hit_positions = np.where(mask.fillna(False).to_numpy())[0]
    for pos in hit_positions:
        tgt = pos + horizon
        if tgt >= len(p) or pos >= len(p) - 1:
            continue  # 未走完 horizon 的不算（含今天）
        fwd = p[tgt] / p[pos] - 1.0
        window = p[pos + 1:tgt + 1]
        mae = float(np.min(window) / p[pos] - 1.0) if window.size else np.nan
        rows.append({"date": idx[pos], "price": float(p[pos]),
                     "fwd_return": float(fwd), "max_drawdown": mae})

    cases = pd.DataFrame(rows)
    _VAL = {"low": "低", "mid": "中", "high": "高"}
    _VOL = {"low_vol": "低波", "mid_vol": "中波", "high_vol": "高波"}
    summary_state = "回撤中" if cur_dd == "in_drawdown" else "近高位"
    if cur_val and cur_val not in ("nan", "None"):
        summary_state += f" · 估值{_VAL.get(cur_val, cur_val)}"
    if cur_vol and cur_vol not in ("nan", "None"):
        summary_state += f" · {_VOL.get(cur_vol, cur_vol)}"

    if cases.empty:
        summary = f"今天状态：{summary_state}。历史上无足够同状态样本（或都未走完 {horizon} 日）。"
    else:
        med = cases["fwd_return"].median()
        winr = (cases["fwd_return"] > 0).mean()
        summary = (f"今天状态：{summary_state}。历史上 {len(cases)} 个同状态日，"
                   f"往后 {horizon} 日实现收益中位 {med:+.1%}、为正比例 {winr:.0%}、"
                   f"途中浮亏中位 {cases['max_drawdown'].median():+.1%}。下表为最近的实例（可核对）。")

    # 取最近 top_k（最贴近当下情形），按日期倒序
    recent = cases.sort_values("date", ascending=False).head(top_k) if not cases.empty else cases
    return {"current": {"drawdown_state": cur_dd, "valuation": cur_val, "vol": cur_vol,
                        "date": str(today.date())},
            "cases": recent, "all_n": int(len(cases)), "summary": summary}


# ---------------------------------------------------------------------------
# 趋势全程分布：像今天这状态，历史后来"再跌多深 / 反弹多高 / 见底多久 / 回到前高多久"
# 铁律：是历史类比的**分布**(带 N/p10-p50-p90)，不是"会进牛/熊市"的预测、不是点位目标。
# ---------------------------------------------------------------------------
def regime_path_distribution(price: pd.Series, window: int = 504, dd_tol: float = 0.05,
                             lookback_high: int = 252, n_min: int = 30) -> dict:
    """从今日回撤深度(距 lookback_high 高点)出发，取历史同深度的日子，统计其后 window 日的
    **全程路径分布**：进一步最大回撤(相对当日)、最大反弹、到谷底天数、回到当时前高的比例与天数。

    返回 {available, n, window, dd_now, further_dd{p10,p50,p90}, runup{p10,p50,p90},
          further_le10(再跌>10%频率), t_trough_med_d, recovery_rate, t_recovery_med_d,
          price_*{现价×(1+分位) 的价位}}。无前视：条件只用当日及之前。"""
    px = price.dropna()
    if len(px) < lookback_high + 60:
        return {"available": False, "n": 0}
    cur = float(px.iloc[-1])
    high = px.rolling(lookback_high, min_periods=lookback_high // 2).max()
    dd = (px / high - 1.0)
    dd_now = float(dd.iloc[-1])
    p = px.to_numpy(dtype=float); hiv = high.to_numpy(dtype=float); ddv = dd.to_numpy()
    mask = (ddv >= dd_now - dd_tol) & (ddv <= dd_now + dd_tol)
    fdd, runup, t_tr, rec_days, rec_flag = [], [], [], [], []
    n_all = len(p)
    for i in np.where(mask)[0]:
        end = min(i + window, n_all - 1)
        if end <= i + 5:
            continue
        win = p[i + 1:end + 1]
        if win.size == 0:
            continue
        j = int(win.argmin())
        fdd.append(win[j] / p[i] - 1.0)            # 进一步最大回撤(相对当日，≤0；正=没再跌破)
        runup.append(win.max() / p[i] - 1.0)       # 全程最大反弹(相对当日)
        t_tr.append(j + 1)                          # 到谷底天数
        peak = hiv[i]                               # 当时的前高(回撤参照)
        rr = np.where(win >= peak)[0]
        if rr.size:
            rec_flag.append(1); rec_days.append(int(rr[0] + 1))
        else:
            rec_flag.append(0)
    n = len(fdd)
    if n < n_min:
        return {"available": False, "n": n}
    pc = lambda a, q: float(np.percentile(a, q))
    return {
        "available": True, "n": n, "window": window, "dd_now": dd_now, "cur": cur,
        "further_dd": {"p10": pc(fdd, 10), "p50": pc(fdd, 50), "p90": pc(fdd, 90)},
        "runup": {"p10": pc(runup, 10), "p50": pc(runup, 50), "p90": pc(runup, 90)},
        "further_le10": float(np.mean(np.asarray(fdd) <= -0.10)),
        "t_trough_med_d": float(np.median(t_tr)),
        "recovery_rate": float(np.mean(rec_flag)),
        "t_recovery_med_d": float(np.median(rec_days)) if rec_days else float("nan"),
        "price_worst": cur * (1 + pc(fdd, 10)), "price_trough_med": cur * (1 + pc(fdd, 50)),
        "price_best": cur * (1 + pc(runup, 90)), "price_runup_med": cur * (1 + pc(runup, 50)),
    }


def format_regime_path(rd: dict) -> dict | None:
    """把趋势全程分布渲染成 {headline, lines[], price_range}。非预测、是历史类比分布。"""
    if not rd or not rd.get("available"):
        return None
    fdd, ru = rd["further_dd"], rd["runup"]
    wm = max(1, round(rd["window"] / 21)); tt = rd["t_trough_med_d"] / 21
    in_dd = rd["dd_now"] < -0.03
    lines = [
        f"📉 **再下行**(相对现价，未来~{wm}个月内)：中位 {fdd['p50']:+.0%}、坏情形(p10) {fdd['p10']:+.0%}"
        f"；{rd['further_le10']:.0%} 的历史情形会再跌过 −10%",
        f"📈 **反弹空间**(相对现价)：中位 {ru['p50']:+.0%}、好情形(p90) {ru['p90']:+.0%}",
        f"⏳ **到谷底**：历史中位约 {tt:.0f} 个月见到该程后最低点",
    ]
    if in_dd:
        rt = rd["t_recovery_med_d"]
        rt_s = f"、其中位用时约 {rt/21:.0f} 个月" if rt == rt else "（多数未在窗口内收复）"
        # 极端比例(≥98%/≤2%)降级措辞，别让"100%"被当成确定性
        _rr = rd["recovery_rate"]
        _rr_s = (f"几乎全部({rd['n']}个样本里 {_rr:.0%})" if _rr >= 0.98 else
                 (f"仅 {_rr:.0%}" if _rr <= 0.5 else f"{_rr:.0%}"))
        lines.append(f"🔁 **回到前高**：{_rr_s} 的情形在 ~{wm}个月内收复前高{rt_s}"
                     + ("（注：历史能进入样本的多是「活下来」的，**幸存者偏差**会高估收复率）" if _rr >= 0.9 else ""))
    price_range = (f"历史类比区间(未来~{wm}个月)：最坏 ≈ {rd['price_worst']:.0f}（{fdd['p10']:+.0%}）"
                   f" → 中位谷底 ≈ {rd['price_trough_med']:.0f} → 最好 ≈ {rd['price_best']:.0f}（{ru['p90']:+.0%}）")
    headline = f"像今天这状态(距1年高 {rd['dd_now']:+.0%})，历史 {rd['n']} 个同深度样本后来的全程分布"
    caveat = ("⚠️ 这是**历史同深度样本的实际走势分布**(非预测、非牛熊判断、非点位目标)；样本为**滚动重叠日**(非独立交易机会)、"
              "且单票含**幸存者偏差**(只统计到活下来的)；上行尾部(p90)常由个别大牛市集中贡献"
              + ("；样本偏少、低置信" if rd["n"] < 60 else "") + "。极端比例(如~100%)务必当「历史如此」而非「未来必然」看。")
    return {"headline": headline, "lines": lines, "price_range": price_range, "n": rd["n"], "caveat": caveat}
