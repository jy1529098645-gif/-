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

    today = price.index[-1]
    cur_dd = str(dd_state.iloc[-1])
    cur_val = str(state["valuation_tercile"].iloc[-1]) if use_valuation else None
    cur_vol = str(vol_st.iloc[-1]) if (use_vol and vol_st is not None) else None

    mask = (dd_state == cur_dd)
    if use_valuation and cur_val == cur_val and cur_val != "nan":
        mask &= (state["valuation_tercile"] == cur_val)
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
