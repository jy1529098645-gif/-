"""建仓 / 撤离 作战卡引擎（v3 验证版）——回答"什么时候建仓、什么时候撤离"。

设计原则：**只用通过回测验证的逻辑，证伪的逻辑一律不进**。每条建议带证据等级。

已验证 ✅（进引擎）
  · 连续波动目标暴露(v3/voltarget)：唯一对同篮子持有做出**显著正 α**的杠杆(DSR 0.999)。
  · 趋势门 + 快重入(站上200线即在场，不等深跌)：v3 把回撤 −54%→−20%、夏普 0.85→1.01。
  · 回撤桶建仓：在优质票回撤中买入，远期正超额在 8/8 显著(N_eff 折算后)。
  · PEAD 窗口：唯一过安慰剂检验的事件信号。
已证伪 ❌（明确排除）
  · 固定 %% 移动止损 / 快速二元出场：回测对同篮子持有**显著负 α**、反拖累夏普。
  · dip-from-high 单独择时：N_eff 折算后超额不显著。
诚实口径 ⚠️
  · 撤离触发是**崩盘保险**：正常年份(26年里22年)闭眼持有更优，撤离的价值只在崩盘年兑现。
  · 没有任何配置在绝对收益上跑赢长牛持有——减暴露必然让出复利。撤离=换风险调整后更优 + 砍回撤。

铁律(与产品总纲一致)：价位是「若到达就行动」的**区间**(非预测/非买卖指令)；
暴露%是**机械规则**(非投资建议)；分布带 N/CI；不给目标价/单一概率。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 风险偏好 → (波动目标年化, 杠杆上限)。中性=v3终审用的 0.25/1.0。
# 🔥杠杆进取(leveraged)= 回测验证的"纪律可控更高利润"档：科技/半导体上年化≈+30%(死拿+29%)、
# 夏普打平、回撤同级；原理=低波动牛市里加杠杆放大、波动一起自动降、200线破位清仓(放大beta非择时)。
PROFILES = {
    "conservative": {"tvol": 0.18, "max_lev": 1.0},
    "moderate":     {"tvol": 0.25, "max_lev": 1.0},
    "aggressive":   {"tvol": 0.35, "max_lev": 1.0},
    "leveraged":    {"tvol": 0.40, "max_lev": 1.5},
}
PROFILE_ZH = {"conservative": "稳健", "moderate": "中性", "aggressive": "进取", "leveraged": "🔥杠杆进取"}


def _ewma_vol_last(ret: pd.Series, lam: float = 0.94) -> float:
    from analysis.quant_edge import ewma_vol
    s = ewma_vol(ret, lam=lam).dropna()
    return float(s.iloc[-1]) if not s.empty else float("nan")


# 上杠杆(暴露>1)的低波动门：仅当已实现波动分位 ≤ 此值，才允许加杠杆(否则封顶 1.0)。
# 思路：高波动常出现在顶部/转折区，"只在低波动+确认趋势时才上杠杆"避免在最危险处放大。
LEV_VOLPCT_MAX = 0.50


def _target_exposure(trend_up: bool, slope_pos: bool, ewmav: float, tvol: float,
                     max_lev: float = 1.0, vol_pct: float | None = None,
                     lev_volpct_max: float = LEV_VOLPCT_MAX) -> float:
    """v3 暴露规则：趋势门 × 波动目标连续定仓(上限 max_lev) × 仅确认趋势死亡才清。

    max_lev>1 即"科技进取·杠杆"：**只在低波动(分位≤lev_volpct_max)+确认趋势**时才上杠杆放大；
    波动一起(分位高)即把杠杆收回 1.0；破位清仓。"""
    if not (ewmav == ewmav) or ewmav <= 0:
        base = 0.0
    else:
        base = min(max_lev, tvol / ewmav)
    # 低波动门：高波动(或分位未知)时不许上杠杆，封顶 1.0
    if max_lev > 1.0 and not (vol_pct is not None and vol_pct == vol_pct and vol_pct <= lev_volpct_max):
        base = min(base, 1.0)
    if trend_up:                       # 站上200线：按波动目标在场(低波动可>1倍)
        return base
    if not slope_pos:                  # 200线下方且斜率转负：确认趋势死亡 → 清
        return 0.0
    return 0.5 * base                  # 200线下方但斜率未转负：半仓过渡


def _exposure_series(price: pd.Series, tvol: float, max_lev: float = 1.0,
                     lev_volpct_max: float = LEV_VOLPCT_MAX) -> pd.Series:
    """全历史暴露序列(供画图/复核/回测)。无前视：t 暴露用到 t 收盘。
    max_lev>1 时**只在低波动分位才上杠杆**(高波动封顶1.0)。"""
    from analysis.quant_edge import ewma_vol
    from regime.observables import realized_vol_percentile
    ret = price.pct_change()
    sma200 = price.rolling(200, min_periods=100).mean()
    slope = (sma200 > sma200.shift(20))
    ewmav = ewma_vol(ret)
    base = (tvol / ewmav).clip(upper=max_lev)
    if max_lev > 1.0:                  # 低波动门：高波动(或分位未知)处把杠杆收回 1.0
        vp = realized_vol_percentile(price, 21, 252)
        high_vol = ~(vp <= lev_volpct_max)   # NaN→True(未知按高波动处理，不上杠杆)
        base = base.where(~high_vol, base.clip(upper=1.0))
    up = price > sma200
    dead = (price < sma200) & (~slope)
    expo = base.where(up, 0.5 * base)
    expo = expo.where(~dead, 0.0)
    return expo.clip(0.0, max_lev).fillna(0.0)


def position_guidance(ticker: str, start: str | None = None, end: str | None = None,
                      horizon: int = 63, broad_pool: bool = True) -> dict:
    """生成单票建仓/撤离作战卡(三档风险偏好)。

    返回 dict：regime(今日状态) / build(建仓) / exit(撤离·三档暴露+触发) / evidence / headline
    + exposure_history(DataFrame: date/price/exposure_moderate 供画图)。
    """
    from data import loader
    from regime import observables as ob
    from analysis import quant_edge as qe

    start = start or ("1995-01-01" if ticker.upper() == "SPY" else "2008-01-01")
    ohlc = loader.load_ohlcv(ticker.upper(), start, end).dropna()
    price = ohlc["close"].dropna()
    ret = price.pct_change()
    px = float(price.iloc[-1])
    asof = str(price.index[-1].date())

    sma200 = price.rolling(200, min_periods=100).mean()
    s200 = float(sma200.iloc[-1])
    slope_pos = bool(sma200.iloc[-1] > sma200.iloc[-21]) if sma200.notna().iloc[-21] else True
    trend_up = bool(px > s200)
    trailing_high = float(price.cummax().iloc[-1])
    dd = float(px / trailing_high - 1.0)
    volp = ob.realized_vol_percentile(price, 21, 252)
    vol_pct = float(volp.iloc[-1]) if volp.notna().iloc[-1] else float("nan")
    ewmav = _ewma_vol_last(ret)
    vol_spike = bool(vol_pct == vol_pct and vol_pct > 0.90)

    regime = {
        "ticker": ticker.upper(), "asof": asof, "price": px, "trailing_high": trailing_high,
        "drawdown_from_high": dd, "sma200": s200, "trend_above_200": trend_up,
        "slope200_positive": slope_pos, "vol_percentile": vol_pct, "ewma_vol_annual": ewmav,
        "vol_spike": vol_spike,
    }

    # --- 今日四档暴露(含🔥杠杆进取·低波动门) ---
    lev_gated_off = bool(vol_pct == vol_pct and vol_pct > LEV_VOLPCT_MAX)  # 高波动→今日不许上杠杆
    exposure = {}
    for k, cfg in PROFILES.items():
        e = _target_exposure(trend_up, slope_pos, ewmav, cfg["tvol"], cfg["max_lev"], vol_pct=vol_pct)
        exposure[k] = {"target_vol": cfg["tvol"], "max_lev": cfg["max_lev"],
                       "exposure": float(round(e, 3)), "exposure_pct": int(round(e * 100)),
                       "leveraged": bool(cfg["max_lev"] > 1.0),
                       "lev_gated_off": bool(cfg["max_lev"] > 1.0 and lev_gated_off)}

    # --- 回撤桶证据(建仓核心) ---
    bucket = {"available": False}
    momentum_trap = None
    try:
        from regime import conditional_returns as cr
        macro = loader.load_macro("1990-01-01", end)
        tab = cr.conditional_forward_returns(price, macro, asset=ticker.upper(),
                                             horizons=(horizon,), groupings=[["in_drawdown"]],
                                             n_boot=250)
        ddrow = tab[tab["bucket"].astype(str).str.contains("in_drawdown=True")]
        if not ddrow.empty:
            r = ddrow.iloc[0]
            sig = bool((r["ci_low"] > 0 or r["ci_high"] < 0) and not r.get("low_power", False))
            momentum_trap = bool(r["excess_median"] <= 0)
            bucket = {"available": True, "median": float(r["median"]),
                      "excess": float(r["excess_median"]), "n_independent": int(r["n_independent"]),
                      "ci": [float(r["ci_low"]), float(r["ci_high"])], "significant": sig,
                      "momentum_trap": momentum_trap, "horizon": horizon}
    except Exception:  # noqa: BLE001
        pass

    # --- PEAD 窗口 ---
    pead = None
    try:
        pn = qe.pead_now(ticker.upper(), start=start, end=end)
        if pn and pn.get("in_window") and pn.get("actionable"):
            pead = {"surprise": pn["surprise"], "beat": pn["beat"], "days_since": pn["days_since"],
                    "drift_median": pn["drift"]["median"], "n": pn["drift"]["n"],
                    "direction": pn.get("direction")}
    except Exception:  # noqa: BLE001
        pass

    # --- 建仓 stance + 区间 ---
    # 诚实分级：单票回撤桶 N 小 → 只能 suggestive；工具**验证级**结论是池化8赢家超额8/8显著。
    in_dd = dd <= -0.10
    excess_pos = bool(bucket.get("available") and bucket.get("excess", 0) > 0)
    if not trend_up and not slope_pos:
        stance, stance_grade = "趋势破位，暂不建仓", "🔴 低"
        stance_why = "价格在200线下方且均线斜率转负(确认趋势死亡=撤离口径)——等站回200线再谈建仓。"
    elif momentum_trap:
        stance, stance_grade = "暂不主动建仓（动量陷阱）", "🔴 低"
        stance_why = (f"在回撤中、但该票回撤桶**超额 {bucket['excess']:+.1%}≤0**——逢跌买历史上不优于随机进场"
                      "(注：桶内绝对收益仍正只因长期是牛股，但相对随机买入无优势)。等趋势确认/波动回落再轻仓。")
    elif in_dd and excess_pos:
        stance, stance_grade = "回撤区有正倾斜（单票样本小·分批轻-中仓）", "🟡 中"
        stance_why = (f"在回撤中、回撤桶超额 {bucket['excess']:+.1%}>0(N独立={bucket['n_independent']})——方向有利，"
                      "但**单票样本小、属 suggestive**；工具验证级结论是**池化8赢家超额8/8显著**，故分批轻-中仓、别重仓抢跌。")
    elif trend_up and not in_dd:
        stance, stance_grade = "趋势内参与（按今日暴露建议）", "🟡 中"
        stance_why = "价格在200线上方、趋势成立——按下方三档暴露参与；回调到支撑分批加，别等深跌空仓踏空。"
    elif in_dd and bucket.get("available"):
        stance, stance_grade = "回撤区轻仓试探（超额≈0/证据弱）", "🟡 中低"
        stance_why = "在回撤中但回撤桶超额不正——只轻仓试探，不重仓抢跌。"
    else:
        stance, stance_grade = "中性观望 / 轻仓", "⚪ 低"
        stance_why = "无显著正倾斜，按低把握处理。"

    # 候选建仓区间(回撤分带=「若到达就行动」的区间，非预测点位；前重后轻)
    bands = [(0.10, "浅档", 0.40), (0.20, "中档", 0.35), (0.30, "深档", 0.25)]
    zones = []
    if stance_grade not in ("🔴 低",):
        for b, name, w in bands:
            lo = trailing_high * (1 - b - 0.03)
            hi = trailing_high * (1 - b + 0.03)
            zones.append({"band": f"距前高 −{int(b*100)}%", "tier": name,
                          "price_low": float(round(lo, 2)), "price_high": float(round(hi, 2)),
                          "size_hint": f"{int(w*100)}% 本票计划仓位",
                          "reached": bool(px <= hi)})

    build = {"stance": stance, "grade": stance_grade, "why": stance_why,
             "drawdown_bucket": bucket, "pead": pead, "zones": zones,
             "sizing_rule": "前重后轻：浅档正期望风险调整最优、越深尾部越肥胜率越低→越深每批越小(回测背书)。",
             "anti_chase": "防踏空：若不回深档直接走高 → 站上MA50/破前高就追小仓，别空仓干等(想等的浅回调多半不来)。"}

    # --- 撤离触发(已验证 + 诚实口径) ---
    if not trend_up and not slope_pos:
        active_trigger = "🔴 趋势死亡已触发（200线下方+斜率转负）→ 清仓口径，今日建议暴露已自动归零。"
    elif not trend_up and slope_pos:
        active_trigger = "🟠 200线下方但斜率未转负 → 半仓过渡，盯斜率是否转负。"
    elif vol_spike:
        active_trigger = "🟠 波动突刺(>90分位) → 暴露按波动目标自动收缩(已体现在今日%)。"
    else:
        active_trigger = "🟢 趋势成立、波动正常 → 按今日暴露建议持有，让赢家在场。"

    exit_blk = {
        "by_profile": exposure,
        "active_trigger": active_trigger,
        "triggers": [
            "✅ 趋势死亡(收盘跌破200线 且 200线斜率持续转负20日) → 清仓。这是DSR验证的崩盘保险。",
            "✅ 200线下方但斜率未转负 → 降到半仓过渡(不一刀清，避免假摔踏空)。",
            "✅ 波动突刺(已实现波动>90分位) → 暴露按波动目标连续收缩(不是二元砍断)。",
            "❌ 不要用固定%移动止损/快速二元止损 → 回测证明对同篮子持有显著负α、反拖累夏普。",
        ],
        "honesty": ("⚠️ 撤离=崩盘保险：26年回测里22年闭眼持有的夏普更高，撤离的正α**只在崩盘年兑现**"
                    "(2008/2022 把回撤砍掉2/3)。稳健/中性/进取(≤1倍)绝对收益上不跑赢长牛持有——换的是"
                    "风险调整后更优 + 回撤腰斩，代价是正常年份让出部分上行。要的是这个保险才用。"),
        "leverage_warning": ("🔥 杠杆进取档(暴露>100%=上杠杆)：**只在低波动(分位≤50%)+确认趋势**时才上杠杆，"
                             "波动一起即把杠杆收回1.0、破位清仓——避免在最危险的高波动顶部区放大。"
                             "回测(科技/半导体 2010-26)是**放大beta、非更聪明择时**，没有免费午餐。"
                             "代价：① 真出现 2000/2008 级结构性深熊会远比这惨(200线过滤滞后挨刀)；"
                             "② −50% 级回撤要扛住不割；③ 全 in-sample、未来不复刻。仅高风险承受力、且严守纪律者用。"),
    }

    # --- 总纲 ---
    em = PROFILE_ZH
    head = (f"{regime['ticker']} @ {px:.2f}（距前高 {dd:+.0%}，{'200线上' if trend_up else '200线下'}"
            f"{'·斜率正' if slope_pos else '·斜率转负'}，波动分位 {vol_pct:.0%}）｜"
            f"建仓：{stance}｜今日建议暴露 稳健{exposure['conservative']['exposure_pct']}%/"
            f"中性{exposure['moderate']['exposure_pct']}%/进取{exposure['aggressive']['exposure_pct']}%/"
            f"🔥杠杆{exposure['leveraged']['exposure_pct']}%｜{active_trigger}")

    # --- 暴露历史(中性档，供画图) ---
    es = _exposure_series(price, PROFILES["moderate"]["tvol"], PROFILES["moderate"]["max_lev"])
    hist = pd.DataFrame({"price": price, "exposure": es}).dropna().tail(750)

    return {"regime": regime, "exposure": exposure, "build": build, "exit": exit_blk,
            "headline": head, "horizon": horizon, "exposure_history": hist,
            "disclaimer": ("机械 if-then 规则，非预测/非买卖指令。暴露%与价位区间为风控纪律，"
                           "分布带N/CI。撤离为崩盘保险口径，绝对收益上不跑赢长牛持有。研究校准用途。")}


def format_guidance(g: dict) -> str:
    """渲染为 Markdown(供 CLI / 报告)。"""
    r, b, x = g["regime"], g["build"], g["exit"]
    ex = g["exposure"]
    L = [f"# 🎖️ 建仓/撤离作战卡 · {r['ticker']}  （{r['asof']}）",
         f"> {g['headline']}", ""]
    # 今日暴露
    L += ["## 📊 今日建议暴露（v3 连续定仓·已验证）",
          f"| 风险偏好 | 波动目标 | 杠杆上限 | 建议暴露 |",
          f"|---|---|---|---|"]
    for k in ("conservative", "moderate", "aggressive", "leveraged"):
        L.append(f"| {PROFILE_ZH[k]} | {ex[k]['target_vol']:.0%} | {ex[k]['max_lev']:g}× | **{ex[k]['exposure_pct']}%** |")
    L += [f"\n_{x['active_trigger']}_", ""]
    if x.get("leverage_warning"):
        L += [f"> {x['leverage_warning']}", ""]
    # 建仓
    L += [f"## 🎯 建仓：{b['stance']}  （把握度 {b['grade']}）", f"> {b['why']}", ""]
    bk = b["drawdown_bucket"]
    if bk.get("available"):
        tag = "⚠️动量陷阱(超额≤0)" if bk["momentum_trap"] else ("🟡正倾斜·suggestive" if bk["excess"] > 0 else "证据弱")
        L.append(f"- 回撤桶证据({tag})：回撤中 {bk['horizon']}日 **超额(vs随机买入) {bk['excess']:+.1%}**，N独立={bk['n_independent']}。"
                 f"（桶内绝对收益中位 {bk['median']:+.1%}、CI[{bk['ci'][0]:+.1%},{bk['ci'][1]:+.1%}]——此CI是绝对收益非超额，牛股恒正不代表择时有优势）")
    if b["zones"]:
        L.append("- 候选建仓区间（**若到达就行动**的区间，非预测点位）：")
        for z in b["zones"]:
            mark = " ←已到达" if z["reached"] else ""
            L.append(f"    - {z['tier']}（{z['band']}）：≈ {z['price_low']:.1f}–{z['price_high']:.1f}，{z['size_hint']}{mark}")
        L.append(f"    - 规则：{b['sizing_rule']}")
        L.append(f"    - {b['anti_chase']}")
    if b["pead"]:
        p = b["pead"]
        L.append(f"- 📅 PEAD窗口：上次财报{'超预期' if p['beat'] else '不及'}(surprise {p['surprise']:+.1f}%)、"
                 f"距今{p['days_since']:.0f}天，历史同类漂移中位 {p['drift_median']:+.1%}(N={p['n']})。")
    L.append("")
    # 撤离
    L += ["## 📉 撤离：触发条件（已验证·崩盘保险口径）"]
    L += [f"- {t}" for t in x["triggers"]]
    L += ["", f"> {x['honesty']}", "", f"_{g['disclaimer']}_"]
    return "\n".join(L)
