"""专业分析师视角（研报式叙事，全面纳入基本面）——**纯叙事，不纳入任何量化计算**。

边界（诚实）：
- 基本面来自 yfinance **当前快照**，含前视/重述偏差，**仅供人读、不入回测/量化**。
- 量化情景(牛/基准/熊)来自工具引擎的**历史远期收益分位**，是分布不是预测；附概率与失效条件。
- 不给"必涨到 X"的承诺；给的是 setup / 赔率 / 催化剂 / 什么会让判断作废 —— 真正风险分析师的语言。
- 真·联网读新闻的深度推理在 Claude 对话的 quant-deep-brief skill 里做；本模块只用工具自身可得数据。

研报结构：一句话论点 → 估值 → 盈利与成长 → 财务健康 → 量化情景 → 催化剂 → 风险与失效条件。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 全面基本面字段（决定取数范围）。越界值由 engine_discipline 体检剔除/标极端。
FULL_FIELDS = [
    "marketCap", "trailingPE", "forwardPE", "pegRatio", "priceToBook", "enterpriseToEbitda",
    "revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth",
    "grossMargins", "operatingMargins", "profitMargins", "returnOnEquity", "returnOnAssets",
    "debtToEquity", "currentRatio", "quickRatio", "freeCashflow", "totalCash", "totalDebt",
    "dividendYield", "payoutRatio", "beta",
    "targetMeanPrice", "targetHighPrice", "targetLowPrice", "recommendationKey",
    "numberOfAnalystOpinions",
]


def _g(info: dict, k):
    v = info.get(k)
    try:
        v = float(v)
        return v if v == v else None
    except (TypeError, ValueError):
        return v if v else None


def _valuation(info: dict) -> dict:
    pe = _g(info, "trailingPE"); fpe = _g(info, "forwardPE"); peg = _g(info, "pegRatio")
    pb = _g(info, "priceToBook"); ev = _g(info, "enterpriseToEbitda")
    notes = []
    if isinstance(pe, float):
        lvl = ("便宜(<15)" if pe < 15 else "合理(15–25)" if pe < 25 else "偏高(25–40)" if pe < 40 else "昂贵(>40)")
        notes.append(f"TTM PE {pe:.1f}（{lvl}）")
    if isinstance(fpe, float):
        notes.append(f"前瞻 PE {fpe:.1f}" + ("（预期盈利改善）" if isinstance(pe, float) and fpe < pe else ""))
    if isinstance(peg, float) and peg > 0:
        notes.append(f"PEG {peg:.2f}（{'估值消化成长' if peg < 1.2 else '成长难撑估值' if peg > 2 else '中性'}）")
    if isinstance(pb, float):
        notes.append(f"PB {pb:.1f}")
    if isinstance(ev, float):
        notes.append(f"EV/EBITDA {ev:.1f}")
    grade = "—"
    if isinstance(pe, float):
        grade = "贵" if pe >= 40 else ("偏贵" if pe >= 25 else "合理" if pe >= 15 else "便宜")
    return {"grade": grade, "notes": notes}


def _growth(info: dict) -> dict:
    rg = _g(info, "revenueGrowth"); eg = _g(info, "earningsGrowth") or _g(info, "earningsQuarterlyGrowth")
    notes = []
    if isinstance(rg, float):
        notes.append(f"营收同比 {rg:+.0%}（{'高增长' if rg > 0.2 else '稳健' if rg > 0.05 else '低/停滞' if rg > -0.02 else '收缩'}）")
    if isinstance(eg, float):
        notes.append(f"盈利同比 {eg:+.0%}")
    grade = "—"
    if isinstance(rg, float):
        grade = "高增长" if rg > 0.2 else "稳健" if rg > 0.05 else "低增长" if rg > -0.02 else "收缩"
    return {"grade": grade, "notes": notes}


def _profitability(info: dict) -> dict:
    gm = _g(info, "grossMargins"); om = _g(info, "operatingMargins"); pm = _g(info, "profitMargins")
    roe = _g(info, "returnOnEquity"); roa = _g(info, "returnOnAssets")
    notes = []
    if isinstance(gm, float): notes.append(f"毛利率 {gm:.0%}")
    if isinstance(om, float): notes.append(f"营业利润率 {om:.0%}")
    if isinstance(pm, float): notes.append(f"净利率 {pm:.0%}")
    if isinstance(roe, float): notes.append(f"ROE {roe:.0%}（{'优秀' if roe > 0.2 else '良好' if roe > 0.1 else '一般/弱'}）")
    if isinstance(roa, float): notes.append(f"ROA {roa:.0%}")
    grade = "—"
    if isinstance(pm, float):
        grade = "高盈利" if pm > 0.2 else "盈利中等" if pm > 0.05 else "薄利/亏损"
    return {"grade": grade, "notes": notes}


def _health(info: dict) -> dict:
    de = _g(info, "debtToEquity"); cr = _g(info, "currentRatio"); fcf = _g(info, "freeCashflow")
    cash = _g(info, "totalCash"); debt = _g(info, "totalDebt")
    notes = []
    if isinstance(de, float):
        de_n = de / 100 if de > 5 else de  # yfinance 有时给百分数
        notes.append(f"负债/权益 {de_n:.1f}（{'低杠杆' if de_n < 0.5 else '适中' if de_n < 1.5 else '高杠杆'}）")
    if isinstance(cr, float):
        notes.append(f"流动比率 {cr:.1f}（{'流动性强' if cr > 1.5 else '尚可' if cr > 1 else '偏紧'}）")
    if isinstance(fcf, float):
        notes.append(f"自由现金流 ${fcf/1e9:+.1f}B（{'正·造血' if fcf > 0 else '负·烧钱'}）")
    if isinstance(cash, float) and isinstance(debt, float):
        net = cash - debt
        notes.append(f"净现金 ${net/1e9:+.0f}B（{'净现金' if net > 0 else '净负债'}）")
    grade = "—"
    if isinstance(fcf, float):
        grade = "稳健" if fcf > 0 else "现金流承压"
    return {"grade": grade, "notes": notes}


def _scenarios(price: pd.Series, horizon: int) -> dict:
    """量化情景：当前回撤档(样本足)或无条件的 未来 horizon 日收益分位 → 牛/基准/熊价位。"""
    px = price.dropna()
    cur = float(px.iloc[-1])
    fwd = (px.shift(-horizon) / px - 1.0)
    high = px.rolling(252, min_periods=120).max()
    dd = px / high - 1.0
    cur_dd = float(dd.iloc[-1])
    # 条件：同等回撤档(±5%)样本
    band = fwd[(dd >= cur_dd - 0.05) & (dd <= cur_dd + 0.05)].dropna()
    use = band if len(band) >= 40 else fwd.dropna()
    cond = "当前回撤档" if len(band) >= 40 else "全样本(回撤档样本不足)"
    arr = use.to_numpy()
    q = {p: float(np.percentile(arr, p)) for p in (10, 25, 50, 75, 90)}
    win = float((arr > 0).mean())
    return {
        "horizon": horizon, "cond": cond, "n": len(arr), "win_rate": win, "current": cur,
        "bear": cur * (1 + q[10]), "base_low": cur * (1 + q[25]), "base": cur * (1 + q[50]),
        "base_high": cur * (1 + q[75]), "bull": cur * (1 + q[90]),
        "ret_bear": q[10], "ret_base": q[50], "ret_bull": q[90],
    }


def analyst_report(ticker: str, price: pd.Series, info: dict, brief: dict | None = None,
                   horizon: int = 63) -> dict:
    """生成专业分析师研报(结构化)。info=yfinance 原始/体检后字段；brief=stock_brief 输出(可选)。"""
    info = info or {}
    brief = brief or {}
    val = _valuation(info); grw = _growth(info); prof = _profitability(info); hlth = _health(info)
    sc = _scenarios(price, horizon)
    tgt = _g(info, "targetMeanPrice"); thi = _g(info, "targetHighPrice"); tlo = _g(info, "targetLowPrice")
    rec = info.get("recommendationKey"); n_an = _g(info, "numberOfAnalystOpinions")
    beta = _g(info, "beta"); dy = _g(info, "dividendYield")

    # 催化剂
    catalysts = []
    if brief.get("next_earnings"):
        d2 = brief.get("days_to_earnings")
        catalysts.append(f"下次财报 {brief['next_earnings']}" + (f"（{int(d2)}天后）" if d2 is not None else ""))
    catalysts.append("财报后漂移(PEAD)是工具唯一过检验的免费 alpha——财报后 1–3 周关注")

    # 风险与失效条件
    risks = []
    if brief.get("momentum_trap"):
        risks.append("⚠️ 动量陷阱：当前回撤档历史超额≤0，越跌越买没优势")
    vp = brief.get("vol_percentile")
    if vp == vp and vp is not None and vp > 0.8:
        risks.append(f"高波动（分位 {vp:.0%}）：仓位需压缩")
    if isinstance(beta, float) and beta > 1.3:
        risks.append(f"高 Beta {beta:.1f}：放大市场涨跌")
    invalidations = ["跌破200日线 → 趋势破位(2.4x未来回撤概率)，减半仓",
                     "市场宽度恶化触发 → 系统性降仓",
                     f"跌破熊市情景 {sc['bear']:.1f}（{sc['ret_bear']:+.0%}）→ 当前 setup 失效"]

    # 一句话论点（综合估值×成长×量化状态，分析师 framing，不预测点位）
    dd = float(price.iloc[-1] / price.rolling(252, min_periods=120).max().iloc[-1] - 1.0)
    thesis = (f"{ticker}：估值{val['grade']}、{grw['grade']}、{prof['grade']}；"
              f"现价距前高 {dd:+.0%}，未来{int(horizon/21)}个月历史区间 "
              f"{sc['ret_bear']:+.0%}~{sc['ret_bull']:+.0%}（基准 {sc['ret_base']:+.0%}、胜率 {sc['win_rate']:.0%}）。"
              + ("⚠️动量陷阱，偏防守。" if brief.get("momentum_trap") else "按情景分批、设失效线。"))

    return {
        "ticker": ticker, "thesis": thesis, "valuation": val, "growth": grw,
        "profitability": prof, "health": hlth, "scenarios": sc,
        "analyst_target": {"mean": tgt, "high": thi, "low": tlo, "rec": rec, "n": n_an},
        "beta": beta, "dividend_yield": dy, "catalysts": catalysts,
        "risks": risks or ["无突出风险标记"], "invalidations": invalidations,
        "disclaimer": "基本面=yfinance当前快照(含前视/重述偏差,仅供人读·不纳入量化)；情景=历史分布非预测；非投资建议。",
    }


def format_report(r: dict) -> str:
    """渲染成 Markdown 研报。"""
    sc = r["scenarios"]; at = r["analyst_target"]
    L = [f"### 📝 {r['ticker']} 分析师视角（叙事·不纳入量化）", "", f"**论点**：{r['thesis']}", ""]
    L.append(f"**估值**（{r['valuation']['grade']}）：" + "；".join(r["valuation"]["notes"] or ["数据不足"]))
    L.append(f"**成长**（{r['growth']['grade']}）：" + "；".join(r["growth"]["notes"] or ["数据不足"]))
    L.append(f"**盈利能力**（{r['profitability']['grade']}）：" + "；".join(r["profitability"]["notes"] or ["数据不足"]))
    L.append(f"**财务健康**（{r['health']['grade']}）：" + "；".join(r["health"]["notes"] or ["数据不足"]))
    if isinstance(at.get("mean"), float):
        L.append(f"**卖方目标价**：均值 {at['mean']:.1f}（区间 {at.get('low','?')}–{at.get('high','?')}）"
                 f"·评级 {at.get('rec','?')}（{at.get('n','?')}家）— *仅参考，卖方系统性偏多*")
    L.append("")
    L.append(f"**量化情景**（未来 {int(sc['horizon']/21)} 个月，基于{sc['cond']} N={sc['n']}·胜率{sc['win_rate']:.0%}）：")
    L.append(f"- 🐂 牛 {sc['bull']:.1f}（{sc['ret_bull']:+.0%}）｜🎯 基准 {sc['base']:.1f}（{sc['ret_base']:+.0%}）｜🐻 熊 {sc['bear']:.1f}（{sc['ret_bear']:+.0%}）")
    L.append("**催化剂**：" + "；".join(r["catalysts"]))
    L.append("**风险**：" + "；".join(r["risks"]))
    L.append("**失效条件**：" + "；".join(r["invalidations"]))
    L.append("")
    L.append(f"> {r['disclaimer']}")
    return "\n".join(L)
