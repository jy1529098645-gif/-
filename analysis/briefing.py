"""多票作战简报（综合层）——把引擎数字 + 技术位 + 财报日程 + 免费新闻 合成一页可读简报。

铁律：可计算层(表/档位/目标止损RR/引擎结论/财报drift)全部来自价格与引擎，诚实可复现；
叙事层(新闻/基本面)来自 data.news，**仅供人读、不入量化**，并显式标注前视风险。
权重是**机械规则**(引擎超额×显著性折扣×反波动率)，标注"非投资建议"。

DO NOT：不输出单一概率/不把新闻当信号/不声称预测；目标价/止损是按**规则从技术位推导**的
风险管理参考(供算盈亏比)，非"预言点位"。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 桶标签翻译
# ---------------------------------------------------------------------------
_BUCKET_CN = {
    "in_drawdown=True": "回撤桶", "in_drawdown=False": "近高位桶",
    "valuation_tercile=low": "估值低位", "valuation_tercile=mid": "估值中位",
    "valuation_tercile=high": "估值高位",
    "credit_trend=widening": "信用走阔", "credit_trend=narrowing": "信用收窄",
}


def _cn_bucket(label: str) -> str:
    return " & ".join(_BUCKET_CN.get(p.strip(), p.strip()) for p in str(label).split("&"))


# ---------------------------------------------------------------------------
# 技术位 + 共振聚簇
# ---------------------------------------------------------------------------
def key_levels(price: pd.Series, ohlcv: pd.DataFrame, bands=(0.10, 0.15, 0.20, 0.25, 0.30)) -> list[dict]:
    """收集关键支撑/参考位：MA50/100/200、近半年/2年 POC、近1年高/近半年低、距前高各回撤档。"""
    from analysis.volume_profile import volume_profile

    price = price.dropna()
    levels = []
    for n, nm in [(50, "MA50"), (100, "MA100"), (200, "MA200")]:
        m = price.rolling(n, min_periods=n // 2).mean().iloc[-1]
        if pd.notna(m):
            levels.append({"name": nm, "price": float(m)})
    try:
        vp_h = volume_profile(ohlcv, bins=50, lookback=126)
        levels.append({"name": "近半年POC", "price": float(vp_h["poc"])})
        vp_y = volume_profile(ohlcv, bins=50, lookback=504)
        levels.append({"name": "近2年POC", "price": float(vp_y["poc"])})
    except Exception:  # noqa: BLE001
        pass
    hi252 = price.rolling(252, min_periods=126).max().iloc[-1]
    lo126 = price.rolling(126, min_periods=63).min().iloc[-1]
    if pd.notna(hi252):
        levels.append({"name": "近1年高点", "price": float(hi252)})
    if pd.notna(lo126):
        levels.append({"name": "近半年低点", "price": float(lo126)})
    trailing_high = float(price.rolling(252, min_periods=126).max().iloc[-1])
    for b in bands:
        levels.append({"name": f"距前高−{b:.0%}档", "price": trailing_high * (1 - b)})
    return levels


def cluster_levels(levels: list[dict], tol: float = 0.025) -> list[dict]:
    """把相距 <tol(相对%) 的技术位聚成"共振簇"，按价位降序返回。每簇：price/members/n。"""
    if not levels:
        return []
    lv = sorted(levels, key=lambda d: d["price"])
    clusters = []
    cur = [lv[0]]
    for x in lv[1:]:
        if abs(x["price"] / cur[-1]["price"] - 1) <= tol:
            cur.append(x)
        else:
            clusters.append(cur)
            cur = [x]
    clusters.append(cur)
    out = []
    for grp in clusters:
        prices = [g["price"] for g in grp]
        out.append({"price": float(np.mean(prices)),
                    "members": [g["name"] for g in grp], "n": len(grp)})
    return sorted(out, key=lambda d: d["price"], reverse=True)


# ---------------------------------------------------------------------------
# 引擎最优桶
# ---------------------------------------------------------------------------
def engine_buckets(tab: pd.DataFrame, horizon: int) -> dict:
    """从 conditional_forward_returns 表抽 h 的各单变量桶（含显著性、盈亏比）。

    返回 {"map": {中文桶名: row}, "all": [rows]}。诚实地保留每个状态桶的真实超额
    （动量股的回撤桶可能为负——这正是"越跌越没优势"的关键信号，不可被最优桶掩盖）。"""
    sub = tab[(tab["horizon"] == horizon) & (tab["grouping"] != "__baseline__")].copy()
    if sub.empty:
        return {"map": {}, "all": []}
    # 显著性：CI 不跨 0 **且** 有效独立窗口足够（low_power=False）。长周期重叠窗口会假性显著。
    lp = sub["low_power"] if "low_power" in sub.columns else pd.Series(False, index=sub.index)
    sub["low_power"] = lp.fillna(False)
    sub["significant"] = ((sub["ci_low"] > 0) | (sub["ci_high"] < 0)) & (~sub["low_power"])
    sub["reward_risk"] = sub.apply(
        lambda r: (r["median"] / abs(r["median_path_drawdown"]))
        if r["median_path_drawdown"] and r["median_path_drawdown"] < 0 else np.nan, axis=1)
    rows, mp = [], {}
    for _, r in sub.sort_values("excess_median", ascending=False).iterrows():
        cn = _cn_bucket(r["bucket"])
        row = {"bucket": cn, "raw_bucket": r["bucket"],
               "median": float(r["median"]), "excess": float(r["excess_median"]),
               "win_rate": float(r["win_rate"]), "n_events": int(r["n_events"]),
               "n_independent": int(r["n_independent"]) if "n_independent" in r and r["n_independent"] == r["n_independent"] else None,
               "low_power": bool(r["low_power"]),
               "reward_risk": float(r["reward_risk"]) if r["reward_risk"] == r["reward_risk"] else float("nan"),
               "ci_low": float(r["ci_low"]), "ci_high": float(r["ci_high"]),
               "significant": bool(r["significant"])}
        rows.append(row)
        mp[cn] = row
    return {"map": mp, "all": rows}


# ---------------------------------------------------------------------------
# 建仓档（浅/中/重）：共振簇 → 目标/止损/RR + 引擎概率
# ---------------------------------------------------------------------------
_TIER_NAMES = ["浅", "中", "重"]


def build_tranches(current_price: float, trailing_high: float, clusters: list[dict],
                   best_bucket: dict | None, stop_buffer: float = 0.02) -> list[dict]:
    """取现价下方最近的最多 3 个共振簇作浅/中/重档；目标=前高，止损=下一簇下沿，算 RR。"""
    below = [c for c in clusters if c["price"] < current_price * 0.999]
    below = sorted(below, key=lambda c: c["price"], reverse=True)[:3]
    tranches = []
    for i, c in enumerate(below):
        entry = c["price"]
        target = max(trailing_high, current_price)  # 目标=前高(或现价兜底)
        # 止损=下一更低簇的价位×(1-buffer)；无更低簇则 entry×(1-8%)
        lower = [x["price"] for x in clusters if x["price"] < entry * 0.995]
        stop = (max(lower) * (1 - stop_buffer)) if lower else entry * 0.92
        rr = (target - entry) / (entry - stop) if entry > stop else float("nan")
        resonance = "（三重共振）" if c["n"] >= 3 else ("（双重共振）" if c["n"] == 2 else "")
        tranches.append({
            "tier": _TIER_NAMES[i],
            "price": entry,
            "what": "、".join(c["members"]) + resonance,
            "target": target, "stop": stop,
            "rr": rr,
            "to_target_pct": target / entry - 1,
            "to_stop_pct": stop / entry - 1,
            "engine_win_rate": best_bucket["win_rate"] if best_bucket else float("nan"),
            "engine_rr": best_bucket["reward_risk"] if best_bucket else float("nan"),
        })
    return tranches


# ---------------------------------------------------------------------------
# 单票简报
# ---------------------------------------------------------------------------
def stock_brief(ticker: str, start: str, end: str | None = None, horizon: int = 63,
                macro: pd.DataFrame | None = None, with_news: bool = True,
                news_sources=("google", "yahoo")) -> dict:
    """合成单票简报：快照 + 引擎桶 + 建仓档 + 财报日程/drift + (可选)免费新闻/基本面。"""
    from data import loader
    from regime import conditional_returns as cr
    from regime import entry_cockpit as ec
    from regime import observables as ob

    price = loader.load_prices([ticker], start, end)[ticker].dropna()
    ohlcv = loader.load_ohlcv(ticker, start, end)
    if macro is None:
        macro = loader.load_macro("1990-01-01", end)

    panel = ob.today_panel(price)
    tp = panel["trend_position"]
    trend = "↑强" if tp > 0.15 else ("↑" if tp > 0 else ("↓深" if tp < -0.15 else "↓"))
    trailing_high = float(price.rolling(252, min_periods=126).max().iloc[-1])

    # 单变量桶：当前状态桶(回撤/近高位) 是"在今天这种状态建仓"的真实历史分布；
    #          估值低位桶 是"买便宜"论据。两者都报，动量股逢跌没优势的真相不被掩盖。
    tab = cr.conditional_forward_returns(price, macro, asset=ticker, horizons=(horizon,),
                                         groupings=[["in_drawdown"], ["valuation_tercile"]], n_boot=300)
    buckets = engine_buckets(tab, horizon)
    bmap = buckets["map"]
    cur_state_cn = "回撤桶" if panel["drawdown_state"] == "in_drawdown" else "近高位桶"
    engine_state = bmap.get(cur_state_cn)            # 匹配当前状态(主结论)
    engine_value = bmap.get("估值低位")               # 买便宜论据
    engine_best = engine_state or engine_value        # 建仓档引擎胜率用"当前状态桶"
    # 动量陷阱：当前在回撤、但回撤桶超额≤0（逢跌买不优于随机）。
    # 注意：引擎回撤桶按**全历史前高(cummax)**分桶；但"陷阱"是给"现在要不要抢这段跌"的告警，
    # 故再要求**近1年也确在回撤**(dd_252≤-5%)，避免"距历史高很深、但已回到近1年高位"的标的(如UNH)
    # 被误标陷阱、与决策卡『追/持有』口径打架（决策卡用近1年高）。
    dd_252 = float(price.iloc[-1] / trailing_high - 1.0) if trailing_high == trailing_high and trailing_high > 0 else 0.0
    momentum_trap = bool(engine_state and panel["drawdown_state"] == "in_drawdown"
                         and engine_state["excess"] <= 0 and dd_252 <= -0.05)
    # 表头桶：常态用"估值低位"(买便宜论据、跨票可比)，动量陷阱时切回撤桶以暴露真相
    engine_headline = (engine_state if momentum_trap else (engine_value or engine_state))

    # 多周期对账：当前状态桶在各 horizon 的结论是否一致（冲突即降置信）
    horizons_reconcile = None
    try:
        from analysis import engine_discipline as ed
        multi_h = (21, 63, 126, 252, 504)
        tab_m = cr.conditional_forward_returns(price, macro, asset=ticker, horizons=multi_h,
                                               groupings=[["in_drawdown"]], n_boot=200)
        per_h = {h: engine_buckets(tab_m, h)["map"].get(cur_state_cn) for h in multi_h}
        horizons_reconcile = ed.reconcile_horizons({h: b for h, b in per_h.items() if b})
    except Exception:  # noqa: BLE001
        horizons_reconcile = None

    # 证据等级（硬闸门：负超额/CI跨0/有效样本不足 永不进 A/B）
    grade = None
    try:
        from analysis import engine_discipline as ed
        conflict = bool(horizons_reconcile and horizons_reconcile.get("conflict"))
        grade = ed.evidence_grade(engine_headline, vol_percentile=panel["vol_percentile"],
                                  horizon_conflict=conflict)
    except Exception:  # noqa: BLE001
        grade = None

    levels = key_levels(price, ohlcv)
    clusters = cluster_levels(levels)
    tranches = build_tranches(float(price.iloc[-1]), trailing_high, clusters, engine_best)

    # ── 统一入场/离场口径：与「个股决策」「建仓扫描」同源(entry_confluence + exit_warning) ──
    # 多票简报不再自成一套裁决，原"建仓档"降为技术参考；主裁决以 entry_confluence(回踩支撑+胜率+
    # regime/飞刀门控)、本票风险以 exit_warning 为准，保证同一只票跨板块结论一致。
    entry = exit_w = None
    risk_txt = "—"
    entry_sup_txt = "—"
    try:
        from analysis import decision as _dec
        exit_w = _dec.exit_warning(price, False, None)   # 本票自身风险(不含市场宽度)，与扫描同口径
        vixp = None
        try:
            _vix = loader.load_ohlcv("^VIX", "1995-01-01", end)["close"].dropna()
            _vp = _vix.rolling(252, min_periods=126).apply(lambda x: (x[-1] >= x).mean(), raw=True)
            if len(_vp) and _vp.iloc[-1] == _vp.iloc[-1]:
                vixp = float(_vp.iloc[-1])
        except Exception:  # noqa: BLE001
            vixp = None
        entry = ec.entry_confluence(ohlcv, asset=ticker, best_entry={"has_zone": False},
                                    warn_red=bool(exit_w["red"]), warn_amber=bool(exit_w["amber"]),
                                    warn_label=exit_w["level"], vix_pctile=vixp)
        # 本票风险标签(买家口径)：红=破位 / 黄=列自身黄灯原因 / 绿=无（与 app.c_build_scan 同口径）
        _own_amber = [s["state"].split(" ", 1)[-1] for s in exit_w.get("signals", [])
                      if "🟡" in s.get("state", "") and s.get("name") != "市场宽度"]
        if exit_w["red"]:
            risk_txt = "🔴 趋势破位"
        elif _own_amber:
            risk_txt = "🟡 " + "/".join(_own_amber[:2])
        else:
            risk_txt = "🟢 无"
        _below = entry.get("supports_below") or []
        _sup = _below[0] if _below else None
        entry_sup_txt = ("现价在支撑共振区" if entry.get("at_support_now")
                         else (f"{_sup['label']} {_sup['price']:.1f}（{_sup['dist_pct']:+.0%}）" if _sup else "—"))
    except Exception:  # noqa: BLE001
        pass

    brief = {
        "ticker": ticker,
        "horizon": int(horizon),
        "price": float(price.iloc[-1]),
        "date": str(price.index[-1].date()),
        "trend": trend, "trend_position": tp,
        "drawdown": panel["drawdown"],
        "vol_percentile": panel["vol_percentile"],
        "vol_state": panel["vol_state"],
        "trailing_high": trailing_high,
        "engine_best": engine_best, "engine_headline": engine_headline,
        "engine_state": engine_state, "engine_value": engine_value,
        "current_state_bucket": cur_state_cn, "momentum_trap": momentum_trap,
        "engine_all": buckets["all"],
        "grade": grade, "horizons": horizons_reconcile,
        "clusters": clusters,
        "tranches": tranches,
        "entry": entry, "exit": exit_w,            # 统一裁决：入场(entry_confluence) + 离场(exit_warning)
        "risk_txt": risk_txt, "entry_sup_txt": entry_sup_txt,
        "next_earnings": None, "days_to_earnings": None,
        "earnings_stats": None, "upcoming": None,
        "news": None, "fundamentals": None, "highlights": None,
        "news_analysis": None, "news_reason": None,
    }

    # 财报日程 + drift（ETF 无财报，跳过）
    try:
        edates = loader.load_earnings_dates(ticker, limit=80)
        up = ec.upcoming_events(price, edates)
        ne = up[up["event"].str.contains("财报", na=False)]
        if not ne.empty:
            brief["next_earnings"] = str(ne.iloc[0]["date"])
            brief["days_to_earnings"] = int(ne.iloc[0]["days_ahead"])
        brief["upcoming"] = up
        brief["earnings_stats"] = ec.earnings_reaction_stats(price, edates)
    except Exception:  # noqa: BLE001
        pass

    if with_news:
        try:
            from data import news as nws
            brief["news"] = nws.stock_news(ticker, limit=8, sources=news_sources)
            brief["fundamentals"] = nws.fundamentals_snapshot(ticker)
            brief["highlights"] = nws.financial_highlights(ticker)
        except Exception:  # noqa: BLE001
            pass
        # 新闻启发式推理（情绪+主题 × 引擎数字，非 LLM、非信号）
        try:
            from analysis.news_reason import analyze_news, reason_paragraph
            brief["news_analysis"] = analyze_news(brief.get("news"))
            brief["news_reason"] = reason_paragraph(brief)
        except Exception:  # noqa: BLE001
            pass

    # 一致性自检：渲染前抓内部自相矛盾（弱证据强评级/止盈乱序/动量陷阱却摊平…）
    brief["consistency"] = []
    try:
        from analysis import engine_discipline as ed
        from analysis.playbook import build_playbook
        pb = build_playbook(brief)
        brief["consistency"] = ed.validate_consistency(brief, pb, grade)
    except Exception:  # noqa: BLE001
        pass

    return brief


# ---------------------------------------------------------------------------
# 自动推荐权重（机械规则）
# ---------------------------------------------------------------------------
def auto_weights(briefs: list[dict]) -> dict:
    """权重 ∝ 引擎超额(显著性折扣) × 反波动率。机械规则，非投资建议。

    返回 {ticker: {weight, role, score}}，weight 为百分比、和为 100。"""
    scores = {}
    for b in briefs:
        # 用"估值低位桶"(买便宜论据)做跨票可比的权重依据；动量陷阱再打折
        thesis = b.get("engine_value") or b.get("engine_best")
        excess = thesis["excess"] if thesis else 0.0
        sig = thesis["significant"] if thesis else False
        raw = max(excess if sig else excess * 0.4, 0.005)  # 不显著打 4 折，给个地板
        if b.get("momentum_trap"):
            raw *= 0.5  # 逢跌无优势的动量票，降权
        volp = b.get("vol_percentile")
        volp = volp if (volp == volp) else 0.5
        risk_factor = 1.0 / (0.3 + volp)  # 波动分位越高，权重越低
        scores[b["ticker"]] = raw * risk_factor
    total = sum(scores.values()) or 1.0
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    grade_of = {b["ticker"]: (b.get("grade") or {}).get("grade") for b in briefs}
    out = {}
    for i, (tk, sc) in enumerate(ranked):
        # 角色按证据等级命名：强语("候选池第1")只给 A/B；弱证据一律降为"观察/小仓"
        gr = grade_of.get(tk)
        if i == 0 and gr in ("A", "B"):
            role = "候选池第1（证据较强）"
        elif gr in ("A", "B"):
            role = "候选池靠前"
        elif gr == "C":
            role = "小仓试探"
        else:
            role = "观察/降权"
        out[tk] = {"weight": round(sc / total * 100, 1), "role": role,
                   "score": sc, "grade": gr}
    return out


# ---------------------------------------------------------------------------
# Markdown 导出
# ---------------------------------------------------------------------------
def _score_cn(s: float | None) -> str:
    """多周期分组得分 → 人话。"""
    if s is None:
        return "无样本"
    if s > 0.5:
        return "偏正✅"
    if s > 0:
        return "弱正"
    if s == 0:
        return "中性"
    return "偏负⚠️"


def _sig_mark(b: dict) -> str:
    be = b.get("engine_headline")
    return "✅" if (be and be["significant"] and be["excess"] > 0) else ""


def render_markdown(briefs: list[dict], weights: dict, horizon: int = 63) -> str:
    """把简报渲染成可读 Markdown（总览表 + 每票 + 权重 + 风控铁律）。"""
    asof = max((b["date"] for b in briefs), default="")
    L = [f"# 多票作战简报（数据截至 {asof}）\n",
         "> 校准式输出：价位带=区间+分布，目标/止损=按技术位规则推导的风险参考，非预测点位。",
         "> 新闻/基本面仅供人读、不入量化、含前视风险。权重为机械规则、非投资建议。\n",
         f"## 一屏总览（引擎周期 h{horizon}）\n",
         "| 标的 | 现价 | 入场裁决 | 合理入场位 | 本票风险 | 趋势/距历史高 | 引擎桶(参考) | 下次财报 |",
         "|---|---|---|---|---|---|---|---|"]
    for b in briefs:
        be = b.get("engine_headline")
        trap = "⚠️" if b.get("momentum_trap") else ""
        bk = (f"{be['bucket']} {be['median']:+.1%}(超额{be['excess']:+.1%}){_sig_mark(b)}{trap}"
              if be else "—")
        ent = b.get("entry") or {}
        verdict = f"{ent.get('grade_tag','')} {ent.get('grade','—')}".strip() or "—"
        ne = b.get("next_earnings") or "—"
        L.append(f"| {b['ticker']} | {b['price']:.1f} | {verdict} | {b.get('entry_sup_txt','—')} | "
                 f"{b.get('risk_txt','—')} | {b['trend']}/{b['drawdown']:.1%} | {bk} | {ne} |")
    L.append("")

    # 每票详情
    for b in briefs:
        be = b.get("engine_best")
        L.append(f"## {b['ticker']} {b['price']:.1f}\n")
        ent = b.get("entry")
        if ent:
            _wn = ent.get("win_now")
            _wntxt = f"现价买入历史胜率 {_wn*100:.0f}%" if (_wn is not None and _wn == _wn) else "现价买胜率样本不足"
            L.append(f"**入场裁决 {ent.get('grade_tag','')} {ent.get('grade','')}** · 合理入场位：{b.get('entry_sup_txt','—')}"
                     f" · {_wntxt} · 本票风险(离场)：{b.get('risk_txt','—')}。\n"
                     f"> _入场/离场裁决与「个股决策」「建仓扫描」同源(entry_confluence)；下方技术参考档/引擎桶为辅助视角。_\n")
        g = b.get("grade")
        if g:
            L.append(f"**证据等级 {g['grade']}**（{g['confidence']}置信，仓位封顶 {g['max_position_fraction']:.0%}）"
                     f"· {g['action']} — {g['meaning']}。\n"
                     f"> _证据等级=历史证据强度，非涨跌预测；负超额/CI跨0/有效样本不足已被硬封顶。_\n")
        hz = b.get("horizons")
        if hz and hz.get("action"):
            cf = "（⚠️多周期冲突）" if hz.get("conflict") else ""
            L.append(f"**多周期对账**{cf}：短期{_score_cn(hz.get('short'))} / 中期{_score_cn(hz.get('medium'))} / "
                     f"长期{_score_cn(hz.get('long'))} → {hz['action']}。\n")
        cons = b.get("consistency") or []
        if cons:
            L.append("> 🚧 **一致性告警**：" + "；".join(c["message"] for c in cons) + "\n")
        fnd = b.get("fundamentals") or {}
        if fnd.get("_data_quality"):
            L.append(f"> ⚠️ **数据质量**：{fnd['_data_quality']}\n")
        if be:
            L.append(f"**当前状态桶（{b.get('current_state_bucket','')}）**：h{horizon} 中位 {be['median']:+.1%}、"
                     f"胜率 {be['win_rate']:.0%}、超额 {be['excess']:+.1%}、"
                     f"CI[{be['ci_low']:+.1%},{be['ci_high']:+.1%}] "
                     f"{'不含0(显著)' if be['significant'] else '含0(证据弱)'}。\n")
        ev = b.get("engine_value")
        if ev:
            L.append(f"**估值低位桶（买便宜论据）**：中位 {ev['median']:+.1%}、超额 {ev['excess']:+.1%}、"
                     f"胜率 {ev['win_rate']:.0%}、盈亏比 "
                     f"{ev['reward_risk']:.2f}{'' if ev['reward_risk']==ev['reward_risk'] else ''} "
                     f"{'✅显著' if ev['significant'] else '证据弱'}。\n")
        if b.get("momentum_trap"):
            L.append("> ⚠️ **动量陷阱**：当前在回撤中，但回撤桶超额≤0——历史上逢跌买并不优于随机进场。"
                     "正确做法是等趋势确认/波动回落，而非抢这段回撤。\n")
        rp = b.get("regime_path")
        if rp:
            L.append(f"**📈📉 趋势全程分布（历史类比·分布非预测）**：{rp['headline']}")
            L += [f"- {ln}" for ln in rp["lines"]]
            L.append(f"- {rp['price_range']}")
            L.append(f"> _{rp['caveat']}_\n")
        if b["tranches"]:
            L.append("| 建仓档 | 价位 | 是什么 | 目标 | 止损 | 盈亏比RR | 引擎胜率 |")
            L.append("|---|---|---|---|---|---|---|")
            for t in b["tranches"]:
                rr = f"{t['rr']:.2f}" if t["rr"] == t["rr"] else "—"
                L.append(f"| {t['tier']} | {t['price']:.1f} | {t['what']} | "
                         f"{t['target']:.0f}({t['to_target_pct']:+.0%}) | "
                         f"{t['stop']:.0f}({t['to_stop_pct']:+.0%}) | {rr} | "
                         f"{t['engine_win_rate']:.0%} |")
            L.append("")
        # 操作预案（if-then）
        try:
            from analysis.playbook import build_playbook, format_playbook
            L.append(format_playbook(build_playbook(b)))
            L.append("")
        except Exception:  # noqa: BLE001
            pass
        es = b.get("earnings_stats")
        if es and es.get("n_events"):
            L.append(f"**财报反应(历史)**：财报日典型波动 ±{es['day_abs_move']['median']:.1%}；"
                     f"财报前{es['pre']}日 drift 中位 {es['pre_drift']['median']:+.1%}(市场提前消化)；"
                     f"财报后{es['post']}日 超预期 {es['post_beat']['median']:+.1%} / "
                     f"不及 {es['post_miss']['median']:+.1%}。")
        if b.get("next_earnings"):
            L.append(f"**下次财报**：{b['next_earnings']}（{b['days_to_earnings']} 天后）。")
        hl = b.get("highlights") or {}
        if hl:
            kv = " · ".join(f"{k} {v}" for k, v in hl.items() if not k.startswith("_"))
            L.append(f"**财报要点(免费结构化·仅供人读)**：{kv}")
        nf = b.get("fundamentals") or {}
        if nf:
            kv = " · ".join(f"{k} {v}" for k, v in nf.items() if not k.startswith("_"))
            L.append(f"**基本面快照(仅供人读)**：{kv}")
        nr = b.get("news_reason")
        if nr:
            L.append(f"**🧠 新闻启发式推理(非信号)**：{nr}")
        nw = b.get("news")
        if nw is not None and not nw.empty:
            na = b.get("news_analysis") or {}
            tag = {it["title"]: it["sentiment"] for it in na.get("items", [])}
            L.append(f"**全网新闻({nw['provider'].nunique()} 家媒体·仅线索)**：")
            for _, r in nw.head(6).iterrows():
                s = tag.get(r["title"], "")
                L.append(f"- [{s}] {r['date']} · {r['title']}（{r['provider']}）")
        L.append("")

    # 权重
    L.append("## 建议权重梯队（机械规则：引擎超额×显著性折扣×反波动率，非投资建议）\n")
    L.append("| 排序 | 标的 | 角色 | 权重 |")
    L.append("|---|---|---|---|")
    for i, (tk, w) in enumerate(sorted(weights.items(), key=lambda kv: kv[1]["weight"], reverse=True), 1):
        L.append(f"| {i} | {tk} | {w['role']} | {w['weight']:.0f}% |")
    L += ["",
          "## 风控铁律",
          "- 同赛道高相关：别同时满仓，分批把集中度风险摊到时间上。",
          "- 高波动/无回撤优势的票仓位放小、止损放宽。",
          "- 不在财报隔夜满仓押方向（gap 风险不对称）。",
          "- 引擎只吃历史条件分布；结构性变化(政策/资本开支证伪/判决)是表外尾部风险，模型吃不进。",
          "",
          "> 本简报为研究校准用途，非投资建议。"]
    return "\n".join(L)
