"""行业动向聚合（半导体 / 科技 等板块）——校准而非预测。

聚合一篮子板块标的的**真实可观测动向**：宽度(多少%在200线上)、相对强度(板块vs大盘·龙头vs板块)、
内部相关性、回撤/波动分布、营收/盈利成长与估值的横截面中位，并用历史类比给**板块级情景分布**
(像今天这种板块状态，历史后来再跌/反弹/见底/修复的分布)。

铁律（与全工具一致）：
- 只给**可观测动向 + 历史类比分布**，不判牛熊、不给点位、不预测拐点；任何"未来情景"都是
  历史频率/区间(带 N)，标注"非预测"。
- 新闻/基本面仅供人读、不入量化（联网拉取，显式标注前视/陈旧风险）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 板块成分（尽量用本地有缓存的标的；缺数据的自动跳过）
SECTORS: dict[str, list[str]] = {
    "🔌 半导体": ["NVDA", "AMD", "AVGO", "TSM", "ASML", "MU", "INTC", "QCOM", "TXN",
                "ARM", "MRVL", "SMCI", "NXPI", "ADI", "AMAT", "SOXX", "SMH"],
    "🖥️ 科技(大盘)": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "ORCL", "CRM",
                   "ADBE", "NOW", "INTU", "AMD", "NFLX", "CSCO", "IBM", "QCOM", "AVGO", "PLTR", "UBER"],
}


def _ew_index(panel: pd.DataFrame) -> pd.Series:
    """等权板块净值指数（日再平衡，从首日=1）。"""
    r = panel.ffill().pct_change().mean(axis=1)
    return (1 + r.fillna(0)).cumprod()


def _pct_rank(s: pd.Series, win: int = 756) -> float:
    """当前值在过去 win 日的分位（0–1，因果）。"""
    x = s.dropna()
    if len(x) < 60:
        return float("nan")
    w = x.tail(win)
    return float((w.iloc[-1] >= w).mean())


def sector_dashboard(tickers: list[str], start: str, end: str | None,
                     spy: pd.Series | None = None, ma: int = 200) -> dict:
    """板块动向仪表盘（全部来自价格，无前视）。返回结构化 dict 供前端渲染。"""
    from data import loader
    from analysis import fragility as fg
    from regime import observables as ob

    px = loader.load_prices(tickers, start, end)
    if isinstance(px, pd.Series):
        px = px.to_frame()
    cols = [c for c in px.columns if px[c].notna().sum() > 250]
    px = px[cols].dropna(how="all")
    if px.shape[1] < 3:
        return {"available": False, "reason": "成分数据不足"}
    if spy is None:
        spy = loader.load_prices(["SPY"], start, end)["SPY"]
    spy = spy.dropna()

    # —— 宽度：多少% 在自身 200 日线上方 + 历史分位（越低越脆弱，领先指数）——
    breadth = fg.breadth_above_ma(px, ma)
    br_now = float(breadth.dropna().iloc[-1]) if breadth.notna().any() else float("nan")
    br_pct = float(fg.rolling_pct_rank(breadth).dropna().iloc[-1]) if breadth.notna().sum() > 300 else float("nan")

    # —— 相对强度：板块等权 vs SPY（近 63/252 日超额）——
    ew = _ew_index(px)
    rel = (ew / ew.reindex(spy.index).ffill()) / (spy / spy.iloc[0])  # 仅看趋势形状
    def _ret(s, n):
        s = s.dropna()
        return float(s.iloc[-1] / s.iloc[-n - 1] - 1) if len(s) > n else float("nan")
    rs_63 = _ret(ew, 63) - _ret(spy, 63)
    rs_252 = _ret(ew, 252) - _ret(spy, 252)

    # —— 内部相关性：高=系统性β主导(分散无效)；低=个股分化(选股行情)——
    try:
        corr = float(ob.mutual_correlation(px, 63).dropna().iloc[-1])
    except Exception:  # noqa: BLE001
        corr = float("nan")

    # —— 龙头 vs 落后：近 63 日各成分收益的分散度 ——
    r63 = {c: _ret(px[c], 63) for c in cols}
    r63 = {k: v for k, v in r63.items() if v == v}
    ranked = sorted(r63.items(), key=lambda kv: kv[1], reverse=True)
    leaders = ranked[:3]; laggards = ranked[-3:]
    dispersion = float(np.std(list(r63.values()))) if r63 else float("nan")

    # —— 每标的快照：距200线 / 距1年高 / 波动分位 / 近63日相对板块 ——
    rows = []
    sec_r63 = _ret(ew, 63)
    for c in cols:
        s = px[c].dropna()
        ma200 = s.rolling(200, min_periods=100).mean().iloc[-1]
        hi = s.rolling(252, min_periods=120).max().iloc[-1]
        rv = s.pct_change().rolling(21, min_periods=10).std() * np.sqrt(252)
        rows.append({
            "标的": c,
            "距200线": float(s.iloc[-1] / ma200 - 1) if ma200 == ma200 else float("nan"),
            "距1年高": float(s.iloc[-1] / hi - 1) if hi == hi else float("nan"),
            "波动分位": _pct_rank(rv),
            "近63日": r63.get(c, float("nan")),
            "vs板块": (r63.get(c, float("nan")) - sec_r63) if r63.get(c) == r63.get(c) else float("nan"),
        })
    table = pd.DataFrame(rows).sort_values("近63日", ascending=False)

    return {
        "available": True, "n": len(cols), "members": cols,
        "breadth": br_now, "breadth_pctile": br_pct,
        "rs_63": rs_63, "rs_252": rs_252, "corr": corr, "dispersion": dispersion,
        "leaders": leaders, "laggards": laggards,
        "ew_index": ew, "table": table,
        "asof": str(px.index[-1].date()),
    }


def sector_state_label(d: dict) -> str:
    """一句话刻画板块当前状态（纯描述，不预测）。"""
    if not d.get("available"):
        return "板块数据不足。"
    br = d["breadth"]; brp = d["breadth_pctile"]; rs = d["rs_252"]; corr = d["corr"]
    parts = []
    parts.append(f"宽度 {br:.0%} 在200线上"
                 + (f"（历史分位 {brp:.0%}{'·偏脆弱' if brp == brp and brp < 0.2 else ''}）" if brp == brp else ""))
    parts.append(f"近1年相对大盘 {rs:+.0%}（{'领涨' if rs > 0.05 else '落后' if rs < -0.05 else '同步'}）")
    if corr == corr:
        parts.append(f"内部相关性 {corr:.2f}（{'系统性β主导·分散无效' if corr > 0.6 else '个股分化·选股行情' if corr < 0.4 else '中性'}）")
    return "；".join(parts) + "。"


def sector_scenarios(d: dict, window: int = 504) -> dict | None:
    """板块级历史类比情景分布：用等权指数跑 regime_path_distribution。
    像今天这种板块深度，历史后来再跌/反弹/见底/修复的分布（非预测）。"""
    if not d.get("available"):
        return None
    from analysis.analogs import regime_path_distribution, format_regime_path
    ew = d["ew_index"].dropna()
    if len(ew) < 600:
        return None
    rd = regime_path_distribution(ew, window=window)
    return format_regime_path(rd)


def _parse_num(v):
    """把 '85.2%' / '$4928B' / '31.1' / '2.2' 解析成 float（百分号→小数；货币/倍数去单位）。"""
    if v is None:
        return float("nan")
    s = str(v).strip()
    is_pct = s.endswith("%")
    s = s.lstrip("$").rstrip("%").rstrip("BxX倍").replace(",", "").strip()
    try:
        x = float(s)
        return x / 100.0 if is_pct else x
    except ValueError:
        return float("nan")


def sector_fundamentals(tickers: list[str]) -> dict:
    """横截面基本面动向（营收/盈利成长、估值的成分中位）——仅供人读、不入量化、含前视/陈旧。

    fundamentals_snapshot 的键是中文格式串（'营收同比'='85.2%'、'TTM 市盈率'='31.1'）。"""
    from data import news as nws
    rg, eg, pe = [], [], []
    n_ok = 0
    for t in tickers:
        try:
            f = nws.fundamentals_snapshot(t)
        except Exception:  # noqa: BLE001
            continue
        if not f:
            continue
        n_ok += 1
        for key, bucket in (("营收同比", rg), ("盈利同比", eg), ("TTM 市盈率", pe)):
            x = _parse_num(f.get(key))
            if x == x:
                bucket.append(x)
    med = lambda a: float(np.median(a)) if a else float("nan")
    return {"n": n_ok, "rev_growth_med": med(rg), "earn_growth_med": med(eg), "pe_med": med(pe),
            "n_growth": len(rg), "n_pe": len(pe)}


def sector_news_themes(tickers: list[str], limit_per: int = 6, n_tickers: int = 12,
                       sources=("google", "yahoo", "gdelt")) -> dict:
    """联网深读：拉板块龙头新闻(含 GDELT 全球外媒) → **板块级主题/情绪聚合**（仅线索·不入量化·含前视）。

    汇总各成分的主题计数(财报/监管/并购/AI产品/宏观…)成"板块主导主题"，再给情绪天平。慢，按钮触发。
    """
    from data import news as nws
    from analysis.news_reason import analyze_news, _THEME_CN
    items, providers = [], set()
    pos = neg = 0
    theme_total: dict[str, int] = {}
    per_ticker_tone: dict[str, int] = {}
    for t in tickers[:n_tickers]:
        try:
            df = nws.stock_news(t, limit=limit_per, sources=sources)
        except Exception:  # noqa: BLE001
            continue
        if df is None or df.empty:
            continue
        an = analyze_news(df) or {}
        per_ticker_tone[t] = int(an.get("tone", 0))
        for th, c in an.get("theme_count", {}).items():
            theme_total[th] = theme_total.get(th, 0) + c
        meta = {it["title"]: it for it in an.get("items", [])}
        for _, r in df.iterrows():
            m = meta.get(r["title"], {})
            s = m.get("sentiment", "")
            pos += (s == "利好"); neg += (s == "利空")
            providers.add(r.get("provider", ""))
            items.append({"ticker": t, "date": str(r.get("date", "")), "title": r.get("title", ""),
                          "sentiment": s, "themes": [_THEME_CN.get(x, x) for x in m.get("themes", [])],
                          "url": r.get("url", "")})
    items.sort(key=lambda x: x["date"], reverse=True)
    top_themes = sorted(theme_total.items(), key=lambda kv: kv[1], reverse=True)
    top_themes_cn = [(_THEME_CN.get(t, t), c) for t, c in top_themes[:5]]
    # 情绪偏多/偏空的成分
    hot_pos = sorted([(k, v) for k, v in per_ticker_tone.items() if v > 0], key=lambda kv: kv[1], reverse=True)[:4]
    hot_neg = sorted([(k, v) for k, v in per_ticker_tone.items() if v < 0], key=lambda kv: kv[1])[:4]
    return {"n": len(items), "providers": len(providers), "pos": pos, "neg": neg,
            "top_themes": top_themes_cn, "hot_pos": hot_pos, "hot_neg": hot_neg,
            "items": items[:30]}
