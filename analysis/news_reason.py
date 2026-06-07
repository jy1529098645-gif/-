"""新闻启发式推理（规则式 NLP，非 LLM、非量化信号）。

边界（重要）：本模块对全网新闻**标题**做透明的规则归纳——金融情绪词典打分 + 主题关键词标注，
再与引擎**已算出的数字**(回撤桶超额/动量陷阱/估值低位桶/财报drift)交叉，合成一段**可溯源的推理**。
每个判断都能追到「数了几条某主题」或「引擎某个超额」。

⚠️ 这是**启发式合理分析**，不是 LLM 深度阅读、更不是买卖信号；新闻永远不进回测/量化。
若未来要 LLM 级正文推理，需接本地模型(ollama)或付费 API（当前不做）。
"""
from __future__ import annotations

import re

import pandas as pd

# 金融情绪词典（标题级，英文）
_POS = {
    "beat", "beats", "surge", "surges", "soar", "soars", "jump", "jumps", "rally", "rallies",
    "upgrade", "upgrades", "raises", "raise", "record", "strong", "growth", "grow", "outperform",
    "bullish", "buy", "gains", "gain", "tops", "boost", "boosts", "expand", "expands", "expansion",
    "breakthrough", "wins", "win", "approval", "approved", "partnership", "demand", "high", "highs",
    "rises", "rise", "climb", "climbs", "top", "best", "leads", "leading", "accelerate", "momentum",
    "profit", "profits", "beat estimates", "all-time",
}
_NEG = {
    "miss", "misses", "plunge", "plunges", "drop", "drops", "fall", "falls", "slump", "slumps",
    "downgrade", "downgrades", "cuts", "cut", "lawsuit", "sued", "probe", "investigation",
    "ban", "bans", "warning", "warns", "weak", "weakness", "decline", "declines", "bearish",
    "sell", "selloff", "loss", "losses", "layoff", "layoffs", "recall", "halt", "fraud", "slowdown",
    "concern", "concerns", "risk", "risks", "sinks", "sink", "tumble", "tumbles", "slips", "slip",
    "crash", "obliterated", "fears", "fear", "pressure", "scrutiny", "antitrust", "fine", "fined",
    "delay", "delays", "disappoint", "disappoints", "sluggish", "headwind", "headwinds",
}

# 主题关键词
_THEMES = {
    "earnings": ["earnings", "revenue", "eps", "guidance", "forecast", "outlook", "quarter",
                 "quarterly", "results", "beat", "miss", "margins"],
    "regulation": ["antitrust", "doj", "ftc", "eu ", "lawsuit", "sued", "probe", "investigation",
                   "regulator", "regulatory", "fine", "ban", "court", "settlement", "subpoena", "scrutiny"],
    "mna": ["acqui", "merger", "buyout", "takeover", "stake", "deal"],
    "capital": ["buyback", "repurchase", "dividend", "offering", "financing", "debt", "equity raise",
                "raises capital", "stock split", "split"],
    "rating": ["upgrade", "downgrade", "price target", "raises target", "cuts target", "rating",
               "analyst", "overweight", "underweight", "initiates", "reiterates"],
    "product_ai": ["ai", "chip", "gpu", "model", "launch", "cloud", "data center", "gemini",
                   "blackwell", "robotaxi", "iphone", "azure", "product", "rubin"],
    "management": ["ceo", "cfo", "resign", "steps down", "departure", "appoint", "board", "executive"],
    "macro": ["fed", "rates", "inflation", "tariff", "china", "export", "recession", "economy", "macro"],
}
_THEME_CN = {"earnings": "财报/指引", "regulation": "监管/诉讼", "mna": "并购", "capital": "融资/回购/分红",
             "rating": "评级/目标价", "product_ai": "产品/AI", "management": "高管变动", "macro": "宏观/政策"}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z\-']+", (text or "").lower())


def _score_item(title: str, summary: str = "") -> tuple[int, list[str]]:
    """单条新闻：情绪分(正词数−负词数) + 命中主题列表。"""
    text = f"{title} {summary}".lower()
    toks = set(_tokens(text))
    pos = len(toks & _POS)
    neg = len(toks & _NEG)
    themes = [t for t, kws in _THEMES.items() if any(k in text for k in kws)]
    return pos - neg, themes


def analyze_news(news: pd.DataFrame) -> dict:
    """对新闻 DataFrame 做情绪 + 主题归纳。返回聚合 dict + 每条标注。"""
    if news is None or news.empty:
        return {"n": 0}
    pos = neg = neu = 0
    theme_count: dict[str, int] = {}
    items = []
    for _, r in news.iterrows():
        sc, themes = _score_item(r.get("title", ""), r.get("summary", ""))
        lab = "正面" if sc > 0 else ("负面" if sc < 0 else "中性")
        pos += sc > 0
        neg += sc < 0
        neu += sc == 0
        for t in themes:
            theme_count[t] = theme_count.get(t, 0) + 1
        items.append({"date": r.get("date"), "title": r.get("title"), "provider": r.get("provider"),
                      "url": r.get("url"), "score": sc, "sentiment": lab, "themes": themes})
    n = len(items)
    net = (pos - neg) / n if n else 0.0
    tone = "偏多" if net > 0.15 else ("偏空" if net < -0.15 else "中性")
    top_themes = sorted(theme_count.items(), key=lambda kv: kv[1], reverse=True)
    return {"n": n, "pos": pos, "neg": neg, "neu": neu, "net_tone": net, "tone_label": tone,
            "themes": [(t, c) for t, c in top_themes], "theme_count": theme_count, "items": items}


def reason_paragraph(brief: dict) -> str:
    """把新闻归纳 + 引擎数字交叉，合成一段可溯源的启发式推理（中文）。"""
    na = brief.get("news_analysis") or {}
    if not na or na.get("n", 0) == 0:
        return "近端无足够新闻可供归纳。"

    es = brief.get("engine_state")           # 当前状态桶
    ev = brief.get("engine_value")           # 估值低位桶
    trap = brief.get("momentum_trap")
    tc = na.get("theme_count", {})
    has_reg = tc.get("regulation", 0) > 0
    has_down = any(("downgrade" in it["title"].lower() or "cuts target" in it["title"].lower())
                   for it in na["items"])
    fundamental_bad = (tc.get("earnings", 0) >= 2 and na["net_tone"] < 0) or has_reg

    P = []
    # 1) 情绪 + 主题
    themes_cn = "、".join(f"{_THEME_CN.get(t, t)}({c})" for t, c in na["themes"][:3]) or "无明显主题"
    P.append(f"近 {na['n']} 条全网新闻情绪**{na['tone_label']}**"
             f"（{na['pos']} 正 / {na['neg']} 空 / {na['neu']} 中，净 {na['net_tone']:+.0%}）；"
             f"主导主题：{themes_cn}。")

    # 2) 与引擎交叉的核心推断
    if trap:
        base = "引擎显示**回撤桶超额≤0（动量陷阱）**——历史上逢跌买不优于随机。"
        if fundamental_bad:
            tail = "叠加新闻含基本面/政策利空，更应谨慎，等利空证伪/确认。"
        elif na["net_tone"] < -0.15:
            tail = ("新闻偏空但主因更像情绪/宏观而非基本面破裂，这段回撤或属情绪冲洗；"
                    "但引擎不支持抢跌，宜等波动回落/趋势确认再加。")
        else:
            tail = ("新闻无明显基本面利空，回撤更像技术性/动量降温；"
                    "但引擎不支持抢跌，宜等波动回落/趋势确认再加。")
        core = base + tail
    elif ev and ev.get("significant") and ev.get("excess", 0) > 0:
        core = (f"引擎**估值低位桶历史显著正超额（{ev['excess']:+.1%}）**——"
                + ("新闻未见基本面破裂迹象，当前回调更可能是**可校准的回踩**，分批有统计背书。"
                   if not fundamental_bad
                   else "但新闻含基本面/政策利空，需对该超额打折，等利空证伪。"))
    elif es:
        core = (f"引擎当前状态桶超额 {es['excess']:+.1%}（"
                + ("证据弱、" if not es.get("significant") else "")
                + "倾斜有限）；新闻情绪" + na["tone_label"] + "，二者need合看，勿单凭其一。")
    else:
        core = "引擎该状态样本不足，难给统计倾斜；新闻仅作情绪参考。"
    P.append(core)

    # 3) 风险/催化补充
    extra = []
    if has_reg:
        extra.append("⚠️ 含**监管/诉讼**主题——结构性悬顶，引擎吃不进，属表外尾部风险。")
    if has_down:
        extra.append("近期出现**评级/目标价下调**，情绪面承压。")
    if tc.get("capital", 0) > 0:
        extra.append("含**融资/回购/分红**主题——融资短期或稀释、回购/分红通常偏正，看具体方向。")
    dte = brief.get("days_to_earnings")
    esr = brief.get("earnings_stats")
    if dte is not None and dte <= 21 and esr and esr.get("pre_drift"):
        extra.append(f"**下次财报临近（{dte} 天）**，历史财报前 drift 中位 "
                     f"{esr['pre_drift']['median']:+.1%}（市场惯于提前抢跑）。")
    if extra:
        P.append(" ".join(extra))

    P.append("_以上为新闻情绪/主题的**启发式归纳** + 引擎数字的合理推断，**非 LLM 深读、非买卖信号**；新闻不入量化。_")
    return " ".join(P)
