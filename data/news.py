"""免费新闻（全网检索）+ 基本面快照。

⚠️ 铁律边界：本模块数据**仅供人读的叙事补充，绝不入量化、不作回测/信号**。
- 新闻：**多源聚合，全网广度**——
    · Google News RSS（索引全网上千家媒体，免费无 key，主引擎）
    · yfinance .news（Yahoo 自家流，补充）
    · GDELT（全球新闻库，含外媒，可选；默认英文过滤）
  去重合并后按时间排序。质量/时效不稳，仅作"最近发生了什么"的线索。
- 基本面快照 / 财报要点：yfinance .info + 季度利润表的**当前**字段——带**前视/重述偏差**
  （最新快照，非 point-in-time），仅用于"现在长什么样"的展示，严禁做历史因子。

CA 修复依赖 data.loader 导入时设置的环境变量（中文路径下 curl 证书问题）。
"""
from __future__ import annotations

import re
import time
from urllib.parse import quote_plus

import pandas as pd

import data.loader  # noqa: F401  —— 触发 _ensure_ascii_ca_bundle() 的 CA 修复

_COLS = ["date", "title", "provider", "url", "summary"]

# 进程内 TTL 缓存：避免每次 app rerun / 重复调用都联网重抓同一只票的新闻。
# 命中条件=同 (ticker, limit, sources, query) 且未过期。Streamlit 端另有 @st.cache_data，
# 这里给非 Streamlit 调用（CLI / brief_cli / 测试）也兜一层。
_NEWS_TTL_SEC = 900  # 15 分钟
_NEWS_CACHE: dict[tuple, tuple[float, pd.DataFrame]] = {}


def clear_news_cache() -> None:
    """清空进程内新闻缓存（测试 / 强制刷新用）。"""
    _NEWS_CACHE.clear()

# 固定池（Mag7 + SPY）公司名，用于提升全网检索召回
COMPANY_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet Google", "AMZN": "Amazon",
    "NVDA": "Nvidia", "META": "Meta Platforms", "TSLA": "Tesla", "SPY": "S&P 500",
}


def _clean_html(s: str) -> str:
    return re.sub(r"<[^>]+>", " ", s or "").strip()


def _norm_title(t: str) -> str:
    """标题归一化（去媒体后缀、非字母数字），用于跨源去重。"""
    t = re.sub(r"\s+-\s+[^-]+$", "", t or "")  # 去掉 " - Publisher" 尾巴
    return re.sub(r"[^a-z0-9]", "", t.lower())[:80]


def _http_get(url: str, timeout: int = 18):
    from curl_cffi import requests as creq
    return creq.get(url, impersonate="chrome", timeout=timeout)


def _query_for(ticker: str) -> str:
    name = COMPANY_NAMES.get(ticker.upper(), "")
    return f"{ticker} {name} stock".strip()


def _google_news(query: str, limit: int = 20) -> list[dict]:
    """Google News RSS 全网检索。返回 dict 列表。失败返回 []。"""
    import xml.etree.ElementTree as ET

    url = (f"https://news.google.com/rss/search?q={quote_plus(query)}"
           "&hl=en-US&gl=US&ceid=US:en")
    try:
        r = _http_get(url)
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception:  # noqa: BLE001
        return []
    out = []
    for it in root.findall(".//item")[:limit]:
        title = it.findtext("title") or ""
        src_el = it.find("source")
        if src_el is None:
            src_el = it.find("{*}source")
        provider = src_el.text if src_el is not None else None
        # 标题尾部常是 " - 媒体名"
        if provider and title.endswith(f" - {provider}"):
            title = title[: -(len(provider) + 3)]
        date = it.findtext("pubDate")
        out.append({
            "date": (pd.to_datetime(date, errors="coerce").date() if date else None),
            "title": title.strip(),
            "provider": provider or "Google News",
            "url": it.findtext("link"),
            "summary": _clean_html(it.findtext("description") or "")[:280],
        })
    return out


def _yahoo_news(ticker: str, limit: int = 12) -> list[dict]:
    """yfinance .news（Yahoo 自家流）。失败返回 []。"""
    try:
        import yfinance as yf

        raw = yf.Ticker(ticker).news or []
    except Exception:  # noqa: BLE001
        return []
    out = []
    for item in raw[:limit]:
        c = item.get("content", item) if isinstance(item, dict) else {}
        if not isinstance(c, dict):
            continue
        title = c.get("title")
        if not title:
            continue
        prov = c.get("provider") or {}
        url_obj = c.get("canonicalUrl") or c.get("clickThroughUrl") or {}
        date = c.get("pubDate") or c.get("displayTime")
        out.append({
            "date": (pd.to_datetime(date, errors="coerce").date() if date else None),
            "title": title.strip(),
            "provider": prov.get("displayName") if isinstance(prov, dict) else str(prov),
            "url": url_obj.get("url") if isinstance(url_obj, dict) else url_obj,
            "summary": (c.get("summary") or c.get("description") or "")[:280],
        })
    return out


def _gdelt_news(query: str, limit: int = 15, english_only: bool = True) -> list[dict]:
    """GDELT 全球新闻库（含外媒）。english_only 时只取英文源。失败返回 []。"""
    q = query + (" sourcelang:english" if english_only else "")
    url = (f"https://api.gdeltproject.org/api/v2/doc/doc?query={quote_plus(q)}"
           f"&mode=ArtList&format=json&maxrecords={limit}&sort=datedesc")
    try:
        r = _http_get(url)
        r.raise_for_status()
        arts = r.json().get("articles", [])
    except Exception:  # noqa: BLE001
        return []
    out = []
    for a in arts:
        sd = a.get("seendate")
        out.append({
            "date": (pd.to_datetime(sd, errors="coerce").date() if sd else None),
            "title": (a.get("title") or "").strip(),
            "provider": a.get("domain") or "GDELT",
            "url": a.get("url"),
            "summary": "",
        })
    return out


def stock_news(ticker: str, limit: int = 12, sources=("google", "yahoo"),
               query: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    """全网多源聚合新闻：列 date/title/provider/url/summary。去重合并、按时间倒序。

    sources：可含 'google'(全网主引擎)/'yahoo'/'gdelt'(全球外媒)。失败的源静默跳过。
    use_cache：命中 15 分钟内的进程缓存则直接返回（避免重复联网）；置 False 强制刷新。
    """
    key = (ticker.upper(), int(limit), tuple(sources), query)
    if use_cache:
        hit = _NEWS_CACHE.get(key)
        if hit is not None and (time.time() - hit[0]) < _NEWS_TTL_SEC:
            return hit[1].copy()

    q = query or _query_for(ticker)
    rows: list[dict] = []
    if "google" in sources:
        rows += _google_news(q, limit=max(20, limit * 2))
    if "yahoo" in sources:
        rows += _yahoo_news(ticker, limit=limit)
    if "gdelt" in sources:
        rows += _gdelt_news(q, limit=limit)

    # 去重（按归一化标题），保留首次出现（google 优先在前）
    seen, uniq = set(), []
    for r in rows:
        if not r.get("title"):
            continue
        k = _norm_title(r["title"])
        if not k or k in seen:
            continue
        seen.add(k)
        uniq.append(r)

    df = pd.DataFrame(uniq, columns=_COLS)
    if not df.empty:
        df = df.sort_values("date", ascending=False, na_position="last").head(limit).reset_index(drop=True)
    if use_cache:
        _NEWS_CACHE[key] = (time.time(), df.copy())
    return df


# 展示用基本面字段 → 中文标签
_FUND_FIELDS = {
    "marketCap": "市值",
    "trailingPE": "TTM 市盈率",
    "forwardPE": "前瞻市盈率",
    "revenueGrowth": "营收同比",
    "earningsGrowth": "盈利同比",
    "profitMargins": "净利率",
    "grossMargins": "毛利率",
    "returnOnEquity": "ROE",
    "dividendYield": "股息率",
    "beta": "Beta",
    "targetMeanPrice": "分析师目标均价",
    "recommendationKey": "评级",
    "numberOfAnalystOpinions": "覆盖分析师数",
}
_PCT_FIELDS = {"revenueGrowth", "earningsGrowth", "profitMargins", "grossMargins",
               "returnOnEquity", "dividendYield"}


def fundamentals_snapshot(ticker: str) -> dict:
    """返回当前基本面快照 dict：{字段中文: 值}。失败返回 {}。

    ⚠️ 仅当前快照（含前视/重述偏差），供人读，不入量化。
    """
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info or {}
    except Exception:  # noqa: BLE001
        return {}

    # 字段体检：归一(股息率单位漂移) + 标记可疑(越界字段不展示)
    from analysis.engine_discipline import sanity_check_fundamentals
    chk = sanity_check_fundamentals(info)
    clean, suspicious, warnings = chk["clean"], chk["suspicious"], chk["warnings"]

    out = {}
    for key, label in _FUND_FIELDS.items():
        if key in suspicious:
            out[label] = "N/A ⚠️(疑似异常，已剔除)"
            continue
        v = clean.get(key, info.get(key))  # 体检过的字段用归一值，其余用原值
        if v is None:
            continue
        if key == "marketCap":
            out[label] = f"${v/1e9:.0f}B" if v >= 1e9 else f"${v/1e6:.0f}M"
        elif key in _PCT_FIELDS:
            out[label] = f"{v:.1%}"
        elif key in ("trailingPE", "forwardPE", "beta"):
            out[label] = f"{v:.1f}"
        elif key == "targetMeanPrice":
            out[label] = f"{v:.1f}"
        else:
            out[label] = str(v)
    if warnings:
        out["_data_quality"] = "；".join(warnings)
    out["_disclaimer"] = "yfinance 当前快照，含前视/重述偏差，仅供人读、不入量化。"
    return out


def fetch_article(url: str, timeout: int = 15) -> str:
    """抓取并抽取新闻正文纯文本。trafilatura 优先，回退 bs4 <p>。失败返回 ""。

    注：Google News 的 link 多为重定向/JS 页，正文抽取常失败；Yahoo/GDELT/直链效果好。"""
    if not url:
        return ""
    try:
        r = _http_get(url, timeout=timeout)
        if r.status_code >= 400 or not r.content:
            return ""
        html = r.text
    except Exception:  # noqa: BLE001
        return ""
    try:
        import trafilatura

        txt = trafilatura.extract(html, include_comments=False, include_tables=False,
                                  favor_precision=True)
        if txt and len(txt) > 200:
            return txt.strip()
    except Exception:  # noqa: BLE001
        pass
    try:  # 回退：bs4 取 <p>
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        ps = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        txt = "\n".join(x for x in ps if len(x) > 40)
        return txt.strip()
    except Exception:  # noqa: BLE001
        return ""


_SALIENT = re.compile(r"\$|%|\b(billion|million|trillion|revenue|earnings|guidance|forecast|"
                      r"raise[sd]?|cut[s]?|lawsuit|probe|acqui|buyback|dividend|upgrade|downgrade|"
                      r"target|growth|beat|miss|surge|plunge|launch|stake|deal)\b", re.I)


def article_highlights(text: str, max_sentences: int = 6) -> list[str]:
    """从正文抽「关键句」：含金额/百分比/数字事件词的句子，按信息量排序取前若干。"""
    if not text:
        return []
    # 粗分句（句号/换行/分号），保留含数字或事件词的句子
    sents = re.split(r"(?<=[.!?])\s+|\n+", text)
    scored = []
    for s in sents:
        s = s.strip()
        if not (25 <= len(s) <= 320):
            continue
        hits = len(_SALIENT.findall(s))
        has_num = bool(re.search(r"\d", s))
        if hits == 0 and not has_num:
            continue
        scored.append((hits + (1 if has_num else 0), s))
    scored.sort(key=lambda x: x[0], reverse=True)
    out, seen = [], set()
    for _, s in scored:
        k = s[:60].lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
        if len(out) >= max_sentences:
            break
    return out


def read_articles(news: pd.DataFrame, limit: int = 5) -> list[dict]:
    """对前 limit 条新闻抓正文 + 抽关键句。返回 [{title,provider,url,date,excerpts,n_chars}]。

    ⚠️ 正文仅供人读、不入量化；含数字句来自原文抽取，未做事实校验。"""
    if news is None or news.empty:
        return []
    out = []
    for _, r in news.head(limit).iterrows():
        body = fetch_article(r.get("url", ""))
        ex = article_highlights(body)
        if not ex:
            continue
        out.append({"title": r.get("title"), "provider": r.get("provider"), "url": r.get("url"),
                    "date": r.get("date"), "excerpts": ex, "n_chars": len(body)})
    return out


def financial_highlights(ticker: str) -> dict:
    """结构化财报要点（免费）：最近季营收/净利同比 + 分析师目标上行 + 评级。

    ⚠️ 走 yfinance 季度利润表 + .info，**含重述/前视偏差**，仅供人读、不入量化。
    用于自动填「已发生事件」的硬数字（如营收 +22%、净利 +81%），而非叙事。"""
    out: dict = {}
    try:
        import yfinance as yf

        t = yf.Ticker(ticker)
        info = t.info or {}
        tgt = info.get("targetMeanPrice")
        cur = info.get("currentPrice") or info.get("regularMarketPrice")
        if tgt and cur:
            out["分析师目标"] = f"{tgt:.0f}（较现价{tgt/cur-1:+.0%}）"
        if info.get("recommendationKey"):
            out["评级"] = f"{info['recommendationKey']}（{info.get('numberOfAnalystOpinions','?')}家）"

        q = t.quarterly_income_stmt

        def _yoy(*names):
            for nm in names:
                if nm in q.index:
                    s = q.loc[nm].dropna()
                    if len(s) >= 5 and s.iloc[4] not in (0, None):
                        return s.index[0], float(s.iloc[0]), float(s.iloc[0] / s.iloc[4] - 1)
            return None

        rev = _yoy("Total Revenue")
        if rev:
            qd, val, g = rev
            out["最近季营收"] = f"{val/1e9:.1f}B（{str(qd)[:7]} 同比 {g:+.0%}）"
        ni = _yoy("Net Income", "Net Income From Continuing Operation Net Minority Interest")
        if ni:
            out["净利同比"] = f"{ni[2]:+.0%}"
    except Exception:  # noqa: BLE001
        return out
    if out:
        out["_disclaimer"] = "yfinance 季度财报/快照（可能重述），仅供人读、不入量化。"
    return out
