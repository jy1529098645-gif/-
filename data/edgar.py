"""SEC EDGAR 事件时间线（免费）。

抓取 SEC filings（10-K/10-Q/8-K 等）+ 财报日历，对齐客观价格反应。
**产出供人复盘理解，不作信号**（主观判断禁止流入回测）。

SEC 要求请求带 User-Agent；可在 config.edgar.user_agent 配置。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data import loader  # 触发 CA 证书设置

_CACHE = loader._CACHE
_UA = {"User-Agent": "quantlab-research personal-use contact@example.com"}

# 默认关注的实质性 filing（排除 Form 4 内幕买卖等噪音；可按需放开）
MATERIAL_FORMS = ["10-K", "10-Q", "8-K", "S-1", "424B", "DEF 14A"]


def _get(url: str, timeout: int = 20):
    from curl_cffi import requests as creq
    r = creq.get(url, headers=_UA, timeout=timeout)
    r.raise_for_status()
    return r


def _cik_map() -> dict[str, str]:
    """ticker → 10 位 CIK，带本地缓存。"""
    path = _CACHE / "sec_cik_map.json"
    if path.exists():
        return json.loads(path.read_text())
    data = json.loads(_get("https://www.sec.gov/files/company_tickers.json").text)
    m = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in data.values()}
    path.write_text(json.dumps(m))
    return m


def load_filings(ticker: str, forms: list[str] | None = None, limit: int = 80, refresh: bool = False) -> pd.DataFrame:
    """返回近期 filings：列 date / form / accession / doc。默认只留实质性 filing。"""
    path = _CACHE / f"edgar_{ticker.upper()}.parquet"
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
    else:
        cik = _cik_map().get(ticker.upper())
        if not cik:
            raise ValueError(f"未找到 {ticker} 的 CIK")
        j = json.loads(_get(f"https://data.sec.gov/submissions/CIK{cik}.json").text)
        rec = j["filings"]["recent"]
        df = pd.DataFrame({
            "date": pd.to_datetime(rec["filingDate"]),
            "form": rec["form"],
            "accession": rec["accessionNumber"],
            "doc": rec.get("primaryDocument", [""] * len(rec["form"])),
        }).sort_values("date", ascending=False)
        df.to_parquet(path)

    if forms is None:
        forms = MATERIAL_FORMS
    mask = df["form"].apply(lambda f: any(f.startswith(x) for x in forms))
    return df[mask].head(limit).reset_index(drop=True)


def event_timeline(ticker: str, prices: pd.Series, forms: list[str] | None = None,
                   horizons: tuple[int, ...] = (1, 5), include_earnings: bool = True) -> pd.DataFrame:
    """合并 filings + 财报，对齐每个事件的客观远期价格反应（次个交易日起）。仅复盘用。"""
    price = prices.dropna()
    pidx = price.index.values.astype("datetime64[ns]")
    pv = price.to_numpy(float)

    rows = []
    fil = load_filings(ticker, forms=forms)
    for _, r in fil.iterrows():
        rows.append({"date": r["date"], "type": "SEC", "label": r["form"]})
    if include_earnings:
        try:
            ed = loader.load_earnings_dates(ticker).dropna(subset=["Surprise(%)"])
            for d, er in ed.iterrows():
                rows.append({"date": pd.Timestamp(d), "type": "财报",
                             "label": f"EPS超预期 {er['Surprise(%)']:+.1f}%"})
        except Exception:  # noqa: BLE001
            pass

    ev = pd.DataFrame(rows)
    if ev.empty:
        return ev
    import numpy as np
    for h in horizons:
        reac = []
        for d in ev["date"].values.astype("datetime64[ns]"):
            r0 = int(np.searchsorted(pidx, d, side="right"))  # 次个交易日
            reac.append(pv[r0 + h - 1] / pv[r0 - 1] - 1.0 if 0 < r0 <= len(pv) - h else float("nan"))
        ev[f"reaction_{h}d"] = reac
    return ev.sort_values("date", ascending=False).reset_index(drop=True)
