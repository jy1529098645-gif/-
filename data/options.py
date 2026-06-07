"""期权当前状态快照（免费 yfinance 期权链）。

⚠️ 明确边界：只反映**当前**快照，**不可回测**（历史期权数据需付费，已砍）。
涉及 IV 的 regime 条件应改用现货历史波动率作 proxy 并标注"非真 IV"。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import loader  # 触发 CA 证书设置


def options_snapshot(ticker: str, n_expiries: int = 3) -> dict:
    """返回当前期权快照：现价、ATM IV、Put/Call(OI)、最大未平仓行权价（call/put 磁吸位）。"""
    import yfinance as yf

    tk = yf.Ticker(ticker)
    expiries = list(tk.options or [])[:n_expiries]
    if not expiries:
        raise ValueError(f"{ticker}: 无可用期权链")

    spot = float(tk.history(period="5d")["Close"].dropna().iloc[-1])

    calls_all, puts_all = [], []
    for ex in expiries:
        ch = tk.option_chain(ex)
        c, p = ch.calls.copy(), ch.puts.copy()
        c["expiry"], p["expiry"] = ex, ex
        calls_all.append(c)
        puts_all.append(p)
    calls = pd.concat(calls_all, ignore_index=True)
    puts = pd.concat(puts_all, ignore_index=True)

    # ATM IV：最接近现价的若干行权价的看涨/看跌 IV 均值
    def _atm_iv(df):
        d = df.dropna(subset=["impliedVolatility"]).copy()
        if d.empty:
            return float("nan")
        d["dist"] = (d["strike"] - spot).abs()
        return float(d.nsmallest(5, "dist")["impliedVolatility"].mean())

    call_oi = float(calls["openInterest"].fillna(0).sum())
    put_oi = float(puts["openInterest"].fillna(0).sum())
    pcr = put_oi / call_oi if call_oi > 0 else float("nan")

    def _max_oi_strike(df):
        d = df.dropna(subset=["openInterest"])
        return float(d.loc[d["openInterest"].idxmax(), "strike"]) if not d.empty else float("nan")

    return {
        "ticker": ticker,
        "spot": spot,
        "expiries": expiries,
        "atm_iv_call": _atm_iv(calls),
        "atm_iv_put": _atm_iv(puts),
        "put_call_oi_ratio": pcr,
        "max_oi_call_strike": _max_oi_strike(calls),   # 上方磁吸/压力参考
        "max_oi_put_strike": _max_oi_strike(puts),     # 下方磁吸/支撑参考
        "calls": calls[["strike", "openInterest", "impliedVolatility", "expiry"]],
        "puts": puts[["strike", "openInterest", "impliedVolatility", "expiry"]],
    }
