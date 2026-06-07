"""信号挖掘（Phase G）：量化对比建仓看涨信号 / 止盈风险信号。

铁律（建仓作战室）：只校准、不预测。所有输出是「历史倾斜 + 分布 + N + CI」，
绝不给目标价/单一概率。多信号扫描=多重检验，故强制：
- 每个信号对比该股**无条件基准**，超额用 block bootstrap 给 CI 与 p 值；
- 跨 K 个信号做 **Benjamini–Hochberg FDR** 校正，标出折扣后是否仍显著；
- 报独立事件数 N（连续在状态内算一次），小 N 不可信。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factors import signals as sg
from regime.conditional_returns import _count_independent_events, _path_min_return

# ---------------------------------------------------------------------------
# 候选状态（全部因果、无前视）
# ---------------------------------------------------------------------------
def _features(price: pd.Series) -> pd.DataFrame:
    p = price.dropna()
    ma50 = p.rolling(50, min_periods=25).mean()
    ma200 = p.rolling(200, min_periods=100).mean()
    dist200 = p / ma200 - 1.0
    high20 = p.rolling(20, min_periods=10).max()
    dip20 = p / high20 - 1.0
    rsi = sg.rsi(p, 14)
    high252 = p.rolling(252, min_periods=120).max()
    rv = p.pct_change().rolling(21, min_periods=10).std()
    rv_pct = rv.rolling(252, min_periods=120).apply(lambda x: (x[-1] >= x).mean(), raw=True)
    dd = p / p.cummax() - 1.0
    return pd.DataFrame({"p": p, "ma50": ma50, "ma200": ma200, "dist200": dist200,
                         "dip20": dip20, "rsi": rsi, "high252": high252,
                         "rv_pct": rv_pct, "dd": dd}, index=p.index)


def entry_states(f: pd.DataFrame) -> dict[str, pd.Series]:
    """看涨/建仓候选（True=处于该状态）。"""
    return {
        "上升趋势中的回调": (f["dist200"] > 0) & (f["dip20"] <= -0.08),
        "RSI超卖": f["rsi"] < 35,
        "深度回调(>15%)": f["dip20"] <= -0.15,
        "回踩50日线": (f["dist200"] > 0) & (f["p"] <= f["ma50"] * 1.01) & (f["p"] >= f["ma50"] * 0.97),
        "金叉后站稳": (f["ma50"] > f["ma200"]) & (f["p"] > f["ma50"]),
        "52周新高突破": f["p"] >= f["high252"].shift(1) * 0.999,
    }


def risk_states(f: pd.DataFrame) -> dict[str, pd.Series]:
    """止盈/风险候选（True=该状态下未来下行风险/回撤可能更大）。"""
    stretch_pct = f["dist200"].rolling(504, min_periods=200).apply(lambda x: (x[-1] >= x).mean(), raw=True)
    return {
        "高波动": f["rv_pct"] >= 0.70,
        "跌破200日线": f["dist200"] <= 0,
        "RSI超买": f["rsi"] > 75,
        "极度拉伸": stretch_pct >= 0.85,
        "已深陷回撤(>20%)": f["dd"] <= -0.20,
    }


# ---------------------------------------------------------------------------
# 统计：block bootstrap + FDR
# ---------------------------------------------------------------------------
def _block_boot(arr: np.ndarray, block: int, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    L = len(arr)
    block = max(1, min(block, L))
    nb = int(np.ceil(L / block))
    starts = rng.integers(0, L, size=(n, nb))
    off = np.arange(block)
    out = np.empty(n)
    for i in range(n):
        idx = (starts[i][:, None] + off[None, :]).ravel()[:L] % L
        out[i] = np.median(arr[idx])
    return out


def _bh_fdr(pvals: list[float], q: float = 0.10) -> list[bool]:
    """Benjamini–Hochberg：返回各 p 是否通过 FDR=q。"""
    m = len(pvals)
    order = np.argsort(pvals)
    passed = [False] * m
    thresh = 0
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= q * rank / m:
            thresh = rank
    for rank, idx in enumerate(order, start=1):
        if rank <= thresh:
            passed[idx] = True
    return passed


def scan(tickers, start="2014-01-01", end=None, horizon=63, dd_h=21,
         n_boot=400, fdr_q=0.10, seed=0) -> pd.DataFrame:
    """扫描所有候选信号，池化跨票，给超额 + CI + p + FDR 校正 + 未来浮亏。"""
    from data import loader

    prices = {t: loader.load_prices([t], start, end)[t].dropna() for t in tickers}
    feats = {t: _features(prices[t]) for t in tickers}
    fwd = {t: prices[t].shift(-horizon) / prices[t] - 1.0 for t in tickers}
    base = {t: float(fwd[t].dropna().median()) for t in tickers}
    pmin = {t: _path_min_return(prices[t], dd_h) for t in tickers}

    catalog = [("entry", entry_states), ("risk", risk_states)]
    rows = []
    for kind, fn in catalog:
        for t in tickers:
            feats[t]  # ensure
        # 收集所有该类信号
        sig_names = list(fn(feats[tickers[0]]).keys())
        for sname in sig_names:
            excess_parts, dd_parts, fwd_parts, n_ev = [], [], [], 0
            for t in tickers:
                st = fn(feats[t])[sname].reindex(prices[t].index).fillna(False)
                f = fwd[t][st].dropna()
                if len(f):
                    excess_parts.append(f.to_numpy() - base[t])  # 相对该股基准的超额
                    fwd_parts.append(f.to_numpy())
                    dd_parts.append(pmin[t][st].dropna().to_numpy())
                    n_ev += _count_independent_events(st)
            if not excess_parts:
                continue
            ex = np.concatenate(excess_parts)
            fw = np.concatenate(fwd_parts)
            dn = np.concatenate(dd_parts) if dd_parts else np.array([np.nan])
            if len(ex) < 20:
                continue
            boot = _block_boot(ex, block=horizon, n=n_boot, seed=seed)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            p = 2.0 * min((boot > 0).mean(), (boot < 0).mean())
            rows.append({
                "kind": kind, "signal": sname, "n_events": n_ev, "n_days": int(len(ex)),
                "median_fwd": float(np.median(fw)), "win_rate": float((fw > 0).mean()),
                "excess": float(np.median(ex)), "ci_low": float(lo), "ci_high": float(hi),
                "pval": float(p), "fwd_drawdown": float(np.nanmedian(dn)),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["sig_raw"] = (df["ci_low"] > 0) | (df["ci_high"] < 0)
    df["sig_fdr"] = _bh_fdr(df["pval"].tolist(), q=fdr_q)
    return df.sort_values(["kind", "excess"], ascending=[True, False]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 建仓分 / 风险分（G2 / G3）——历史倾斜，非预测
# ---------------------------------------------------------------------------
def humanize_scan(df: pd.DataFrame):
    """把扫描原始表翻成"人话"显示表 + 一句白话总结。返回 (display_df, summary)。"""
    if df is None or df.empty:
        return pd.DataFrame(), "没有足够样本的信号。"

    def verdict(r):
        arrow = "↑偏涨" if r["excess"] > 0 else "↓偏弱"
        if r["sig_fdr"]:
            badge = "✅ 稳健显著"
        elif r["sig_raw"]:
            badge = "🟡 边缘显著"
        else:
            badge = "⚪ 不显著"
        return f"{badge} · {arrow}"

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "类别": "📈 建仓信号" if r["kind"] == "entry" else "⚠️ 风险信号",
            "信号(状态)": r["signal"],
            "历史次数": int(r["n_events"]),
            "胜率": f"{r['win_rate']:.0%}",
            "未来中位涨幅": f"{r['median_fwd']:+.1%}",
            "比闭眼买多/少": f"{r['excess']:+.1%}",
            "95%可信区间": f"[{r['ci_low']:+.0%}, {r['ci_high']:+.0%}]",
            "结论": verdict(r),
            "进场后典型浮亏": f"{r['fwd_drawdown']:+.1%}",
        })
    disp = pd.DataFrame(rows)

    # 白话总结：最强建仓倾斜 / 最该回避 / 是否有统计显著
    ent = df[df["kind"] == "entry"].sort_values("excess", ascending=False)
    sig = df[df["sig_raw"]]
    parts = []
    if not ent.empty:
        top = ent.iloc[0]
        parts.append(f"历史上**「{top['signal']}」时建仓倾斜最强**（比闭眼买多 {top['excess']:+.1%}、胜率 {top['win_rate']:.0%}）")
        worst = ent.iloc[-1]
        if worst["excess"] < 0:
            parts.append(f"**「{worst['signal']}」反而偏弱**（{worst['excess']:+.1%}，少追）")
    if sig.empty:
        parts.append("但**没有一个信号在统计上达到显著**（可信区间都跨 0）——说明在这只票上择时 edge 很微弱，别当圣杯，仅作分批建仓的倾斜参考")
    else:
        names = "、".join(sig["signal"].head(3))
        parts.append(f"其中 **{names}** 达到边缘显著（可信区间不跨 0）")
    return disp, "；".join(parts) + "。"


def _pct_rank(s: pd.Series) -> pd.Series:
    return s.rolling(504, min_periods=120).apply(lambda x: (x[-1] >= x).mean(), raw=True)


def score_series(ticker: str, start="2014-01-01", end=None,
                 entry_weights: dict | None = None, risk_weights: dict | None = None) -> pd.DataFrame:
    """单股逐日「建仓分」「风险分」(0–100)。

    建仓分 = 当日处于的看涨信号(按其历史正超额加权)之和的历史分位。
    风险分 = 当日处于的风险信号(按其未来浮亏严重度加权)之和的历史分位。
    纯历史倾斜，不预测、不给点位。
    """
    from data import loader

    p = loader.load_prices([ticker], start, end)[ticker].dropna()
    f = _features(p)
    es, rs = entry_states(f), risk_states(f)

    ew = entry_weights or {k: 1.0 for k in es}
    rw = risk_weights or {k: 1.0 for k in rs}
    raw_e = sum(es[k].astype(float) * max(0.0, ew.get(k, 0.0)) for k in es)
    raw_r = sum(rs[k].astype(float) * max(0.0, rw.get(k, 0.0)) for k in rs)
    bull = (_pct_rank(raw_e) * 100).rename("建仓分")
    risk = (_pct_rank(raw_r) * 100).rename("风险分")
    return pd.concat([p.rename("price"), bull, risk], axis=1)


def cross_section_today(tickers, start="2014-01-01", end=None,
                        entry_weights=None, risk_weights=None) -> pd.DataFrame:
    """七股今日综合：当前建仓分 / 风险分排名（相对，非买卖指令）。"""
    rows = []
    for t in tickers:
        s = score_series(t, start, end, entry_weights, risk_weights).dropna(subset=["建仓分", "风险分"])
        if s.empty:
            continue
        last = s.iloc[-1]
        rows.append({"标的": t, "建仓分": round(float(last["建仓分"]), 0),
                     "风险分": round(float(last["风险分"]), 0), "现价": round(float(last["price"]), 1)})
    df = pd.DataFrame(rows).sort_values("建仓分", ascending=False).reset_index(drop=True)
    return df
