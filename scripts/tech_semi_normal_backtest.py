"""大规模科技+半导体回测：详细 alpha/beta，并给"剔除超级熊市"的正常市场口径。

超级熊市(十年一遇)用规则定义：QQQ 距前高回撤 > 20% 的确认熊市日(自动涵盖 2022 / 2020COVID / 2018Q4)。
剔除这些天后重算 = "平时(非超级熊市)能指望多少"。诚实：剔熊市后降仓策略更弱、杠杆策略更强。

策略(全部走已上线引擎 position_guidance)：
  HOLD       闭眼持有
  v3_mod     引擎中性档(波动目标25% ≤1×)
  v3_lev     引擎🔥杠杆进取档(波动目标40% ≤1.5×)
输出：每票全期 CAGR + α/β；组合全期 vs 正常市场口径的 CAGR/夏普/回撤/α/β。
"""
import warnings
from pathlib import Path
import sys

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from data import loader
from analysis import quant_edge as qe
from analysis import position_guidance as pg
import config

_CFG = config.load_config()
FEE = float(_CFG["costs"]["fees"]) + float(_CFG["costs"]["slippage"])
END = "2026-06-06"
START = "2010-01-01"
BEAR_DD = 0.20  # 超级熊市阈值：指数距前高回撤 > 20%

SEMI = ["NVDA", "AMD", "AVGO", "TSM", "ASML", "QCOM", "MU", "TXN", "INTC", "AMAT", "LRCX", "KLAC", "ADI", "MRVL", "ON"]
SOFT = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "ORCL", "CRM", "ADBE", "NFLX"]
GROUP = {**{t: "半导体" for t in SEMI}, **{t: "软件/互联网" for t in SOFT}}
BASKET = SEMI + SOFT


def section(t):
    print("\n" + "=" * 96 + f"\n{t}\n" + "-" * 96)


def perf_from_ret(r, mask=None):
    """从日收益序列算 CAGR/夏普/回撤；mask=保留的日(布尔)。"""
    r = r.dropna()
    if mask is not None:
        r = r[mask.reindex(r.index).fillna(False)]
    if len(r) < 60:
        return {"cagr": np.nan, "sharpe": np.nan, "maxdd": np.nan, "n": len(r)}
    nav = (1 + r).cumprod()
    cagr = nav.iloc[-1] ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    return {"cagr": float(cagr), "sharpe": float(cagr / vol) if vol > 0 else np.nan,
            "maxdd": float((nav / nav.cummax() - 1).min()), "n": len(r)}


def strat_ret(price, tvol, ml):
    ret = price.pct_change()
    expo = pg._exposure_series(price, tvol, ml).shift(1).fillna(0.0)
    return expo * ret - expo.diff().abs().fillna(0.0) * FEE, expo


# 基准 + 超级熊市掩码(用 QQQ)
spy = loader.load_ohlcv("SPY", "2000-01-01", END)["close"].dropna()
spy_ret = spy.pct_change()
qqq = loader.load_ohlcv("QQQ", START, END)["close"].dropna()
qqq_dd = qqq / qqq.cummax() - 1.0
bear = (qqq_dd < -BEAR_DD)                       # True = 超级熊市日(剔除)
normal = ~bear                                    # 正常市场日(保留)

# ---------------------------------------------------------------------------
section(f"0. 超级熊市定义：QQQ 距前高回撤 > {BEAR_DD:.0%} 的确认熊市日(剔除)")
seg, in_bear, st0 = [], False, None
for d, b in bear.items():
    if b and not in_bear:
        in_bear, st0 = True, d
    elif not b and in_bear:
        in_bear = False
        seg.append((st0, prev))
    prev = d
if in_bear:
    seg.append((st0, prev))
print(f"  剔除天数 {int(bear.sum())} / {len(bear)} ({bear.mean():.0%})；剔除时段：")
for s, e in seg:
    if (e - s).days >= 20:
        print(f"    {s.date()} ~ {e.date()}  (QQQ最深回撤 {qqq_dd.loc[s:e].min():.0%})")

# ---------------------------------------------------------------------------
section("1. 每票全期：HOLD / 中性 / 🔥杠杆 的 年化 + α/β(vs SPY)")
hold_r, mod_r, lev_r = {}, {}, {}
rows = []
for t in BASKET:
    try:
        p = loader.load_ohlcv(t, START, END)["close"].dropna()
        if p.shape[0] < 252:
            continue
        hr = p.pct_change()
        mr, mexpo = strat_ret(p, 0.25, 1.0)
        lr, lexpo = strat_ret(p, 0.40, 1.5)
        hold_r[t], mod_r[t], lev_r[t] = hr, mr, lr
        ph = perf_from_ret(hr); pm = perf_from_ret(mr); pl = perf_from_ret(lr)
        abm = qe.alpha_beta(mr, spy_ret, n_boot=250)
        abl = qe.alpha_beta(lr, spy_ret, n_boot=250)
        rows.append({"票": t, "组": GROUP[t], "持有年化": ph["cagr"], "中性年化": pm["cagr"],
                     "杠杆年化": pl["cagr"], "杠杆在场%": float(lexpo.gt(0).mean()),
                     "杠杆均暴露": float(lexpo[lexpo > 0].mean()),
                     "中性α": abm["alpha_ann"], "中性β": abm["beta"],
                     "杠杆α": abl["alpha_ann"], "杠杆β": abl["beta"], "杠杆α显著": abl["alpha_significant"]})
    except Exception as e:
        print(f"  {t} 跳过 {type(e).__name__}")
df = pd.DataFrame(rows).set_index("票")
print(f"{'票':<6}{'组':<11}{'持有年化':>8}{'中性年化':>8}{'杠杆年化':>8}{'杠杆均暴露':>9}{'中性α':>8}{'中性β':>6}{'杠杆α':>8}{'杠杆β':>6}{'杠杆α显著':>8}")
for t, r in df.iterrows():
    print(f"{t:<6}{r['组']:<10}{r['持有年化']:>+8.0%}{r['中性年化']:>+8.0%}{r['杠杆年化']:>+8.0%}"
          f"{r['杠杆均暴露']:>9.2f}{r['中性α']:>+8.1%}{r['中性β']:>6.2f}{r['杠杆α']:>+8.1%}{r['杠杆β']:>6.2f}{str(r['杠杆α显著']):>8}")

# ---------------------------------------------------------------------------
def portfolio(rdict):
    d = pd.DataFrame(rdict)
    common = d.dropna(how="all").index
    return d.loc[common].mean(axis=1)


hp, mp, lp = portfolio(hold_r), portfolio(mod_r), portfolio(lev_r)

section("2. 组合层·全期(含超级熊市)：CAGR/夏普/回撤 + α/β")
print(f"{'策略':<14}{'年化':>8}{'夏普':>7}{'回撤':>8}{'α vs SPY':>10}{'β':>6}{'α vs 持有':>11}{'α显著':>6}")
for nm, r in [("闭眼持有", hp), ("中性(≤1×)", mp), ("🔥杠杆(≤1.5×)", lp)]:
    pf = perf_from_ret(r)
    ab = qe.alpha_beta(r, spy_ret, n_boot=600)
    abh = qe.alpha_beta(r, hp, n_boot=600) if nm != "闭眼持有" else {"alpha_ann": 0.0, "alpha_significant": False}
    print(f"{nm:<14}{pf['cagr']:>+8.1%}{pf['sharpe']:>7.2f}{pf['maxdd']:>8.0%}{ab['alpha_ann']:>+10.1%}{ab['beta']:>6.2f}"
          f"{abh['alpha_ann']:>+11.1%}{str(abh['alpha_significant']):>6}")

section(f"3. 组合层·正常市场(剔除 QQQ>{BEAR_DD:.0%} 回撤的超级熊市日)：这才是平时能指望的")
print(f"{'策略':<14}{'年化':>8}{'夏普':>7}{'回撤':>8}{'α vs SPY':>10}{'β':>6}{'α vs 持有':>11}{'α显著':>6}")
spy_n = spy_ret[normal.reindex(spy_ret.index).fillna(False)]
hp_n = hp[normal.reindex(hp.index).fillna(False)]
for nm, r in [("闭眼持有", hp), ("中性(≤1×)", mp), ("🔥杠杆(≤1.5×)", lp)]:
    pf = perf_from_ret(r, mask=normal)
    rn = r[normal.reindex(r.index).fillna(False)]
    ab = qe.alpha_beta(rn, spy_ret, n_boot=600)
    abh = qe.alpha_beta(rn, hp_n, n_boot=600) if nm != "闭眼持有" else {"alpha_ann": 0.0, "alpha_significant": False}
    print(f"{nm:<14}{pf['cagr']:>+8.1%}{pf['sharpe']:>7.2f}{pf['maxdd']:>8.0%}{ab['alpha_ann']:>+10.1%}{ab['beta']:>6.2f}"
          f"{abh['alpha_ann']:>+11.1%}{str(abh['alpha_significant']):>6}")

section("4. 分组·正常市场年化中位(剔超级熊市)：半导体 vs 软件互联网")
for g in ["半导体", "软件/互联网", "全池"]:
    tks = [t for t in df.index if (g == "全池" or df.loc[t, "组"] == g)]
    h = np.median([perf_from_ret(hold_r[t], normal)["cagr"] for t in tks])
    m = np.median([perf_from_ret(mod_r[t], normal)["cagr"] for t in tks])
    l = np.median([perf_from_ret(lev_r[t], normal)["cagr"] for t in tks])
    print(f"  【{g}】N={len(tks)}  持有 {h:+.1%} | 中性 {m:+.1%} | 🔥杠杆 {l:+.1%}")

print("\n" + "=" * 96 + "\n完成。\n" + "=" * 96)
