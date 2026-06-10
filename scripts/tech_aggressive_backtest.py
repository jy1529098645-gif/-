"""科技进取版策略族：追求更高年化(代价=更高回撤/爆仓风险)。诚实测每种的收益/风险。

核心认知(前面回测已证)：择时赢不了死拿，"更大利润"只能来自**更大暴露**——
杠杆 / 集中 / 趋势过滤的杠杆ETF。本脚本把它们和"闭眼持有"摆一起，看谁真的年化更高、代价多大。

策略(科技+半导体篮子 + 杠杆ETF)：
  HOLD        闭眼持有(基准)
  VT40_x1     波动目标40% 不加杠杆(只缩放)
  VT40_x1.5   波动目标40% 上限1.5倍(低波动牛市里加杠杆放大)
  VT40_x2     波动目标40% 上限2倍
  TREND_x2    站上200线持2倍杠杆、跌破清仓(趋势过滤的杠杆)
  LEVETF_HOLD 直接死拿 3x ETF(TQQQ/SOXL)
  LEVETF_TREND 3x ETF + 200线过滤(站上才持、跌破走)
  MOM_TOP3    每月轮动持有篮子里近12-1动量最强的3只(集中)
全部含费、对比 SPY、α 带 CI。诚实：高年化几乎必然伴随更深回撤。
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
import config

_CFG = config.load_config()
FEE = float(_CFG["costs"]["fees"]) + float(_CFG["costs"]["slippage"])
END = "2026-06-06"
START = "2010-01-01"

SEMI = ["NVDA", "AMD", "AVGO", "TSM", "ASML", "QCOM", "MU", "TXN", "INTC", "AMAT", "LRCX", "KLAC", "ADI", "MRVL", "ON"]
SOFT = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "ORCL", "CRM", "ADBE", "NFLX"]
BASKET = SEMI + SOFT


def perf(value):
    v = value.dropna()
    n = len(v)
    if n < 60:
        return {"cagr": np.nan, "sharpe": np.nan, "maxdd": np.nan}
    cagr = (v.iloc[-1] / v.iloc[0]) ** (252 / n) - 1
    r = v.pct_change().dropna()
    vol = r.std() * np.sqrt(252)
    return {"cagr": float(cagr), "sharpe": float(cagr / vol) if vol > 0 else np.nan,
            "maxdd": float((v / v.cummax() - 1).min())}


def section(t):
    print("\n" + "=" * 92 + f"\n{t}\n" + "-" * 92)


spy = loader.load_ohlcv("SPY", "2000-01-01", END)["close"].dropna()
spy_ret = spy.pct_change()

# 载入篮子价格
px = {}
for t in BASKET:
    try:
        p = loader.load_ohlcv(t, START, END)["close"].dropna()
        if p.shape[0] >= 252:
            px[t] = p
    except Exception:
        pass
prices = pd.DataFrame(px).dropna(how="all")


def vt_returns(price, tvol, maxlev):
    """波动目标(带杠杆上限)逐日收益。无前视：EWMA已shift。"""
    ret = price.pct_change()
    rv = qe.ewma_vol(ret)
    expo = (tvol / rv).clip(upper=maxlev).fillna(0.0)
    turn = expo.diff().abs().fillna(0.0)
    return expo * ret - turn * FEE


def trend_lev_returns(price, lev=2.0, ma=200):
    """站上200线持 lev 倍、跌破清仓。"""
    ret = price.pct_change()
    sma = price.rolling(ma, min_periods=ma // 2).mean()
    sig = (price > sma).shift(1).fillna(False).astype(float) * lev
    turn = sig.diff().abs().fillna(0.0)
    return sig * ret - turn * FEE


def equal_weight(ret_dict):
    df = pd.DataFrame(ret_dict)
    common = df.dropna(how="all").index
    return df.loc[common].mean(axis=1)


# ---------------------------------------------------------------------------
section("1. 篮子等权：闭眼持有 vs 波动目标带杠杆 vs 趋势杠杆（年化/夏普/回撤/α）")
strat_rets = {}
strat_rets["HOLD"] = equal_weight({t: prices[t].pct_change() for t in prices})
strat_rets["VT40_x1"] = equal_weight({t: vt_returns(prices[t], 0.40, 1.0) for t in prices})
strat_rets["VT40_x1.5"] = equal_weight({t: vt_returns(prices[t], 0.40, 1.5) for t in prices})
strat_rets["VT40_x2"] = equal_weight({t: vt_returns(prices[t], 0.40, 2.0) for t in prices})
strat_rets["TREND_x2"] = equal_weight({t: trend_lev_returns(prices[t], 2.0) for t in prices})

print(f"{'策略':<12}{'年化':>8}{'夏普':>7}{'最大回撤':>9}{'α vs SPY':>10}{'β':>6}{'α显著':>6}")
for k, r in strat_rets.items():
    pf = perf((1 + r.fillna(0)).cumprod())
    ab = qe.alpha_beta(r, spy_ret, n_boot=400)
    print(f"{k:<12}{pf['cagr']:>+8.1%}{pf['sharpe']:>7.2f}{pf['maxdd']:>9.0%}{ab['alpha_ann']:>+10.1%}{ab['beta']:>6.2f}{str(ab['alpha_significant']):>6}")

# ---------------------------------------------------------------------------
section("2. 杠杆ETF：直接死拿 3x  vs  3x+200线趋势过滤  vs  1x持有")
pairs = [("TQQQ", "QQQ"), ("SOXL", "SMH")]
print(f"{'组合':<22}{'年化':>8}{'夏普':>7}{'最大回撤':>9}")
for lev_t, base_t in pairs:
    try:
        levp = loader.load_ohlcv(lev_t, START, END)["close"].dropna()
        basep = loader.load_ohlcv(base_t, START, END)["close"].dropna()
        common = levp.index.intersection(basep.index)
        levp, basep = levp.loc[common], basep.loc[common]
        # 1x 持有
        pf1 = perf(basep / basep.iloc[0])
        # 3x 死拿
        pf3 = perf(levp / levp.iloc[0])
        # 3x + 200线过滤(用底层指数的200线作信号)
        sma = basep.rolling(200, min_periods=100).mean()
        sig = (basep > sma).shift(1).fillna(False).astype(float)
        ret = levp.pct_change()
        turn = sig.diff().abs().fillna(0.0)
        sret = sig * ret - turn * FEE
        pf3t = perf((1 + sret.fillna(0)).cumprod())
        print(f"{base_t+' 1x 持有':<22}{pf1['cagr']:>+8.1%}{pf1['sharpe']:>7.2f}{pf1['maxdd']:>9.0%}")
        print(f"{lev_t+' 3x 死拿':<22}{pf3['cagr']:>+8.1%}{pf3['sharpe']:>7.2f}{pf3['maxdd']:>9.0%}")
        print(f"{lev_t+' 3x+200线过滤':<20}{pf3t['cagr']:>+8.1%}{pf3t['sharpe']:>7.2f}{pf3t['maxdd']:>9.0%}")
        print()
    except Exception as e:
        print(f"  {lev_t} 跳过 {type(e).__name__}: {str(e)[:40]}")

# ---------------------------------------------------------------------------
section("3. 动量集中：每月轮动持有近(12-1)动量最强的 K 只 vs 等权全篮子")
ret_all = prices.pct_change()
mom = prices.shift(21) / prices.shift(252) - 1.0
for K in [3, 5, 8]:
    rebal = prices.index[252::21]
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for dt in rebal:
        row = mom.loc[dt].dropna()
        if len(row) < K:
            continue
        top = row.nlargest(K).index
        weights.loc[dt, top] = 1.0 / K
    weights = weights.replace(0.0, np.nan).ffill().fillna(0.0)
    turn = weights.diff().abs().sum(axis=1).fillna(0.0)
    pr = (weights.shift(1) * ret_all).sum(axis=1) - turn * FEE
    pf = perf((1 + pr.fillna(0)).cumprod())
    ab = qe.alpha_beta(pr, spy_ret, n_boot=400)
    print(f"  动量Top{K}: 年化 {pf['cagr']:+.1%} | 夏普 {pf['sharpe']:.2f} | 回撤 {pf['maxdd']:.0%}"
          f" | α vs SPY {ab['alpha_ann']:+.1%}({'显著' if ab['alpha_significant'] else '不显著'})")
pf_eq = perf((1 + strat_rets["HOLD"].fillna(0)).cumprod())
print(f"  对照·等权全篮子持有: 年化 {pf_eq['cagr']:+.1%} | 夏普 {pf_eq['sharpe']:.2f} | 回撤 {pf_eq['maxdd']:.0%}")

print("\n" + "=" * 92 + "\n完成。\n" + "=" * 92)
