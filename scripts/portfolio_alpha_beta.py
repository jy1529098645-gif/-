"""组合层 Alpha/Beta 评估：把"工具能否直接用于交易"落到可交易的组合口径。

三件事，全部对比 SPY、给 block bootstrap α 显著性：
  A. 等权组合(降偏差宽池) + 波动目标 overlay  vs  SPY        —— 工具"系统"层 α/β
  B. 同一篮子: overlay 组合  vs  同篮子闭眼持有(等权 buy&hold) —— overlay **自身**增量 α/β
  C. 纯 SPY 上的 overlay     vs  SPY                          —— 剥离选股、只看风控的 α/β

口径诚实：overlay 只降不加杠杆(EWMA 波动目标 20%)；含费(在单票 overlay 内不另计交易费，
        但年化/夏普/回撤为净值口径)；等权日度再平衡用日收益近似(无再平衡摩擦，故 overlay
        的真实可交易 α 会略低于此)。
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

END = "2026-06-06"
START = "2014-01-01"
# 降偏差宽池：11 个十年赢家 + 10 个走平/下跌大盘股（不是纯赢家池，压低选股偏差）
WINNERS = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "AMD", "TSM", "NFLX"]
LAGGARDS = ["INTC", "PYPL", "DIS", "BA", "F", "T", "VZ", "NKE", "MMM", "QCOM"]
UNIVERSE = WINNERS + LAGGARDS


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "-" * 78)


def perf(value):
    v = value.dropna()
    n = len(v)
    cagr = (v.iloc[-1] / v.iloc[0]) ** (252 / n) - 1
    ret = v.pct_change().dropna()
    vol = float(ret.std() * np.sqrt(252))
    sharpe = float(cagr / vol) if vol > 0 else float("nan")
    maxdd = float((v / v.cummax() - 1).min())
    return {"cagr": float(cagr), "vol": vol, "sharpe": sharpe, "maxdd": maxdd, "n": n}


# 载入价格
px = loader.load_prices(UNIVERSE + ["SPY"], START, END).dropna(how="all").ffill()
spy = px["SPY"].dropna()
spy_ret = spy.pct_change()

# 各票 overlay 净值（波动目标 20%，只降不加杠杆）与等权 buy&hold 净值
overlay_navs, hold_navs = {}, {}
for t in UNIVERSE:
    if t not in px or px[t].notna().sum() < 252:
        continue
    p = px[t].dropna()
    r = qe.vol_target_backtest(p, target_vol=0.20)
    overlay_navs[t] = r["equity"]["波动目标"]
    hold_navs[t] = p / p.iloc[0]

ov = pd.DataFrame(overlay_navs).dropna(how="all")
hd = pd.DataFrame(hold_navs).dropna(how="all")
common = ov.index.intersection(hd.index).intersection(spy.index)
ov, hd = ov.loc[common], hd.loc[common]

# 等权组合日收益 = 各票日收益横截面均值（每日再平衡近似）
ov_port_ret = ov.pct_change().mean(axis=1)
hd_port_ret = hd.pct_change().mean(axis=1)
ov_port_nav = (1 + ov_port_ret.fillna(0)).cumprod()
hd_port_nav = (1 + hd_port_ret.fillna(0)).cumprod()

section(f"样本：{len(UNIVERSE)} 票降偏差宽池（{len(WINNERS)}赢家+{len(LAGGARDS)}走平/跌）"
        f" {common[0].date()}~{common[-1].date()}，基准 SPY")

# ---------------------------------------------------------------------------
section("A. 系统层：等权宽池 + 波动目标 overlay  vs  SPY（含选股 + 风控）")
pa_ov = perf(ov_port_nav)
pa_spy = perf(spy.loc[common] / spy.loc[common].iloc[0])
ab_a = qe.alpha_beta(ov_port_ret, spy_ret.loc[common], n_boot=1000)
print(f"  组合年化 {pa_ov['cagr']:+.1%} | 夏普 {pa_ov['sharpe']:.2f} | 回撤 {pa_ov['maxdd']:.0%} | 波动 {pa_ov['vol']:.0%}")
print(f"  SPY 年化 {pa_spy['cagr']:+.1%} | 夏普 {pa_spy['sharpe']:.2f} | 回撤 {pa_spy['maxdd']:.0%} | 波动 {pa_spy['vol']:.0%}")
print(f"  → α年化 {ab_a['alpha_ann']:+.1%}  CI[{ab_a['alpha_ann_ci'][0]:+.1%},{ab_a['alpha_ann_ci'][1]:+.1%}]"
      f"  β {ab_a['beta']:.2f}  R² {ab_a['r2']:.2f}  显著?{ab_a['alpha_significant']}")
print(f"  {ab_a['verdict']}")

# ---------------------------------------------------------------------------
section("B. overlay 自身增量：overlay 组合  vs  同篮子等权闭眼持有（剥离选股、只看风控）")
pb_hold = perf(hd_port_nav)
ab_b = qe.alpha_beta(ov_port_ret, hd_port_ret, n_boot=1000)
print(f"  overlay 组合 年化 {pa_ov['cagr']:+.1%} | 夏普 {pa_ov['sharpe']:.2f} | 回撤 {pa_ov['maxdd']:.0%}")
print(f"  等权闭眼持有 年化 {pb_hold['cagr']:+.1%} | 夏普 {pb_hold['sharpe']:.2f} | 回撤 {pb_hold['maxdd']:.0%}")
print(f"  → α年化(对同篮子) {ab_b['alpha_ann']:+.1%}  CI[{ab_b['alpha_ann_ci'][0]:+.1%},{ab_b['alpha_ann_ci'][1]:+.1%}]"
      f"  β {ab_b['beta']:.2f}  R² {ab_b['r2']:.2f}  显著?{ab_b['alpha_significant']}")
print(f"  夏普增量 {pa_ov['sharpe']-pb_hold['sharpe']:+.2f} | 回撤改善 {pa_ov['maxdd']-pb_hold['maxdd']:+.0%}（正=overlay回撤更浅）")
print(f"  {ab_b['verdict']}")

# ---------------------------------------------------------------------------
section("C. 纯风控(无选股)：SPY 上的波动目标 overlay  vs  SPY")
r_spy = qe.vol_target_backtest(spy, target_vol=0.20)
spy_ov_nav = r_spy["equity"]["波动目标"]
spy_ov_ret = spy_ov_nav.pct_change()
cc = spy_ov_ret.dropna().index.intersection(spy_ret.dropna().index)
ab_c = qe.alpha_beta(spy_ov_ret.loc[cc], spy_ret.loc[cc], n_boot=1000)
po, ph = r_spy["overlay"], r_spy["hold"]
print(f"  SPY+overlay 年化 {po['cagr']:+.1%} | 夏普 {po['sharpe']:.2f} | 回撤 {po['maxdd']:.0%} | 平均暴露 {r_spy['avg_exposure']:.0%}")
print(f"  SPY 持有     年化 {ph['cagr']:+.1%} | 夏普 {ph['sharpe']:.2f} | 回撤 {ph['maxdd']:.0%}")
print(f"  → α年化 {ab_c['alpha_ann']:+.1%}  CI[{ab_c['alpha_ann_ci'][0]:+.1%},{ab_c['alpha_ann_ci'][1]:+.1%}]"
      f"  β {ab_c['beta']:.2f}  显著?{ab_c['alpha_significant']}")
print(f"  夏普增量 {po['sharpe']-ph['sharpe']:+.2f} | 回撤改善 {po['maxdd']-ph['maxdd']:+.0%}")
print(f"  {ab_c['verdict']}")

print("\n" + "=" * 78 + "\n完成。\n" + "=" * 78)
