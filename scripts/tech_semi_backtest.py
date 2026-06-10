"""大规模回测：科技 + 半导体(+NFLX)为主，附带所有能测的分析。

主回测用**已上线的 v3 引擎**(analysis.position_guidance._exposure_series)——即测产品本身。
全部含费、对比 SPY、α 带 block bootstrap CI；诚实口径(撤离=崩盘保险，不跑赢长牛绝对收益)。

分节：
  1. 每票 v3 vs 闭眼持有 vs SPY：年化/夏普/回撤/在场% + α/β(显著性)
  2. 分组汇总(半导体 / 软件互联网 / ETF)：v3 跑赢持有率、α显著占比、回撤改善
  3. 组合层：等权 v3 vs 等权持有 vs SPY 的 α/β
  4. PEAD(财报后漂移)：科技+半导体池的领先 IC vs 假财报对照(安慰剂)
  5. 横截面相对排名 edge(动量+低波，beta中性化+NW-t+deflated)
  6. 今日作战卡快照：龙头名的建仓/撤离建议(三档暴露)
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
TVOL = 0.25  # v3 中性档

SEMI = ["NVDA", "AMD", "AVGO", "TSM", "ASML", "QCOM", "MU", "TXN", "INTC",
        "AMAT", "LRCX", "KLAC", "ADI", "MRVL", "ON", "ARM"]
SOFT = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "ORCL", "CRM", "ADBE", "NFLX"]
ETFS = ["SMH", "SOXX", "QQQ", "XLK"]
GROUP = {**{t: "半导体" for t in SEMI}, **{t: "软件/互联网" for t in SOFT}, **{t: "ETF" for t in ETFS}}
ALL = SEMI + SOFT + ETFS


def section(t):
    print("\n" + "=" * 96 + f"\n{t}\n" + "-" * 96)


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


# 基准 SPY
spy = loader.load_ohlcv("SPY", "2000-01-01", END)["close"].dropna()
spy_ret = spy.pct_change()

# ---------------------------------------------------------------------------
section("1. 每票 v3(已上线引擎) vs 闭眼持有 vs SPY —— 科技+半导体(+NFLX)")
rows = []
v3_rets, hold_rets = {}, {}
for tk in ALL:
    try:
        ohlc = loader.load_ohlcv(tk, START, END).dropna()
        if ohlc.shape[0] < 252:
            print(f"  {tk:<6} 样本不足，跳过")
            continue
        c = ohlc["close"]
        ret = c.pct_change()
        expo = pg._exposure_series(c, TVOL).shift(1).fillna(0.0)   # 无前视
        turn = expo.diff().abs().fillna(0.0)
        sret = expo * ret - turn * FEE
        v3_nav = (1 + sret.fillna(0)).cumprod()
        hold_nav = c / c.iloc[0]
        pv3, ph = perf(v3_nav), perf(hold_nav)
        ab = qe.alpha_beta(sret, spy_ret, n_boot=300)
        rows.append({"票": tk, "组": GROUP[tk], "持有年化": ph["cagr"], "v3年化": pv3["cagr"],
                     "持有夏普": ph["sharpe"], "v3夏普": pv3["sharpe"], "持有DD": ph["maxdd"],
                     "v3DD": pv3["maxdd"], "在场%": float((expo > 0).mean()),
                     "α": ab["alpha_ann"], "β": ab["beta"], "α显著": ab["alpha_significant"]})
        v3_rets[tk], hold_rets[tk] = sret, ret
    except Exception as e:
        print(f"  {tk:<6} 跳过 {type(e).__name__}: {str(e)[:40]}")

df = pd.DataFrame(rows).set_index("票")
print(f"\n{'票':<6}{'组':<12}{'持有年化':>8}{'v3年化':>8}{'持有夏普':>8}{'v3夏普':>8}{'持有DD':>8}{'v3DD':>8}{'在场%':>7}{'β':>6}{'α':>8}{'α显著':>6}")
for tk, r in df.iterrows():
    print(f"{tk:<6}{r['组']:<11}{r['持有年化']:>+8.0%}{r['v3年化']:>+8.0%}{r['持有夏普']:>8.2f}{r['v3夏普']:>8.2f}"
          f"{r['持有DD']:>8.0%}{r['v3DD']:>8.0%}{r['在场%']:>7.0%}{r['β']:>6.2f}{r['α']:>+8.1%}{str(r['α显著']):>6}")

# ---------------------------------------------------------------------------
section("2. 分组汇总：v3 跑赢持有率 / α显著 / 回撤改善")
for g in ["半导体", "软件/互联网", "ETF", "全池"]:
    d = df if g == "全池" else df[df["组"] == g]
    if d.empty:
        continue
    print(f"  【{g}】 N={len(d)}")
    print(f"    年化中位  持有 {d['持有年化'].median():+.1%} | v3 {d['v3年化'].median():+.1%}")
    print(f"    夏普中位  持有 {d['持有夏普'].median():.2f} | v3 {d['v3夏普'].median():.2f}（v3更高?{d['v3夏普'].median()>d['持有夏普'].median()}）")
    print(f"    回撤中位  持有 {d['持有DD'].median():.0%} | v3 {d['v3DD'].median():.0%}（改善 {d['v3DD'].median()-d['持有DD'].median():+.0%}）")
    print(f"    v3 绝对年化跑赢持有占比: {(d['v3年化']>d['持有年化']).mean():.0%}"
          f" | v3 夏普跑赢持有占比: {(d['v3夏普']>d['持有夏普']).mean():.0%}"
          f" | α显著为正占比: {((d['α显著'])&(d['α']>0)).mean():.0%}")

# ---------------------------------------------------------------------------
section("3. 组合层：等权 v3 vs 等权闭眼持有 vs SPY（α/β）")
ov = pd.DataFrame(v3_rets)
hd = pd.DataFrame(hold_rets)
common = ov.dropna(how="all").index.intersection(hd.dropna(how="all").index)
ovr, hdr = ov.loc[common].mean(axis=1), hd.loc[common].mean(axis=1)
pov, phd = perf((1 + ovr.fillna(0)).cumprod()), perf((1 + hdr.fillna(0)).cumprod())
ab_spy = qe.alpha_beta(ovr, spy_ret, n_boot=1000)
ab_hold = qe.alpha_beta(ovr.loc[common], hdr.loc[common], n_boot=1000)
print(f"  等权 v3 组合   年化 {pov['cagr']:+.1%} | 夏普 {pov['sharpe']:.2f} | 回撤 {pov['maxdd']:.0%}")
print(f"  等权闭眼持有   年化 {phd['cagr']:+.1%} | 夏普 {phd['sharpe']:.2f} | 回撤 {phd['maxdd']:.0%}")
print(f"  v3 vs SPY :       α {ab_spy['alpha_ann']:+.1%} CI[{ab_spy['alpha_ann_ci'][0]:+.1%},{ab_spy['alpha_ann_ci'][1]:+.1%}] β {ab_spy['beta']:.2f} 显著?{ab_spy['alpha_significant']}")
print(f"  v3 vs 同篮子持有: α {ab_hold['alpha_ann']:+.1%} CI[{ab_hold['alpha_ann_ci'][0]:+.1%},{ab_hold['alpha_ann_ci'][1]:+.1%}] β {ab_hold['beta']:.2f} 显著?{ab_hold['alpha_significant']}")

# ---------------------------------------------------------------------------
section("4. PEAD 财报后漂移：科技+半导体池 领先IC vs 假财报对照(安慰剂)")
try:
    from evaluation import earnings_eval as ee
    names = ["NVDA", "AMD", "AVGO", "TSM", "QCOM", "MU", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX"]
    prices = {t: loader.load_prices([t], START, END)[t].dropna() for t in names}
    edates = {t: loader.load_earnings_dates(t) for t in names}
    ic = ee.earnings_drift_ic(prices, edates, horizons=(1, 5, 21, 63), n_control=60)
    print(ic["ic_table"].to_string(index=False, formatters={
        "ic_real": "{:+.3f}".format, "ic_fake_mean": "{:+.3f}".format, "perm_pvalue": "{:.3f}".format}))
    print(f"  事件数 N={ic['n_events']}；真实IC远超假财报对照(≈0)+p<0.05 = 真信号")
except Exception as e:
    print(f"  PEAD 跳过 {type(e).__name__}: {str(e)[:60]}")

# ---------------------------------------------------------------------------
section("5. 横截面相对排名 edge：科技+半导体池(动量+低波·beta中性化·NW-t·deflated)")
try:
    cs_uni = [t for t in SEMI + SOFT if t not in ("ARM",)]
    cs = qe.cross_section_edge(loader.load_prices(list(dict.fromkeys(cs_uni)), "2013-01-01", END), n_boot=300)
    print(f"  多空夏普 {cs['sharpe']:.2f} CI[{cs['sharpe_ci_low']:.2f},{cs['sharpe_ci_high']:.2f}] | 年化 {cs['ann_return']:+.0%}")
    print(f"  IC均 {cs['ic_mean']:.3f} | NW|t| {abs(cs['ic_t_newey_west']):.2f} | deflated概率 {cs['deflated_sharpe_prob']:.0%} | 稳健?{cs['robust']}")
except Exception as e:
    print(f"  横截面 跳过 {type(e).__name__}: {str(e)[:60]}")

# ---------------------------------------------------------------------------
section("6. 今日作战卡快照：龙头名建仓/撤离建议(三档暴露·已上线引擎)")
for tk in ["NVDA", "AMD", "AVGO", "TSM", "ASML", "MU", "NFLX", "META", "MSFT"]:
    try:
        g = pg.position_guidance(tk, start=START, end=END)
        ex = g["exposure"]
        r = g["regime"]
        print(f"  {tk:<6} @{r['price']:.1f} 距前高{r['drawdown_from_high']:+.0%} "
              f"{'200上' if r['trend_above_200'] else '200下'}{'·斜率正' if r['slope200_positive'] else '·斜率负'} "
              f"波动{r['vol_percentile']:.0%} | 暴露 稳{ex['conservative']['exposure_pct']}/中{ex['moderate']['exposure_pct']}/进{ex['aggressive']['exposure_pct']}% "
              f"| 建仓:{g['build']['stance']} {g['build']['grade']}")
    except Exception as e:
        print(f"  {tk:<6} 跳过 {type(e).__name__}: {str(e)[:40]}")

print("\n" + "=" * 96 + "\n完成。\n" + "=" * 96)
