"""大规模回测：用工具自己的机器，诚实回答"年化多少 / 信号有没有效"。

全部走已实现的工具函数；输出含 基准对比 / CI / 样本量 / 显著性，不报光秃秃单点。
"""
import warnings
from pathlib import Path
import sys

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from data import loader
from backtest import strategies as bt
from analysis import quant_edge as qe

END = "2026-06-06"
WINNERS = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "AMD", "TSM", "NFLX"]
LAGGARDS = ["INTC", "PYPL", "DIS", "WBA", "BA", "F", "T", "VZ", "NKE", "MMM"]
ETFS = ["SPY", "QQQ"]
LEV = ["TQQQ", "SOXL"]
ALL = WINNERS + LAGGARDS + ETFS + LEV


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "-" * 78)


# ===========================================================================
section("1. 策略 vs 闭眼持有：每票年化 / 夏普 / 回撤（含费用，深跌买×让利润奔跑）")
spy = loader.load_prices(["SPY"], "2010-01-01", END)["SPY"].pct_change()
rows = []
for t in ALL:
    try:
        pv = bt.strategy_vs_hold(t, "2014-01-01", END)
        s, h = pv["strategy"], pv["hold"]
        ab = qe.alpha_beta(pv["equity"]["策略"].pct_change(), spy, n_boot=300)
        rows.append({
            "票": t, "策略年化": s["cagr"], "持有年化": h["cagr"], "差": s["cagr"] - h["cagr"],
            "策略夏普": s["sharpe"], "持有夏普": h["sharpe"],
            "策略回撤": s["maxdd"], "持有回撤": h["maxdd"], "在场%": s.get("in_market", np.nan),
            "α年化": ab["alpha_ann"], "α显著": ab["alpha_significant"],
        })
    except Exception as e:
        print(f"  {t} 跳过: {type(e).__name__}")
df = pd.DataFrame(rows).set_index("票")
pd.set_option("display.width", 200, "display.max_columns", 20)
print(df.to_string(formatters={
    "策略年化": "{:+.0%}".format, "持有年化": "{:+.0%}".format, "差": "{:+.0%}".format,
    "策略夏普": "{:.2f}".format, "持有夏普": "{:.2f}".format,
    "策略回撤": "{:.0%}".format, "持有回撤": "{:.0%}".format, "在场%": "{:.0%}".format,
    "α年化": "{:+.1%}".format, "α显著": str}))

section("1b. 汇总（剔除杠杆ETF，避免复利失真）")
core = df.drop(index=[x for x in LEV if x in df.index])
print(f"  样本 {len(core)} 票（{len(WINNERS)}赢家+{len(LAGGARDS)}输家+{len(ETFS)}ETF）")
print(f"  策略年化 中位 {core['策略年化'].median():+.1%} | 持有年化 中位 {core['持有年化'].median():+.1%}")
print(f"  策略跑赢持有的票占比: {(core['差'] > 0).mean():.0%}")
print(f"  策略夏普 中位 {core['策略夏普'].median():.2f} | 持有夏普 中位 {core['持有夏普'].median():.2f}")
print(f"  策略回撤 中位 {core['策略回撤'].median():.0%} | 持有回撤 中位 {core['持有回撤'].median():.0%}（策略回撤更浅?{core['策略回撤'].median() > core['持有回撤'].median()}）")
print(f"  扣市场β后 α 显著为正的票数: {int((core['α显著'] & (core['α年化'] > 0)).sum())}/{len(core)}")
print(f"  赢家池 策略年化中位 {df.loc[[t for t in WINNERS if t in df.index],'策略年化'].median():+.1%}"
      f" vs 输家池 {df.loc[[t for t in LAGGARDS if t in df.index],'策略年化'].median():+.1%}（差距=选股偏差证据）")

# ===========================================================================
section("2. 入场信号有效吗：Walk-forward 样本外（参数固定，看跨期稳定）")
wf_rows = []
for t in WINNERS[:8] + LAGGARDS[:5]:
    try:
        wf = qe.walkforward_oos(t, "2010-01-01", END)
        if wf.get("n_windows"):
            wf_rows.append({"票": t, "OOS窗口": wf["n_windows"], "跑赢持有比例": wf["beat_rate"],
                            "策略年化中位": wf["strat_cagr_median"], "持有年化中位": wf["hold_cagr_median"]})
    except Exception as e:
        print(f"  {t} 跳过: {type(e).__name__}")
wfd = pd.DataFrame(wf_rows).set_index("票")
print(wfd.to_string(formatters={"跑赢持有比例": "{:.0%}".format,
                                "策略年化中位": "{:+.0%}".format, "持有年化中位": "{:+.0%}".format}))
print(f"\n  >> 全样本 OOS 跑赢持有比例 中位: {wfd['跑赢持有比例'].median():.0%}"
      f"（≈50% 即无稳定择时优势；明显<50% 说明入场择时反而拖累）")

# ===========================================================================
section("3. 入场状态有没有超额：条件桶（回撤桶 vs 无条件基准，带CI/N）")
from regime import conditional_returns as cr
macro = loader.load_macro("1990-01-01", END)
bucket_rows = []
for t in WINNERS[:8]:
    try:
        px = loader.load_prices([t], "2010-01-01", END)[t]
        tab = cr.conditional_forward_returns(px, macro, asset=t, horizons=(63,),
                                             groupings=[["in_drawdown"]], n_boot=250)
        dd = tab[(tab["bucket"].str.contains("in_drawdown=True"))]
        if not dd.empty:
            r = dd.iloc[0]
            sig = (r["ci_low"] > 0 or r["ci_high"] < 0) and not r.get("low_power", False)
            bucket_rows.append({"票": t, "回撤桶中位": r["median"], "超额": r["excess_median"],
                                "N独立": int(r["n_independent"]), "CI低": r["ci_low"], "CI高": r["ci_high"],
                                "显著": sig})
    except Exception as e:
        print(f"  {t} 跳过: {type(e).__name__}")
bd = pd.DataFrame(bucket_rows).set_index("票")
print(bd.to_string(formatters={"回撤桶中位": "{:+.1%}".format, "超额": "{:+.1%}".format,
                               "CI低": "{:+.1%}".format, "CI高": "{:+.1%}".format, "显著": str}))
print(f"\n  >> 回撤桶超额>0 的票: {(bd['超额'] > 0).sum()}/{len(bd)}；其中**显著**的: {int(bd['显著'].sum())}/{len(bd)}")

# ===========================================================================
section("4. PEAD（财报后漂移）—— 工具唯一过安慰剂检验的免费信号")
from evaluation import earnings_eval as ee
mag7 = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]
prices = {t: loader.load_prices([t], "2010-01-01", END)[t] for t in mag7}
edates = {t: loader.load_earnings_dates(t) for t in mag7}
ic = ee.earnings_drift_ic(prices, edates, horizons=(1, 5, 21, 63), n_control=60)
print(ic["ic_table"].to_string(index=False, formatters={
    "ic_real": "{:+.3f}".format, "ic_fake_mean": "{:+.3f}".format, "perm_pvalue": "{:.3f}".format}))
print(f"  事件数 N={ic['n_events']}；真实IC远超假财报日对照(≈0) + p<0.05 = 真信号")

# ===========================================================================
section("5. 横截面相对排名（动量+低波，winsorize+beta中性化+NW-t+deflated）")
cs_uni = WINNERS + ["AMD", "QCOM", "TXN", "MU", "ORCL", "CRM", "ADBE"]
cs = qe.cross_section_edge(loader.load_prices(list(dict.fromkeys(cs_uni)), "2015-01-01", END), n_boot=300)
print(f"  多空夏普 {cs['sharpe']:.2f}  CI[{cs['sharpe_ci_low']:.2f},{cs['sharpe_ci_high']:.2f}]")
print(f"  年化 {cs['ann_return']:+.0%} | IC均 {cs['ic_mean']:.3f} | NW|t| {abs(cs['ic_t_newey_west']):.2f}"
      f" | deflated概率 {cs['deflated_sharpe_prob']:.0%} | 稳健?{cs['robust']}")

# ===========================================================================
section("6. 入场方法对比：一次性 vs 定投 vs 越跌越补（赢家 NVDA / 输家 INTC）")
for t in ["NVDA", "INTC", "SPY"]:
    try:
        r = bt.compare_entry_strategies(loader.load_prices([t], "2010-01-01", END), asset=t)
        ps = r["per_strategy"]
        print(f"  {t}: 一次性中位{ps['lump_sum']['median']:+.0%} | 定投{ps['dca']['median']:+.0%}"
              f" | 越跌越补{ps['average_down']['median']:+.0%}"
              f"（越跌越补 vs 一次性 差 {r['vs_lump_sum']['average_down']['median_diff']:+.0%},"
              f"显著?{r['vs_lump_sum']['average_down']['significant']}）")
    except Exception as e:
        print(f"  {t} 跳过: {type(e).__name__}")

print("\n" + "=" * 78 + "\n完成。\n" + "=" * 78)
