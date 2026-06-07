"""临时：扫描 GOOGL / MSFT / NVDA / NFLX 当前建仓点位（一次性脚本）。"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from data.loader import load_prices, load_ohlcv, load_macro
from regime.observables import today_panel
from regime.conditional_returns import conditional_forward_returns, current_fingerprint
from analysis.volume_profile import volume_profile

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

TICKERS = ["GOOGL", "MSFT", "NVDA", "NFLX"]
START = "2010-01-01"

print("== 拉取数据 ==")
panel = load_prices(TICKERS, start=START)
macro = load_macro(start=START)
print("价格区间:", panel.index.min().date(), "->", panel.index.max().date())
print("最新收盘:")
print(panel.ffill().iloc[-1].round(2).to_string())


def rr(reward, risk):
    risk = abs(risk)
    return reward / risk if risk > 1e-9 else float("nan")


for t in TICKERS:
    print("\n" + "=" * 72)
    print(f"### {t}")
    print("=" * 72)
    px = panel[t].dropna()
    last = float(px.iloc[-1])
    print(f"最新价: {last:.2f}  ({px.index[-1].date()})")

    tp = today_panel(px)
    print("\n[今日状态]")
    print(f"  趋势: {tp['trend_state']}  (距200日均线 {tp['trend_position']*100:+.1f}%)")
    print(f"  距前高回撤: {tp['drawdown']*100:+.1f}%  ({tp['drawdown_state']})")
    print(f"  波动: {tp['vol_state']}  (年化{tp['realized_vol']*100:.0f}%, 分位{tp['vol_percentile']*100:.0f}%)")

    ma20 = float(px.rolling(20).mean().iloc[-1])
    ma50 = float(px.rolling(50).mean().iloc[-1])
    ma100 = float(px.rolling(100).mean().iloc[-1])
    ma200 = float(px.rolling(200).mean().iloc[-1])
    hi = float(px.cummax().iloc[-1])
    hi52 = float(px.tail(252).max())
    print("\n[关键参考位]")
    print(f"  历史前高: {hi:.2f}   近1年高点: {hi52:.2f}   现价距前高 {(last/hi-1)*100:+.1f}%")
    print(f"  MA20:{ma20:.2f}({(last/ma20-1)*100:+.1f}%)  MA50:{ma50:.2f}({(last/ma50-1)*100:+.1f}%)  MA100:{ma100:.2f}({(last/ma100-1)*100:+.1f}%)  MA200:{ma200:.2f}({(last/ma200-1)*100:+.1f}%)")
    print("  从近1年高点回撤台阶:", "  ".join(f"-{int(d*100)}%={hi52*(1-d):.1f}" for d in (0.05,0.10,0.15,0.20,0.25)))

    # Volume Profile：多窗口
    print("\n[筹码密集区]")
    for start_d, lb, tag in [("2024-01-01",126,"近半年"),("2024-01-01",63,"近3月"),("2022-01-01",504,"近2年")]:
        try:
            o = load_ohlcv(t, start=start_d)
            vp = volume_profile(o, bins=40, lookback=lb)
            print(f"  {tag}: POC={vp['poc']:.2f}  价值区(70%量)={vp['value_area'][0]:.2f}~{vp['value_area'][1]:.2f}")
        except Exception as e:
            print(f"  {tag} VP失败: {e}")

    # 建仓引擎
    try:
        cfr = conditional_forward_returns(panel, macro, asset=t, n_boot=400)
        for h in (21, 63):
            sub = cfr[cfr["horizon"] == h]
            base = sub[sub["grouping"] == "__baseline__"]
            print(f"\n[建仓引擎 h={h}日]")
            if not base.empty:
                b = base.iloc[0]
                ratio = rr(b['median'], b['median_path_drawdown'])
                print(f"  无条件基准:    中位{b['median']*100:+.1f}% 胜率{b['win_rate']*100:.0f}% p10{b['p10']*100:+.1f}% p90{b['p90']*100:+.1f}% 途中浮亏中位{b['median_path_drawdown']*100:.1f}% 盈亏比≈{ratio:.2f}")
            # in_drawdown 边际桶
            dd = sub[(sub["grouping"]=="in_drawdown") & (sub["bucket"].str.contains("True", na=False))]
            for _, r in dd.iterrows():
                ratio = rr(r['median'], r['median_path_drawdown'])
                sig = "显著✅" if (r['ci_low']>0) else "不显著"
                print(f"  回撤>10%桶:    中位{r['median']*100:+.1f}% (超额{r['excess_median']*100:+.1f}pt) 胜率{r['win_rate']*100:.0f}% p10{r['p10']*100:+.1f}% 途中浮亏{r['median_path_drawdown']*100:.1f}% 盈亏比≈{ratio:.2f} N={int(r['n_events'])} CI[{r['ci_low']*100:+.1f},{r['ci_high']*100:+.1f}] {sig}")
            # 估值低位桶
            val = sub[(sub["grouping"]=="valuation_tercile") & (sub["bucket"].str.contains("low", na=False))]
            for _, r in val.iterrows():
                ratio = rr(r['median'], r['median_path_drawdown'])
                sig = "显著✅" if (r['ci_low']>0) else "不显著"
                print(f"  估值低位桶:    中位{r['median']*100:+.1f}% (超额{r['excess_median']*100:+.1f}pt) 胜率{r['win_rate']*100:.0f}% 途中浮亏{r['median_path_drawdown']*100:.1f}% 盈亏比≈{ratio:.2f} N={int(r['n_events'])} CI[{r['ci_low']*100:+.1f},{r['ci_high']*100:+.1f}] {sig}")
    except Exception as e:
        import traceback; traceback.print_exc()

print("\n完成。")
