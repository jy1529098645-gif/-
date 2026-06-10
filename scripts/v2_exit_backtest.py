"""出场族头对头烤测：入场固定(dip-15%)，只换出场，看哪种能净跑赢闭眼持有。

逐日事件模拟器(真 ATR + 分批撤离 + 无前视：t 日暴露用到 t 收盘信息，作用于 t→t+1 收益)。
空仓日现金 0 收益(诚实承担择时的踏空代价)。暴露变动按 fees+slippage 计摩擦。

变体(全部共用 dip-from-high(20,15%) 入场)：
  v1   固定 30% 移动止损(现行工具定稿)
  B    收盘跌破 200 线即清(用户嫌迟钝的版本)
  C    Chandelier 贴顶移动止损 = 入场以来最高 − 3×ATR(22)
  v2   Chandelier(过热时 k:3→1.5) + 分批撤离(20日低破/50线斜率拐头/波动突刺 各减 1/3)

输出：每票 v1/B/C/v2/持有 的 年化/夏普/回撤/在场% + v2 对 SPY 的 α/β + 逐年 OOS 跑赢持有率。
"""
import warnings
from pathlib import Path
import sys

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from data import loader
from factors import signals as sg
from analysis import quant_edge as qe
import config

_CFG = config.load_config()
FEE = float(_CFG["costs"]["fees"]) + float(_CFG["costs"]["slippage"])

END = "2026-06-06"
WINNERS = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "AMD", "TSM", "NFLX"]
LAGGARDS = ["INTC", "PYPL", "DIS", "BA", "F", "T", "VZ", "NKE", "MMM", "QCOM"]
POOL = WINNERS + LAGGARDS


def atr(ohlc, n=22):
    h, l, c = ohlc["high"], ohlc["low"], ohlc["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n // 2).mean()


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


def simulate(ohlc, variant):
    """返回逐日策略收益序列(已含空仓现金0、暴露摩擦)。"""
    c = ohlc["close"]
    ret = c.pct_change().fillna(0.0)
    entry = sg.dip_from_high(c, lookback=20, pct=0.15).reindex(c.index).fillna(False).values
    a = atr(ohlc, 22).values
    sma50 = c.rolling(50, min_periods=25).mean()
    sma200 = c.rolling(200, min_periods=100).mean()
    slope50 = (sma50 > sma50.shift(10)).values           # 50线10日斜率>0
    ext = ((c - sma50) / sma50)                           # 距50线
    ext_std = ext.rolling(252, min_periods=120).std()
    overheat = (ext > 2 * ext_std).values                 # 过热(距50线>2σ)
    low20 = c.rolling(20, min_periods=10).min().shift(1).values
    from regime import observables as ob
    vpx = ob.realized_vol_percentile(c, 21, 252).reindex(c.index).values
    px = c.values
    N = len(px)

    expo = np.zeros(N)
    in_pos = False
    peak = np.nan
    cur = 0.0
    for t in range(N):
        if not in_pos:
            if entry[t]:
                in_pos = True
                peak = px[t]
                cur = 1.0
            else:
                cur = 0.0
        else:
            peak = max(peak, px[t])
            if variant == "v1":
                if px[t] < peak * (1 - 0.30):
                    in_pos, cur = False, 0.0
                else:
                    cur = 1.0
            elif variant == "B":
                if px[t] < sma200.values[t]:
                    in_pos, cur = False, 0.0
                else:
                    cur = 1.0
            elif variant == "C":
                k = 3.0
                stop = peak - k * (a[t] if a[t] == a[t] else peak * 0.05)
                if px[t] < stop:
                    in_pos, cur = False, 0.0
                else:
                    cur = 1.0
            elif variant == "v2":
                k = 1.5 if overheat[t] else 3.0
                stop = peak - k * (a[t] if a[t] == a[t] else peak * 0.05)
                if px[t] < stop:
                    in_pos, cur = False, 0.0
                else:
                    trims = 0
                    if low20[t] == low20[t] and px[t] < low20[t]:
                        trims += 1
                    if not slope50[t]:
                        trims += 1
                    if vpx[t] == vpx[t] and vpx[t] > 0.90:
                        trims += 1
                    cur = max(0.0, 1.0 - trims / 3.0)
                    if cur == 0.0:                       # 全减完=退出该仓，需重新入场
                        in_pos = False
        expo[t] = cur

    expo_s = pd.Series(expo, index=c.index).shift(1).fillna(0.0)   # 无前视
    turnover = expo_s.diff().abs().fillna(0.0)
    strat_ret = expo_s * ret - turnover * FEE
    return strat_ret, float((expo_s > 0).mean())


def section(t):
    print("\n" + "=" * 88 + f"\n{t}\n" + "-" * 88)


# ---------------------------------------------------------------------------
spy_ohlc = loader.load_ohlcv("SPY", "2000-01-01", END)
spy_ret = spy_ohlc["close"].pct_change()

rows = []
v2_port_rets = {}
hold_port_rets = {}
for tk in POOL + ["SPY"]:
    try:
        start = "2000-01-01"
        ohlc = loader.load_ohlcv(tk, start, END).dropna()
        if ohlc.shape[0] < 252:
            continue
        c = ohlc["close"]
        hold_nav = c / c.iloc[0]
        out = {"票": tk, "持有": perf(hold_nav)}
        rets = {}
        for v in ["v1", "B", "C", "v2"]:
            sr, inmkt = simulate(ohlc, v)
            nav = (1 + sr).cumprod()
            out[v] = perf(nav)
            out[v]["inmkt"] = inmkt
            rets[v] = sr
        # v2 alpha/beta vs SPY
        ab = qe.alpha_beta(rets["v2"], spy_ret, n_boot=400)
        out["v2α"] = ab["alpha_ann"]
        out["v2β"] = ab["beta"]
        out["v2α显著"] = ab["alpha_significant"]
        # 逐年 OOS：v2 年收益 vs 持有年收益
        yr_s = (1 + rets["v2"]).groupby(rets["v2"].index.year).prod() - 1
        yr_h = (1 + c.pct_change()).groupby(c.index.year).prod() - 1
        yj = pd.concat([yr_s.rename("s"), yr_h.rename("h")], axis=1).dropna()
        yj = yj[yj.index < 2026]
        out["v2逐年胜"] = float((yj["s"] > yj["h"]).mean()) if len(yj) else np.nan
        out["年数"] = len(yj)
        rows.append(out)
        v2_port_rets[tk] = rets["v2"]
        hold_port_rets[tk] = c.pct_change()
    except Exception as e:
        print(f"  {tk} 跳过: {type(e).__name__}: {str(e)[:50]}")

section("各票：年化 / 最大回撤 / 在场%  (入场统一 dip-15%，只换出场)")
print(f"{'票':<6}{'持有年化':>8}{'v1年化':>8}{'B年化':>8}{'C年化':>8}{'v2年化':>8} | "
      f"{'持有DD':>7}{'v2DD':>7}{'v2在场':>7} | {'v2-持有':>8}{'v2β':>6}{'v2α':>8}{'α显著':>6}")
for r in rows:
    h, v1, B, C, v2 = r["持有"], r["v1"], r["B"], r["C"], r["v2"]
    print(f"{r['票']:<6}{h['cagr']:>+8.0%}{v1['cagr']:>+8.0%}{B['cagr']:>+8.0%}{C['cagr']:>+8.0%}"
          f"{v2['cagr']:>+8.0%} | {h['maxdd']:>7.0%}{v2['maxdd']:>7.0%}{v2['inmkt']:>7.0%} | "
          f"{v2['cagr']-h['cagr']:>+8.0%}{r['v2β']:>6.2f}{r['v2α']:>+8.1%}{str(r['v2α显著']):>6}")

# ---------------------------------------------------------------------------
df = pd.DataFrame([{"票": r["票"], "持有": r["持有"]["cagr"], "v1": r["v1"]["cagr"],
                   "B": r["B"]["cagr"], "C": r["C"]["cagr"], "v2": r["v2"]["cagr"],
                   "持有DD": r["持有"]["maxdd"], "v2DD": r["v2"]["maxdd"],
                   "持有Sh": r["持有"]["sharpe"], "v2Sh": r["v2"]["sharpe"],
                   "v2α显著": r["v2α显著"], "v2α": r["v2α"], "v2逐年胜": r["v2逐年胜"]}
                  for r in rows]).set_index("票")
core = df.drop(index=["SPY"], errors="ignore")
win = core.loc[[t for t in WINNERS if t in core.index]]
lag = core.loc[[t for t in LAGGARDS if t in core.index]]

section("汇总（剔 SPY 看个股池；含 SPY 单列）")
for name, d in [("全池", core), ("赢家池", win), ("走平/跌池", lag)]:
    print(f"\n  【{name}】 N={len(d)}")
    print(f"    年化中位  持有 {d['持有'].median():+.1%} | v1 {d['v1'].median():+.1%} | "
          f"B {d['B'].median():+.1%} | C {d['C'].median():+.1%} | v2 {d['v2'].median():+.1%}")
    print(f"    夏普中位  持有 {d['持有Sh'].median():.2f} | v2 {d['v2Sh'].median():.2f}")
    print(f"    回撤中位  持有 {d['持有DD'].median():.0%} | v2 {d['v2DD'].median():.0%}")
    print(f"    v2 净跑赢持有(年化)占比: {(d['v2'] > d['持有']).mean():.0%}"
          f" | v2 α显著为正占比: {((d['v2α显著']) & (d['v2α'] > 0)).mean():.0%}")
    print(f"    v2 逐年跑赢持有率中位: {d['v2逐年胜'].median():.0%}")
if "SPY" in df.index:
    s = df.loc["SPY"]
    print(f"\n  【SPY 2000至今】 持有年化 {s['持有']:+.1%}(DD {s['持有DD']:.0%}) | "
          f"v2 {s['v2']:+.1%}(DD {s['v2DD']:.0%}) | v2 净跑赢? {s['v2']>s['持有']}")

# ---------------------------------------------------------------------------
section("组合层：等权 v2 组合 vs 等权闭眼持有 vs SPY（α/β）")
ov = pd.DataFrame(v2_port_rets).drop(columns=["SPY"], errors="ignore")
hd = pd.DataFrame(hold_port_rets).drop(columns=["SPY"], errors="ignore")
common = ov.dropna(how="all").index.intersection(hd.dropna(how="all").index)
ovr = ov.loc[common].mean(axis=1)
hdr = hd.loc[common].mean(axis=1)
spc = spy_ret.loc[spy_ret.index.intersection(common)]
ov_nav = (1 + ovr.fillna(0)).cumprod()
hd_nav = (1 + hdr.fillna(0)).cumprod()
pov, phd = perf(ov_nav), perf(hd_nav)
ab_spy = qe.alpha_beta(ovr, spy_ret, n_boot=1000)
ab_hold = qe.alpha_beta(ovr.loc[common], hdr.loc[common], n_boot=1000)
print(f"  等权 v2 组合   年化 {pov['cagr']:+.1%} | 夏普 {pov['sharpe']:.2f} | 回撤 {pov['maxdd']:.0%}")
print(f"  等权闭眼持有   年化 {phd['cagr']:+.1%} | 夏普 {phd['sharpe']:.2f} | 回撤 {phd['maxdd']:.0%}")
print(f"  v2 vs SPY :  α {ab_spy['alpha_ann']:+.1%} CI[{ab_spy['alpha_ann_ci'][0]:+.1%},{ab_spy['alpha_ann_ci'][1]:+.1%}]"
      f"  β {ab_spy['beta']:.2f}  显著?{ab_spy['alpha_significant']}")
print(f"  v2 vs 同篮子持有: α {ab_hold['alpha_ann']:+.1%} CI[{ab_hold['alpha_ann_ci'][0]:+.1%},{ab_hold['alpha_ann_ci'][1]:+.1%}]"
      f"  β {ab_hold['beta']:.2f}  显著?{ab_hold['alpha_significant']}")

print("\n" + "=" * 88 + "\n完成。\n" + "=" * 88)
