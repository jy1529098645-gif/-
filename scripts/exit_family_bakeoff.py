"""出场族总烤测：把对话里提过的所有出场方案放进同一逐日引擎，用统一数学标准排名。

入场统一 dip-15%(控制变量)，除非标注[改重入]/[连续定仓]。无前视(t暴露用到t收盘，作用于t→t+1)；
空仓现金0；暴露变动按 fees+slip 计摩擦。

候选：
  hold      闭眼持有(基准)
  v1        固定30%移动止损(现行)
  B         收盘跌破200线即清(迟钝版)
  slope200  200线斜率转负即清(领先于破线)
  macd      MACD 下穿信号线即清
  C         Chandelier 贴顶 ATR×3
  chand5    Chandelier 宽止损 ATR×5(让利润更奔跑)
  ratchet   利润棘轮：盈利越多止损越紧(40%→15%)
  v2        Chandelier动态k + 分批撤离(快，上一轮的)
  voltarget [连续定仓] 波动目标(始终在场、只缩放，不二元进出)
  v3        [改重入+连续定仓] 趋势门(站上200线即在场，不等深跌) × 波动目标 × 仅"200下方且斜率转负"才清

数学排名标准(可审计)：组合层(等权) **夏普** 为主，并报 **对同篮子闭眼持有的 α(信息比口径)** +
显著性 + 回撤。"最成立" = 夏普最高且 α 不显著为负(择时未损害风险调整后回报)。
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
from regime import observables as ob
import config

_CFG = config.load_config()
FEE = float(_CFG["costs"]["fees"]) + float(_CFG["costs"]["slippage"])
TVOL = 0.25  # 波动目标(年化)

END = "2026-06-06"
WINNERS = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "AMD", "TSM", "NFLX"]
LAGGARDS = ["INTC", "PYPL", "DIS", "BA", "F", "T", "VZ", "NKE", "MMM", "QCOM"]
POOL = WINNERS + LAGGARDS
VARIANTS = ["v1", "B", "slope200", "macd", "C", "chand5", "ratchet", "v2", "voltarget", "v3"]


def atr(o, n=22):
    h, l, c = o["high"], o["low"], o["close"]
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


def simulate_all(o):
    """一次性算出所有变体的逐日策略收益。返回 {variant: ret_series}, inmkt dict。"""
    c = o["close"]
    ret = c.pct_change().fillna(0.0)
    entry = sg.dip_from_high(c, lookback=20, pct=0.15).reindex(c.index).fillna(False).values
    a = atr(o, 22).values
    sma50 = c.rolling(50, min_periods=25).mean()
    sma200 = c.rolling(200, min_periods=100).mean().values
    s200 = c.rolling(200, min_periods=100).mean()
    slope200 = (s200 > s200.shift(20)).values
    ext = (c - sma50) / sma50
    ext_std = ext.rolling(252, min_periods=120).std()
    overheat = (ext > 2 * ext_std).values
    low20 = c.rolling(20, min_periods=10).min().shift(1).values
    vpx = ob.realized_vol_percentile(c, 21, 252).reindex(c.index).values
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    macd_up = (macd > sig).values
    ewmav = qe.ewma_vol(ret).values  # 年化 ex-ante 波动
    px = c.values
    N = len(px)

    def vt(t):  # 波动目标暴露(无前视，ewma 已 shift)
        v = ewmav[t]
        if not (v == v) or v <= 0:
            return 0.0
        return float(min(1.0, TVOL / v))

    expo = {v: np.zeros(N) for v in VARIANTS}
    inmkt = {}
    for v in VARIANTS:
        in_pos = False
        peak = entry_px = np.nan
        for t in range(N):
            cur = 0.0
            if v == "voltarget":            # 始终在场，仅缩放
                cur = vt(t)
            elif v == "v3":                 # 趋势门 + 重入快 + 连续定仓
                if px[t] > sma200[t]:
                    cur = vt(t)
                elif (px[t] < sma200[t]) and (not slope200[t]):
                    cur = 0.0
                else:
                    cur = vt(t) * 0.5       # 200下方但斜率未转负：半仓过渡
            else:                            # 二元/dip重入族
                if not in_pos:
                    if entry[t]:
                        in_pos, peak, entry_px, cur = True, px[t], px[t], 1.0
                else:
                    peak = max(peak, px[t])
                    if v == "v1":
                        cur = 0.0 if px[t] < peak * 0.70 else 1.0
                    elif v == "B":
                        cur = 0.0 if px[t] < sma200[t] else 1.0
                    elif v == "slope200":
                        cur = 0.0 if not slope200[t] else 1.0
                    elif v == "macd":
                        cur = 0.0 if not macd_up[t] else 1.0
                    elif v == "C":
                        stop = peak - 3.0 * (a[t] if a[t] == a[t] else peak * .05)
                        cur = 0.0 if px[t] < stop else 1.0
                    elif v == "chand5":
                        stop = peak - 5.0 * (a[t] if a[t] == a[t] else peak * .05)
                        cur = 0.0 if px[t] < stop else 1.0
                    elif v == "ratchet":
                        gain = peak / entry_px - 1.0
                        trail = max(0.15, 0.40 - 0.6 * gain)
                        cur = 0.0 if px[t] < peak * (1 - trail) else 1.0
                    elif v == "v2":
                        k = 1.5 if overheat[t] else 3.0
                        stop = peak - k * (a[t] if a[t] == a[t] else peak * .05)
                        if px[t] < stop:
                            cur = 0.0
                        else:
                            tr = 0
                            if low20[t] == low20[t] and px[t] < low20[t]:
                                tr += 1
                            if not slope200[t]:
                                tr += 1
                            if vpx[t] == vpx[t] and vpx[t] > 0.90:
                                tr += 1
                            cur = max(0.0, 1.0 - tr / 3.0)
                    if cur == 0.0:
                        in_pos = False
            expo[v][t] = cur

    rets = {}
    for v in VARIANTS:
        es = pd.Series(expo[v], index=c.index).shift(1).fillna(0.0)
        turn = es.diff().abs().fillna(0.0)
        rets[v] = es * ret - turn * FEE
        inmkt[v] = float((es > 0).mean())
    return rets, inmkt


def section(t):
    print("\n" + "=" * 92 + f"\n{t}\n" + "-" * 92)


spy_o = loader.load_ohlcv("SPY", "2000-01-01", END)
spy_ret = spy_o["close"].pct_change()

per_stock = {}      # tk -> {variant: perf, 'hold': perf}
port_rets = {v: {} for v in VARIANTS}
hold_rets = {}
for tk in POOL:
    try:
        o = loader.load_ohlcv(tk, "2000-01-01", END).dropna()
        if o.shape[0] < 252:
            continue
        c = o["close"]
        rets, inmkt = simulate_all(o)
        rec = {"hold": perf(c / c.iloc[0])}
        for v in VARIANTS:
            rec[v] = perf((1 + rets[v]).cumprod())
            rec[v]["inmkt"] = inmkt[v]
            port_rets[v][tk] = rets[v]
        per_stock[tk] = rec
        hold_rets[tk] = c.pct_change()
    except Exception as e:
        print(f"  {tk} 跳过 {type(e).__name__}: {str(e)[:40]}")

# ---------------------------------------------------------------------------
section("组合层(等权全池)：年化 / 夏普 / 回撤 / 在场% + α(vs SPY) + α(vs 同篮子持有) —— 数学排名")
hd = pd.DataFrame(hold_rets)
common = hd.dropna(how="all").index
hold_port = hd.loc[common].mean(axis=1)
hold_perf = perf((1 + hold_port.fillna(0)).cumprod())

table = []
for v in VARIANTS:
    pr = pd.DataFrame(port_rets[v]).reindex(common)
    r = pr.mean(axis=1)
    pf = perf((1 + r.fillna(0)).cumprod())
    inmkt_med = float(np.median([per_stock[t][v]["inmkt"] for t in per_stock]))
    ab_spy = qe.alpha_beta(r, spy_ret, n_boot=600)
    ab_hold = qe.alpha_beta(r.loc[common], hold_port.loc[common], n_boot=600)
    table.append({"出场": v, "年化": pf["cagr"], "夏普": pf["sharpe"], "回撤": pf["maxdd"],
                  "在场%": inmkt_med, "βvsSPY": ab_spy["beta"],
                  "αvsSPY": ab_spy["alpha_ann"], "αSPY显著": ab_spy["alpha_significant"],
                  "αvs持有": ab_hold["alpha_ann"], "α持有CI": ab_hold["alpha_ann_ci"],
                  "α持有显著": ab_hold["alpha_significant"]})
# 加入 hold 自身
table.append({"出场": "hold", "年化": hold_perf["cagr"], "夏普": hold_perf["sharpe"],
              "回撤": hold_perf["maxdd"], "在场%": 1.0, "βvsSPY": np.nan, "αvsSPY": np.nan,
              "αSPY显著": False, "αvs持有": 0.0, "α持有CI": (0, 0), "α持有显著": False})
T = pd.DataFrame(table).set_index("出场")
T = T.sort_values("夏普", ascending=False)
print(f"{'出场':<10}{'年化':>7}{'夏普':>7}{'回撤':>7}{'在场%':>7}{'βvsSPY':>8}{'αvsSPY':>8}{'αvs持有':>9}{'α持有CI':>18}{'显著':>5}")
for v, r in T.iterrows():
    ci = f"[{r['α持有CI'][0]:+.1%},{r['α持有CI'][1]:+.1%}]" if v != "hold" else "    —"
    print(f"{v:<10}{r['年化']:>+7.1%}{r['夏普']:>7.2f}{r['回撤']:>7.0%}{r['在场%']:>7.0%}"
          f"{r['βvsSPY']:>8.2f}{r['αvsSPY']:>+8.1%}{r['αvs持有']:>+9.1%}{ci:>18}{str(r['α持有显著']):>5}")

# ---------------------------------------------------------------------------
section("分池年化中位(剔SPY)：看哪类票上哪种出场更优")
rows = []
for v in ["hold"] + VARIANTS:
    w = np.median([per_stock[t][v]["cagr"] for t in per_stock if t in WINNERS])
    l = np.median([per_stock[t][v]["cagr"] for t in per_stock if t in LAGGARDS])
    a = np.median([per_stock[t][v]["cagr"] for t in per_stock])
    dd = np.median([per_stock[t][v]["maxdd"] for t in per_stock])
    rows.append({"出场": v, "赢家中位": w, "走平跌中位": l, "全池中位": a, "全池回撤中位": dd})
R = pd.DataFrame(rows).set_index("出场")
print(f"{'出场':<10}{'赢家中位':>9}{'走平跌中位':>11}{'全池中位':>9}{'全池回撤中位':>12}")
for v, r in R.iterrows():
    print(f"{v:<10}{r['赢家中位']:>+9.1%}{r['走平跌中位']:>+11.1%}{r['全池中位']:>+9.1%}{r['全池回撤中位']:>12.0%}")

# ---------------------------------------------------------------------------
section("数学最成立的出场(夏普最高且 α 不显著为负)")
cand = T[~((T.index != "hold") & (T["α持有显著"]) & (T["αvs持有"] < 0))]
best = cand["夏普"].idxmax()
br = T.loc[best]
print(f"  >>> {best}：组合夏普 {br['夏普']:.2f}(持有 {hold_perf['sharpe']:.2f}) | 年化 {br['年化']:+.1%}"
      f"(持有 {hold_perf['cagr']:+.1%}) | 回撤 {br['回撤']:.0%}(持有 {hold_perf['maxdd']:.0%})")
print(f"      α vs SPY {br['αvsSPY']:+.1%}(显著{br['αSPY显著']}) | α vs 同篮子持有 {br['αvs持有']:+.1%}(显著{br['α持有显著']})")

print("\n" + "=" * 92 + "\n完成。\n" + "=" * 92)
