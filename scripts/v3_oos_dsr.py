"""v3 终审：walk-forward 分段 OOS + 剔危机年稳健性 + Deflated Sharpe(多重检验折扣)。

v3 参数固定(非逐窗拟合)，故 OOS 检验的是"优势是否跨期稳定、是否只靠少数危机年喂出来"。
Deflated Sharpe 把"我们试了 10 个出场方案才挑出 v3"这件事计入折扣——治数据窥探。
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
from stats.deflated_sharpe import deflated_sharpe_ratio, probabilistic_sharpe_ratio, expected_max_sharpe
import config

_CFG = config.load_config()
FEE = float(_CFG["costs"]["fees"]) + float(_CFG["costs"]["slippage"])
TVOL = 0.25
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


def sharpe_daily(r):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan


def simulate_all(o):
    c = o["close"]
    ret = c.pct_change().fillna(0.0)
    entry = sg.dip_from_high(c, lookback=20, pct=0.15).reindex(c.index).fillna(False).values
    a = atr(o, 22).values
    sma50 = c.rolling(50, min_periods=25).mean()
    s200 = c.rolling(200, min_periods=100).mean()
    sma200 = s200.values
    slope200 = (s200 > s200.shift(20)).values
    ext = (c - sma50) / sma50
    ext_std = ext.rolling(252, min_periods=120).std()
    overheat = (ext > 2 * ext_std).values
    low20 = c.rolling(20, min_periods=10).min().shift(1).values
    vpx = ob.realized_vol_percentile(c, 21, 252).reindex(c.index).values
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sigl = macd.ewm(span=9, adjust=False).mean()
    macd_up = (macd > sigl).values
    ewmav = qe.ewma_vol(ret).values
    px = c.values
    N = len(px)

    def vt(t):
        v = ewmav[t]
        return float(min(1.0, TVOL / v)) if (v == v and v > 0) else 0.0

    expo = {v: np.zeros(N) for v in VARIANTS}
    for v in VARIANTS:
        in_pos = False
        peak = entry_px = np.nan
        for t in range(N):
            cur = 0.0
            if v == "voltarget":
                cur = vt(t)
            elif v == "v3":
                if px[t] > sma200[t]:
                    cur = vt(t)
                elif (px[t] < sma200[t]) and (not slope200[t]):
                    cur = 0.0
                else:
                    cur = vt(t) * 0.5
            else:
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
                        cur = 0.0 if px[t] < peak - 3.0 * (a[t] if a[t] == a[t] else peak * .05) else 1.0
                    elif v == "chand5":
                        cur = 0.0 if px[t] < peak - 5.0 * (a[t] if a[t] == a[t] else peak * .05) else 1.0
                    elif v == "ratchet":
                        gain = peak / entry_px - 1.0
                        trail = max(0.15, 0.40 - 0.6 * gain)
                        cur = 0.0 if px[t] < peak * (1 - trail) else 1.0
                    elif v == "v2":
                        k = 1.5 if overheat[t] else 3.0
                        if px[t] < peak - k * (a[t] if a[t] == a[t] else peak * .05):
                            cur = 0.0
                        else:
                            tr = (low20[t] == low20[t] and px[t] < low20[t]) + (not slope200[t]) + \
                                 (vpx[t] == vpx[t] and vpx[t] > 0.90)
                            cur = max(0.0, 1.0 - tr / 3.0)
                    if cur == 0.0:
                        in_pos = False
            expo[v][t] = cur
    rets = {}
    for v in VARIANTS:
        es = pd.Series(expo[v], index=c.index).shift(1).fillna(0.0)
        rets[v] = es * ret - es.diff().abs().fillna(0.0) * FEE
    return rets


def section(t):
    print("\n" + "=" * 90 + f"\n{t}\n" + "-" * 90)


spy_o = loader.load_ohlcv("SPY", "2000-01-01", END)
spy_ret = spy_o["close"].pct_change()

allv = {v: {} for v in VARIANTS}
hold_rets = {}
for tk in POOL:
    try:
        o = loader.load_ohlcv(tk, "2000-01-01", END).dropna()
        if o.shape[0] < 252:
            continue
        rets = simulate_all(o)
        for v in VARIANTS:
            allv[v][tk] = rets[v]
        hold_rets[tk] = o["close"].pct_change()
    except Exception as e:
        print(f"  {tk} 跳过 {type(e).__name__}")

hd = pd.DataFrame(hold_rets)
common = hd.dropna(how="all").index
hold_port = hd.loc[common].mean(axis=1)
v3 = pd.DataFrame(allv["v3"]).reindex(common).mean(axis=1)
spy = spy_ret.reindex(common)

# ---------------------------------------------------------------------------
section("1) 分年 OOS：v3 vs 同篮子持有 vs SPY（参数固定，逐年即样本外）")
print(f"{'年份':<6}{'v3':>8}{'持有篮子':>9}{'SPY':>8}{'v3夏普':>8}{'持有夏普':>9}{'v3超额':>8}  胜")
yrs = sorted(set(common.year))
beat_ret, beat_sh, crisis_years = 0, 0, []
n_yr = 0
for y in yrs:
    if y >= 2026:
        continue
    m = common.year == y
    if m.sum() < 60:
        continue
    v3y, hdy, spyy = v3[m], hold_port[m], spy[m]
    v3r = (1 + v3y).prod() - 1
    hdr = (1 + hdy).prod() - 1
    spyr = (1 + spyy.fillna(0)).prod() - 1
    sv, sh = sharpe_daily(v3y), sharpe_daily(hdy)
    crisis = hdr < -0.10 or spyr < -0.10
    if crisis:
        crisis_years.append(y)
    bw = v3r > hdr
    bs = (sv > sh) if (sv == sv and sh == sh) else False
    beat_ret += bw
    beat_sh += bs
    n_yr += 1
    print(f"{y:<6}{v3r:>+8.0%}{hdr:>+9.0%}{spyr:>+8.0%}{sv:>8.2f}{sh:>9.2f}{v3r-hdr:>+8.0%}"
          f"  {'✅' if bw else ' '}{'★夏普' if bs else ''}{'  ⚠危机' if crisis else ''}")
print(f"\n  逐年 v3 收益跑赢持有: {beat_ret}/{n_yr} ({beat_ret/n_yr:.0%}) | "
      f"v3 夏普跑赢持有: {beat_sh}/{n_yr} ({beat_sh/n_yr:.0%})")
print(f"  危机年(篮子或SPY跌>10%): {crisis_years}")

# ---------------------------------------------------------------------------
section("2) 剔危机年稳健性：edge 是不是只靠崩盘那几年喂出来的？")
for label, mask in [("全样本", np.ones(len(common), bool)),
                    ("剔危机年", ~np.isin(common.year, crisis_years)),
                    ("仅危机年", np.isin(common.year, crisis_years))]:
    idx = common[mask]
    ab = qe.alpha_beta(v3.loc[idx], hold_port.loc[idx], n_boot=800)
    abs_ = qe.alpha_beta(v3.loc[idx], spy.loc[idx], n_boot=800)
    print(f"  {label:<8} N={mask.sum():>5} | v3夏普 {sharpe_daily(v3.loc[idx]):>5.2f} "
          f"持有夏普 {sharpe_daily(hold_port.loc[idx]):>5.2f} | "
          f"α vs持有 {ab['alpha_ann']:+.1%}{'✔' if ab['alpha_significant'] else '✘'} "
          f"CI[{ab['alpha_ann_ci'][0]:+.1%},{ab['alpha_ann_ci'][1]:+.1%}] | "
          f"α vsSPY {abs_['alpha_ann']:+.1%}{'✔' if abs_['alpha_significant'] else '✘'}")

# ---------------------------------------------------------------------------
section("3) Deflated Sharpe：把'试了10个出场方案才挑出v3'计入多重检验折扣")
trial_sh = []
for v in VARIANTS:
    r = pd.DataFrame(allv[v]).reindex(common).mean(axis=1)
    trial_sh.append(sharpe_daily(r) / np.sqrt(252))   # 日度口径
trial_sh = np.array([s for s in trial_sh if s == s])
sr_v3_d = sharpe_daily(v3) / np.sqrt(252)
n_obs = int(v3.dropna().shape[0])
sk = float(v3.dropna().skew())
ku = float(v3.dropna().kurt() + 3.0)   # pandas kurt 是超额峰度，+3 得全峰度
sr0 = expected_max_sharpe(float(trial_sh.std(ddof=1)), n_trials=len(trial_sh))
dsr = deflated_sharpe_ratio(sr_v3_d, float(trial_sh.std(ddof=1)), n_trials=len(trial_sh),
                            n_obs=n_obs, skew=sk, kurtosis=ku)
psr0 = probabilistic_sharpe_ratio(sr_v3_d, 0.0, n_obs, skew=sk, kurtosis=ku)
sr_hold_d = sharpe_daily(hold_port) / np.sqrt(252)
psr_h = probabilistic_sharpe_ratio(sr_v3_d, sr_hold_d, n_obs, skew=sk, kurtosis=ku)
print(f"  v3 日度夏普 {sr_v3_d:.4f} (年化 {sr_v3_d*np.sqrt(252):.2f}) | 样本 {n_obs} 日 | 偏度 {sk:+.2f} 峰度 {ku:.1f}")
print(f"  试验数 N={len(trial_sh)}，各方案日夏普 std={trial_sh.std(ddof=1):.4f} → 期望最大夏普(应扣基准) SR0={sr0:.4f}")
print(f"  PSR(vs 0)        = {psr0:.3f}  (夏普真>0 的概率)")
print(f"  PSR(vs 持有夏普) = {psr_h:.3f}  (v3 夏普真>持有 的概率)")
print(f"  **Deflated Sharpe = {dsr:.3f}**  (扣多重检验后仍>0 的概率；>0.95 算稳健) → {'稳健 ✅' if dsr > 0.95 else '存疑 ⚠️'}")

# ---------------------------------------------------------------------------
section("4) 终审结论")
robust_oos = beat_sh / n_yr >= 0.55
robust_excrisis = True  # 由上表人读
print(f"  · 跨期稳定: 逐年夏普跑赢持有 {beat_sh}/{n_yr} ({beat_sh/n_yr:.0%}) → {'稳 ✅' if robust_oos else '一般 ⚠️'}")
print(f"  · 多重检验: Deflated Sharpe {dsr:.2f} → {'过 0.95 闸 ✅' if dsr > 0.95 else '未过 0.95 ⚠️'}")
print(f"  · 看上方'剔危机年'行判断 edge 是否依赖崩盘年。")
print("\n" + "=" * 90 + "\n完成。\n" + "=" * 90)
