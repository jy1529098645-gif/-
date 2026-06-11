"""实测：完全按工具(position_guidance 生产引擎)给的建仓/离场暴露回测，输出全套指标。

口径：
  · 暴露序列 = analysis.position_guidance._exposure_series（UI/作战卡实际调用的函数本体，
    含 200线趋势门 × 波动目标连续定仓 × 低波动杠杆门），无前视：t 暴露只用到 t 收盘。
  · 当日策略收益 = 昨日暴露 × 当日收益 − |暴露变动| × 单边成本(10bps)。
  · 对照：闭眼买入持有(exposure≡1) 与 SPY。
  · 指标：CAGR / 年化波动 / 夏普 / Sortino / 最大回撤 / Calmar / 在场占比 / 年换手 /
    逐年夏普胜率 / α·β(vs 持有, bootstrap CI)。
不预测、不择时圣杯——验证工具自己的诚实口径：撤离=崩盘保险，风险调整更优+砍回撤，绝对收益不跑赢长牛。
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from data import loader
from analysis import position_guidance as pg
from analysis import quant_edge as qe
import config

CFG = config.load_config()
FEE = float(CFG["costs"]["fees"]) + float(CFG["costs"]["slippage"])  # 单边 10bps
END = "2026-06-06"
START = "2010-01-01"
MAG7 = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]
LAGGARDS = ["INTC", "PYPL", "DIS", "BA", "QCOM", "NKE"]
PROFKEYS = ["conservative", "moderate", "aggressive", "leveraged"]
ANN = 252


def metrics(r: pd.Series, expo: pd.Series | None = None) -> dict:
    r = r.dropna()
    if r.empty:
        return {}
    n = len(r)
    cum = (1 + r).prod()
    cagr = cum ** (ANN / n) - 1
    vol = r.std() * np.sqrt(ANN)
    sharpe = r.mean() / r.std() * np.sqrt(ANN) if r.std() > 0 else np.nan
    downside = r[r < 0].std() * np.sqrt(ANN)
    sortino = r.mean() * ANN / downside if downside and downside == downside else np.nan
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0)
    maxdd = float(dd.min())
    calmar = cagr / abs(maxdd) if maxdd < 0 else np.nan
    out = {"CAGR": cagr, "vol": vol, "sharpe": sharpe, "sortino": sortino,
           "maxdd": maxdd, "calmar": calmar, "total": cum - 1, "n": n}
    if expo is not None:
        e = expo.reindex(r.index).fillna(0.0)
        out["in_mkt"] = float((e > 0).mean())
        out["avg_expo"] = float(e.mean())
        # 年换手：日均|Δ暴露| × 252
        out["turnover_yr"] = float(e.diff().abs().mean() * ANN)
    return out


def strat_returns(price: pd.Series, tvol: float, max_lev: float):
    ret = price.pct_change()
    expo = pg._exposure_series(price, tvol, max_lev)
    pos = expo.shift(1)
    sr = pos * ret - pos.diff().abs().fillna(0.0) * FEE
    return sr.dropna(), expo


def yearly_beat(strat: pd.Series, hold: pd.Series):
    idx = strat.dropna().index.intersection(hold.dropna().index)
    s, h = strat.loc[idx], hold.loc[idx]
    yrs = sorted(set(idx.year))
    bsh = bret = nb = 0
    rows = []
    for y in yrs:
        if y >= 2026:
            continue
        m = idx.year == y
        if m.sum() < 60:
            continue
        sy, hy = s[m], h[m]
        sr_s = sy.mean()/sy.std()*np.sqrt(ANN) if sy.std() > 0 else np.nan
        sr_h = hy.mean()/hy.std()*np.sqrt(ANN) if hy.std() > 0 else np.nan
        rs = (1+sy).prod()-1; rh = (1+hy).prod()-1
        b_s = sr_s > sr_h if (sr_s==sr_s and sr_h==sr_h) else False
        b_r = rs > rh
        bsh += b_s; bret += b_r; nb += 1
        rows.append((y, rs, rh, sr_s, sr_h, b_s, rh < -0.10))
    return rows, bsh, bret, nb


def fmt(m, keys):
    return "  ".join(f"{k}={m.get(k, float('nan')):+.2f}" if isinstance(m.get(k), float) else f"{k}={m.get(k)}" for k in keys)


print("="*100)
print(f"完全按工具(position_guidance)建仓/离场暴露回测 · {START}→{END} · 成本{FEE*1e4:.0f}bps单边")
print("="*100)

# SPY 基准收益
spy_px = loader.load_ohlcv("SPY", START, END)["close"].dropna()
spy_ret = spy_px.pct_change()

# ---- 1) 逐票（中性档 moderate）----
print("\n【1】逐票 · 中性档(moderate, 波动目标25%·上限1.0×) · 工具暴露 vs 闭眼持有")
print(f"{'票':<6}{'年化':>7}{'持有年化':>9}{'夏普':>6}{'持有夏普':>9}{'Sortino':>8}{'最大回撤':>9}{'持有回撤':>9}{'Calmar':>7}{'在场':>6}{'年换手':>7}")
pool_strat, pool_hold = {}, {}
for tk in MAG7:
    try:
        px = loader.load_ohlcv(tk, START, END)["close"].dropna()
        if len(px) < 252:
            print(f"{tk:<6} 数据不足"); continue
        sr, expo = strat_returns(px, pg.PROFILES["moderate"]["tvol"], pg.PROFILES["moderate"]["max_lev"])
        hr = px.pct_change().dropna()
        ms = metrics(sr, expo); mh = metrics(hr)
        pool_strat[tk] = sr; pool_hold[tk] = hr
        print(f"{tk:<6}{ms['CAGR']:>+7.0%}{mh['CAGR']:>+9.0%}{ms['sharpe']:>6.2f}{mh['sharpe']:>9.2f}"
              f"{ms['sortino']:>8.2f}{ms['maxdd']:>+9.0%}{mh['maxdd']:>+9.0%}{ms['calmar']:>7.2f}"
              f"{ms['in_mkt']:>6.0%}{ms['turnover_yr']:>7.1f}")
    except Exception as e:
        print(f"{tk:<6} 跳过 {type(e).__name__}: {e}")

# ---- 2) 池化等权组合（中性档）vs 持有 vs SPY ----
print("\n【2】Mag7 等权池化组合 · 中性档 · vs 闭眼持有 vs SPY")
sdf = pd.DataFrame(pool_strat); hdf = pd.DataFrame(pool_hold)
common = sdf.dropna(how="all").index.intersection(hdf.dropna(how="all").index)
strat_port = sdf.reindex(common).mean(axis=1)
hold_port = hdf.reindex(common).mean(axis=1)
spy_c = spy_ret.reindex(common)
mS = metrics(strat_port); mH = metrics(hold_port); mY = metrics(spy_c.dropna())
for nm, m in [("工具暴露组合", mS), ("闭眼持有组合", mH), ("SPY", mY)]:
    print(f"  {nm:<12} CAGR {m['CAGR']:+.1%} | 波动 {m['vol']:.1%} | 夏普 {m['sharpe']:.2f} | "
          f"Sortino {m['sortino']:.2f} | 最大回撤 {m['maxdd']:+.1%} | Calmar {m['calmar']:.2f}")
ab = qe.alpha_beta(strat_port.loc[common], hold_port.loc[common], n_boot=1000)
aby = qe.alpha_beta(strat_port.loc[common], spy_c.loc[common].fillna(0), n_boot=1000)
print(f"  α vs 持有: {ab['alpha_ann']:+.1%} (β {ab['beta']:.2f}, CI[{ab['alpha_ann_ci'][0]:+.1%},{ab['alpha_ann_ci'][1]:+.1%}], "
      f"{'显著✅' if ab['alpha_significant'] else '不显著'})")
print(f"  α vs SPY:  {aby['alpha_ann']:+.1%} (β {aby['beta']:.2f}, CI[{aby['alpha_ann_ci'][0]:+.1%},{aby['alpha_ann_ci'][1]:+.1%}], "
      f"{'显著✅' if aby['alpha_significant'] else '不显著'})")

# ---- 3) 逐年 OOS（池化中性档） ----
print("\n【3】逐年 · 工具组合 vs 持有组合（参数固定→逐年即样本外）")
rows, bsh, bret, nb = yearly_beat(strat_port, hold_port)
print(f"{'年':<6}{'工具':>8}{'持有':>8}{'工具夏普':>9}{'持有夏普':>9}  胜夏普 危机")
for y, rs, rh, ss, sh, b_s, crisis in rows:
    print(f"{y:<6}{rs:>+8.0%}{rh:>+8.0%}{ss:>9.2f}{sh:>9.2f}  {'★' if b_s else ' '}    {'⚠' if crisis else ''}")
print(f"  逐年夏普跑赢持有 {bsh}/{nb} ({bsh/nb:.0%}) | 逐年收益跑赢持有 {bret}/{nb} ({bret/nb:.0%})")
cri = [r for r in rows if r[6]]
if cri:
    print(f"  危机年(持有跌>10%): " + ", ".join(f"{r[0]}(工具{r[1]:+.0%} vs 持有{r[2]:+.0%})" for r in cri))

# ---- 4) 四档风险偏好对比（池化） ----
print("\n【4】四档风险偏好 · Mag7池化 · 暴露规则差异（年化/夏普/回撤/Calmar/在场/年换手）")
print(f"{'档':<14}{'波动目标':>8}{'杠杆':>6}{'CAGR':>8}{'夏普':>6}{'最大回撤':>9}{'Calmar':>7}{'在场':>6}{'年换手':>7}")
# 预取各票价格一次，复用
_px_cache = {tk: loader.load_ohlcv(tk, START, END)["close"].dropna() for tk in MAG7}
for pk in PROFKEYS:
    cfg = pg.PROFILES[pk]
    cols, inms, tovs = {}, [], []
    for tk, px in _px_cache.items():
        sr, expo = strat_returns(px, cfg["tvol"], cfg["max_lev"])
        cols[tk] = sr
        mm = metrics(sr, expo)
        inms.append(mm.get("in_mkt", np.nan)); tovs.append(mm.get("turnover_yr", np.nan))
    port = pd.DataFrame(cols).reindex(common).mean(axis=1)
    m = metrics(port)
    print(f"{pg.PROFILE_ZH[pk]:<14}{cfg['tvol']:>8.0%}{cfg['max_lev']:>5.1f}×{m['CAGR']:>+8.0%}{m['sharpe']:>6.2f}"
          f"{m['maxdd']:>+9.0%}{m['calmar']:>7.2f}{np.nanmean(inms):>6.0%}{np.nanmean(tovs):>7.1f}")

# ---- 5) 落后股 sanity（不是只在赢家上成立？）----
print("\n【5】落后/平庸股 sanity（中性档）· 工具暴露 vs 持有")
print(f"{'票':<6}{'工具CAGR':>9}{'持有CAGR':>9}{'工具夏普':>9}{'持有夏普':>9}{'工具回撤':>9}{'持有回撤':>9}")
lag_s, lag_h = {}, {}
for tk in LAGGARDS:
    try:
        px = loader.load_ohlcv(tk, START, END)["close"].dropna()
        if len(px) < 252: continue
        sr, expo = strat_returns(px, pg.PROFILES["moderate"]["tvol"], pg.PROFILES["moderate"]["max_lev"])
        hr = px.pct_change().dropna()
        ms, mh = metrics(sr, expo), metrics(hr)
        lag_s[tk] = sr; lag_h[tk] = hr
        print(f"{tk:<6}{ms['CAGR']:>+9.0%}{mh['CAGR']:>+9.0%}{ms['sharpe']:>9.2f}{mh['sharpe']:>9.2f}{ms['maxdd']:>+9.0%}{mh['maxdd']:>+9.0%}")
    except Exception as e:
        print(f"{tk:<6} 跳过 {type(e).__name__}")
if lag_s:
    lc = pd.DataFrame(lag_s).dropna(how="all").index.intersection(pd.DataFrame(lag_h).dropna(how="all").index)
    lsp = pd.DataFrame(lag_s).reindex(lc).mean(axis=1); lhp = pd.DataFrame(lag_h).reindex(lc).mean(axis=1)
    mls, mlh = metrics(lsp), metrics(lhp)
    print(f"  落后池组合 工具: CAGR {mls['CAGR']:+.1%} 夏普 {mls['sharpe']:.2f} 回撤 {mls['maxdd']:+.0%}  |  "
          f"持有: CAGR {mlh['CAGR']:+.1%} 夏普 {mlh['sharpe']:.2f} 回撤 {mlh['maxdd']:+.0%}")

print("\n" + "="*100)
print("完成。解读见脚本末尾结论。")
print("="*100)
