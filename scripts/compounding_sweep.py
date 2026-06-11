"""复利迭代研究：系统扫描各种暴露方案，找"复利逼近持有 + 保留回撤保护"的更优默认。

诊断：现行 v3 的 200线趋势门"破位砍到 0"是复利主拖累（V型反弹割在低点、追在高点）。
本扫描把暴露引擎参数化，系统试：破位地板(不砍到0) / 纯波动目标 / 提高波动目标 / 低波动加杠杆 /
深跌加仓 / 组合，并在 Mag7 + 落后股 + SPY 上、全样本与分段都评，抗过拟合。

口径与生产一致：无前视(t 暴露只用到 t 收盘)；当日收益=昨暴露×当日收益−|Δ暴露|×10bps。
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from data import loader
from analysis.quant_edge import ewma_vol
from regime.observables import realized_vol_percentile
import config

FEE = float(config.load_config()["costs"]["fees"]) * 2 if False else 0.0010  # 单边 10bps
END = "2026-06-06"; START = "2010-01-01"; ANN = 252
MAG7 = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]
LAGGARDS = ["INTC", "PYPL", "DIS", "BA", "QCOM", "NKE", "F", "VZ"]


# ---------------------------------------------------------------------------
# 参数化暴露引擎（无前视）
# ---------------------------------------------------------------------------
def exposure(price, tvol=0.25, max_lev=1.0, ma_win=200, slope_win=20,
             break_floor=0.0, slope_floor=0.5, lev_volpct_max=0.50,
             dip_add=0.0, dip_thresh=0.15, gate=True):
    ret = price.pct_change()
    ewmav = ewma_vol(ret)
    base = (tvol / ewmav).clip(upper=max_lev)
    if max_lev > 1.0:                                  # 低波动门：高波动处把杠杆收回 1.0
        vp = realized_vol_percentile(price, 21, 252)
        base = base.where(vp <= lev_volpct_max, base.clip(upper=1.0))
    if not gate:
        expo = base
    else:
        ma = price.rolling(ma_win, min_periods=ma_win // 2).mean()
        slope_up = ma > ma.shift(slope_win)
        up = price > ma
        dead = (price < ma) & (~slope_up)
        expo = base.where(up, slope_floor * base)      # 线下但斜率未死 → slope_floor 档
        expo = expo.where(~dead, break_floor * base)   # 确认趋势死亡 → break_floor 档(现行=0)
    if dip_add > 0:                                    # 深跌加仓(趋势仍在时，越跌越补)
        dd = price / price.cummax() - 1.0
        in_dip = (dd <= -dip_thresh)
        if gate:
            in_dip = in_dip & (price > price.rolling(ma_win, min_periods=ma_win // 2).mean())
        expo = expo + in_dip.astype(float) * dip_add
    cap = max_lev + (dip_add if dip_add > 0 else 0.0)
    return expo.clip(0.0, cap).fillna(0.0)


def strat_ret(price, **kw):
    ret = price.pct_change()
    expo = exposure(price, **kw)
    pos = expo.shift(1)
    return (pos * ret - pos.diff().abs().fillna(0.0) * FEE).dropna(), expo


def metrics(r, expo=None):
    r = r.dropna()
    if len(r) < 60:
        return {}
    n = len(r); cum = (1 + r).prod()
    cagr = cum ** (ANN / n) - 1
    vol = r.std() * np.sqrt(ANN)
    sharpe = r.mean() / r.std() * np.sqrt(ANN) if r.std() > 0 else np.nan
    dvol = r[r < 0].std() * np.sqrt(ANN)
    sortino = r.mean() * ANN / dvol if dvol and dvol == dvol else np.nan
    eq = (1 + r).cumprod(); dd = eq / eq.cummax() - 1.0
    mdd = float(dd.min()); calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    out = dict(CAGR=cagr, vol=vol, sharpe=sharpe, sortino=sortino, mdd=mdd, calmar=calmar,
               mult=cum, n=n)
    if expo is not None:
        e = expo.reindex(r.index).fillna(0.0)
        out["inmkt"] = float((e > 0).mean()); out["turnover"] = float(e.diff().abs().mean() * ANN)
    return out


# ---------------------------------------------------------------------------
# 方案册
# ---------------------------------------------------------------------------
STRATS = {
    "持有(baseline)":        dict(_hold=True),
    "v3现行(门→0,tv25)":      dict(tvol=0.25, break_floor=0.0, slope_floor=0.5),
    "纯波动目标(无门,tv25)":   dict(tvol=0.25, gate=False),
    "纯波动目标(无门,tv40)":   dict(tvol=0.40, gate=False),
    "软地板0.3(tv25)":        dict(tvol=0.25, break_floor=0.30, slope_floor=0.65),
    "软地板0.5(tv25)":        dict(tvol=0.25, break_floor=0.50, slope_floor=0.75),
    "软地板0.3+高目标(tv40)":  dict(tvol=0.40, break_floor=0.30, slope_floor=0.65),
    "软地板0.5+高目标(tv50)":  dict(tvol=0.50, break_floor=0.50, slope_floor=0.75),
    "低波动杠杆(tv40,1.5x)":   dict(tvol=0.40, max_lev=1.5, break_floor=0.0, slope_floor=0.5),
    "低波杠杆+软地板0.4":      dict(tvol=0.40, max_lev=1.5, break_floor=0.40, slope_floor=0.70),
    "深跌加仓(v3+dip.5)":     dict(tvol=0.25, break_floor=0.0, slope_floor=0.5, dip_add=0.5, dip_thresh=0.15),
    "软地板0.4+深跌加仓":      dict(tvol=0.35, break_floor=0.40, slope_floor=0.70, dip_add=0.5, dip_thresh=0.15),
    "高目标+软地板0.5+杠杆1.3":dict(tvol=0.45, max_lev=1.3, break_floor=0.50, slope_floor=0.75),
}


def pooled(universe, cfg, idx_period=None, pxmap=None):
    pxmap = pxmap if pxmap is not None else _PX
    cols = {}
    expos = []
    for tk in universe:
        px = pxmap.get(tk)
        if px is None or len(px) < 252:
            continue
        if cfg.get("_hold"):
            r = px.pct_change().dropna(); e = pd.Series(1.0, index=px.index)
        else:
            r, e = strat_ret(px, **{k: v for k, v in cfg.items() if not k.startswith("_")})
        cols[tk] = r; expos.append(e)
    if not cols:
        return {}, None
    df = pd.DataFrame(cols)
    common = df.dropna(how="all").index
    if idx_period is not None:
        common = common[(common >= idx_period[0]) & (common < idx_period[1])]
    port = df.reindex(common).mean(axis=1)
    avg_e = pd.concat(expos, axis=1).reindex(common).mean(axis=1) if expos else None
    return metrics(port, avg_e), port


def show_table(title, universe, period=None, pxmap=None):
    print("\n" + "=" * 112)
    print(title)
    print("-" * 112)
    print(f"{'方案':<26}{'CAGR':>7}{'终值x':>8}{'波动':>7}{'夏普':>6}{'Sortino':>8}{'最大回撤':>9}{'Calmar':>7}{'在场':>6}{'换手':>6}")
    hold_m = None
    rows = []
    for name, cfg in STRATS.items():
        m, _ = pooled(universe, cfg, period, pxmap)
        if not m:
            continue
        if cfg.get("_hold"):
            hold_m = m
        rows.append((name, m))
    for name, m in rows:
        flag = ""
        if hold_m and not STRATS[name].get("_hold"):
            # 标注：复利达持有≥85% 且 回撤显著更浅 且 Calmar>持有
            comp_ok = m["CAGR"] >= 0.85 * hold_m["CAGR"]
            dd_ok = m["mdd"] > hold_m["mdd"] * 0.75   # 回撤至少浅 25%
            cal_ok = m["calmar"] > hold_m["calmar"]
            if comp_ok and cal_ok:
                flag = " ✅复利达标+风调更优"
            elif cal_ok and dd_ok:
                flag = " 🟡风调更优(复利仍让)"
        im = f"{m.get('inmkt', float('nan')):.0%}" if m.get("inmkt") == m.get("inmkt") else "—"
        tv = f"{m.get('turnover', float('nan')):.1f}" if m.get("turnover") == m.get("turnover") else "—"
        print(f"{name:<26}{m['CAGR']:>+7.0%}{m['mult']:>7.1f}x{m['vol']:>7.0%}{m['sharpe']:>6.2f}"
              f"{m['sortino']:>8.2f}{m['mdd']:>+9.0%}{m['calmar']:>7.2f}{im:>6}{tv:>6}{flag}")


# 预取价格
print("加载数据 ...")
_PX = {}
for tk in set(MAG7 + LAGGARDS + ["SPY"]):
    try:
        _PX[tk] = loader.load_ohlcv(tk, START, END)["close"].dropna()
    except Exception as e:
        print(f"  {tk} 跳过 {type(e).__name__}")

show_table("【A】Mag7 池 · 全样本 2010→2026（核心：能否复利逼近持有+保回撤）", MAG7)
show_table("【B】落后/平庸股池 · 全样本（抗过拟合：不是只在赢家成立？）", LAGGARDS)
show_table("【C】Mag7+落后股 全池 · 全样本（最宽稳健性）", MAG7 + LAGGARDS)
show_table("【D】SPY 单标的 · 全样本（大盘上的表现）", ["SPY"])
# 分段稳定性（Mag7）
show_table("【E】Mag7 · 前半段 2010→2018（OOS 稳定性）", MAG7, (pd.Timestamp("2010-01-01"), pd.Timestamp("2018-01-01")))
show_table("【F】Mag7 · 后半段 2018→2026（OOS 稳定性）", MAG7, (pd.Timestamp("2018-01-01"), pd.Timestamp("2026-07-01")))

# ---------------------------------------------------------------------------
# 长历史压测：含 2000 互联网泡沫 + 2008 金融危机（关键抗过拟合/真崩盘检验）
# ---------------------------------------------------------------------------
LONG = ["SPY", "AAPL", "MSFT", "QCOM", "INTC"]
START_LONG = "1996-01-01"
print("\n加载长历史 ...")
_PXL = {}
for tk in LONG:
    try:
        _PXL[tk] = loader.load_ohlcv(tk, START_LONG, END)["close"].dropna()
    except Exception as e:
        print(f"  {tk} 跳过 {type(e).__name__}")

show_table("【G】长历史池 SPY+AAPL+MSFT+QCOM+INTC · 1996→2026（含 dotcom+GFC 全样本）", LONG, pxmap=_PXL)

CRISES = [
    ("互联网泡沫 2000-09→2003-01", "2000-09-01", "2003-01-31"),
    ("金融危机 2007-10→2009-04", "2007-10-01", "2009-04-30"),
    ("新冠急跌 2020-02→2020-05", "2020-02-01", "2020-05-31"),
    ("加息熊 2022 全年", "2022-01-01", "2022-12-31"),
]
print("\n" + "=" * 112)
print("【H】崩盘窗口压测（长历史池）：每个方案在危机里的 区间总收益 / 期间最大回撤 —— 验证软地板+杠杆是否会爆")
print("-" * 112)
FOCUS = ["持有(baseline)", "v3现行(门→0,tv25)", "纯波动目标(无门,tv25)", "软地板0.5(tv25)",
         "低波杠杆+软地板0.4", "高目标+软地板0.5+杠杆1.3"]
hdr = f"{'方案':<26}" + "".join(f"{c[0][:14]:>17}" for c in CRISES)
print(hdr)
for name in FOCUS:
    cfg = STRATS[name]
    cells = []
    for _, s, e in CRISES:
        _, port = pooled(LONG, cfg, (pd.Timestamp(s), pd.Timestamp(e)), _PXL)
        if port is None or len(port) < 10:
            cells.append("    —"); continue
        tot = (1 + port).prod() - 1
        eq = (1 + port).cumprod(); mdd = float((eq / eq.cummax() - 1).min())
        cells.append(f"{tot:+5.0%}/{mdd:+4.0%}")
    print(f"{name:<26}" + "".join(f"{c:>17}" for c in cells))
print("  (格式：区间总收益 / 期间最大回撤；越靠近 0 越抗跌)")

print("\n" + "=" * 112)
print("读法：✅=复利达持有85%以上且Calmar优于持有(=合格候选)；🟡=风险调整更优但复利仍明显让。")
print("=" * 112)
