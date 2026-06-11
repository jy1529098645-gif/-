"""v3.1 软地板 · Walk-forward 前向滚动验证：floor 是不是全样本挑出来的过拟合？

最严口径：每个 OOS 窗口只用**之前**的训练窗口选 floor(按 Calmar)，再到没见过的 OOS 窗口跑。
对照三条：① WF自适应(每窗重选floor) ② 固定floor=0.5(v3.1默认) ③ 固定floor=0(硬清零·旧) ④ 持有。
若"固定0.5"的拼接OOS≈"WF自适应"，且选出的floor稳定簇拢，则 floor=0.5 非过拟合、可上实盘。
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from data import loader
from analysis import position_guidance as pg

FEE = 0.0010; ANN = 252; END = "2026-06-06"; START = "2000-01-01"
POOL = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "AMD", "TSM", "NFLX",
        "INTC", "PYPL", "DIS", "BA", "F", "T", "VZ", "NKE", "MMM", "QCOM"]
FLOORS = [0.0, 0.25, 0.5, 0.75, 1.0]
SLOPE_FLOOR = 0.70          # 固定，只研究 break floor 这一杠杆
TVOL = 0.25


def pooled_returns(floor):
    cols = {}
    for tk, px in PX.items():
        ret = px.pct_change()
        expo = pg._exposure_series(px, TVOL, 1.0, floor=floor, slope_floor=SLOPE_FLOOR)
        pos = expo.shift(1)
        cols[tk] = pos * ret - pos.diff().abs().fillna(0) * FEE
    return pd.DataFrame(cols).mean(axis=1)


def met(r):
    r = r.dropna()
    if len(r) < 40:
        return dict(CAGR=np.nan, sharpe=np.nan, mdd=np.nan, calmar=np.nan, mult=np.nan)
    n = len(r); cum = (1 + r).prod()
    eq = (1 + r).cumprod(); mdd = float((eq / eq.cummax() - 1).min())
    cagr = cum ** (ANN / n) - 1
    return dict(CAGR=cagr, sharpe=r.mean() / r.std() * np.sqrt(ANN) if r.std() > 0 else np.nan,
                mdd=mdd, calmar=cagr / abs(mdd) if mdd < 0 else np.nan, mult=cum)


print("加载宽池 2000→2026 ...")
PX = {}
for tk in POOL:
    try:
        s = loader.load_ohlcv(tk, START, END)["close"].dropna()
        if len(s) >= 252:
            PX[tk] = s
    except Exception:
        pass
print(f"  可用 {len(PX)} 只")

RET = {f: pooled_returns(f) for f in FLOORS}      # 每个 floor 的全样本池化日收益
hold = pd.DataFrame({tk: px.pct_change() for tk, px in PX.items()}).mean(axis=1)
idx = RET[0.5].dropna().index

# Walk-forward：训练 4 年 → OOS 2 年，步进 2 年
oos_start = 2004
windows = []
while oos_start + 2 <= 2026:
    tr = (pd.Timestamp(f"{oos_start-4}-01-01"), pd.Timestamp(f"{oos_start}-01-01"))
    oo = (pd.Timestamp(f"{oos_start}-01-01"), pd.Timestamp(f"{oos_start+2}-01-01"))
    windows.append((tr, oo)); oos_start += 2

print("\n" + "=" * 96)
print("Walk-forward：每窗用训练期 Calmar 选 floor，再跑 OOS（floor 是否稳定/非过拟合）")
print("=" * 96)
print(f"{'OOS窗口':<14}{'训练选floor':>11}{'WF自适应OOS':>26}{'固定0.5 OOS':>22}{'持有 OOS':>20}")
print(f"{'':14}{'(Calmar最优)':>11}{'CAGR/夏普/回撤/Cal':>26}{'CAGR/回撤/Cal':>22}{'CAGR/回撤':>20}")
adaptive_oos, fixed05_oos, fixed00_oos, hold_oos = [], [], [], []
sel_floors = []
for tr, oo in windows:
    tr_mask = (idx >= tr[0]) & (idx < tr[1])
    oo_mask = (idx >= oo[0]) & (idx < oo[1])
    tr_idx, oo_idx = idx[tr_mask], idx[oo_mask]
    # 训练期按 Calmar 选 floor
    best_f, best_cal = 0.5, -1e9
    for f in FLOORS:
        c = met(RET[f].reindex(tr_idx)).get("calmar", np.nan)
        if c == c and c > best_cal:
            best_cal, best_f = c, f
    sel_floors.append(best_f)
    m_ad = met(RET[best_f].reindex(oo_idx))
    m_05 = met(RET[0.5].reindex(oo_idx))
    m_00 = met(RET[0.0].reindex(oo_idx))
    m_hd = met(hold.reindex(oo_idx))
    adaptive_oos.append(RET[best_f].reindex(oo_idx)); fixed05_oos.append(RET[0.5].reindex(oo_idx))
    fixed00_oos.append(RET[0.0].reindex(oo_idx)); hold_oos.append(hold.reindex(oo_idx))
    yr = f"{oo[0].year}-{oo[1].year}"
    print(f"{yr:<14}{best_f:>11.2f}"
          f"{m_ad['CAGR']:>+8.0%}{m_ad['sharpe']:>6.2f}{m_ad['mdd']:>+6.0%}{m_ad['calmar']:>5.2f}"
          f"{m_05['CAGR']:>+9.0%}{m_05['mdd']:>+7.0%}{m_05['calmar']:>6.2f}"
          f"{m_hd['CAGR']:>+11.0%}{m_hd['mdd']:>+7.0%}")

# 拼接 OOS 整体
def stitch(lst): return pd.concat(lst).sort_index()
print("\n" + "-" * 96)
print("拼接全 OOS（2004→2026 全程样本外）整体表现：")
for nm, series in [("WF自适应(每窗重选)", stitch(adaptive_oos)),
                   ("固定 floor=0.5 (v3.1默认)", stitch(fixed05_oos)),
                   ("固定 floor=0.0 (硬清零·旧)", stitch(fixed00_oos)),
                   ("闭眼持有", stitch(hold_oos))]:
    m = met(series)
    print(f"  {nm:<26} CAGR {m['CAGR']:+.1%} | 终值 {m['mult']:.1f}x | 夏普 {m['sharpe']:.2f} | "
          f"回撤 {m['mdd']:+.0%} | Calmar {m['calmar']:.2f}")

print(f"\n  各窗训练选出的 floor：{sel_floors}")
print(f"  floor 均值 {np.mean(sel_floors):.2f} · 中位 {np.median(sel_floors):.2f} · "
      f"≥0.5 的窗占比 {np.mean([f>=0.5 for f in sel_floors]):.0%}")
print("\n读法：若 固定0.5 ≈ WF自适应 且 选出floor多在0.5上下 → floor=0.5 非过拟合、可上实盘。")
print("=" * 96)
