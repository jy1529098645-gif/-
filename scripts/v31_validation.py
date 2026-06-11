"""v3.1 改前/改后对照：直接用生产函数 position_guidance._exposure_series 验证软地板的效果。

改前 = 硬清零(floor=0, slope_floor=0.5)；改后 = 各档新参数(中性 floor=0.5)。
在 Mag7(2010→) 与 长历史(1996→, 含 dotcom+GFC) 两套池子上对照，并看四档崩盘窗口表现。
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from data import loader
from analysis import position_guidance as pg

FEE = 0.0010; ANN = 252; END = "2026-06-06"
MAG7 = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]
LONG = ["SPY", "AAPL", "MSFT", "QCOM", "INTC"]


def bt(price, **kw):
    ret = price.pct_change()
    expo = pg._exposure_series(price, **kw)
    pos = expo.shift(1)
    return (pos * ret - pos.diff().abs().fillna(0) * FEE).dropna()


def met(r):
    r = r.dropna(); n = len(r); cum = (1 + r).prod()
    eq = (1 + r).cumprod(); mdd = float((eq / eq.cummax() - 1).min())
    cagr = cum ** (ANN / n) - 1; vol = r.std() * np.sqrt(ANN)
    sh = r.mean() / r.std() * np.sqrt(ANN)
    return dict(CAGR=cagr, mult=cum, sharpe=sh, mdd=mdd, calmar=cagr / abs(mdd) if mdd < 0 else np.nan)


def pool(univ, start, cfg, period=None):
    cols = {}
    for tk in univ:
        try:
            px = loader.load_ohlcv(tk, start, END)["close"].dropna()
            if cfg is None:
                r = px.pct_change().dropna()
            else:
                r = bt(px, **cfg)
            cols[tk] = r
        except Exception:
            pass
    df = pd.DataFrame(cols); idx = df.dropna(how="all").index
    if period: idx = idx[(idx >= period[0]) & (idx < period[1])]
    return met(df.reindex(idx).mean(axis=1))


OLD = dict(tvol=0.25, max_lev=1.0, floor=0.0, slope_floor=0.5)   # 改前·中性(硬清零)
NEW = dict(tvol=0.25, max_lev=1.0, floor=pg.PROFILES["moderate"]["floor"],
           slope_floor=pg.PROFILES["moderate"]["slope_floor"])    # 改后·中性(软地板0.5)

print("=" * 92)
print("v3.1 软地板 改前(硬清零) vs 改后(floor=0.5) · 中性档 · 生产函数 _exposure_series")
print("=" * 92)
for label, univ, start in [("Mag7 池 2010→2026", MAG7, "2010-01-01"),
                           ("长历史池 1996→2026(含dotcom+GFC)", LONG, "1996-01-01")]:
    hold = pool(univ, start, None)
    old = pool(univ, start, OLD)
    new = pool(univ, start, NEW)
    print(f"\n# {label}")
    print(f"{'':<14}{'CAGR':>8}{'终值x':>9}{'夏普':>7}{'最大回撤':>10}{'Calmar':>8}")
    for nm, m in [("闭眼持有", hold), ("改前(硬清零)", old), ("改后(软地板)", new)]:
        print(f"{nm:<14}{m['CAGR']:>+8.0%}{m['mult']:>8.1f}x{m['sharpe']:>7.2f}{m['mdd']:>+10.0%}{m['calmar']:>8.2f}")
    print(f"  → 改后 vs 改前：复利 {old['mult']:.1f}x→{new['mult']:.1f}x  夏普 {old['sharpe']:.2f}→{new['sharpe']:.2f}  "
          f"Calmar {old['calmar']:.2f}→{new['calmar']:.2f}  回撤 {old['mdd']:+.0%}→{new['mdd']:+.0%}")

# 四档全览（改后）· 长历史
print("\n" + "=" * 92)
print("改后·四档风险偏好 全览 · 长历史池 1996→2026")
print("=" * 92)
print(f"{'档':<14}{'floor':>6}{'CAGR':>8}{'终值x':>9}{'夏普':>7}{'最大回撤':>10}{'Calmar':>8}")
holdL = pool(LONG, "1996-01-01", None)
print(f"{'闭眼持有':<14}{'—':>6}{holdL['CAGR']:>+8.0%}{holdL['mult']:>8.1f}x{holdL['sharpe']:>7.2f}{holdL['mdd']:>+10.0%}{holdL['calmar']:>8.2f}")
for k in ("conservative", "moderate", "aggressive", "leveraged"):
    c = pg.PROFILES[k]
    m = pool(LONG, "1996-01-01", dict(tvol=c["tvol"], max_lev=c["max_lev"], floor=c["floor"], slope_floor=c["slope_floor"]))
    print(f"{pg.PROFILE_ZH[k]:<14}{c['floor']:>6.1f}{m['CAGR']:>+8.0%}{m['mult']:>8.1f}x{m['sharpe']:>7.2f}{m['mdd']:>+10.0%}{m['calmar']:>8.2f}")

# 崩盘窗口（改前 vs 改后 中性 + 稳健）
print("\n" + "=" * 92)
print("崩盘窗口·区间总收益/期间最大回撤 · 长历史池（验证软地板没把保护丢掉）")
print("=" * 92)
CRISES = [("dotcom 00-03", "2000-09-01", "2003-01-31"), ("GFC 07-09", "2007-10-01", "2009-04-30"),
          ("covid 2020", "2020-02-01", "2020-05-31"), ("2022熊", "2022-01-01", "2022-12-31")]
def crisis_row(name, cfg):
    cells = []
    for _, s, e in CRISES:
        m = pool(LONG, "1996-01-01", cfg, (pd.Timestamp(s), pd.Timestamp(e)))
        cells.append(f"{m['CAGR'] if False else (m['mult']-1):+5.0%}/{m['mdd']:+4.0%}")
    print(f"{name:<18}" + "".join(f"{c:>16}" for c in cells))
print(f"{'':<18}" + "".join(f"{c[0]:>16}" for c in CRISES))
crisis_row("闭眼持有", None)
crisis_row("改前中性(硬清零)", OLD)
crisis_row("改后中性(软地板)", NEW)
crisis_row("改后稳健(仍清零)", dict(tvol=0.18, max_lev=1.0, floor=0.0, slope_floor=0.40))
print("\n完成。")
