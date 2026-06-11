"""v3.1 软地板 · Deflated Sharpe 多重检验复核（扣"试了 N 个暴露方案才挑出软地板"的折扣）。

口径同原 v3 终审(scripts/v3_oos_dsr.py)：2000 起、21 只赢家+落后股宽池、含 dotcom+GFC。
用**生产函数** position_guidance._exposure_series（已含 floor），把我搜过的暴露方案当 trials，
对选定的「中性 floor=0.5」做 Deflated Sharpe / PSR(vs0) / PSR(vs持有)；并看剔危机年的 α。
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from data import loader
from analysis import position_guidance as pg
from analysis import quant_edge as qe
from stats.deflated_sharpe import deflated_sharpe_ratio, probabilistic_sharpe_ratio, expected_max_sharpe

FEE = 0.0010; ANN = 252; END = "2026-06-06"; START = "2000-01-01"
WINNERS = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "AMD", "TSM", "NFLX"]
LAGGARDS = ["INTC", "PYPL", "DIS", "BA", "F", "T", "VZ", "NKE", "MMM", "QCOM"]
POOL = WINNERS + LAGGARDS

# 搜索过的暴露方案(trials)：tvol × max_lev × floor × slope_floor
TRIALS = {
    "tv18_f0":  dict(tvol=0.18, max_lev=1.0, floor=0.0, slope_floor=0.40),
    "tv25_f0":  dict(tvol=0.25, max_lev=1.0, floor=0.0, slope_floor=0.50),   # 旧 v3 中性
    "tv25_f03": dict(tvol=0.25, max_lev=1.0, floor=0.30, slope_floor=0.65),
    "tv25_f05": dict(tvol=0.25, max_lev=1.0, floor=0.50, slope_floor=0.70),  # ← 选定:v3.1 中性
    "tv25_nogate": dict(tvol=0.25, max_lev=1.0, floor=1.0, slope_floor=1.0),
    "tv35_f0":  dict(tvol=0.35, max_lev=1.0, floor=0.0, slope_floor=0.50),
    "tv35_f05": dict(tvol=0.35, max_lev=1.0, floor=0.50, slope_floor=0.70),
    "tv40_f0":  dict(tvol=0.40, max_lev=1.0, floor=0.0, slope_floor=0.50),
    "tv40_nogate": dict(tvol=0.40, max_lev=1.0, floor=1.0, slope_floor=1.0),
    "tv50_f05": dict(tvol=0.50, max_lev=1.0, floor=0.50, slope_floor=0.75),
    "lev15_f0": dict(tvol=0.40, max_lev=1.5, floor=0.0, slope_floor=0.50),
    "lev15_f04": dict(tvol=0.40, max_lev=1.5, floor=0.40, slope_floor=0.70),
    "lev13_f05": dict(tvol=0.45, max_lev=1.3, floor=0.50, slope_floor=0.75),
}
SELECTED = "tv25_f05"

print("加载宽池 2000→2026 ...")
PX = {}
for tk in POOL:
    try:
        s = loader.load_ohlcv(tk, START, END)["close"].dropna()
        if len(s) >= 252:
            PX[tk] = s
    except Exception:
        pass
print(f"  可用 {len(PX)}/{len(POOL)} 只")


def strat_pool(cfg):
    cols = {}
    for tk, px in PX.items():
        ret = px.pct_change()
        expo = pg._exposure_series(px, **cfg)
        pos = expo.shift(1)
        cols[tk] = (pos * ret - pos.diff().abs().fillna(0) * FEE)
    return pd.DataFrame(cols)


def sharpe_d(r):
    r = r.dropna()
    return float(r.mean() / r.std()) if r.std() > 0 else np.nan   # 日度


# 各 trial 池化日收益 + 日度夏普
trial_ret = {name: strat_pool(cfg) for name, cfg in TRIALS.items()}
common = None
for df in trial_ret.values():
    idx = df.dropna(how="all").index
    common = idx if common is None else common.intersection(idx)
hold = pd.DataFrame({tk: px.pct_change() for tk, px in PX.items()}).reindex(common).mean(axis=1)
trial_port = {name: df.reindex(common).mean(axis=1) for name, df in trial_ret.items()}
trial_sh = {name: sharpe_d(r) for name, r in trial_port.items()}

sel = trial_port[SELECTED].dropna()
sr_sel_d = sharpe_d(sel)
n_obs = len(sel); sk = float(sel.skew()); ku = float(sel.kurt() + 3.0)
sh_arr = np.array([s for s in trial_sh.values() if s == s])
sr0 = expected_max_sharpe(float(sh_arr.std(ddof=1)), n_trials=len(sh_arr))
dsr = deflated_sharpe_ratio(sr_sel_d, float(sh_arr.std(ddof=1)), n_trials=len(sh_arr),
                            n_obs=n_obs, skew=sk, kurtosis=ku)
psr0 = probabilistic_sharpe_ratio(sr_sel_d, 0.0, n_obs, skew=sk, kurtosis=ku)
sr_hold_d = sharpe_d(hold.reindex(sel.index))
psr_h = probabilistic_sharpe_ratio(sr_sel_d, sr_hold_d, n_obs, skew=sk, kurtosis=ku)

print("\n" + "=" * 92)
print("v3.1 选定方案 = 中性 floor=0.5（tv25_f05）· Deflated Sharpe 复核 · 宽池 2000→2026")
print("=" * 92)
print(f"  trials(搜过的暴露方案) N={len(sh_arr)} · 各方案日度夏普 std={sh_arr.std(ddof=1):.4f}")
print(f"  选定日度夏普 {sr_sel_d:.4f}(年化 {sr_sel_d*np.sqrt(ANN):.2f}) | 样本 {n_obs} 日 | 偏度 {sk:+.2f} 峰度 {ku:.1f}")
print(f"  期望最大夏普(应扣基准) SR0={sr0:.4f}(年化 {sr0*np.sqrt(ANN):.2f})")
print(f"  PSR(vs 0)        = {psr0:.3f}  (夏普真>0 的概率)")
print(f"  PSR(vs 持有夏普) = {psr_h:.3f}  (v3.1 夏普真>持有 的概率) · 持有日夏普年化 {sr_hold_d*np.sqrt(ANN):.2f}")
print(f"  **Deflated Sharpe = {dsr:.3f}** (扣{len(sh_arr)}方案多重检验后仍>0 的概率；>0.95 稳健) → "
      f"{'稳健 ✅' if dsr > 0.95 else '存疑 ⚠️'}")

# 各 trial 年化夏普一览（看选定是不是靠运气挑的极值）
print("\n  各 trial 年化夏普(看选定非靠运气挑极值)：")
for name in sorted(trial_sh, key=lambda k: -trial_sh[k]):
    star = " ←选定" if name == SELECTED else ""
    print(f"    {name:<14} {trial_sh[name]*np.sqrt(ANN):>5.2f}{star}")

# 剔危机年的 α vs 持有（edge 是否只靠崩盘年）
print("\n" + "=" * 92)
print("剔危机年稳健性：v3.1中性 α vs 持有（edge 是否只靠崩盘年喂出来）")
print("=" * 92)
yrs = sorted(set(common.year))
crisis_years = []
for y in yrs:
    if y >= 2026: continue
    m = common.year == y
    if m.sum() < 60: continue
    hy = hold.reindex(common)[m]
    if (1 + hy).prod() - 1 < -0.10:
        crisis_years.append(y)
for label, mask in [("全样本", np.ones(len(common), bool)),
                    ("剔危机年", ~np.isin(common.year, crisis_years)),
                    ("仅危机年", np.isin(common.year, crisis_years))]:
    idx = common[mask]
    ab = qe.alpha_beta(sel.reindex(idx), hold.reindex(idx), n_boot=800)
    print(f"  {label:<8} N={mask.sum():>5} | v3.1夏普 {sharpe_d(sel.reindex(idx))*np.sqrt(ANN):>5.2f} "
          f"持有夏普 {sharpe_d(hold.reindex(idx))*np.sqrt(ANN):>5.2f} | "
          f"α {ab['alpha_ann']:+.1%}{'✔' if ab['alpha_significant'] else '✘'} "
          f"CI[{ab['alpha_ann_ci'][0]:+.1%},{ab['alpha_ann_ci'][1]:+.1%}] β{ab['beta']:.2f}")
print(f"  危机年: {crisis_years}")
print("\n完成。")
