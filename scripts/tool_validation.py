"""大规模工具可行性体检：从科学角度验证「这个量化引擎是否诚实/可信」。

不评判它能否赚钱(不可知)，而是检验四件可证伪的事：
 A 诚实(无假阳性)：随机输入应得 ≈0 的结果
 B 识破过拟合/选股偏差：同规则在赢家池 vs 宽池差多少；deflated Sharpe 是否打折
 C 真信号可测：文献已知的 PEAD 应被检出且显著
 D 机制正确：成本被计入；永远有基准对比
每项给具体数值 + PASS/FAIL。
"""
import warnings
from pathlib import Path
import sys

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from data import loader
from factors import price_factors as pf, signals as sg
from evaluation import factor_eval as fe, rule_eval as re, earnings_eval as ee, rule_select as rs
from stats.bootstrap import block_bootstrap_ci

MAG7 = loader.load_universe("mag7")
results = []
def rec(cat, name, value, ok, expect):
    results.append((cat, name, value, "✅" if ok else "❌", expect))


print("加载数据…")
px7 = loader.load_prices(MAG7, "2012-01-01", "2026-06-07")

# ===== A 诚实：随机输入 ≈0 =====
print("A 诚实性(无假阳性)…")
rf = pf.random_factor(px7, seed=42)
ic = fe.evaluate_factor(rf, px7, periods=(1, 5, 21, 63))["ic"]["IC_mean"].abs().max()
rec("A 诚实", "随机因子 IC(应≈0)", f"{ic:.3f}", ic < 0.03, "<0.03")

def rand_entry(p):
    rng = np.random.default_rng(len(p)); return pd.Series(rng.random(len(p)) < 0.07, index=p.index)
rr = re.evaluate_rule(rand_entry, {"trailing_stop": 0.2, "time_stop": 63}, tickers=MAG7,
                      start="2012-01-01", end="2026-06-07", n_boot=300)["pooled"]
rec("A 诚实", "随机进场 超额(应≈0)", f"{rr['excess_median']:+.1%}", abs(rr["excess_median"]) < 0.03, "≈0")
rec("A 诚实", "随机进场 显著?(应否)", "显著" if rr["excess_significant"] else "不显著", not rr["excess_significant"], "不显著")

# bootstrap CI 覆盖已知真值
rng = np.random.default_rng(0); x = rng.normal(0.001, 0.01, 3000)
pt, lo, hi = block_bootstrap_ci(x, np.mean, block_size=21, n=800)
rec("A 诚实", "BootstrapCI 覆盖真值0.001", f"[{lo:.4f},{hi:.4f}]", lo < 0.001 < hi, "含真值")

# ===== B 识破过拟合/选股偏差 =====
print("B 识破偏差…")
entry = lambda p: sg.dip_from_high(p, lookback=20, pct=0.15)
ex_spec = {"trailing_stop": 0.20, "take_profit": 0.25, "time_stop": 63}
e7 = re.evaluate_rule(entry, ex_spec, tickers=MAG7, start="2012-01-01", end="2026-06-07", n_boot=300)["pooled"]
try:
    div = loader.load_universe("diversified")
    ed = re.evaluate_rule(entry, ex_spec, tickers=div, start="2012-01-01", end="2026-06-07", n_boot=300)["pooled"]
    drop = e7["excess_median"] - ed["excess_median"]
    rec("B 识破", "同规则 赢家池→宽池 超额", f"{e7['excess_median']:+.1%}→{ed['excess_median']:+.1%}", drop > 0.01, "宽池明显更低")
except Exception as ee2:
    rec("B 识破", "宽池对照", f"失败:{ee2}", False, "-")

grid = rs.candidate_grid_from([("dip_from_high", 0.15, 0.0)], "and", ex_spec)
d = rs.deflated_rule_sharpe(candidates=grid, tickers=MAG7, start="2012-01-01", end="2026-06-07", min_trades=15)
rec("B 识破", "deflated Sharpe 多重检验折扣", f"prob={d['deflated_sharpe_prob']:.2f}", True, "会打折(信息性)")

# ===== C 真信号可测 (PEAD) =====
print("C 真信号检测(PEAD)…")
prices = {t: loader.load_prices([t], "2012-01-01", "2026-06-07")[t].dropna() for t in MAG7}
edates = {t: loader.load_earnings_dates(t) for t in MAG7}
icp = ee.earnings_drift_ic(prices, edates, horizons=(1, 5, 21, 63), n_control=60)["ic_table"]
r5 = icp[icp["horizon"] == 5].iloc[0]
rec("C 真信号", "PEAD 5日 真实IC(应>0显著)", f"{r5['ic_real']:+.3f} p={r5['perm_pvalue']:.3f}", r5["ic_real"] > 0.03 and r5["significant"], ">0.03且p<.05")
rec("C 真信号", "PEAD 假财报日 平均IC(应≈0)", f"{icp['ic_fake_mean'].abs().max():.3f}", icp["ic_fake_mean"].abs().max() < 0.03, "<0.03")

# ===== D 机制正确 =====
print("D 机制…")
from backtest import exits as exm
p = prices["NVDA"]; ent = entry(p)
pf_cost = exm.run_trades(p, ent, ex_spec)  # 默认含费用
pf_free = exm.run_trades(p, ent, ex_spec, fees=0.0, slippage=0.0)
t_cost = exm.extract_trades(pf_cost, p)["return"].mean()
t_free = exm.extract_trades(pf_free, p)["return"].mean()
rec("D 机制", "成本被计入(含费<免费)", f"{t_cost:.3%} < {t_free:.3%}", t_cost < t_free, "含费更低")
rec("D 机制", "永远有基准对比", "baseline_median 存在" if "baseline_median" in e7 else "缺失", "baseline_median" in e7, "存在")

# ===== 输出 =====
print("\n" + "=" * 78)
print(f"{'类别':<8}{'检验项':<30}{'数值':<26}{'判定'}")
print("-" * 78)
for cat, name, val, mark, exp in results:
    print(f"{cat:<8}{name:<30}{str(val):<26}{mark}  (期望:{exp})")
npass = sum(1 for r in results if r[3] == "✅")
print("-" * 78)
print(f"通过 {npass}/{len(results)} 项")
