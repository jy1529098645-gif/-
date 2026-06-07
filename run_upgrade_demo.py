"""升级验证：验收闸门 + 宽池(降偏差) + 长周期低功效标记。"""
import warnings, time
warnings.filterwarnings("ignore")
import pandas as pd

from data import loader
from evaluation import rule_eval as re_, acceptance as acc
from factors import signals as sg
from regime import conditional_returns as cr

EXIT = {"trailing_stop": 0.20, "take_profit": 0.25, "time_stop": 63}
entry = sg.build_entry([("dip_from_high", 0.15, 0.0)], "and")
def hr(t): print("\n"+"="*72+f"\n{t}\n"+"="*72)

# 1) 验收闸门（mag7）
hr("① 验收闸门：dip-15% 规则（七姐妹）跑赢基准+OOS夏普≥1+回撤可承受")
mag7 = loader.load_universe("mag7")
res = re_.evaluate_rule(entry, EXIT, tickers=mag7, start="2013-01-01", n_boot=300)
gate = acc.acceptance_gate(res, entry, EXIT, mag7, start="2013-01-01")
print(acc.format_gate(gate))

# 2) 宽池 vs 七姐妹：降低选股偏差后超额还在不在
hr("② 选股偏差检验：同一 dip 规则，七姐妹 vs 降偏差宽池")
t0=time.time()
div = loader.load_universe("diversified")
res_div = re_.evaluate_rule(entry, EXIT, tickers=div, start="2013-01-01", n_boot=300)
for name, r in [("七姐妹(已知赢家)", res), ("降偏差宽池", res_div)]:
    p = r["pooled"]
    print(f"{name:<14} N={p['n_trades']:>4} N_eff≈{p['n_eff']:>3.0f} 胜率{p['win_rate']:.0%} "
          f"收益中位{p['median_return']:+.1%} | 超额{p['excess_median']:+.1%} "
          f"CI[{p['excess_ci_low']:+.1%},{p['excess_ci_high']:+.1%}] "
          f"{'显著' if p['excess_significant'] else '不显著'}")
print(f"(宽池实际纳入 {len(res_div['per_ticker'])} 票, 用时 {time.time()-t0:.0f}s)")

# 3) 长周期低功效标记
hr("③ 长周期 CI 诚实化：h252 重叠窗口 → 低功效标记 → 显著性降级")
macro = loader.load_macro("1990-01-01")
print(f"{'标的':<6}{'h':>5}{'超额':>9}{'名义N':>7}{'独立窗口':>9}{'低功效?':>8}")
for tk in ["AAPL","NVDA","TSLA"]:
    price = loader.load_prices([tk],"2013-01-01")[tk].dropna()
    for h in (63, 252):
        tab = cr.conditional_forward_returns(price, macro, asset=tk, horizons=(h,),
                                             groupings=[["in_drawdown"]], n_boot=150)
        d = tab[(tab.bucket=="in_drawdown=True")&(tab.horizon==h)]
        if d.empty: continue
        d=d.iloc[0]
        print(f"{tk:<6}{h:>5}{d['excess_median']:>+9.1%}{int(d['n_days']):>7}"
              f"{int(d['n_independent']):>9}{'⚠️是' if d['low_power'] else '否':>8}")
print("→ h252 名义 N 大但独立窗口少→标⚠️低功效；引擎/简报据此把'显著'降级，不再假性显著。")
hr("完成")
