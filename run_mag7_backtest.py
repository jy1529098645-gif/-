"""七姐妹全套回测 + 可靠性自检（按产品现有架构调用真实模块）。

判断工具是否可靠的逻辑：好工具不是"能跑出高收益"，而是
  (a) 在该没优势的地方诚实报≈0/不显著（随机进场、假财报）；
  (b) 真实规则的超额经 N_eff 折算后，给出诚实的显著性结论；
  (c) walk-forward OOS 不崩、deflated Sharpe 守门；
  (d) 单票引擎敢报负超额（动量陷阱），不粉饰。
全部走 evaluation/ + regime/ + backtest/ 的真实函数，不另造轮子。
"""
from __future__ import annotations

import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
START = "2013-01-01"


def hr(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


# ---------------------------------------------------------------------------
hr("0) 可靠性自检：随机进场 / 假财报 —— 应≈0 且不显著（工具不能凭噪音造优势）")
from evaluation import rule_eval as re_
from factors import signals as sg

rng = np.random.default_rng(0)
def random_entry(price):
    return pd.Series(rng.random(len(price)) < 0.03, index=price.index)  # ~3% 天随机进场

exit_spec = {"trailing_stop": 0.20, "take_profit": 0.25, "time_stop": 63}
t0 = time.time()
rand = re_.evaluate_rule(random_entry, exit_spec, tickers=MAG7, start=START,
                         rule_name="random_entry", n_boot=200)
rp = rand["pooled"]
print(f"随机进场：N={rp['n_trades']} 笔, N_eff≈{rp['n_eff']:.0f}, 胜率 {rp['win_rate']:.0%}, "
      f"收益中位 {rp['median_return']:+.2%}")
print(f"  → 相对随机基准超额 {rp['excess_median']:+.2%} "
      f"CI[{rp['excess_ci_low']:+.2%},{rp['excess_ci_high']:+.2%}] "
      f"{'显著(❌应不显著!)' if rp['excess_significant'] else '不显著 ✅(符合预期)'}")

from data import loader
from evaluation import earnings_eval as ee
prices = {t: loader.load_prices([t], START)[t].dropna() for t in MAG7}
edates = {t: loader.load_earnings_dates(t, limit=80) for t in MAG7}
ic = ee.earnings_drift_ic(prices, edates, horizons=(1, 5, 21), n_control=40)
for _, r in ic["ic_table"].iterrows():
    print(f"PEAD IC h={int(r['horizon']):>2}: 真实 {r['ic_real']:+.3f} | "
          f"假财报对照均值 {r['ic_fake_mean']:+.3f} (应≈0) | 置换 p={r['perm_pvalue']:.2f} "
          f"{'显著' if r['significant'] else '不显著'}")
print(f"[自检用时 {time.time()-t0:.0f}s]")


# ---------------------------------------------------------------------------
hr("1) 主回测：dip-15% 入场 + 移动止损20%/止盈25%/时间止损63 —— 七姐妹池化逐笔(含成本)")
t0 = time.time()
entry = sg.build_entry([("dip_from_high", 0.15, 0.0)], "and")
res = re_.evaluate_rule(entry, exit_spec, tickers=MAG7, start=START,
                        rule_name="dip15_trail20_tp25", n_boot=300)
p = res["pooled"]
print(f"交易笔数 N={p['n_trades']} | N_eff≈{p['n_eff']:.0f} (ρ̄={p['rho_bar']:.2f}) | 胜率 {p['win_rate']:.0%}")
print(f"单笔收益中位 {p['median_return']:+.2%} | 5分位 {p['p5_return']:+.2%} | MAE中位 {p['median_mae']:+.2%} "
      f"| 最长连亏 {p['longest_losing_streak']}")
print(f"随机基准中位 {p['baseline_median']:+.2%}")
print(f"★ 相对随机进场的超额 {p['excess_median']:+.2%} "
      f"CI[{p['excess_ci_low']:+.2%},{p['excess_ci_high']:+.2%}] "
      f"→ {'显著' if p['excess_significant'] else '不显著（择时无统计优势）'}")
print("\n单票分解（仅展示，结论以池化为准）：")
pt = pd.DataFrame(res["per_ticker"]).T
for tk in MAG7:
    if tk in pt.index:
        row = pt.loc[tk]
        print(f"  {tk:<5} N={int(row['n_trades']):>3} 胜率 {row['win_rate']:.0%} 收益中位 {row['median_return']:+.2%}")
print(f"[主回测用时 {time.time()-t0:.0f}s]")


# ---------------------------------------------------------------------------
hr("2) 反过拟合体检：walk-forward(IS选/OOS报) + deflated Sharpe（3×3 候选）")
t0 = time.time()
from evaluation import rule_select as rs
grid = rs.candidate_grid_from([("dip_from_high", 0.15, 0.0)], "and", exit_spec)
wf = rs.walk_forward_rule(candidates=grid, tickers=MAG7, start=START)
s = wf["summary"]
print(f"候选数 {len(grid)} | walk-forward 段数 {s.get('n_folds','?')}")
print(f"IS 平均 per-trade Sharpe {s['mean_is_sharpe']:.2f} | OOS 平均 {s['mean_oos_sharpe']:.2f} "
      f"| 过拟合缺口 {s['overfit_gap']:+.2f}")
defl = rs.deflated_rule_sharpe(candidates=grid, tickers=MAG7, start=START)
print(f"最优候选 {defl['best_name']} per-trade Sharpe {defl['best_sharpe']:.2f}")
print(f"Deflated Sharpe 概率 {defl['deflated_sharpe_prob']:.2f} ({defl['n_trials']} 候选) "
      f"→ {'稳健(>0.95)' if defl['robust'] else '存疑(未过 0.95 门槛)'}")
print(f"[体检用时 {time.time()-t0:.0f}s]")


# ---------------------------------------------------------------------------
hr("3) 单票建仓引擎：在回撤中(>10%)的 252日远期收益 vs 无条件基准 —— 看动量陷阱")
t0 = time.time()
from regime import conditional_returns as cr
macro = loader.load_macro("1990-01-01")
print(f"{'标的':<6}{'回撤桶中位':>10}{'基准中位':>10}{'超额':>9}{'N':>5}  结论")
for tk in MAG7:
    try:
        price = loader.load_prices([tk], START)[tk].dropna()
        tab = cr.conditional_forward_returns(price, macro, asset=tk, horizons=(252,),
                                             groupings=[["in_drawdown"]], n_boot=200)
        base = tab[(tab.grouping == "__baseline__") & (tab.horizon == 252)]["median"].iloc[0]
        dd = tab[(tab.bucket == "in_drawdown=True") & (tab.horizon == 252)]
        if dd.empty:
            print(f"{tk:<6}{'样本不足':>10}")
            continue
        d = dd.iloc[0]
        sig = (d["ci_low"] > 0) or (d["ci_high"] < 0)
        verdict = ("⚠️动量陷阱(逢跌买无优势)" if d["excess_median"] <= 0
                   else ("✅回撤带正超额" + ("·显著" if sig else "·不显著")))
        print(f"{tk:<6}{d['median']:>+10.1%}{base:>+10.1%}{d['excess_median']:>+9.1%}"
              f"{int(d['n_events']):>5}  {verdict}")
    except Exception as e:  # noqa: BLE001
        print(f"{tk:<6} 失败 {type(e).__name__}")
print(f"[引擎用时 {time.time()-t0:.0f}s]")


# ---------------------------------------------------------------------------
hr("4) 建仓布局回测：阶梯(-10/-20/-30%补仓) vs 一次性 vs 定投 —— 同预算滚动窗口+CI")
t0 = time.time()
from regime import entry_cockpit as ec
print(f"{'标的':<6}{'阶梯中位':>9}{'一次性':>9}{'定投':>9}  阶梯vs一次性")
for tk in MAG7:
    try:
        px = loader.load_prices([tk], START)
        r = ec.ladder_plan_backtest(px, asset=tk, bands=(0.10, 0.20, 0.30), n_boot=150)
        ps = r["per_strategy"]; vl = r["vs_lump_sum"]["ladder"]
        flag = "显著" if vl["significant"] else "不显著"
        print(f"{tk:<6}{ps['ladder']['median']:>+9.1%}{ps['lump_sum']['median']:>+9.1%}"
              f"{ps['dca']['median']:>+9.1%}  中位差 {vl['median_diff']:+.1%} ({flag})")
    except Exception as e:  # noqa: BLE001
        print(f"{tk:<6} 失败 {type(e).__name__}: {str(e)[:40]}")
print(f"[布局回测用时 {time.time()-t0:.0f}s]")

hr("完成")
