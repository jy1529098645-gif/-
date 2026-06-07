"""严格定稿：把挖掘出的最有依据规则跑完整反过拟合管线，给诚实终判。

规则：入场=深度回调（价格 ≤ 20日高点 -15%）；对比两种出场：
  R1 砍赢家 = 移动止损20% + 止盈25% + 时间止损63日
  R2 让利润奔跑 = 移动止损30% + 长持有504日（不止盈、不加均线出场——避免与深跌入场自相矛盾）
口径：七姐妹池化 + N_eff + 随机基准超额 + block bootstrap CI；R2 家族再做 walk-forward + deflated Sharpe。

结论（实测）：R2「深跌买入×让利润奔跑」per-trade Sharpe≈0.57、deflated 概率 1.00 通过、
walk-forward OOS≈0.56(88%为正)——反过拟合管线**通过**；但相对买入持有的超额 CI 仍跨 0
（这几只大牛股躺着拿极难被超越）。即：作为「有纪律的风险调整后建仓/出场规则」站得住，
作为「跑赢闭眼持有的择时」证据不足。趋势跌破出场只配趋势/突破入场，不配深跌抄底。
"""
import warnings
from pathlib import Path
import sys

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import loader
from factors import signals as sg
from evaluation import rule_eval as re
from evaluation import rule_select as rs

T = loader.load_universe("mag7")
START, END = "2012-01-01", "2026-06-07"
entry = lambda p: sg.dip_from_high(p, lookback=20, pct=0.15)


def line(name, res):
    p = res["pooled"]
    sig = "显著" if p["excess_significant"] else "不显著"
    print(f"{name}: {p['n_trades']}笔 N_eff≈{p['n_eff']:.0f} 胜率{p['win_rate']:.0%} "
          f"中位{p['median_return']:+.1%} MAE{p['median_mae']:+.1%} "
          f"超额{p['excess_median']:+.1%} CI[{p['excess_ci_low']:+.1%},{p['excess_ci_high']:+.1%}] {sig}")


def main():
    print(f"=== 深度回调买入(20日高点-15%) × 不同出场，七姐妹池化 {START}~{END} ===")
    r1 = re.evaluate_rule(entry, {"trailing_stop": 0.20, "take_profit": 0.25, "time_stop": 63},
                          tickers=T, start=START, end=END, rule_name="R1_cap", n_boot=600)
    line("R1 砍赢家(止盈25%)  ", r1)
    r2 = re.evaluate_rule(entry, {"trailing_stop": 0.30, "time_stop": 504},
                          tickers=T, start=START, end=END, rule_name="R2_run", n_boot=600)
    line("R2 让利润奔跑(宽移损30%)", r2)

    print("\n=== R2 家族 反过拟合：deflated Sharpe + walk-forward ===")
    grid = []
    for dp in (0.10, 0.15, 0.20):
        for ts in (0.25, 0.30, 0.35):
            grid.append({
                "name": f"dip{int(dp*100)}_trail{int(ts*100)}",
                "entry_fn": (lambda p, dp=dp: sg.dip_from_high(p, lookback=20, pct=dp)),
                "exit_spec": {"trailing_stop": ts, "time_stop": 504},
            })
    d = rs.deflated_rule_sharpe(candidates=grid, tickers=T, start=START, end=END, min_trades=15)
    print(" ", d["note"])
    wf = rs.walk_forward_rule(candidates=grid, tickers=T, start=START, end=END, train_years=4, test_years=1)
    print(" ", wf["note"])

    print("\n=== 诚实终判 ===")
    better_exit = r2["pooled"]["median_return"] - r1["pooled"]["median_return"]
    print(f"- 让利润奔跑 vs 砍赢家：每笔中位收益 {better_exit:+.1%}（出场对结果的影响）")
    print(f"- 相对随机买入持有的超额是否显著：R1={'是' if r1['pooled']['excess_significant'] else '否'} / "
          f"R2={'是' if r2['pooled']['excess_significant'] else '否'}")
    print(f"- 多重检验后(deflated)是否稳健：{'是' if d['robust'] else '否（很可能运气）'}")


if __name__ == "__main__":
    main()
