"""每日自动追踪：定时跑(不用打开 app)，留痕当日指导 + 回填评估历史判断准确性。

做两件事：
  1. 对关注清单(默认聚焦科技/半导体/ETF)逐票生成当日指导(决策卡+引擎+入场锚点)，
     落库 signals 表(按 标的+日期+周期 去重，一天一条)。
  2. 回填所有已成熟(走完 horizon)的历史信号，用真实价格算实现收益，**标注每条判断对/错**，
     打印长期有效性摘要(准确率/实现命中率 vs 预测/Brier/实现超额)。

用法(可挂定时任务，美股收盘后跑)：
    .venv/Scripts/python -m scripts.daily_track            # 默认清单
    .venv/Scripts/python -m scripts.daily_track NVDA AAPL  # 指定标的
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WATCHLIST = ["SPY", "QQQ", "XLK", "SMH", "NVDA", "AAPL", "MSFT", "GOOGL",
             "AMZN", "META", "TSLA", "AVGO", "AMD", "TSM", "CRM"]
HORIZONS = (21, 63)


def run(tickers=None, horizons=HORIZONS):
    from analysis import briefing as bf, decision as dc, journal as jn
    from regime import entry_cockpit as ec
    from data import loader

    tickers = tickers or WATCHLIST
    logged = skipped = errs = 0
    for t in tickers:
        start = "1995-01-01" if t == "SPY" else "2008-01-01"
        try:
            px = loader.load_prices([t], start, None)[t].dropna()
            for h in horizons:
                b = bf.stock_brief(t, start, None, horizon=h, with_news=False)
                try:
                    bez = ec.best_entry_across_horizons(px, asset=t, single_name=(t not in
                          ("SPY", "QQQ", "DIA", "IWM", "XLK", "SMH", "SOXX")), n_boot=150)
                    card = dc.decision_card(t, px, bez, fragile_now=False)
                    extra = {"decision_state": card.get("state"), "decision_action": card.get("action"),
                             "entry_anchor": (card.get("entry") or {}).get("anchor")}
                except Exception:  # noqa: BLE001
                    extra = {}
                if jn.log_from_brief(b, extra=extra):
                    logged += 1
                else:
                    skipped += 1
        except Exception as e:  # noqa: BLE001
            errs += 1
            print(f"  ⚠️ {t}: {type(e).__name__}")

    # 回填评估历史
    df = jn.load_signals()
    ev = jn.evaluate(df) if len(df) else df
    cal = jn.calibration_summary(ev) if len(df) else {}
    print(f"\n[daily_track] 留痕新增 {logged} 条 · 去重跳过 {skipped} · 失败 {errs} · 累计 {len(df)} 条")
    if cal.get("n_matured"):
        acc = cal.get("accuracy")
        print(f"  历史有效性（{cal['n_matured']} 条已成熟，{cal.get('n_judged',0)} 条可判对错）:")
        print(f"    · 判断准确率: {acc:.0%}" if acc == acc else "    · 判断准确率: —")
        print(f"    · 实现命中率 {cal['realized_hit']:.0%} vs 引擎预测胜率 {cal['pred_win_rate_mean']:.0%}")
        print(f"    · 实现超额(均) {cal['realized_excess_mean']:+.1%} · Brier {cal['brier']:.2f}(越低越准)")
    else:
        print(f"  历史: 已留痕但尚无走完 horizon 的成熟信号(需时间积累)。")
    return cal


if __name__ == "__main__":
    args = [a.upper() for a in sys.argv[1:]] or None
    run(args)
