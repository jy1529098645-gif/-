"""策略验收闸门（把三铁律合成一个 go/no-go 裁决）。

三条硬标准（全过才 PASS）：
  1. 跑赢基准：池化超额中位 > 0 且**经 N_eff 折算后显著**（CI 不跨 0）。
  2. 出样本夏普 ≥ 阈值：规则**日度策略收益**在 walk-forward OOS 段的**年化夏普** ≥ 1.0（默认）。
     —— 固定规则(无拟合参数)的 OOS = 按时间切出的样本外段；含未持仓的现金日（诚实口径）。
  3. 回撤可承受：池化日度净值的**最大回撤** ≤ 容忍阈值（默认 35%）。

铁律：PASS ≠ 买入信号，只代表「这条规则过了反过拟合及格线」；FAIL = 别信。仍是校准非预测。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stats.bootstrap import sharpe
from stats.walkforward import walk_forward_splits


def _rule_daily_returns(entry_fn, exit_spec, price: pd.Series) -> pd.Series:
    """单票跑规则，返回日度策略收益（vectorbt pf.returns()）。"""
    from backtest.exits import run_trades

    price = price.dropna()
    entries = entry_fn(price)
    pf = run_trades(price, entries, exit_spec)
    return pf.returns().reindex(price.index).fillna(0.0)


def rule_daily_panel(entry_fn, exit_spec, tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    """各票日度策略收益面板（列=ticker）。失败的票静默跳过。"""
    from data import loader

    cols = {}
    for tk in tickers:
        try:
            price = loader.load_prices([tk], start, end)[tk]
            cols[tk] = _rule_daily_returns(entry_fn, exit_spec, price)
        except Exception:  # noqa: BLE001
            continue
    return pd.DataFrame(cols)


def acceptance_gate(
    rule_eval_result: dict,
    entry_fn,
    exit_spec: dict,
    tickers: list[str],
    start: str = "2013-01-01",
    end: str | None = None,
    oos_sharpe_min: float = 1.0,
    max_dd_tol: float = 0.35,
    train: int = 1000,
    test: int = 250,
) -> dict:
    """对一条规则跑三条验收标准，返回结构化裁决。

    rule_eval_result：evaluation.rule_eval.evaluate_rule(...) 的返回（取 pooled.excess*）。
    """
    p = rule_eval_result["pooled"]

    # 标准 1：跑赢基准（N_eff 折算后显著）
    c1_pass = bool(p["excess_median"] > 0 and p["excess_significant"])

    # 池化日度收益（等权，跳过未持仓票的 NaN）
    panel = rule_daily_panel(entry_fn, exit_spec, tickers, start, end)
    pooled = panel.mean(axis=1).dropna() if not panel.empty else pd.Series(dtype=float)

    # 标准 2：OOS 年化夏普
    oos_sharpe = float("nan")
    n_oos = 0
    if len(pooled) > train + test:
        splits = walk_forward_splits(pooled.index, train=train, test=test)
        oos_chunks = [pooled.iloc[s.test_start:s.test_end].to_numpy() for s in splits]
        if oos_chunks:
            oos = np.concatenate(oos_chunks)
            n_oos = int(len(oos))
            oos_sharpe = float(sharpe(oos)) if len(oos) > 5 else float("nan")
    elif len(pooled) > 30:
        oos_sharpe = float(sharpe(pooled.to_numpy()))  # 历史太短，退化用全样本（标注）
        n_oos = int(len(pooled))
    c2_pass = bool(oos_sharpe == oos_sharpe and oos_sharpe >= oos_sharpe_min)

    # 标准 3：最大回撤可承受
    max_dd = float("nan")
    if len(pooled) > 5:
        eq = (1.0 + pooled).cumprod()
        max_dd = float((eq / eq.cummax() - 1.0).min())
    c3_pass = bool(max_dd == max_dd and max_dd >= -abs(max_dd_tol))

    overall = c1_pass and c2_pass and c3_pass
    return {
        "overall": overall,
        "criteria": {
            "beat_baseline": {
                "pass": c1_pass,
                "excess_median": float(p["excess_median"]),
                "ci": [float(p["excess_ci_low"]), float(p["excess_ci_high"])],
                "significant": bool(p["excess_significant"]),
                "n_eff": float(p["n_eff"]),
                "label": "跑赢随机基准(N_eff折算后显著)",
            },
            "oos_sharpe": {
                "pass": c2_pass, "value": oos_sharpe, "threshold": oos_sharpe_min,
                "n_oos_days": n_oos, "label": f"OOS 年化夏普 ≥ {oos_sharpe_min}",
            },
            "drawdown": {
                "pass": c3_pass, "max_drawdown": max_dd, "tolerance": -abs(max_dd_tol),
                "label": f"最大回撤 ≤ {abs(max_dd_tol):.0%}",
            },
        },
        "params": {"oos_sharpe_min": oos_sharpe_min, "max_dd_tol": max_dd_tol,
                   "tickers": list(tickers), "start": start},
    }


def format_gate(gate: dict) -> str:
    """把裁决渲染成一句话措辞。"""
    c = gate["criteria"]
    mk = lambda b: "✅" if b else "❌"  # noqa: E731
    b1 = c["beat_baseline"]; b2 = c["oos_sharpe"]; b3 = c["drawdown"]
    head = "🟢 PASS（过反过拟合及格线，非买入信号）" if gate["overall"] else "🔴 FAIL（未达标，别信）"
    return (
        f"{head}　|　"
        f"{mk(b1['pass'])} 跑赢基准 超额 {b1['excess_median']:+.1%} "
        f"CI[{b1['ci'][0]:+.1%},{b1['ci'][1]:+.1%}]{'显著' if b1['significant'] else '不显著'}"
        f"(N_eff≈{b1['n_eff']:.0f})　|　"
        f"{mk(b2['pass'])} OOS夏普 {b2['value']:.2f}{'(NA)' if b2['value']!=b2['value'] else ''}≥{b2['threshold']}　|　"
        f"{mk(b3['pass'])} 最大回撤 {b3['max_drawdown']:.0%}≤{abs(b3['tolerance']):.0%}"
    )
