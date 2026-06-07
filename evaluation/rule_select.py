"""规则选择的 walk-forward + deflated Sharpe 包裹（补充规格 A，S3）。

铁律：
- 参数只能在 IS（训练段）选、在 OOS（测试段）报；禁止单票/全样本寻优。
- 候选组数 >1 时，对"最优"结果做 deflated Sharpe 折扣（治多重检验/数据窥探）。
- 每个"最优规则"必须带出样本表现 + 折扣后夏普。

per-trade Sharpe = 每笔收益的 mean/std（同口径用于 deflated Sharpe，n_obs=笔数）。
"""
from __future__ import annotations

from itertools import product
from typing import Callable

import numpy as np
import pandas as pd

import config
from backtest import exits as ex
from factors import signals as sg
from stats.deflated_sharpe import deflated_sharpe_ratio
from stats.walkforward import walk_forward_splits

_CFG = config.load_config()


def per_trade_sharpe(returns: np.ndarray) -> float:
    """每笔收益的夏普（mean/std），非年化。样本不足或零方差返回 nan/0。"""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size < 3:
        return float("nan")
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


# ---------------------------------------------------------------------------
# 候选规则网格
# ---------------------------------------------------------------------------
def candidate_grid(
    dip_pcts=(0.10, 0.15, 0.20),
    trail_stops=(0.15, 0.20, 0.25),
    take_profit=0.25,
    time_stop=63,
) -> list[dict]:
    """生成"入场×出场"候选组合。每个候选含 name / entry_fn / exit_spec。"""
    grid = []
    for dp, ts in product(dip_pcts, trail_stops):
        grid.append(
            {
                "name": f"dip{int(dp*100)}_trail{int(ts*100)}",
                "entry_fn": (lambda p, dp=dp: sg.dip_from_high(p, pct=dp)),
                "exit_spec": {"trailing_stop": ts, "take_profit": take_profit, "time_stop": time_stop},
            }
        )
    return grid


def candidate_grid_from(specs: list[tuple], op: str, base_exit: dict,
                        p1_factors=(0.7, 1.0, 1.3), trail_factors=(0.75, 1.0, 1.25)) -> list[dict]:
    """围绕**当前规则**动态生成候选：缩放首信号的 p1 与移动止损，做反过拟合体检。

    specs：当前入场信号 [(name,p1,p2),...]；op：组合；base_exit：当前出场规格。
    """
    base_p1 = specs[0][1]
    base_trail = base_exit.get("trailing_stop") or base_exit.get("stop_loss") or 0.20
    rest = list(specs[1:])
    grid = []
    for pf in p1_factors:
        for tf in trail_factors:
            new_p1 = round(base_p1 * pf, 4)
            new_specs = [(specs[0][0], new_p1, specs[0][2]), *rest]
            ex_spec = dict(base_exit)
            if "trailing_stop" in ex_spec or "stop_loss" in ex_spec:
                key = "trailing_stop" if "trailing_stop" in ex_spec else "stop_loss"
                ex_spec[key] = round(base_trail * tf, 4)
            grid.append({
                "name": f"{specs[0][0][:4]}{new_p1}_st{round(base_trail*tf,3)}",
                "entry_fn": (lambda p, s=tuple(new_specs), o=op: sg.build_entry(list(s), o)(p)),
                "exit_spec": ex_spec,
            })
    return grid


def _candidate_trades(cand: dict, prices: dict[str, pd.Series]) -> pd.DataFrame:
    """一个候选在所有标的上的全历史池化逐笔交易（带 entry_date 便于按窗口切分）。"""
    frames = []
    for t, p in prices.items():
        entries = cand["entry_fn"](p)
        pf = ex.run_trades(p, entries, cand["exit_spec"])
        tr = ex.extract_trades(pf, p)
        tr["ticker"] = t
        frames.append(tr)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("entry_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# deflated Sharpe（全样本多候选）
# ---------------------------------------------------------------------------
def deflated_rule_sharpe(
    candidates: list[dict] | None = None,
    tickers: list[str] | None = None,
    start: str = "2010-01-01",
    end: str | None = None,
    min_trades: int = 20,
    extra_trials: int = 0,
) -> dict:
    """对候选网格算每个的池化 per-trade Sharpe，取最优并做 deflated Sharpe 折扣。

    extra_trials：本次自动网格**之外**已经手工试过的规则数（如前端 AND/OR 组合构建器里
    用户拖出来的若干组合、之前几轮调参）。这些"看不见的尝试"同样消耗显著性预算，
    必须计入 n_trials，否则 DSR 会系统性高估稳健度。n_trials = 合格候选数 + extra_trials。
    """
    from data import loader

    candidates = candidates or candidate_grid()
    tickers = tickers or loader.load_universe(_CFG["single_name"]["universe"])
    prices = {t: loader.load_prices([t], start, end)[t].dropna() for t in tickers}

    rows = []
    for c in candidates:
        tr = _candidate_trades(c, prices)
        r = tr["return"].dropna().to_numpy()
        rows.append({"name": c["name"], "n_trades": len(r), "sharpe": per_trade_sharpe(r)})
    tab = pd.DataFrame(rows).dropna(subset=["sharpe"])
    elig = tab[tab["n_trades"] >= min_trades]
    if elig.empty:
        raise ValueError("没有候选满足最小交易数")

    best = elig.loc[elig["sharpe"].idxmax()]
    sr_trials_std = float(elig["sharpe"].std(ddof=1)) if len(elig) > 1 else 0.0
    n_trials = len(elig) + max(0, int(extra_trials))  # 计入网格外的手工尝试
    dsr = deflated_sharpe_ratio(
        sr=float(best["sharpe"]),
        sr_trials_std=sr_trials_std if sr_trials_std > 0 else 1e-6,
        n_trials=n_trials,
        n_obs=int(best["n_trades"]),
    )
    return {
        "table": tab.sort_values("sharpe", ascending=False).reset_index(drop=True),
        "best_name": best["name"],
        "best_sharpe": float(best["sharpe"]),
        "n_trials": n_trials,
        "sr_trials_std": sr_trials_std,
        "deflated_sharpe_prob": dsr,
        "robust": bool(dsr > 0.95),
        "note": (
            f"从 {n_trials} 个候选中选出最优（{best['name']}，per-trade Sharpe {best['sharpe']:.3f}）。"
            f"Deflated Sharpe 概率={dsr:.2f}（>0.95 才算稳健）。"
            f"{'通过' if dsr > 0.95 else '未通过——最优很可能是多重检验下的运气，慎用。'}"
        ),
    }


# ---------------------------------------------------------------------------
# walk-forward：IS 选、OOS 报
# ---------------------------------------------------------------------------
def walk_forward_rule(
    candidates: list[dict] | None = None,
    tickers: list[str] | None = None,
    start: str = "2010-01-01",
    end: str | None = None,
    train_years: int = 4,
    test_years: int = 1,
    min_trades: int = 8,
) -> dict:
    """walk-forward：每段在 IS 选最优候选，在紧接的 OOS 报其表现，暴露过拟合缺口。"""
    from data import loader

    candidates = candidates or candidate_grid()
    tickers = tickers or loader.load_universe(_CFG["single_name"]["universe"])
    prices = {t: loader.load_prices([t], start, end)[t].dropna() for t in tickers}

    # 每个候选预算全历史交易，后按窗口用 entry_date 切分（避免重复回测）
    cand_trades = {c["name"]: _candidate_trades(c, prices) for c in candidates}

    # 用任一标的的交易日索引作为时间轴
    cal = sorted(set().union(*[set(p.index) for p in prices.values()]))
    cal = pd.DatetimeIndex(cal)
    train = train_years * 252
    test = test_years * 252
    splits = walk_forward_splits(cal, train=train, test=test, step=test)

    def _sharpe_in(name, lo, hi):
        tr = cand_trades[name]
        sel = tr[(tr["entry_date"] >= lo) & (tr["entry_date"] < hi)]["return"].dropna().to_numpy()
        return per_trade_sharpe(sel), len(sel)

    rows = []
    for sp in splits:
        tr_idx, te_idx = sp.slice(cal)
        is_lo, is_hi = tr_idx[0], tr_idx[-1]
        oos_lo, oos_hi = te_idx[0], te_idx[-1]

        # IS 选最优
        best_name, best_is = None, -np.inf
        for c in candidates:
            s, n = _sharpe_in(c["name"], is_lo, is_hi)
            if n >= min_trades and np.isfinite(s) and s > best_is:
                best_is, best_name = s, c["name"]
        if best_name is None:
            continue
        oos_s, oos_n = _sharpe_in(best_name, oos_lo, oos_hi)
        rows.append(
            {
                "is_start": is_lo.date(), "oos_start": oos_lo.date(),
                "selected": best_name, "is_sharpe": best_is,
                "oos_sharpe": oos_s, "oos_n_trades": oos_n,
            }
        )

    wf = pd.DataFrame(rows)
    valid = wf.dropna(subset=["oos_sharpe"])
    summary = {
        "n_splits": len(wf),
        "mean_is_sharpe": float(valid["is_sharpe"].mean()) if not valid.empty else float("nan"),
        "mean_oos_sharpe": float(valid["oos_sharpe"].mean()) if not valid.empty else float("nan"),
        "oos_positive_rate": float((valid["oos_sharpe"] > 0).mean()) if not valid.empty else float("nan"),
    }
    summary["overfit_gap"] = summary["mean_is_sharpe"] - summary["mean_oos_sharpe"]
    return {"table": wf, "summary": summary,
            "note": (
                f"{summary['n_splits']} 段 walk-forward：IS 平均 Sharpe {summary['mean_is_sharpe']:.2f} "
                f"vs OOS 平均 {summary['mean_oos_sharpe']:.2f}（过拟合缺口 {summary['overfit_gap']:+.2f}）；"
                f"OOS 为正比例 {summary['oos_positive_rate']:.0%}。缺口越大、OOS 越差，越像过拟合。"
            )}
