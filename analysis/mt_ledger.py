"""全局多重检验账本（Multiple-Testing Ledger）。

整个工具跨页会检验很多假设（PEAD 各 horizon、横截面因子、各状态桶…）。单看某一处"显著"
没意义——挖得够多总能撞上假阳性。本账本把所有跑过的检验落库，对全体做 Benjamini-Hochberg
FDR 校正，诚实回答："扣除挖掘后，到底还剩几个真显著？"

铁律：只记录与校正，不改结论。p 值缺失的检验(只有 CI 的)不强行编 p，单独计数。
存储：复用 data/quantlab.db 的 mt_tests 表。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import config

_DB = Path(config.ROOT) / "data" / "quantlab.db"


def _conn(db_path: str | Path | None = None):
    p = Path(db_path) if db_path else _DB
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(p)
    c.execute(
        "CREATE TABLE IF NOT EXISTS mt_tests ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, family TEXT, name TEXT, p_value REAL, "
        "stat REAL, logged TEXT, UNIQUE(family, name))"
    )
    return c


def log_test(family: str, name: str, p_value: float, stat: float | None = None,
             db_path: str | Path | None = None) -> None:
    """记录一个假设检验（family+name 去重，重复则更新 p）。p_value 为 None/NaN 则跳过。"""
    if p_value is None or (isinstance(p_value, float) and p_value != p_value):
        return
    with _conn(db_path) as c:
        c.execute(
            "INSERT INTO mt_tests(family,name,p_value,stat,logged) VALUES(?,?,?,?,datetime('now')) "
            "ON CONFLICT(family,name) DO UPDATE SET p_value=excluded.p_value, stat=excluded.stat, "
            "logged=excluded.logged",
            (family, name, float(p_value), None if stat is None else float(stat)),
        )


def load_tests(db_path: str | Path | None = None) -> pd.DataFrame:
    with _conn(db_path) as c:
        return pd.read_sql_query("SELECT family,name,p_value,stat,logged FROM mt_tests ORDER BY p_value", c)


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.10) -> tuple[np.ndarray, float]:
    """BH-FDR：返回 (是否拒绝原假设的布尔数组, 阈值 p*)。控制错误发现率 ≤ alpha。"""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=bool), float("nan")
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    below = ranked <= thresh
    k = np.where(below)[0].max() + 1 if below.any() else 0
    pstar = ranked[k - 1] if k > 0 else 0.0
    reject = np.zeros(m, dtype=bool)
    if k > 0:
        reject[order[:k]] = True
    return reject, float(pstar)


def fdr_report(alpha: float = 0.10, db_path: str | Path | None = None) -> dict:
    """对账本里全部检验做 BH-FDR。返回 {n_tests, n_sig_raw, n_sig_bh, p_star, table, note}。"""
    df = load_tests(db_path)
    if df.empty:
        return {"n_tests": 0, "n_sig_raw": 0, "n_sig_bh": 0, "p_star": float("nan"),
                "table": df, "note": "账本为空——运行 PEAD / 横截面等带 p 值的分析后会自动累计。"}
    p = df["p_value"].to_numpy()
    reject, pstar = benjamini_hochberg(p, alpha=alpha)
    df = df.assign(显著_未校正=(p < 0.05), 显著_BH=reject)
    return {
        "n_tests": int(len(df)), "n_sig_raw": int((p < 0.05).sum()),
        "n_sig_bh": int(reject.sum()), "p_star": pstar, "alpha": alpha, "table": df,
        "note": (f"共 {len(df)} 个检验：未校正下 {int((p<0.05).sum())} 个 p<0.05；"
                 f"Benjamini-Hochberg(FDR≤{alpha:.0%})校正后仅 {int(reject.sum())} 个存活(阈值 p*≤{pstar:.4f})。"
                 "差额就是被挖掘出来的假阳性——这才是诚实的'剩下几个真信号'。"),
    }
