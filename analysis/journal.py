"""信号日志 + 校准追踪（自验证闭环）。

把工具每次给出的"当前状态/证据等级/引擎预期"落库；待 horizon 走完后，用真实价格回填实现收益，
比对"当时的预测"(引擎胜率/超额) 与"事后的实现"(是否为正/实现超额)——这是把工具从
**后视镜回测**升级成**前瞻校准**的关键：让它自己知道自己准不准。

铁律：只记录与比对，不改任何结论；校准曲线用胜率分箱(可靠性图)，诚实呈现"说的 vs 做到的"。
存储：复用 data/quantlab.db（SQLite）的 signals 表。
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import config

def _conn(db_path: str | Path | None = None):
    p = Path(db_path) if db_path else config.user_db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(p)
    c.execute(
        "CREATE TABLE IF NOT EXISTS signals ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT, signal_date TEXT, horizon INTEGER, "
        "price REAL, grade TEXT, bucket TEXT, pred_win_rate REAL, pred_excess REAL, "
        "baseline_median REAL, momentum_trap INTEGER, payload TEXT, logged TEXT, "
        "UNIQUE(ticker, signal_date, horizon))"
    )
    return c


def log_signal(record: dict, db_path: str | Path | None = None) -> bool:
    """落库一条信号。按 (ticker, signal_date, horizon) 去重；已存在返回 False。"""
    cols = ("ticker", "signal_date", "horizon", "price", "grade", "bucket",
            "pred_win_rate", "pred_excess", "baseline_median", "momentum_trap")
    vals = [record.get(c) for c in cols]
    vals[9] = int(bool(record.get("momentum_trap")))
    payload = json.dumps({k: v for k, v in record.items() if k not in cols},
                         ensure_ascii=False, default=str)
    try:
        with _conn(db_path) as c:
            c.execute(
                "INSERT INTO signals(ticker,signal_date,horizon,price,grade,bucket,"
                "pred_win_rate,pred_excess,baseline_median,momentum_trap,payload,logged) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,datetime('now'))",
                (*vals, payload),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def log_from_brief(brief: dict, db_path: str | Path | None = None, extra: dict | None = None) -> bool:
    """从 stock_brief 输出抽取关键字段落库。extra=额外留痕(如决策卡状态/建议仓位/历史常驻价)，并入 payload。"""
    be = brief.get("engine_headline") or brief.get("engine_best") or {}
    g = brief.get("grade") or {}
    rec = {
        "ticker": brief.get("ticker"), "signal_date": brief.get("date"),
        "horizon": int(brief.get("horizon", 63)), "price": brief.get("price"),
        "grade": g.get("grade"), "bucket": be.get("bucket"),
        "pred_win_rate": be.get("win_rate"), "pred_excess": be.get("excess"),
        "baseline_median": (be.get("median", 0) - be.get("excess", 0)) if be else None,
        "momentum_trap": brief.get("momentum_trap"),
        "confidence": g.get("confidence"), "max_position_fraction": g.get("max_position_fraction"),
    }
    if extra:
        rec.update(extra)
    return log_signal(rec, db_path)


def load_signals(db_path: str | Path | None = None) -> pd.DataFrame:
    with _conn(db_path) as c:
        df = pd.read_sql_query("SELECT * FROM signals ORDER BY signal_date DESC", c)
    if not df.empty:
        df["signal_date"] = pd.to_datetime(df["signal_date"])
    return df


def evaluate(df: pd.DataFrame, prices: dict[str, pd.Series] | None = None,
             asof: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """对已成熟(signal_date + horizon 交易日 ≤ asof)的信号回填实现收益。

    prices: {ticker: 价格 Series}；缺省则用 loader 拉取。返回带 realized_return / realized_pos /
    realized_excess / matured 列的 DataFrame。"""
    if df.empty:
        return df.assign(matured=[], realized_return=[], realized_pos=[], realized_excess=[])
    if prices is None:
        from data import loader
        prices = {}
        for t in df["ticker"].unique():
            try:
                prices[t] = loader.load_prices([t], "2005-01-01", None)[t].dropna()
            except Exception:  # noqa: BLE001
                pass

    rows = []
    for _, r in df.iterrows():
        p = prices.get(r["ticker"])
        rr = rpos = rexc = np.nan
        matured = False
        if p is not None and len(p):
            idx = p.index
            sd = pd.Timestamp(r["signal_date"])
            pos = int(np.searchsorted(idx.values.astype("datetime64[ns]"),
                                      np.datetime64(sd, "ns"), side="left"))
            tgt = pos + int(r["horizon"])
            # asof：只认"到 asof 为止已走完 horizon"的信号成熟，且只用 asof 前的价格回填，
            # 防止以过去某日做校准复盘时用到未来价(默认 asof=None→用全部已加载价格,到今天,无前视)。
            asof_ok = (asof is None) or (tgt < len(p) and pd.Timestamp(idx[tgt]) <= pd.Timestamp(asof))
            if 0 <= pos < len(p) and tgt < len(p) and asof_ok:
                rr = float(p.iloc[tgt] / p.iloc[pos] - 1.0)
                rpos = float(rr > 0)
                if r.get("baseline_median") == r.get("baseline_median"):
                    rexc = float(rr - r["baseline_median"])
                matured = True
        # 判断是否准确：引擎方向(看涨buckets pred_excess>0)与实现方向是否一致
        correct = np.nan
        if matured:
            pe = r.get("pred_excess"); trap = bool(r.get("momentum_trap"))
            if trap:  # 动量陷阱=防守判断, 实现≤基准(rexc≤0)才算"判对(成功避险)"
                correct = float(rexc <= 0) if rexc == rexc else np.nan
            elif pe == pe and pe is not None:
                correct = float((pe > 0) == (rexc > 0)) if rexc == rexc else np.nan
            elif rr == rr:
                correct = float(rr > 0)  # 无超额口径则看绝对方向
        d = r.to_dict()
        d.update({"matured": matured, "realized_return": rr, "realized_pos": rpos,
                  "realized_excess": rexc, "correct": correct})
        rows.append(d)
    return pd.DataFrame(rows)


def calibration_summary(ev: pd.DataFrame) -> dict:
    """校准摘要：成熟样本的实现命中率 vs 引擎预测胜率(可靠性分箱)、实现超额、Brier 分。"""
    if ev.empty or "matured" not in ev.columns:
        return {"n_total": int(len(ev)), "n_matured": 0, "note": "暂无成熟信号可评估。"}
    m = ev[ev["matured"] == True]  # noqa: E712
    n_mat = int(len(m))
    if n_mat == 0:
        return {"n_total": int(len(ev)), "n_matured": 0, "note": "已有信号但尚未走完 horizon。"}

    realized_hit = float(m["realized_pos"].mean())
    pred_mean = float(m["pred_win_rate"].mean()) if m["pred_win_rate"].notna().any() else float("nan")
    realized_excess = float(m["realized_excess"].mean()) if m["realized_excess"].notna().any() else float("nan")
    # Brier：预测胜率 vs 实现(0/1)
    valid = m.dropna(subset=["pred_win_rate", "realized_pos"])
    brier = float(((valid["pred_win_rate"] - valid["realized_pos"]) ** 2).mean()) if len(valid) else float("nan")

    # 可靠性分箱（预测胜率分箱 → 实现命中率）
    bins = []
    if len(valid):
        valid = valid.assign(_b=pd.cut(valid["pred_win_rate"], [0, 0.4, 0.5, 0.6, 0.7, 1.01]))
        for b, grp in valid.groupby("_b", observed=True):
            bins.append({"pred_bin": str(b), "n": int(len(grp)),
                         "pred_mean": float(grp["pred_win_rate"].mean()),
                         "realized_hit": float(grp["realized_pos"].mean())})

    # 按证据等级的实现表现
    by_grade = []
    for g, grp in m.groupby("grade", observed=True):
        by_grade.append({"grade": g, "n": int(len(grp)),
                         "realized_hit": float(grp["realized_pos"].mean()),
                         "realized_excess": float(grp["realized_excess"].mean()) if grp["realized_excess"].notna().any() else float("nan")})

    accuracy = float(m["correct"].mean()) if "correct" in m.columns and m["correct"].notna().any() else float("nan")
    n_judged = int(m["correct"].notna().sum()) if "correct" in m.columns else 0
    return {
        "n_total": int(len(ev)), "n_matured": n_mat,
        "accuracy": accuracy, "n_judged": n_judged,
        "realized_hit": realized_hit, "pred_win_rate_mean": pred_mean,
        "realized_excess_mean": realized_excess, "brier": brier,
        "reliability": bins, "by_grade": by_grade,
        "note": ("校准追踪：'预测胜率'是落库时引擎对该状态的历史胜率，'实现命中率'是事后真实为正的比例；"
                 "两者越接近越校准。Brier 越低越好(0=完美,0.25=瞎猜)。实现超额>0 才说明信号事后真有优势。"),
    }
