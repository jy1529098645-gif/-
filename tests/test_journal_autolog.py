"""自动留痕(决策卡 extra + 去重)测试。"""
from __future__ import annotations
import json, os, tempfile
from analysis import journal as jn


def _db():
    import time
    return os.path.join(tempfile.gettempdir(), f"jtest_{os.getpid()}.db")


def test_log_extra_and_dedup():
    db = _db()
    if os.path.exists(db):
        try: os.remove(db)
        except OSError: pass
    brief = {"ticker": "NVDA", "date": "2026-06-08", "horizon": 63, "price": 208.0,
             "engine_headline": {"bucket": "in_drawdown", "win_rate": 0.7, "excess": 0.03, "median": 0.09},
             "grade": {"grade": "B", "confidence": "中", "max_position_fraction": 0.6},
             "momentum_trap": False}
    ok = jn.log_from_brief(brief, db_path=db, extra={"decision_state": "🟢追", "rec_position": 0.67, "entry_anchor": 206.9})
    dup = jn.log_from_brief(brief, db_path=db, extra={"decision_state": "🟢追"})
    assert ok is True and dup is False          # 同日同票去重
    df = jn.load_signals(db_path=db)
    assert len(df) == 1
    pl = json.loads(df.iloc[0]["payload"])
    assert pl.get("decision_state") == "🟢追" and pl.get("rec_position") == 0.67


def test_evaluate_and_calibration_runs():
    db = _db().replace(".db", "_b.db")
    if os.path.exists(db):
        try: os.remove(db)
        except OSError: pass
    jn.log_from_brief({"ticker": "X", "date": "2020-01-02", "horizon": 21, "price": 100.0,
                       "engine_headline": {"win_rate": 0.6, "excess": 0.02, "median": 0.05},
                       "grade": {"grade": "B"}}, db_path=db)
    ev = jn.evaluate(jn.load_signals(db_path=db), prices={"X": __import__("pandas").Series(
        [100.0] * 30, index=__import__("pandas").date_range("2020-01-02", periods=30, freq="B"))})
    cal = jn.calibration_summary(ev)
    assert "n_total" in cal
