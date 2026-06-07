"""用户数据 备份/恢复 + 持久化路径 测试。"""
from __future__ import annotations

import os

import config
from analysis import userdata as ud
from analysis import journal as jn
from analysis import event_radar as er
from analysis import mt_ledger as mt


def test_user_db_path_env_override(monkeypatch, tmp_path):
    target = tmp_path / "sub" / "my.db"
    monkeypatch.setenv("QUANTLAB_DB_PATH", str(target))
    p = config.user_db_path()
    assert p == target
    assert p.parent.exists()  # 目录被创建
    monkeypatch.delenv("QUANTLAB_DB_PATH", raising=False)
    assert config.user_db_path().name == "quantlab.db"  # 回退默认


def test_export_import_roundtrip(tmp_path):
    db = tmp_path / "a.db"
    # 写入各类用户数据
    jn.log_signal({"ticker": "NVDA", "signal_date": "2026-06-05", "horizon": 63, "price": 100.0,
                   "grade": "D", "bucket": "回撤桶", "pred_win_rate": 0.6, "pred_excess": -0.01,
                   "baseline_median": 0.05, "momentum_trap": True}, db_path=db)
    er.add_event("2026-06-13", "SpaceX 上市", scope="全市场", category="大型IPO",
                 impact="抽流动性", severity="高", db_path=db)
    mt.log_test("PEAD", "h5", 0.001, db_path=db)

    payload = ud.export_userdata(db_path=db)
    assert payload["_counts"]["signals"] == 1
    assert payload["_counts"]["event_watch"] == 1
    assert payload["_counts"]["mt_tests"] == 1

    # 导入到新库
    db2 = tmp_path / "b.db"
    written = ud.import_userdata(payload, mode="replace", db_path=db2)
    assert written.get("signals") == 1
    assert written.get("event_watch") == 1
    # 验证恢复内容
    assert len(jn.load_signals(db2)) == 1
    assert er.manual_events(db2)[0]["title"] == "SpaceX 上市"


def test_export_json_parseable(tmp_path):
    import json
    db = tmp_path / "c.db"
    er.add_event("2026-07-01", "测试事件", db_path=db)
    s = ud.export_json(db_path=db)
    parsed = json.loads(s)
    assert parsed["_schema"] == "quantlab_userdata_v1"


def test_import_bad_payload_raises():
    import pytest
    with pytest.raises(ValueError):
        ud.import_userdata({"nope": 1})


def test_import_merge_dedup(tmp_path):
    db = tmp_path / "d.db"
    mt.log_test("PEAD", "h5", 0.001, db_path=db)
    payload = ud.export_userdata(db_path=db)
    # merge 同样数据：mt_tests 有 UNIQUE(family,name) → 不重复
    ud.import_userdata(payload, mode="merge", db_path=db)
    assert len(mt.load_tests(db)) == 1
