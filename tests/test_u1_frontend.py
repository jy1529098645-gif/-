"""U1 验收：OHLCV 数据层 + 多信号组合 + 规则持久化。

OHLCV 联网（缓存/skip）；build_entry / SQLite 持久化为纯逻辑（不联网）。
"""
import numpy as np
import pandas as pd
import pytest

from factors import signals as sg


def test_build_entry_and_or():
    idx = pd.bdate_range("2020-01-01", periods=10)
    a = pd.Series([True, True, False, False, True, False, True, True, False, True], index=idx)
    b = pd.Series([True, False, True, False, True, True, False, True, False, False], index=idx)

    # 用 monkeypatch 风格：直接验证 combine 逻辑
    e_and = sg.combine_and(a, b)
    e_or = sg.combine_or(a, b)
    assert list(e_and) == [x and y for x, y in zip(a, b)]
    assert list(e_or) == [x or y for x, y in zip(a, b)]


def test_make_single_and_build_entry_runs():
    idx = pd.bdate_range("2018-01-01", periods=400)
    rng = np.random.default_rng(0)
    price = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, 400))), index=idx)

    f = sg.make_single("dip_from_high", 0.10, 0.0)
    assert f(price).dtype == bool

    entry = sg.build_entry([("dip_from_high", 0.10, 0.0), ("rsi_oversold", 14, 40)], op="and")
    out = entry(price)
    assert out.dtype == bool and out.shape[0] == price.shape[0]
    # AND 组合不多于任一单信号触发数
    s1 = sg.make_single("dip_from_high", 0.10, 0.0)(price).fillna(False)
    assert out.sum() <= s1.sum() + 1


def test_store_roundtrip(tmp_path, monkeypatch):
    import config
    from frontend import store

    monkeypatch.setattr(store, "_DB", tmp_path / "t.db")
    spec = {"specs": (("dip_from_high", 0.1, 0.0),), "op": "and", "trailing": 0.2,
            "tp": 0.25, "time_stop": 63, "cond_kind": "none", "cond_window": 20, "rule_name": "r"}
    store.save_rule("r1", spec)
    got = store.get_rule("r1")
    assert got["op"] == "and" and got["specs"][0][0] == "dip_from_high"
    assert any(r["名称"] == "r1" for r in store.list_rules())
    store.delete_rule("r1")
    assert store.get_rule("r1") is None


def test_load_ohlcv_columns():
    from data import loader

    try:
        df = loader.load_ohlcv("AAPL", "2018-01-01", "2022-01-01")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"数据不可用：{type(e).__name__}: {e}")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.shape[0] > 500
    # high>=low，OHLC 自洽
    assert (df["high"] >= df["low"]).all()
    assert not df[["open", "high", "low", "close"]].isna().any().any()
