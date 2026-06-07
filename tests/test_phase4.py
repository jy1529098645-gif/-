"""Phase 4 数据层增强测试（离线）。"""
from __future__ import annotations

import pytest


def test_macro_cache_versioned():
    from data import loader
    assert loader.CACHE_VERSION in loader._macro_cache_path("fred_BAA10Y").name


def test_pit_fundamentals_refuses_without_paid_source():
    from factors import fundamentals as fd
    with pytest.raises(NotImplementedError):
        fd.load_pit_fundamentals("AAPL")


def test_earnings_default_limit_increased():
    import inspect
    from data import loader
    sig = inspect.signature(loader.load_earnings_dates)
    assert sig.parameters["limit"].default >= 160
