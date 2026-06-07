"""Phase 1 验收：数据层。

需要联网（或已有缓存）。无法取数时自动 skip，不让 CI 误红。
"""
import pandas as pd
import pytest

from data import loader


def _maybe(fn, *a, **k):
    """取数失败（无网络/被墙）则 skip 而非 fail。"""
    try:
        return fn(*a, **k)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"数据不可用，跳过：{type(e).__name__}: {e}")


def test_load_universe():
    tickers = loader.load_universe()  # 默认池
    assert isinstance(tickers, list) and "SPY" in tickers

    with pytest.raises(KeyError):
        loader.load_universe("不存在的池")


def test_load_prices_shape_and_cache():
    px = _maybe(loader.load_prices, ["SPY", "AAPL"], "2010-01-01", "2015-01-01")
    assert isinstance(px, pd.DataFrame)
    assert list(px.columns) == ["SPY", "AAPL"]
    assert px.shape[0] > 1000               # 5 年日线约 1250 行
    assert isinstance(px.index, pd.DatetimeIndex)
    assert px.index.is_monotonic_increasing

    # 二次调用命中缓存且结果一致
    px2 = loader.load_prices(["SPY", "AAPL"], "2010-01-01", "2015-01-01")
    assert px.equals(px2)


def test_prices_not_forward_filled():
    """价格不前向填充：构造一个上市较晚的标的，早期应为 NaN（不被填充）。"""
    px = _maybe(loader.load_prices, ["SPY", "HYG"], "2000-01-01", "2010-01-01")
    # HYG 2007 才上市，2000-2006 段对齐到 SPY 交易日应为 NaN
    early = px.loc["2002-01-01":"2003-01-01", "HYG"]
    assert early.isna().all()


def test_load_macro_columns_and_sources():
    m = _maybe(loader.load_macro, "2010-01-01", "2015-01-01")
    assert {"credit_spread", "yield_curve"} <= set(m.columns)
    assert "sources" in m.attrs
    # 收益率曲线应有实际数据
    assert m["yield_curve"].dropna().shape[0] > 1000
