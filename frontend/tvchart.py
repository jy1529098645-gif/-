"""TradingView 原生图表（Lightweight Charts，TradingView 官方开源引擎）。

与 TradingView 一致的操作：十字光标(带轴价/时间标签)、滚轮缩放、拖动平移、缩放自动 y 适配、
量价副图、买卖箭头标记。守住产品边界：**不提供画线工具、不标买卖点/目标价**，
K 线上只放规则触发的进出场箭头。
"""
from __future__ import annotations

import pandas as pd
from streamlit_lightweight_charts import renderLightweightCharts

from frontend import theme as _theme


def _up() -> str:
    return _theme.tokens()["candle_up"]


def _dn() -> str:
    return _theme.tokens()["candle_down"]

# ---------------------------------------------------------------------------
# 周期切换辅助（纯 pandas，不引入 streamlit）：时间范围 / K线粒度 / 标记吸附
# ---------------------------------------------------------------------------
PERIODS = ["1M", "3M", "6M", "1Y", "3Y", "全部"]
TIMEFRAMES = ["日", "周", "月"]
_PERIOD_OFFSET = {
    "1M": pd.DateOffset(months=1), "3M": pd.DateOffset(months=3),
    "6M": pd.DateOffset(months=6), "1Y": pd.DateOffset(years=1),
    "3Y": pd.DateOffset(years=3),
}
_TF_RULE = {"日": None, "周": "W", "月": "ME"}
_OHLC_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def resample_ohlcv(df: pd.DataFrame, timeframe: str = "日") -> pd.DataFrame:
    """把日线 OHLCV 聚合到 周/月 蜡烛。日线原样返回。"""
    rule = _TF_RULE.get(timeframe)
    if rule is None or df is None or df.empty:
        return df
    agg = {c: _OHLC_AGG[c] for c in _OHLC_AGG if c in df.columns}
    out = df.resample(rule).agg(agg)
    return out.dropna(subset=["close"]) if "close" in out.columns else out.dropna(how="all")


def slice_period(df: pd.DataFrame, period: str = "全部") -> pd.DataFrame:
    """按时间范围切片（相对数据最后一根 bar 往回推）。"""
    off = _PERIOD_OFFSET.get(period)
    if off is None or df is None or df.empty:
        return df
    return df.loc[df.index >= (df.index.max() - off)]


def snap_markers_to_bars(trades: "pd.DataFrame | None", index: pd.DatetimeIndex) -> "pd.DataFrame | None":
    """周/月聚合后，把每笔 entry/exit 日期吸附到重采样后存在的 bar，避免 lightweight-charts 丢标记。

    日线（index 与原 trades 同粒度）下吸附是恒等映射，无副作用。
    """
    if trades is None or len(trades) == 0 or index is None or len(index) == 0:
        return trades
    idx = pd.DatetimeIndex(index).sort_values()

    def _snap(ts):
        d = pd.Timestamp(ts)
        pos = idx.searchsorted(d, side="right") - 1   # 落到 ≤ d 的最近一根 bar
        pos = max(0, pos)
        return idx[pos]

    out = trades.copy()
    for col in ("entry_date", "exit_date"):
        if col in out.columns:
            out[col] = out[col].map(_snap)
    return out


def _candles(ohlcv: pd.DataFrame) -> list[dict]:
    df = ohlcv.dropna(subset=["open", "high", "low", "close"])
    return [{"time": d.strftime("%Y-%m-%d"), "open": float(o), "high": float(h), "low": float(l), "close": float(c)}
            for d, o, h, l, c in zip(df.index, df["open"], df["high"], df["low"], df["close"])]


def _volume(ohlcv: pd.DataFrame) -> list[dict]:
    df = ohlcv.dropna(subset=["volume"])
    up = (df["close"] >= df["open"]).to_numpy()
    tk = _theme.tokens()
    vup, vdn = tk["vol_up"], tk["vol_down"]
    return [{"time": d.strftime("%Y-%m-%d"), "value": float(v),
             "color": (vup if u else vdn)}
            for d, v, u in zip(df.index, df["volume"], up)]


def _markers(trades: pd.DataFrame) -> list[dict]:
    m = []
    up, dn = _up(), _dn()
    for _, t in trades.iterrows():
        col = up if t["return"] > 0 else dn
        m.append({"time": pd.Timestamp(t["entry_date"]).strftime("%Y-%m-%d"),
                  "position": "belowBar", "color": col, "shape": "arrowUp", "text": "买"})
        m.append({"time": pd.Timestamp(t["exit_date"]).strftime("%Y-%m-%d"),
                  "position": "aboveBar", "color": col, "shape": "arrowDown", "text": "卖"})
    m.sort(key=lambda x: x["time"])
    return m


def _chart_opts(height: int, log: bool = False) -> dict:
    """统一的 TradingView 图表外观（跟随主题 + 磁吸十字光标 + 缩放/平移）。"""
    tk = _theme.tokens()
    return {
        "height": height,
        "layout": {"background": {"type": "solid", "color": "rgba(0,0,0,0)"}, "textColor": tk["text"],
                   "fontFamily": "Inter, Segoe UI, sans-serif"},
        "grid": {"vertLines": {"color": tk["grid"]}, "horzLines": {"color": tk["grid"]}},
        "crosshair": {"mode": 1},  # 1=磁吸十字光标（TradingView 手感）
        "timeScale": {"timeVisible": True, "rightOffset": 6, "borderColor": tk["border"]},
        "rightPriceScale": {"borderColor": tk["border"], "mode": 1 if log else 0},  # 1=对数
    }


def _price_lines(price_lines: list[dict]) -> list[dict]:
    _info = _theme.tokens()["info"]
    return [{"price": float(p["price"]), "color": p.get("color", _info), "lineWidth": 1,
             "lineStyle": 2, "axisLabelVisible": True, "title": p.get("title", "")} for p in price_lines]


def tv_line(series: pd.Series, markers: list[dict] | None = None, price_lines: list[dict] | None = None,
            key: str = "tvl", height: int = 460, color: str | None = None, log: bool = True) -> None:
    """渲染 TradingView 原生折线图（事件时间线等），支持事件标记 + 横向价位线。"""
    s = series.dropna()
    if color is None:
        color = _theme.tokens()["primary"]
    data = [{"time": d.strftime("%Y-%m-%d"), "value": float(v)} for d, v in s.items()]
    line = {"type": "Line", "data": data,
            "options": {"color": color, "lineWidth": 2, "priceLineVisible": False, "lastValueVisible": True}}
    if markers:
        line["markers"] = sorted(markers, key=lambda m: m["time"])
    if price_lines:
        line["priceLines"] = _price_lines(price_lines)
    renderLightweightCharts([{"chart": _chart_opts(height, log), "series": [line]}], key)


def tv_candles(ohlcv: pd.DataFrame, trades: pd.DataFrame | None = None,
               price_lines: list[dict] | None = None, key: str = "tv",
               height: int = 540, log: bool = False) -> None:
    """渲染 TradingView 原生 K 线 + 成交量副图 + 买卖箭头 + 横向价位线（POC 等）。"""
    chart = _chart_opts(height, log)
    _u, _d, _info = _up(), _dn(), _theme.tokens()["info"]
    candle = {
        "type": "Candlestick",
        "data": _candles(ohlcv),
        "options": {"upColor": _u, "downColor": _d, "wickUpColor": _u, "wickDownColor": _d,
                    "borderVisible": False},
    }
    if trades is not None and len(trades):
        candle["markers"] = _markers(trades)
    if price_lines:
        candle["priceLines"] = [
            {"price": float(p["price"]), "color": p.get("color", _info), "lineWidth": 1,
             "lineStyle": 2, "axisLabelVisible": True, "title": p.get("title", "")}
            for p in price_lines
        ]
    volume = {
        "type": "Histogram",
        "data": _volume(ohlcv),
        "options": {"priceFormat": {"type": "volume"}, "priceScaleId": ""},
        "priceScale": {"scaleMargins": {"top": 0.8, "bottom": 0}},
    }
    renderLightweightCharts([{"chart": chart, "series": [candle, volume]}], key)
