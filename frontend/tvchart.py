"""TradingView 原生图表（Lightweight Charts，TradingView 官方开源引擎）。

与 TradingView 一致的操作：十字光标(带轴价/时间标签)、滚轮缩放、拖动平移、缩放自动 y 适配、
量价副图、买卖箭头标记。守住产品边界：**不提供画线工具、不标买卖点/目标价**，
K 线上只放规则触发的进出场箭头。
"""
from __future__ import annotations

import pandas as pd
from streamlit_lightweight_charts import renderLightweightCharts

_UP, _DN = "#2BE6A8", "#FF5C7A"


def _candles(ohlcv: pd.DataFrame) -> list[dict]:
    df = ohlcv.dropna(subset=["open", "high", "low", "close"])
    return [{"time": d.strftime("%Y-%m-%d"), "open": float(o), "high": float(h), "low": float(l), "close": float(c)}
            for d, o, h, l, c in zip(df.index, df["open"], df["high"], df["low"], df["close"])]


def _volume(ohlcv: pd.DataFrame) -> list[dict]:
    df = ohlcv.dropna(subset=["volume"])
    up = (df["close"] >= df["open"]).to_numpy()
    return [{"time": d.strftime("%Y-%m-%d"), "value": float(v),
             "color": ("rgba(43,230,168,0.5)" if u else "rgba(255,92,122,0.5)")}
            for d, v, u in zip(df.index, df["volume"], up)]


def _markers(trades: pd.DataFrame) -> list[dict]:
    m = []
    for _, t in trades.iterrows():
        col = _UP if t["return"] > 0 else _DN
        m.append({"time": pd.Timestamp(t["entry_date"]).strftime("%Y-%m-%d"),
                  "position": "belowBar", "color": col, "shape": "arrowUp", "text": "买"})
        m.append({"time": pd.Timestamp(t["exit_date"]).strftime("%Y-%m-%d"),
                  "position": "aboveBar", "color": col, "shape": "arrowDown", "text": "卖"})
    m.sort(key=lambda x: x["time"])
    return m


def _chart_opts(height: int, log: bool = False) -> dict:
    """统一的 TradingView 图表外观（暗色 + 磁吸十字光标 + 缩放/平移）。"""
    return {
        "height": height,
        "layout": {"background": {"type": "solid", "color": "rgba(0,0,0,0)"}, "textColor": "#E6E9EF",
                   "fontFamily": "Inter, Segoe UI, sans-serif"},
        "grid": {"vertLines": {"color": "rgba(255,255,255,0.06)"}, "horzLines": {"color": "rgba(255,255,255,0.06)"}},
        "crosshair": {"mode": 1},  # 1=磁吸十字光标（TradingView 手感）
        "timeScale": {"timeVisible": True, "rightOffset": 6, "borderColor": "rgba(255,255,255,0.2)"},
        "rightPriceScale": {"borderColor": "rgba(255,255,255,0.2)", "mode": 1 if log else 0},  # 1=对数
    }


def _price_lines(price_lines: list[dict]) -> list[dict]:
    return [{"price": float(p["price"]), "color": p.get("color", "#00D4FF"), "lineWidth": 1,
             "lineStyle": 2, "axisLabelVisible": True, "title": p.get("title", "")} for p in price_lines]


def tv_line(series: pd.Series, markers: list[dict] | None = None, price_lines: list[dict] | None = None,
            key: str = "tvl", height: int = 460, color: str = "#7C5CFC", log: bool = True) -> None:
    """渲染 TradingView 原生折线图（事件时间线等），支持事件标记 + 横向价位线。"""
    s = series.dropna()
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
    candle = {
        "type": "Candlestick",
        "data": _candles(ohlcv),
        "options": {"upColor": _UP, "downColor": _DN, "wickUpColor": _UP, "wickDownColor": _DN,
                    "borderVisible": False},
    }
    if trades is not None and len(trades):
        candle["markers"] = _markers(trades)
    if price_lines:
        candle["priceLines"] = [
            {"price": float(p["price"]), "color": p.get("color", "#00D4FF"), "lineWidth": 1,
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
