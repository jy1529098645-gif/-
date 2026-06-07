"""出场规则（补充规格 A，重点）。与入场**成对**传入回测。

支持：take_profit / stop_loss / trailing_stop / time_stop / signal_exit，
用 vectorbt.Portfolio.from_signals（sl_stop / sl_trail / tp_stop / 显式 exits）实现，含 fees+slippage。

铁律：出场对收益分布影响通常大于入场，故只评估"入场×出场"组合，不单独优化入场。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

_CFG = config.load_config()
_FEES = float(_CFG["costs"]["fees"])
_SLIP = float(_CFG["costs"]["slippage"])


def _time_stop_exits(entries: pd.Series, n: int) -> pd.Series:
    """时间止损：入场信号 n 个交易日**之后**标记一次出场。

    方向核对（重要，曾被误读）：`shift(+n)` 把布尔信号沿时间轴**向后**搬 n 个 bar，
    即 t 日的入场信号会在 t+n 日出现 True → 这正是"入场 n 天后出场"，**不是** shift(-n)
    （shift(-n) 会把出场放到入场之前，制造未来函数）。见 test_time_stop_direction。

    近似说明：这是对入场**信号**序列平移，而非对实际成交日平移。在 accumulate=False 下，
    同一持仓期内的重复入场信号被忽略，故平移出的出场点对绝大多数单仓场景与"成交后 n 天"一致；
    极端密集信号下可能有 ≤1 个 bar 的对齐误差（vectorbt 1.0 无原生 td_stop，已知局限）。
    n<1 视为不启用时间止损。
    """
    if n < 1:
        return pd.Series(False, index=entries.index)
    return entries.shift(n, fill_value=False).astype(bool)


def vol_scaled_trail(price: pd.Series, k: float = 3.0, lookback: int = 21) -> float:
    """波动自适应移动止损宽度：k × 该票典型日波动(年化前的日 std 中位)。

    思路同 ATR/Chandelier——安静的票给更紧的止损、波动大的票给更宽的止损，避免被正常噪音扫出。
    用收盘价的日收益滚动 std 的中位作 ATR 代理(close-only 管线兼容)；夹在 [5%,50%] 防极端。
    返回一个**移动止损比例**(供 vectorbt sl_stop + sl_trail=True 用)。
    """
    vol = price.pct_change().rolling(lookback, min_periods=max(2, lookback // 2)).std()
    typical = float(vol.median())
    if not (typical == typical) or typical <= 0:
        return 0.20
    return float(min(0.50, max(0.05, k * typical)))


def build_exit_kwargs(price: pd.Series, entries: pd.Series, exit_spec: dict) -> dict:
    """把出场规格 dict 翻成 from_signals 的 kwargs。

    exit_spec 可含：take_profit, stop_loss, trailing_stop（与 stop_loss 互斥，取前者），
                    time_stop（int 天），signal_exit（布尔 Series）。
    """
    kw: dict = {}
    # 波动自适应移动止损：vol_trail=k → 止损宽度按该票典型波动缩放（优先于固定 trailing/stop）
    if exit_spec.get("vol_trail"):
        kw["sl_stop"] = vol_scaled_trail(price, k=float(exit_spec["vol_trail"]))
        kw["sl_trail"] = True
    elif "trailing_stop" in exit_spec and exit_spec["trailing_stop"] is not None:
        kw["sl_stop"] = float(exit_spec["trailing_stop"])
        kw["sl_trail"] = True
    elif "stop_loss" in exit_spec and exit_spec["stop_loss"] is not None:
        kw["sl_stop"] = float(exit_spec["stop_loss"])
    if exit_spec.get("take_profit") is not None:
        kw["tp_stop"] = float(exit_spec["take_profit"])

    exits = pd.Series(False, index=price.index)
    if exit_spec.get("time_stop"):
        exits = exits | _time_stop_exits(entries, int(exit_spec["time_stop"]))
    if exit_spec.get("ma_exit"):  # 趋势跌破出场：收盘跌破 N 日均线（让利润奔跑、破势才走）
        w = int(exit_spec["ma_exit"])
        ma = price.rolling(w, min_periods=max(2, w // 2)).mean()
        exits = exits | (price < ma).reindex(price.index).fillna(False).astype(bool)
    if exit_spec.get("signal_exit") is not None:
        exits = exits | exit_spec["signal_exit"].reindex(price.index).fillna(False).astype(bool)
    if exits.any():
        kw["exits"] = exits
    return kw


def run_trades(
    price: pd.Series,
    entries: pd.Series,
    exit_spec: dict,
    fees: float | None = None,
    slippage: float | None = None,
    init_cash: float = 10000.0,
):
    """跑单票的"入场→出场"逐笔交易，返回 vectorbt Portfolio。accumulate=False（一次一仓）。"""
    import vectorbt as vbt

    price = price.dropna()
    entries = entries.reindex(price.index).fillna(False).astype(bool)
    kw = build_exit_kwargs(price, entries, exit_spec)
    explicit_exits = kw.pop("exits", pd.Series(False, index=price.index))

    return vbt.Portfolio.from_signals(
        price,
        entries=entries,
        exits=explicit_exits,
        direction="longonly",
        accumulate=False,
        fees=fees if fees is not None else _FEES,
        slippage=slippage if slippage is not None else _SLIP,
        init_cash=init_cash,
        freq="1D",
        **kw,
    )


def extract_trades(pf, price: pd.Series) -> pd.DataFrame:
    """从 Portfolio 提取逐笔交易，并自算 MAE（途中最大浮亏）。

    返回每笔：entry_date, exit_date, entry_price, exit_price, return, duration_days, mae。
    """
    rec = pf.trades.records_readable
    if rec.empty:
        return pd.DataFrame(
            columns=["entry_date", "exit_date", "entry_price", "exit_price", "return",
                     "duration_days", "duration_bars", "mae"]
        )

    price = price.dropna()
    p = price.to_numpy(dtype=float)

    def _col(*names):
        for n in names:
            if n in rec.columns:
                return rec[n]
        raise KeyError(f"找不到列 {names}，实际列：{list(rec.columns)}")

    entry_ts = pd.to_datetime(_col("Entry Index", "Entry Timestamp"))
    exit_ts = pd.to_datetime(_col("Exit Index", "Exit Timestamp"))
    ret = _col("Return", "Return [%]").astype(float)
    if "Return [%]" in rec.columns:
        ret = ret / 100.0
    entry_px = _col("Avg Entry Price").astype(float)
    exit_px = _col("Avg Exit Price").astype(float)

    pos = {d: i for i, d in enumerate(price.index)}
    rows = []
    for ets, xts, r, ep, xp in zip(entry_ts, exit_ts, ret, entry_px, exit_px):
        i0, i1 = pos.get(ets), pos.get(xts)
        if i0 is None or i1 is None or i1 < i0:
            mae, bars = np.nan, np.nan
        else:
            path = p[i0 : i1 + 1]
            mae = float(np.nanmin(path) / p[i0] - 1.0) if path.size else np.nan
            bars = int(i1 - i0)  # 持有交易日数
        rows.append(
            {
                "entry_date": ets,
                "exit_date": xts,
                "entry_price": float(ep),
                "exit_price": float(xp),
                "return": float(r),
                "duration_days": int((xts - ets).days),
                "duration_bars": bars,
                "mae": mae,
            }
        )
    return pd.DataFrame(rows)


# 默认出场规格（来自 config）
def default_exit_spec() -> dict:
    e = _CFG["single_name"]["exit"]
    return {
        "trailing_stop": e.get("trailing_stop"),
        "take_profit": e.get("take_profit"),
        "time_stop": e.get("time_stop"),
    }
