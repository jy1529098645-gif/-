"""数据预热：把 SPY + 七姐妹的价格/OHLCV/财报 + 宏观全部拉好缓存，首次使用前端即秒开。

运行：.venv/Scripts/python scripts/warmup.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import loader  # noqa: E402

MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
ALL = ["SPY"] + MAG7


def main():
    print("预热宏观（FRED）…")
    try:
        m = loader.load_macro("1990-01-01")
        print(f"  macro {m.shape} 来源 {m.attrs.get('sources')}")
    except Exception as e:  # noqa: BLE001
        print(f"  宏观失败：{e}")

    for t in ALL:
        try:
            px = loader.load_prices([t], "1995-01-01")
            print(f"  {t} 价格 {px.shape[0]} 行", end="")
        except Exception as e:  # noqa: BLE001
            print(f"  {t} 价格失败：{e}"); continue
        try:
            loader.load_ohlcv(t, "2010-01-01")
            print(" · OHLCV ✓", end="")
        except Exception as e:  # noqa: BLE001
            print(f" · OHLCV 失败：{e}", end="")
        if t != "SPY":
            try:
                ed = loader.load_earnings_dates(t)
                print(f" · 财报 {ed.shape[0]} 条")
            except Exception as e:  # noqa: BLE001
                print(f" · 财报失败：{e}")
        else:
            print()
    print("预热完成。")


if __name__ == "__main__":
    main()
