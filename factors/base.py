"""因子协议：输入价格面板 → 输出同形状的因子值面板（截面可比）。

每个因子是一个纯函数。Phase 2 实现具体因子。
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd


class Factor(Protocol):
    """因子协议。实现者为纯函数：prices(DataFrame) -> factor_values(DataFrame)。"""

    def __call__(self, prices: pd.DataFrame) -> pd.DataFrame:
        ...
