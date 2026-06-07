"""滚动 IS/OOS（walk-forward）切分。

任何参数选择只能在 IS（训练段）做、在紧接其后的 OOS（测试段）报。
反复多段，比单次切分更难自欺。本模块只负责生成切分，不做拟合。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Split:
    """一次 walk-forward 切分。索引为相对 index 的整数位置。"""

    train_start: int
    train_end: int   # 不含（train = [train_start, train_end)）
    test_start: int
    test_end: int    # 不含（test = [test_start, test_end)）

    def slice(self, index: pd.Index) -> tuple[pd.Index, pd.Index]:
        """返回 (train_index, test_index) 两段标签。"""
        return index[self.train_start:self.train_end], index[self.test_start:self.test_end]


def walk_forward_splits(
    index: pd.Index | np.ndarray | int,
    train: int,
    test: int,
    step: int | None = None,
    anchored: bool = False,
) -> list[Split]:
    """生成滚动 IS/OOS 切分。

    参数
    ----
    index : 可传 DatetimeIndex / 数组 / 或整数长度 n
    train : 训练段长度
    test : 测试段长度
    step : 每次前移步长（默认 = test，即相邻 OOS 段不重叠、无缝衔接）
    anchored : True 则训练段起点固定在 0（扩张窗口）；False 为滚动定长窗口

    返回
    ----
    list[Split]：训练段严格早于其测试段（train_end == test_start，无泄漏）。
    """
    n = int(index) if isinstance(index, (int, np.integer)) else len(index)
    step = step or test
    if train < 1 or test < 1:
        raise ValueError("train 和 test 都必须 ≥ 1")
    if train + test > n:
        raise ValueError(f"train+test={train + test} 超过样本长度 {n}")

    splits: list[Split] = []
    test_start = train
    while test_start + test <= n:
        train_start = 0 if anchored else test_start - train
        splits.append(
            Split(
                train_start=train_start,
                train_end=test_start,     # 训练段结束即测试段开始：无重叠、无泄漏
                test_start=test_start,
                test_end=test_start + test,
            )
        )
        test_start += step
    return splits
