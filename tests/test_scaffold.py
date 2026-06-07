"""Phase 0 验收：脚手架完整性检查。所有模块可导入、配置可加载。"""
import importlib

import pytest


def test_config_loads():
    import config

    cfg = config.load_config()
    assert "universe" in cfg
    assert "dates" in cfg
    assert cfg["dates"]["start"] == "2000-01-01"


@pytest.mark.parametrize(
    "module",
    [
        "data.loader",
        "factors.base",
        "factors.price_factors",
        "factors.fundamental_factors",
        "evaluation.factor_eval",
        "regime.conditional_returns",
        "stats.bootstrap",
        "stats.walkforward",
        "stats.deflated_sharpe",
        "backtest.strategies",
    ],
)
def test_module_imports(module):
    importlib.import_module(module)
