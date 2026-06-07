"""多票作战简报综合层验收（技术位/共振/建仓档/引擎桶/权重/markdown）。

新闻/基本面走网络，测试里用 with_news=False 保证离线可跑、快。
"""
import numpy as np
import pandas as pd
import pytest

from analysis import briefing as bf


@pytest.fixture(scope="module")
def brief_googl():
    return bf.stock_brief("GOOGL", "2010-01-01", "2024-12-31", horizon=63, with_news=False)


def test_cluster_levels_merges_close():
    lv = [{"name": "A", "price": 100.0}, {"name": "B", "price": 101.0},
          {"name": "C", "price": 130.0}]
    cl = bf.cluster_levels(lv, tol=0.025)
    # 100/101 应合并成一簇(双重)，130 单独
    assert len(cl) == 2
    big = max(cl, key=lambda c: c["n"])
    assert big["n"] == 2 and set(big["members"]) == {"A", "B"}
    # 降序返回
    assert cl[0]["price"] >= cl[1]["price"]


def test_cn_bucket_translation():
    assert bf._cn_bucket("valuation_tercile=low") == "估值低位"
    assert "回撤桶" in bf._cn_bucket("in_drawdown=True & valuation_tercile=low")


def test_stock_brief_shape(brief_googl):
    b = brief_googl
    assert b["ticker"] == "GOOGL"
    assert b["price"] > 0
    assert b["trend"] in {"↑强", "↑", "↓", "↓深"}
    # 引擎最优桶有显著性标记与盈亏比
    assert b["engine_best"] is not None
    assert set(["bucket", "median", "excess", "win_rate", "significant", "reward_risk"]) <= set(b["engine_best"])
    # 建仓档：价位单调递减、目标>价位>止损、RR 有限
    tr = b["tranches"]
    assert len(tr) >= 1
    prices = [t["price"] for t in tr]
    assert prices == sorted(prices, reverse=True)
    for t in tr:
        assert t["target"] >= t["price"] > t["stop"]
        assert np.isfinite(t["rr"])


def test_build_tranches_rr_math():
    clusters = [{"price": 90.0, "members": ["MA50", "POC", "档"], "n": 3},
                {"price": 80.0, "members": ["MA200"], "n": 1},
                {"price": 70.0, "members": ["低点"], "n": 1}]
    best = {"win_rate": 0.7, "reward_risk": 2.0, "excess": 0.05, "significant": True}
    tr = bf.build_tranches(current_price=100.0, trailing_high=110.0, clusters=clusters, best_bucket=best)
    assert tr[0]["tier"] == "浅" and abs(tr[0]["price"] - 90.0) < 1e-9
    assert "三重共振" in tr[0]["what"]  # n>=3
    # RR = (target-entry)/(entry-stop)；浅档 stop=80*0.98=78.4 → (110-90)/(90-78.4)
    assert abs(tr[0]["rr"] - (110 - 90) / (90 - 80 * 0.98)) < 1e-6


def test_auto_weights_sum_100(brief_googl):
    b2 = bf.stock_brief("MSFT", "2010-01-01", "2024-12-31", horizon=63, with_news=False)
    w = bf.auto_weights([brief_googl, b2])
    assert abs(sum(v["weight"] for v in w.values()) - 100.0) < 0.5
    assert all(v["weight"] >= 0 for v in w.values())
    # 角色按证据等级命名（弱证据不再叫"核心超配"）；强语只给 A/B
    _ROLES = {"候选池第1（证据较强）", "候选池靠前", "小仓试探", "观察/降权"}
    assert all(v["role"] in _ROLES for v in w.values())
    top = max(w.items(), key=lambda kv: kv[1]["weight"])
    if top[1].get("grade") in ("A", "B"):
        assert "候选池" in top[1]["role"]
    else:
        assert top[1]["role"] in {"小仓试探", "观察/降权"}


def test_render_markdown(brief_googl):
    w = bf.auto_weights([brief_googl])
    md = bf.render_markdown([brief_googl], w, horizon=63)
    assert "多票作战简报" in md and "一屏总览" in md
    assert "建仓档" in md and "风控铁律" in md
    assert "GOOGL" in md
    # 不得出现"目标价预言/最佳买点"等措辞
    assert "预言" not in md
