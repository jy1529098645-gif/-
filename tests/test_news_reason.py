"""新闻启发式推理验收（规则式情绪 + 主题 + 与引擎交叉的推理段）。

纯离线：用构造的新闻 DataFrame，不走网络。
"""
import pandas as pd

from analysis import news_reason as nr


def _news(titles):
    return pd.DataFrame([{"date": "2026-06-06", "title": t, "provider": "X", "url": "u", "summary": ""}
                         for t in titles])


def test_sentiment_scoring():
    sc_pos, _ = nr._score_item("Nvidia beats estimates, stock surges to record high")
    sc_neg, _ = nr._score_item("Stock plunges after earnings miss and downgrade")
    assert sc_pos > 0 and sc_neg < 0


def test_theme_tagging():
    _, themes = nr._score_item("DOJ antitrust lawsuit probes Google search deal")
    assert "regulation" in themes
    _, t2 = nr._score_item("Nvidia launches new Blackwell AI GPU for data center")
    assert "product_ai" in t2


def test_analyze_news_aggregate():
    df = _news(["Apple surges on strong iPhone demand and record revenue",
                "Apple beats earnings, raises guidance",
                "Apple faces antitrust probe in EU",
                "Analyst maintains rating on Apple"])
    a = nr.analyze_news(df)
    assert a["n"] == 4
    assert a["pos"] >= 2 and a["neg"] >= 1
    assert -1.0 <= a["net_tone"] <= 1.0
    assert a["tone_label"] in {"偏多", "偏空", "中性"}
    # 每条都带情绪与主题标注
    assert all("sentiment" in it and "themes" in it for it in a["items"])


def test_analyze_empty():
    assert nr.analyze_news(pd.DataFrame()) == {"n": 0}
    assert nr.analyze_news(None) == {"n": 0}


def test_reason_paragraph_momentum_trap():
    brief = {
        "news_analysis": nr.analyze_news(_news([
            "Stock tumbles amid market selloff and macro fears",
            "Tech shares slip on rate worries"])),
        "engine_state": {"excess": -0.012, "significant": True},
        "engine_value": {"excess": -0.018, "significant": True, "reward_risk": 1.0},
        "momentum_trap": True, "days_to_earnings": 80,
    }
    txt = nr.reason_paragraph(brief)
    assert "动量陷阱" in txt
    assert "非买卖信号" in txt  # 必带免责
    assert "新闻" in txt


def test_reason_paragraph_value_thesis_with_regulation():
    brief = {
        "news_analysis": nr.analyze_news(_news([
            "Company beats earnings and raises outlook",
            "DOJ files antitrust lawsuit against company",
            "Strong cloud growth reported"])),
        "engine_state": {"excess": 0.02, "significant": False},
        "engine_value": {"excess": 0.05, "significant": True, "reward_risk": 2.5},
        "momentum_trap": False, "days_to_earnings": 200,
    }
    txt = nr.reason_paragraph(brief)
    assert "估值低位桶" in txt
    assert "监管/诉讼" in txt  # 含监管风险提示


def test_reason_empty_news():
    assert "无足够新闻" in nr.reason_paragraph({"news_analysis": {"n": 0}})


def test_article_highlights_extracts_salient():
    from data import news as nws
    text = ("The company reported quarterly revenue of $81.6 billion, up 85% year over year. "
            "The weather was nice in the morning. "
            "Management raised full-year guidance and announced an $80 billion buyback. "
            "He likes coffee. "
            "Analysts at a major bank cut their price target citing margin pressure.")
    ex = nws.article_highlights(text, max_sentences=4)
    assert 1 <= len(ex) <= 4
    joined = " ".join(ex)
    # 含数字/事件的句子被选中，闲句被剔除
    assert "revenue" in joined or "buyback" in joined or "target" in joined
    assert "coffee" not in joined and "weather" not in joined


def test_article_highlights_empty():
    from data import news as nws
    assert nws.article_highlights("") == []
    assert nws.article_highlights("short. tiny. no info here at all friend.") == []


def test_brief_cli_build_offline():
    from analysis import brief_cli
    out = brief_cli.build("AAPL", horizon=63, broad=False, with_news=False)
    assert out["ticker"] == "AAPL"
    assert "engine_state_bucket" in out and "entry_tranches" in out
    assert isinstance(out["entry_tranches"], list)
    assert out["price"] > 0
