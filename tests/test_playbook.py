"""操作预案验收：动量陷阱→防守口径；价值倾斜→可补；格式完整。"""
from analysis import playbook as pb


def _trap_brief():
    return {
        "ticker": "NVDA", "horizon": 63, "price": 205.0, "vol_percentile": 0.97,
        "momentum_trap": True,
        "engine_state": {"median": 0.073, "excess": -0.019, "significant": True},
        "engine_value": {"median": 0.071, "excess": -0.021, "significant": True},
        "tranches": [
            {"tier": "浅", "price": 201.7, "what": "MA50", "target": 235.5, "stop": 189.7,
             "rr": 2.8, "to_target_pct": 0.17, "to_stop_pct": -0.06, "engine_win_rate": 0.65},
            {"tier": "重", "price": 188.2, "what": "MA200+POC", "target": 235.5, "stop": 173.1,
             "rr": 3.1, "to_target_pct": 0.25, "to_stop_pct": -0.08, "engine_win_rate": 0.65},
        ],
        "days_to_earnings": 82, "earnings_stats": {"pre": 10, "day_abs_move": {"median": 0.052},
                                                   "pre_drift": {"median": 0.028}},
    }


def _value_brief():
    b = _trap_brief()
    b["ticker"] = "GOOGL"; b["momentum_trap"] = False; b["vol_percentile"] = 0.6
    b["engine_state"] = {"median": 0.10, "excess": 0.05, "significant": True}
    b["engine_value"] = {"median": 0.10, "excess": 0.049, "significant": True}
    b["days_to_earnings"] = 12
    return b


def test_momentum_trap_defensive():
    p = pb.build_playbook(_trap_brief())
    assert "低" in p["conviction"]
    entry = " ".join(p["entry"])
    down = " ".join(p["if_down"])
    assert "不主动在回撤中建仓" in entry or "轻仓" in entry
    assert "不要越跌越补" in down  # 关键：动量陷阱不许摊平
    assert "止损" in down


def test_value_allows_add_on_dip():
    p = pb.build_playbook(_value_brief())
    down = " ".join(p["if_down"])
    assert "可按计划补仓" in down  # 价值倾斜允许补
    # 涨了有减仓 + 移动止损
    up = " ".join(p["if_up"])
    assert "减" in up and ("移动止损" in up or "止损" in up)
    # 财报临近(12天)给 gap 警告
    assert any("财报" in x for x in p["time_event"])


def test_gate_fail_lowers_conviction():
    b = _value_brief()
    p = pb.build_playbook(b, gate={"overall": False})
    # 未过闸门 → headline 转防守
    assert "低把握" in p["headline"] or "低" in p["conviction"] or "余地" in p["conviction_basis"]


def test_format_has_all_sections():
    txt = pb.format_playbook(pb.build_playbook(_value_brief()))
    for sec in ["建仓", "涨了", "跌了", "时间", "风控"]:
        assert sec in txt
    assert "非买卖指令" in txt
