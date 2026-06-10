"""操作预案生成器（基于量化结果的 if-then 条件操作指导）。

把工具算出的 引擎桶/价位档/盈亏比/MAE/动量陷阱/财报drift/验收闸门 → 翻成可执行的
**条件预案**：在哪些价位区间建仓、涨了怎么操作、跌了怎么操作、时间/事件怎么处理、风控。

铁律（关键）：这是**机械的 if-then 规则**，不是预测、不是买卖指令——
- 价位是「**若到达就行动**」的区间(回前高目标=技术参考)，不是「会涨到 X」的预言；
- **当引擎不支持时（动量陷阱 / 未过验收闸门），预案会明确转为「别补、轻仓、按止损」**，
  绝不在没有统计优势的地方给"越跌越买"的危险指导；
- 把握度(conviction)由 验收闸门 + 引擎显著性 + 动量陷阱 共同决定。
"""
from __future__ import annotations


def _pct(x):
    return f"{x:+.0%}" if x == x else "—"


def conviction_tier(brief: dict, gate: dict | None) -> tuple[str, str]:
    """把握度分档 → (标签, 一句话依据)。"""
    trap = brief.get("momentum_trap")
    ev = brief.get("engine_value") or {}
    es = brief.get("engine_state") or {}
    gate_pass = gate.get("overall") if gate else None

    if trap:
        return "🔴 低（动量陷阱）", "当前在回撤、但回撤桶超额≤0——逢跌买历史上不优于随机进场。"
    if gate_pass is True and (ev.get("significant") and ev.get("excess", 0) > 0):
        return "🟢 中高", "验收闸门通过 + 估值低位桶显著正超额，统计与样本外都站得住。"
    if ev.get("significant") and ev.get("excess", 0) > 0:
        return "🟡 中（价值倾斜）", "估值低位桶显著正超额，但未全过验收闸门，留余地。"
    if es.get("significant") and es.get("excess", 0) > 0:
        return "🟡 中（状态倾斜）", "当前状态桶有正超额，证据中等。"
    return "⚪ 低/中性", "引擎未给出显著正倾斜，按低把握处理。"


def build_playbook(brief: dict, gate: dict | None = None) -> dict:
    """生成结构化操作预案。返回各分节 + steps 文本列表。"""
    tk = brief.get("ticker", "")
    price = brief.get("price", float("nan"))
    volp = brief.get("vol_percentile", float("nan"))
    horizon = brief.get("horizon", 63)
    trap = bool(brief.get("momentum_trap"))
    tranches = brief.get("tranches", []) or []
    es = brief.get("engine_state") or {}
    ev = brief.get("engine_value") or {}
    up_median = es.get("median") if es.get("median") is not None else ev.get("median")
    dte = brief.get("days_to_earnings")
    esr = brief.get("earnings_stats") or {}

    tier, basis = conviction_tier(brief, gate)
    high_vol = (volp == volp and volp >= 0.8)

    pb = {"ticker": tk, "conviction": tier, "conviction_basis": basis, "price": price,
          "entry": [], "if_up": [], "if_down": [], "time_event": [], "risk": [], "headline": ""}

    # ---------- 建仓 ----------
    if tranches:
        zone_txt = "；".join(
            f"{t['tier']}档 ≈{t['price']:.1f}（{t['what']}，回前高目标 {t['target']:.0f}/{_pct(t['to_target_pct'])}，"
            f"技术止损 {t['stop']:.0f}/{_pct(t['to_stop_pct'])}，盈亏比 {t['rr']:.1f}）"
            for t in tranches if t.get("rr") == t.get("rr"))
        pb["entry"].append("分批价位区间（到达才行动，非预测点位）：" + zone_txt)
    if trap:
        pb["entry"].append("⚠️ 动量陷阱：**不主动在回撤中建仓**。等趋势确认（站回 MA50/MA100）或"
                           "波动分位回落后，**只在浅档轻仓试探**，不要重仓抢跌。")
        pb["entry"].append("建议起始仓位 ≤ 计划仓位的 1/3，且每加一档都要同时上移整体止损。")
    else:
        size_hint = "高波动(分位 %.0f%%)→先建计划仓位的 1/3 试探，波动回落再补中/重档" % (volp * 100) if high_vol \
            else "可按 浅30%%/中40%%/重30%% 分批落地（指该标的计划仓位，非全部资金）"
        pb["entry"].append("分批落地：" + size_hint + "。")
        # Plan B（防踏空）：避免"只挂深档、价格直接走高踏空整波"——在场>择时
        pb["entry"].append("🪂 **防踏空 Plan B**：若价格不回深档、直接走高 → 站上 MA50 / 突破前高确认就**追小仓**，"
                           "别空仓干等（历史上想等的浅回调多数不来，踏空常比追高更亏）。深档与追势两手都留子弹。")

    # ---------- 涨了（减仓位必须按价位从低到高，先到先减）----------
    sc1 = (price * (1 + up_median)) if (up_median == up_median) else None
    tgt = tranches[0]["target"] if tranches else None
    tp_levels = []  # (价位, 文案) —— 排序后输出，杜绝"高位先减、低位后减"的乱序
    if sc1 and sc1 > price:
        tp_levels.append((sc1, f"≈ {sc1:.1f}（+{up_median:.0%}，引擎该状态中位上行）：减 1/3 锁部分利润。"))
    if tgt and tgt > price:
        tp_levels.append((tgt, f"≈ 回前高 {tgt:.0f}：再减一档。"))
    tp_levels.sort(key=lambda x: x[0])  # 价格升序 = 先到先减
    for i, (_, txt) in enumerate(tp_levels):
        pb["if_up"].append(("第一减仓区 " if i == 0 else "下一减仓区 ") + txt)
    pb["if_up"].append("其余仓位用**移动止损**(回吐约 20% 出场，回测验证的出场)让利润奔跑。")
    pb["if_up"].append("趋势仍强（站上 MA200）且未到减仓区 → **持有，让赢家跑**，别过早全平。")

    # ---------- 跌了（诚实分支）----------
    stops = [t for t in tranches if t.get("stop") == t.get("stop")]
    hard_stop = min((t["stop"] for t in stops), default=None)
    if trap:
        pb["if_down"].append("**不要越跌越补**——历史上该票逢跌买无统计优势(回撤桶超额≤0)。")
        if tranches:
            pb["if_down"].append(f"跌破 浅档止损 {tranches[0]['stop']:.0f} 即减/离场，**认错优先于摊平**。")
    else:
        if (es.get("excess", 0) > 0 or ev.get("excess", 0) > 0):
            mids = [t for t in tranches[1:]]
            if mids:
                add_txt = "、".join(f"{t['tier']}档 {t['price']:.1f}" for t in mids)
                pb["if_down"].append(f"跌到下一共振支撑（{add_txt}）**可按计划补仓**——历史上该深度远期分布仍正偏；"
                                     "但每补一档必须**同步下移整体止损**。")
        else:
            pb["if_down"].append("引擎未给正超额 → 跌了**不加仓**，按止损纪律执行。")
    if hard_stop is not None:
        pb["if_down"].append(f"**硬止损 {hard_stop:.0f}**（重档技术失守位）：无条件离场，不恋战。")

    # ---------- 时间 / 事件 ----------
    pb["time_event"].append(f"时间维度：约 {horizon} 个交易日是引擎的校准窗口；"
                            f"持有超过且无进展、趋势走平 → 释放仓位（机会成本）。")
    if dte is not None and dte <= 21 and esr.get("day_abs_move"):
        dm = esr["day_abs_move"]["median"]
        pb["time_event"].append(f"⚠️ 财报临近（{dte} 天）：财报日典型波动 ±{dm:.0%}、隔夜 gap 不对称——"
                                "**别在财报前满仓押方向**；财报后按 超预期/不及 的历史漂移再调仓。")
    elif esr.get("pre_drift"):
        pb["time_event"].append(f"财报前 {esr.get('pre',10)} 日历史 drift 中位 "
                                f"{esr['pre_drift']['median']:+.0%}（市场惯于提前抢跑），临近财报留意。")

    # ---------- 风控 ----------
    if high_vol:
        pb["risk"].append(f"波动分位 {volp:.0%}（极高）→ 仓位放小、止损放宽，别在高波动里追。")
    pb["risk"].append("同赛道高相关：别多只同时满仓；单笔仓位与把握度匹配（低把握=小仓）。")
    pb["risk"].append("结构性变化（政策/资本开支证伪/判决）是引擎吃不进的表外尾部风险。")

    # ---------- 一句话总纲 ----------
    if trap:
        pb["headline"] = (f"{tier}：别抢跌。等确认再轻仓，跌破浅档止损就走——这是动量票，摊平是陷阱。")
    elif gate and not gate.get("overall"):
        pb["headline"] = (f"{tier}：规则未过验收闸门，按**低把握**操作——轻仓、严格止损、不重仓不摊平。")
    else:
        pb["headline"] = (f"{tier}：在 浅/中/重 共振价位分批建仓，涨了分段减仓+移动止损，"
                          f"跌到下一档可补但同步下移止损，跌破硬止损离场。")
    return pb


def format_playbook(pb: dict) -> str:
    """渲染为 Markdown。"""
    L = [f"### 📋 操作预案 · {pb['ticker']}（把握度 {pb['conviction']}）",
         f"> {pb['headline']}",
         f"> _依据：{pb['conviction_basis']}_", ""]
    for title, key in [("🎯 建仓", "entry"), ("📈 涨了怎么操作", "if_up"),
                       ("📉 跌了怎么操作", "if_down"), ("⏱️ 时间 / 事件", "time_event"),
                       ("🛡️ 风控", "risk")]:
        items = pb.get(key) or []
        if not items:
            continue
        L.append(f"**{title}**")
        L += [f"- {x}" for x in items]
        L.append("")
    L.append("_以上为基于历史条件分布的**机械 if-then 预案**，价位是「若到达就行动」的区间(非预测)，"
             "**非买卖指令**；过不了验收闸门/动量陷阱时已自动转为防守口径。研究校准用途，非投资建议。_")
    return "\n".join(L)
