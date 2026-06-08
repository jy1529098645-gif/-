"""统一决策卡：把"入场点 / 没入怎么办 / 入场后处理 / 离场"合成一张专业、简单、可执行的卡。

设计原则（量化专业视角）：
- 只输出**经回测校准**的动作，按**资产类别**区别对待（ETF标定结论嵌入）：
    指数ETF/宽基：逢跌普遍有效，跌20-30%是高置信建仓区；
    科技ETF：跌5-15%即有效（最佳逢跌标的）；
    半导体(ETF/个股)：只有深跌(>20~30%)才有edge，浅/中跌不接(常负超额、价值/whipsaw)；
    单只科技/七姐妹：深跌靠"会不会回来"，需基本面把握。
- 市场脆弱性(宽度恶化)为总开关：触发→一切偏防守/降仓。
- 离场=趋势破位(跌破200线) 或 市场脆弱性触发；非预测，是机械纪律。
- 一切是"若到达就行动"的区间+概率，非保证、非买卖指令。

离场规则经**端到端回测验证**（v1→v3 迭代）：
  "趋势破位(跌破200线)或市场宽度恶化 → 减到半仓(非清仓)"在 **ETF 上夏普 7/7 改善、
  回撤 -39% vs 持有 -55%、2015+ 样本外仍 6/7 占优**——这是工具唯一 OOS 验证过的可交易规则。
  个股(七姐妹/半导体)上该规则只降回撤、不升夏普(高动量个股长持更优)，故按资产类别区别给法。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_SEMI = {"NVDA", "AMD", "AVGO", "TSM", "ASML", "MU", "INTC", "QCOM", "TXN", "ARM", "SMCI", "MRVL",
         "SMH", "SOXX", "SOXL"}
_INDEX_ETF = {"SPY", "QQQ", "DIA", "IWM", "XLK", "XLF", "XLE", "XLY", "VTI", "VOO"}
_LEV_ETF = {"TQQQ", "SOXL", "UPRO", "TECL", "FNGU", "TNA"}


def classify(asset: str) -> str:
    a = asset.upper()
    if a in _LEV_ETF:
        return "leveraged_etf"
    if a in {"SMH", "SOXX"}:
        return "semi_etf"
    if a in _INDEX_ETF:
        return "index_etf"
    if a in _SEMI:
        return "semi_single"
    return "single"


def _trend_break(px: pd.Series) -> bool:
    ma200 = px.rolling(200, min_periods=100).mean()
    return bool(px.iloc[-1] < ma200.iloc[-1]) if ma200.notna().iloc[-1] else False


def decision_card(asset: str, px: pd.Series, best_entry: dict, fragile_now: bool,
                  market_light: str = "") -> dict:
    """合成单标的决策卡。best_entry=entry_cockpit.best_entry_across_horizons 的返回。"""
    px = px.dropna()
    cls = classify(asset)
    cur = float(px.iloc[-1])
    high = px.rolling(252, min_periods=120).max().iloc[-1]
    dd = float(cur / high - 1.0) if high == high else 0.0
    trend_broken = _trend_break(px)
    is_etf_index = cls in ("index_etf",)
    is_semi = cls in ("semi_etf", "semi_single")

    # ---- 状态机：现在该做什么（按资产类别校准）----
    if fragile_now and dd > -0.10:
        state, color = "🔴 防守", "#FF5C7A"
        action = "降仓 / 不追高——市场宽度恶化，历史上未来数月大跌概率翻倍"
    elif dd <= -0.20:
        if is_etf_index:
            state, color = "🟢 建仓区(高置信)", "#2BE6A8"
            action = "分批加重——指数/宽基ETF深跌是回测里胜率最高、最干净的建仓带"
        elif is_semi:
            state, color = "🟢 建仓区(半导体)", "#2BE6A8"
            action = "分批加重——半导体只有深跌(-20%+)才有正edge，正是此区；但波动大、严设止损"
        else:
            state, color = "🟡 深跌·需把握", "#FFD166"
            action = "仅在你确信基本面会回来时分批——单票深跌可能是价值陷阱(输家此档历史负超额)"
    elif dd <= -0.10:
        if is_semi:
            state, color = "🟠 半导体中度回撤·别急", "#FF9F45"
            action = "半导体浅/中跌(10-20%)历史无edge甚至负——别接，等深跌(-20%+)或趋势重新站上200线"
        else:
            state, color = "🟡 分批", "#FFD166"
            action = "中度回撤，分档建仓、留子弹给更深的档；指数/科技ETF此区尚可，单票看把握"
    elif dd <= -0.05:
        state, color = "🟡 浅回撤", "#FFD166"
        action = ("科技/宽基ETF浅跌(5-10%)历史已有正超额，可分批" if cls in ("index_etf",)
                  else "浅回撤无显著edge，按计划小批进、别空等")
    else:
        state, color = "🟢 追 / 持有", "#2BE6A8"
        action = "高位附近：直接分批进/持有——别等浅回调(回测70%等不到且更亏，在场>择时)"

    # ---- 入场点（来自 best_entry 跨周期择优）----
    entry = None
    if best_entry and best_entry.get("has_zone"):
        band = best_entry.get("price_band", [None, None])
        entry = {
            "anchor": best_entry.get("anchor_price"),
            "band_low": band[0], "band_high": band[1],
            "zone": best_entry.get("zone_label"),
            "horizon_months": round(best_entry.get("horizon", 63) / 21),
            "tier": best_entry.get("tier"), "confident": best_entry.get("confident"),
            "excess": best_entry.get("excess_median"), "dsr": best_entry.get("dsr"),
        }

    # ---- 入场后处理 ----
    post_entry = {
        "add": ("跌到下一更深档再补一批（按价位带分档）" if dd > -0.20 else "已在深档，剩余子弹小步补、别梭哈"),
        "trim": "反弹到前高附近/引擎中位上行后，分批减 1/3，移动止损让利润奔跑",
        "stop": ("严设止损（杠杆/半导体波动大）" if cls in ("leveraged_etf", "semi_etf", "semi_single")
                 else "跌破建仓档下沿或200日线则减仓"),
    }

    # ---- 离场 / 降仓（规则经端到端回测验证；ETF 上夏普7/7改善、OOS稳健）----
    is_etf = cls in ("index_etf", "semi_etf")
    derisk = trend_broken or fragile_now
    exit_rules = []
    if derisk:
        why = []
        if trend_broken:
            why.append("跌破200日线(2.4x未来回撤概率)")
        if fragile_now:
            why.append("市场宽度恶化(3x概率·略领先指数)")
        trig = " + ".join(why)
        if is_etf:
            exit_rules.append(f"🔴 {trig} → **减到半仓**（ETF验证：减半仓比清仓更优，夏普7/7改善、"
                              "回撤-39%vs-55%、OOS稳健；站回200线上方且宽度healthy再加回）")
        else:
            exit_rules.append(f"🟠 {trig} → 个股可减仓控回撤，但实测'减半仓'在个股只降回撤不升夏普——"
                              "长持高动量龙头可不减；要控回撤则减半仓")
    else:
        if is_etf:
            exit_rules.append("🟢 无离场信号(未破200线·宽度healthy)→ 持满仓（ETF半仓叠加规则：此状态满仓）")
        else:
            exit_rules.append("🟢 无离场信号(未破200线·宽度healthy)→ 持有")
    if cls == "leveraged_etf":
        exit_rules.append("⚠️ 杠杆ETF每日复利衰减、不可长持，必须设止损/止盈（叠加规则不适用）")

    return {
        "asset": asset, "asset_class": cls, "current_price": cur, "drawdown": dd,
        "trend_broken": trend_broken, "fragile": fragile_now, "market_light": market_light,
        "state": state, "color": color, "action": action,
        "entry": entry, "post_entry": post_entry, "exit_rules": exit_rules,
    }


def format_card(card: dict) -> str:
    """一句话核心裁决（用于顶部横幅）。"""
    e = card.get("entry")
    if e and e.get("anchor") == e.get("anchor"):
        lvl = (f"｜入场参考 ≈ {e['anchor']:.1f}（{e['zone']}·持有~{e['horizon_months']}月·"
               f"{'稳健' if e.get('confident') else '低置信'}）")
    else:
        lvl = ""
    return f"{card['state']}：{card['action']}{lvl}"
