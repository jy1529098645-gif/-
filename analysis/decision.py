"""统一决策卡：把"入场点 / 没入怎么办 / 入场后处理 / 离场"合成一张专业、简单、可执行的卡。

设计原则（量化专业视角）：
- 只输出**经回测校准**的动作，按**资产类别**区别对待（ETF标定结论嵌入）：
    指数ETF/宽基：逢跌普遍有效，跌20-30%是高置信建仓区；
    科技ETF：跌5-15%即有效（最佳逢跌标的）；
    半导体(ETF/个股)：只有深跌(>20~30%)才有edge，浅/中跌不接(常负超额、价值/whipsaw)；
    单只科技/七姐妹：深跌靠"会不会回来"，需基本面把握。
  ⚠️ **幸存者偏差警示**：上述"逢跌有效/深跌edge"是用 2010–2026 的**已知赢家池**(NVDA/七姐妹/科技半导体
     ETF)回测得到的 in-sample 结论——这些标的"跌了还能回来"本身是事后才知道的。换个年代(2000/2008)的赢家
     池是另一批(思科/花旗/GE)，逢跌买未必回来。故这些资产类别结论是**先验偏向、非保证**，遇结构性证伪
     (资本开支崩/判决/政策)模型吃不进。单票深跌务必结合基本面判断"会不会回来"，别只看历史回撤分布。
- 市场脆弱性(宽度恶化)为总开关：触发→一切偏防守/降仓。
- 离场=趋势破位(跌破200线) 或 市场脆弱性触发；非预测，是机械纪律。
- 一切是"若到达就行动"的区间+概率，非保证、非买卖指令。

离场规则经**端到端回测验证**（v1→v3 迭代；**业绩数字以 analysis/overlay.py 头注为唯一口径**，勿在此另写一套）：
  "趋势破位(跌破200线)或市场宽度恶化 → 减到半仓(非清仓)"——ETF 全程夏普 0.77 vs 持有 0.70、
  2015+ OOS 1.00 vs 0.91、回撤约砍 40%(-33% vs -58%)；个股全程 0.84 vs 0.83。这是工具的可部署风控规则。
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
                  market_light: str = "", momentum_trap: bool = False,
                  grade: dict | None = None) -> dict:
    """合成单标的决策卡。best_entry=entry_cockpit.best_entry_across_horizons 的返回。

    momentum_trap / grade / best_entry 一起作为**引擎纪律覆盖**：当引擎判定逢跌无优势
    (动量陷阱)、证据等级 F、或各回撤档都跑不赢基准时，**绝不在回撤里硬给"建仓"裁决**，
    与下方"操作预案 / 最佳入场区"口径对齐，杜绝顶层结论与引擎层前后矛盾。"""
    px = px.dropna()
    cls = classify(asset)
    cur = float(px.iloc[-1])
    high = px.rolling(252, min_periods=120).max().iloc[-1]
    dd = float(cur / high - 1.0) if high == high else 0.0
    trend_broken = _trend_break(px)
    is_etf_index = cls in ("index_etf",)
    is_semi = cls in ("semi_etf", "semi_single")
    is_lev = cls == "leveraged_etf"

    # ---- 状态机：现在该做什么（按资产类别校准）----
    # posture: defend(防守降仓) / build(逢跌建仓) / chase(高位追·持有) / wait(无edge·分批留子弹)
    # 脆弱性阈值 -0.15 与 fragility.wait_or_chase 对齐：脆弱+尚未进入深跌edge区(>-15%)才转防守，
    # 避免"决策卡说分批、市场环境横幅说降仓减半"这类前后矛盾。
    if fragile_now and dd > -0.15:
        state, color, posture = "🔴 防守", "#FF5C7A", "defend"
        action = "降仓 / 不追高——市场宽度恶化(且尚未进入深跌edge区)，历史上未来数月大跌概率翻倍"
    elif dd <= -0.20:
        if is_lev:
            state, color, posture = "🟠 杠杆ETF深跌·别摊平", "#FF9F45", "caution"
            action = ("3x杠杆深跌：每日复利衰减+路径依赖，反弹弹性大但**不可久持/重仓摊平**——"
                      "只在做短线波段、严设止损时小仓参与，长持者反而该减")
        elif is_etf_index:
            state, color, posture = "🟢 建仓区(高置信)", "#2BE6A8", "build"
            action = ("分批加重——指数/宽基ETF深跌在 2010–26 回测里是胜率最高的一档建仓带"
                      "(宽基不易整体归零、比单票稳；仍是 in-sample 统计、非保证)")
        elif is_semi:
            state, color, posture = "🟡 半导体深跌·小批认尾部", "#FFD166", "build"
            action = ("半导体深跌长周期反弹大、绝对收益高，但回测显示**此档尾部最肥(p10可达-25%)、胜率最低**——"
                      "分批小步进、严设止损、认尾部，别'越深越重仓'(浅跌0-10%正期望档才是风险调整最优)")
        else:
            state, color, posture = "🟡 深跌·需把握", "#FFD166", "build"
            action = "仅在你确信基本面会回来时分批——单票深跌可能是价值陷阱(输家此档历史负超额)"
    elif dd <= -0.10:
        if is_lev:
            state, color, posture = "🟠 杠杆ETF回撤·小仓波段", "#FF9F45", "wait"
            action = "3x杠杆每日复利衰减、不宜久持——只做短线波段、严设止损，别越跌越补"
        elif is_semi:
            state, color, posture = "🟠 半导体中度回撤·别急", "#FF9F45", "wait"
            action = ("半导体中跌(10-20%)是回测里**最弱的一档**(edge最薄甚至为负)——别因'跌得多'就接；"
                      "等引擎此档转正期望再小批，或趋势站回200线。注意**深跌(-20%+)尾部更肥、胜率更低，不是越深越安全**")
        else:
            state, color, posture = "🟡 分批", "#FFD166", "wait"
            action = ("中度回撤，分档建仓；但别一味'留子弹等更深'——回测里更深档尾部更肥、胜率更低，"
                      "浅档(0-10%)正期望反而风险调整最优。指数/科技ETF此区尚可，单票看引擎与把握")
    elif dd <= -0.05:
        state, color, posture = "🟡 浅回撤", "#FFD166", "wait"
        action = ("科技/宽基ETF浅跌(5-10%)历史已有正超额，可分批" if cls in ("index_etf",)
                  else "浅跌(0-10%)正期望档是回测里**风险调整最优**(胜率最高、尾部最浅)——引擎此档转正即可分批，别空等更深(更深≠更好)")
    else:
        state, color, posture = "🟢 追 / 持有", "#2BE6A8", "chase"
        action = "高位附近：直接分批进/持有——别等浅回调(回测70%等不到且更亏，在场>择时)"

    # ---- 引擎纪律覆盖：动量陷阱 / 弱证据(F) / 引擎未找到稳健入场区 →
    #      绝不在没有统计优势的回撤里硬建仓（与操作预案/最佳入场区一致，杜绝前后矛盾）----
    grade_letter = (grade or {}).get("grade") if isinstance(grade, dict) else None
    best_defensive = bool(best_entry and not best_entry.get("has_zone"))
    engine_override = None
    if posture in ("build", "wait") and dd <= -0.05:
        if momentum_trap:
            state, color, posture = "🟠 回撤但无统计优势", "#FF9F45", "caution"
            action = ("别越跌越补——该票此回撤档历史超额≤0(动量陷阱)，逢跌买不优于随机进场。"
                      "等站回 MA50/MA100 或波动回落后再轻仓试探。")
            engine_override = "momentum_trap"
        elif grade_letter == "F":
            state, color, posture = "🔴 弱证据·不建仓", "#FF5C7A", "defend"
            action = "证据等级 F：该状态历史显著吃亏(负超额+左尾差)，不主动建仓，等趋势确认再说。"
            engine_override = "grade_F"
        elif best_defensive:
            state, color, posture = "🟡 深跌但引擎未找到稳健入场点", "#FFD166", "caution"
            action = ("各回撤档历史上都没跑赢无条件基准——不硬给买点，观望或仅极轻仓试探、严设止损；"
                      "把'会不会回来'交给你的基本面判断。")
            engine_override = "no_robust_zone"

    # ---- 趋势破位注记：建仓视角仍建议分批、但已跌破200线时，明确提示与下方'已建仓'减仓口径的关系，
    #      避免"顶部说分批建仓、已建仓卡说减仓防守"被误读为自相矛盾 ----
    if trend_broken and posture in ("build", "wait"):
        action += "　⚠️已跌破200线(趋势破位)：新仓分批更慢、严设止损；**已持仓者**见下方『已建仓怎么办』的减仓口径。"

    # ---- 是否建议"现在建新仓"：供前端门控顶部三联卡（仓位/入场价/持有），
    #      杜绝"决策卡说别接、却同屏显示建仓仓位+入场参考价"的矛盾 ----
    no_enter = (
        posture in ("defend", "caution")                 # 防守/弱证据/动量陷阱/无稳健入场区
        or (is_semi and -0.20 < dd <= -0.10)              # 半导体浅/中跌：历史无edge甚至负
        or (is_lev and dd <= -0.05)                       # 杠杆ETF回撤里不建标准仓（只短线波段）
    )
    enter_ok = not no_enter

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
    # add 必须与裁决一致：凡"现在别建新仓"(enter_ok=False，含半导体中浅跌/动量陷阱/弱证据等)
    # 一律不说"补一批"，杜绝"别接"却又"越跌越补"的卡内矛盾
    if not enter_ok:
        add_txt = "⚠️不越跌越补——引擎未给统计优势，等趋势确认(站回MA50/MA100)再轻仓，每加一档同步上移止损"
    elif is_lev:
        add_txt = "杠杆ETF不摊平：只做短线波段、严设止损，别越跌越补"
    elif dd > -0.20:
        add_txt = "可在更深档小批分档补（按价位带）——但越深尾部越肥、胜率越低，每深一档减码、别越深越重仓"
    else:
        add_txt = "已在深档（历史尾部最肥、胜率最低）——只用剩余子弹小步试探、认尾部，别梭哈摊平"
    post_entry = {
        "add": add_txt,
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
            exit_rules.append(f"🔴 {trig} → **减半仓 + 按波动降仓**（验证：ETF夏普7/7改善、回撤-39%vs-55%、"
                              "2015+OOS 1.00vs0.91；站回200线上方且宽度healthy再加回。见脆弱性页净值曲线）")
        else:
            exit_rules.append(f"🟠 {trig} → **减半仓 + 按波动降仓**（验证：个股也升夏普0.84vs0.83、"
                              "回撤-35%vs-56%、7/9占优；高动量龙头若只追收益可不减）")
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
        "state": state, "color": color, "action": action, "posture": posture,
        "engine_override": engine_override, "momentum_trap": bool(momentum_trap),
        "grade": grade_letter, "enter_ok": enter_ok,
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


def holding_advice(card: dict, brief: dict | None = None,
                   best_entry: dict | None = None) -> dict:
    """已建仓者怎么办：守 / 加 / 减 / 离 + 触发式止盈止损。

    与 decision_card 共用同一套引擎纪律(回撤/趋势/脆弱/动量陷阱/证据等级)，保证
    "建仓视角"与"已持仓视角"口径一致、不自相矛盾。返回
    {stance, color, headline, actions[], triggers[]}。
    """
    brief = brief or {}
    cls = card.get("asset_class", "single")
    dd = float(card.get("drawdown", 0.0) or 0.0)
    cur = float(card.get("current_price", float("nan")))
    trend_broken = bool(card.get("trend_broken"))
    fragile = bool(card.get("fragile"))
    trap = bool(brief.get("momentum_trap"))
    grade_letter = (brief.get("grade") or {}).get("grade")
    is_lev = cls == "leveraged_etf"
    is_etf = cls in ("index_etf", "semi_etf")
    es = brief.get("engine_state") or {}
    ev = brief.get("engine_value") or {}
    up_median = es.get("median") if es.get("median") is not None else ev.get("median")
    has_edge = (grade_letter in ("A", "B")) or bool(best_entry and best_entry.get("confident"))
    tranches = brief.get("tranches") or []
    dte = brief.get("days_to_earnings")

    actions: list[str] = []
    triggers: list[str] = []

    # ---- 立场（优先级：风控开关 > 动量陷阱 > 弱证据 > 高位 > 有edge深跌 > 默认）----
    if trend_broken or fragile:
        why = " + ".join(w for w in ("跌破200日线" if trend_broken else "",
                                     "市场宽度恶化" if fragile else "") if w)
        stance, color = "🔴 减仓 / 防守", "#FF5C7A"
        headline = f"{why} → 按已验证规则**减到半仓 + 按波动降仓**(非清仓)，等修复再加回。"
        if is_etf:
            actions.append("减半仓+按波动降仓——ETF上该规则夏普7/7改善、回撤-39%vs-55%、2015+OOS稳健。")
        else:
            actions.append("减半仓+按波动降仓——个股也升夏普(0.84vs0.83)、回撤-35%vs-56%、7/9占优；"
                           "若是高动量龙头、你只追长期收益，可少减、用移动止损替代清仓。")
        actions.append("**站回200日线上方且市场宽度转 healthy** → 再把仓位加回去。")
    elif trap and dd <= -0.05:
        stance, color = "🟠 别补·守止损", "#FF9F45"
        headline = "你已套在动量陷阱里(该回撤档历史超额≤0)——**认错优先于摊平**，绝不越跌越补。"
        actions.append("不要越跌越补：该票逢跌买历史上无统计优势。")
        if tranches:
            actions.append(f"跌破浅档止损 {tranches[0]['stop']:.0f} → 减仓/离场。")
        actions.append("等**站回 MA50/MA100** 或波动回落确认趋势，再决定是否重新轻仓参与。")
    elif grade_letter == "F":
        stance, color = "🔴 降低暴露", "#FF5C7A"
        headline = "证据等级 F：当前状态历史显著吃亏——主动**降低这只票的暴露**，不加仓。"
        actions.append("分批减仓到你睡得着的水平，严设止损，别死等'回本'。")
    elif dd > -0.05:
        stance, color = "🟢 持有·让利润跑", "#2BE6A8"
        headline = "接近高位且趋势在：**持有**，用移动止损保护，到减仓区再分段止盈——别因怕回调过早全平。"
        actions.append("用移动止损(回吐约20%出场)让赢家继续跑——锁浮盈的纪律、非择时alpha(固定止损在高波动票上会被甩下车)；真正护回撤靠破200线/宽度→减半仓。")
        actions.append("趋势仍站上200线、未到减仓区 → 持有，别过早全平。")
    elif has_edge and dd <= -0.15 and not trap:
        stance, color = "🟢 持有·可在更深档补", "#2BE6A8"
        headline = "深跌且引擎仍有正倾斜：**持有**，跌到下一共振档可按计划补，但每补一档同步下移整体止损。"
        if len(tranches) >= 2:
            add_txt = "、".join(f"{t['tier']}档 {t['price']:.1f}" for t in tranches[1:])
            actions.append(f"跌到下一共振支撑({add_txt})可补仓——历史上该深度远期分布仍正偏；补一档下移一次止损。")
        actions.append("反弹到引擎中位/前高附近分批减 1/3，移动止损锁利润。")
    else:
        stance, color = "🟡 持有·按纪律", "#FFD166"
        headline = "无明确加/减信号：**持有**，按机械止盈止损纪律走，不追不砍。"
        actions.append("到下方止盈区分批减、跌破硬止损离场；其余仓位用移动止损。")

    # ---- 通用触发式止盈/止损（价格升序，先到先动；排序后再编号，杜绝"低位标再减、高位标第一减"的乱序）----
    tp = []
    if up_median == up_median and up_median is not None and cur == cur and up_median > 0:
        sc1 = cur * (1 + up_median)
        if sc1 > cur:
            tp.append((sc1, f"≈{sc1:.1f}(+{up_median:.0%}，引擎该状态中位上行)"))
    tgt = tranches[0]["target"] if tranches else None
    if tgt and tgt == tgt and cur == cur and tgt > cur:
        tp.append((tgt, f"≈前高 {tgt:.0f}"))
    tp.sort(key=lambda x: x[0])
    for i, (_, desc) in enumerate(tp):
        lead = "涨到第一减仓区 " if i == 0 else "涨到下一减仓区 "
        triggers.append(f"{lead}{desc}：分批减 1/3、移动止损保护其余。")
    stops = [t["stop"] for t in tranches if t.get("stop") == t.get("stop")]
    if stops:
        triggers.append(f"跌破**硬止损 ≈{min(stops):.0f}**(重档技术失守)：无条件离场，不恋战。")
    triggers.append("跌破200日线 或 市场宽度恶化 → 减到半仓(已验证的降仓规则)，别扛。")
    if dte is not None and dte <= 21:
        triggers.append(f"财报临近({dte}天)：别在财报前满仓押方向，财报后按超预期/不及的历史漂移再调。")
    if is_lev:
        triggers.append("⚠️ 杠杆ETF每日复利衰减、不可长持，必须设硬止损/止盈。")

    return {"asset": card.get("asset"), "stance": stance, "color": color,
            "headline": headline, "actions": actions, "triggers": triggers}


def exit_warning(price: pd.Series, fragile_now: bool = False,
                 breadth_pctile: float | None = None) -> dict:
    """🚨 撤离预警状态灯：把'撤离/减仓'从触发式升级成**分级预警**，并补上现在缺的"防过热止盈"维度。

    综合四个**已有但没整合**的信号，全部历史校准、不预测：
      ① 距200线临近度——离撤离线(200日均线)还有多少；跌破=红、4%内=黄(接近)。
      ② 市场宽度脆弱——一篮子跌破自身200线比例的历史分位(领先指数·诚实误报~60%)。
      ③ 波动跳升——近21日已实现波动的历史分位。
      ④ 过热/乖离——价格距200线的乖离处历史高分位 → 分批止盈(防过热，红/黄规则不含此维度)。
    返回 {level, color, action, signals[], dist_ma200, vol_pctile, overext_pctile, red, amber}。
    红=触发已验证的减半仓规则；黄=提前预警(收紧止损/止盈/降仓)；绿=无撤离信号。
    """
    px = price.dropna()
    if len(px) < 120:
        return {"level": "—", "color": "#8A93A6", "action": "样本不足，撤离预警暂不可用。",
                "signals": [], "red": False, "amber": False,
                "dist_ma200": float("nan"), "vol_pctile": float("nan"), "overext_pctile": float("nan")}
    cur = float(px.iloc[-1])
    ma200 = px.rolling(200, min_periods=100).mean()
    dist = float(cur / ma200.iloc[-1] - 1.0) if ma200.notna().iloc[-1] else float("nan")
    tp = (px / ma200 - 1.0).dropna()
    overext = float((tp.iloc[-1] >= tp.tail(756)).mean()) if len(tp) > 120 else float("nan")
    rv = px.pct_change().rolling(21, min_periods=10).std() * np.sqrt(252)
    vol_pct = float((rv.iloc[-1] >= rv.tail(252)).mean()) if rv.notna().iloc[-1] else float("nan")

    signals: list[dict] = []
    red = amber = False
    # ① 趋势 / 距200线
    if dist == dist:
        if dist < 0:
            signals.append({"name": "趋势", "state": "🔴 已跌破200线",
                            "detail": f"距200线 {dist:+.0%}（趋势破位·历史 2.4x 未来回撤概率）"}); red = True
        elif dist < 0.04:
            signals.append({"name": "趋势", "state": "🟡 接近减仓线",
                            "detail": f"距200线仅 {dist:+.0%}（快到撤离阈值，收紧止损、待确认）"}); amber = True
        else:
            signals.append({"name": "趋势", "state": "🟢 趋势健康", "detail": f"距200线 {dist:+.0%}（在均线上方）"})
    # ② 市场宽度
    if fragile_now:
        signals.append({"name": "市场宽度", "state": "🔴 宽度恶化",
                        "detail": "一篮子跌破自身200线比例处历史低分位（约 3x 未来大跌概率·略领先指数）"}); red = True
    elif breadth_pctile is not None and breadth_pctile == breadth_pctile and breadth_pctile < 0.25:
        signals.append({"name": "市场宽度", "state": "🟡 宽度转弱",
                        "detail": f"宽度分位 {breadth_pctile:.0%}（接近脆弱阈值 15%，留意领先恶化）"}); amber = True
    else:
        signals.append({"name": "市场宽度", "state": "🟢 宽度healthy",
                        "detail": (f"宽度分位 {breadth_pctile:.0%}" if (breadth_pctile is not None and breadth_pctile == breadth_pctile) else "正常")})
    # ③ 波动
    if vol_pct == vol_pct:
        if vol_pct > 0.90:
            signals.append({"name": "波动", "state": "🟡 波动飙升",
                            "detail": f"近21日已实现波动分位 {vol_pct:.0%}（高波动→按波动降仓）"}); amber = True
        else:
            signals.append({"name": "波动", "state": "🟢 波动正常", "detail": f"波动分位 {vol_pct:.0%}"})
    # ④ 过热 / 乖离（防过热止盈——红/黄减半仓规则不含此维度）
    if overext == overext:
        if overext > 0.90 and dist == dist and dist > 0.05:
            signals.append({"name": "过热", "state": "🟡 乖离过大",
                            "detail": f"距200线 {dist:+.0%} 处近3年第 {overext:.0%} 分位（高位拉伸→分批止盈、移动止损）"}); amber = True
        else:
            signals.append({"name": "过热", "state": "🟢 不过热", "detail": f"乖离分位 {overext:.0%}"})

    if red:
        level, color = "🔴 撤离预警", "#FF5C7A"
        action = "**减到半仓 + 按波动降仓**（已验证规则：历史砍回撤约 40%）；站回200线上方且宽度healthy 再加回。"
    elif amber:
        level, color = "🟡 撤离黄灯（提前预警）", "#FFD166"
        acts = []
        if any("接近减仓线" in s["state"] for s in signals): acts.append("接近撤离线→收紧止损、别加仓、待确认")
        if any("乖离过大" in s["state"] for s in signals): acts.append("高位拉伸→分批止盈减 1/3、移动止损让利润奔跑")
        if any("波动飙升" in s["state"] for s in signals): acts.append("波动高→按波动降仓")
        if any("宽度转弱" in s["state"] for s in signals): acts.append("宽度转弱→停止加仓、盯领先恶化")
        action = "；".join(acts) or "观察为主、收紧止损。"
    else:
        level, color = "🟢 无撤离信号", "#2BE6A8"
        action = "趋势健康 + 宽度healthy + 波动正常 + 不过热——持有，让赢家跑（移动止损保护）。"
    return {"level": level, "color": color, "action": action, "signals": signals,
            "dist_ma200": dist, "vol_pctile": vol_pct, "overext_pctile": overext,
            "fragile": bool(fragile_now), "red": red, "amber": amber}
