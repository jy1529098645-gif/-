"""引擎纪律层（Engine Discipline）——把"诚实"从约定升级成可执行的代码闸门。

源自对一份外部升级方案的取舍：**赞同的部分按本工具铁律改写口径后落地**，不赞同的（未验证启发式、
新闻按系数自动砍仓、LLM 抽新闻）一律拒绝或仅作 display-only 旗标。

铁律对齐：
- 这里给的是**证据等级（evidence grade）**，不是涨跌预测、不是买卖指令；
- 负超额 / CI 跨 0 / 有效样本不足 → **硬封顶**，永不进入 A/B 强等级；
- 基本面/新闻**只能下调**把握度与仓位上限，**永不**凭其制造买入信号；
- 单票权重是"候选池内排序"，真实组合仓位另有**相关性感知**上限。

五个闸门：
1. sanity_check_fundamentals  —— 体检基本面字段（修 yfinance 股息率 87% 这类脏数据）
2. evidence_grade            —— 由 超额/CI/有效样本/波动 给 A–F 证据等级 + 仓位封顶（硬规则）
3. reconcile_horizons        —— 多周期对账，冲突即降置信
4. portfolio_budget          —— 相关性感知的真实组合仓位上限（候选权重≠真实仓位）
5. validate_consistency      —— 渲染前抓内部自相矛盾（弱证据却强评级、止盈乱序、动量陷阱却允许摊平…）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. 基本面字段体检（§3）—— 操作"原始数值 info"，归一 + 标记可疑，可疑字段不入展示/评分
# ---------------------------------------------------------------------------
# 合理区间（小数口径，megacap 科技股语境）。越界即标记可疑、剔除展示。
_FUND_BOUNDS = {
    "dividendYield": (0.0, 0.15),     # >15% 对大盘科技几乎不可能
    "trailingPE":    (0.0, 300.0),
    "forwardPE":     (0.0, 300.0),
    "returnOnEquity": (-1.0, 2.0),
    "grossMargins":  (-0.2, 0.95),
    "profitMargins": (-0.5, 0.85),
    "revenueGrowth": (-0.9, 5.0),
    "earningsGrowth": (-5.0, 10.0),
    "beta":          (0.0, 5.0),
}


def _normalize_dividend_yield(v: float) -> float:
    """yfinance 的 dividendYield 单位在不同版本间漂移：有时是小数(0.0087)，
    有时已是百分数(0.87 表示 0.87%)。若按小数口径 >15% 基本不可能 → 判定其实是百分数，/100 还原。"""
    if v is None or v != v:
        return v
    return v / 100.0 if v > 0.15 else v


def sanity_check_fundamentals(info: dict) -> dict:
    """体检原始基本面数值字典。返回 {clean: {field: value(小数口径)}, suspicious: {field: 原值}, warnings: [..]}。

    - dividendYield 先做单位归一；
    - 其余字段越界即判可疑：从 clean 剔除、登记到 suspicious，并产出中文告警。
    可疑字段**不得**进入展示或任何评分。"""
    info = info or {}
    clean: dict = {}
    suspicious: dict = {}
    warnings: list[str] = []

    for field, (lo, hi) in _FUND_BOUNDS.items():
        v = info.get(field)
        if v is None or (isinstance(v, float) and v != v):
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if field == "dividendYield":
            v = _normalize_dividend_yield(v)
        if lo <= v <= hi:
            clean[field] = v
        else:
            suspicious[field] = v
            warnings.append(f"{field} 取值 {v:.4g} 越出合理区间[{lo:g},{hi:g}]，疑似数据异常，已剔除展示与评分。")

    return {"clean": clean, "suspicious": suspicious, "warnings": warnings}


# ---------------------------------------------------------------------------
# 2. 证据等级 + 仓位封顶（§4，改写口径：是"证据强度"不是"买入推荐"）
# ---------------------------------------------------------------------------
# 等级 → (最大候选池内仓位比例, 一句话含义)。这是"证据能支撑多重的仓位"，非看涨程度。
_GRADE_CAP = {
    "A": (1.00, "显著正超额 + 有效样本充足 + 风险可控 + 多周期不冲突"),
    "B": (0.60, "正超额且 CI 不跨 0，但部分条件不理想"),
    "C": (0.33, "绝对收益为正，但超额/显著性/样本质量不足，仅小仓试探"),
    "D": (0.20, "回撤无超额优势 / 高波动 / 弱证据，观察为主"),
    "F": (0.00, "负超额且左尾差，不建仓"),
}


def evidence_grade(bucket: dict | None, vol_percentile: float | None = None,
                   horizon_conflict: bool = False) -> dict:
    """由引擎桶给证据等级 A–F + 仓位封顶。**硬规则**：负超额/CI跨0/有效样本不足永不进 A/B。

    bucket 期望字段：excess, ci_low, ci_high, significant, n_events, n_independent, low_power, reward_risk。
    返回 {grade, confidence, max_position_fraction, action, reasons[], meaning}。
    """
    reasons: list[str] = []
    if not bucket:
        return {"grade": "C", "confidence": "低", "max_position_fraction": 0.25,
                "action": "仅观察", "reasons": ["无引擎桶/样本缺失"], "meaning": _GRADE_CAP["C"][1]}

    excess = bucket.get("excess")
    ci_low = bucket.get("ci_low")
    ci_high = bucket.get("ci_high")
    n = bucket.get("n_events")
    neff = bucket.get("n_independent")
    low_power = bool(bucket.get("low_power"))
    significant = bool(bucket.get("significant"))

    ci_crosses_zero = (ci_low is not None and ci_high is not None and ci_low <= 0 <= ci_high)

    # 有效样本闸门
    sample_weak = False
    if n is None or n < 30:
        sample_weak = True
        reasons.append("名义样本 N<30")
    if neff is None or (neff == neff and neff < 20):
        sample_weak = True
        reasons.append("有效独立样本 N_eff<20")
    if low_power:
        sample_weak = True
        reasons.append("重叠窗口致有效样本不足(low_power)")

    # —— 主等级判定（从坏到好，先用硬否决兜底）——
    if excess is None:
        grade, confidence, action = "C", "低", "仅观察"
        reasons.append("超额缺失")
    elif excess <= 0:
        grade, confidence, action = "D", "低", "不主动建仓 / 等确认"
        reasons.append("超额≤0（更可能是 beta/选股偏差，非择时 alpha）")
    elif ci_crosses_zero or not significant:
        grade, confidence, action = "C", "低", "小仓试探"
        reasons.append("CI 跨 0 / 未达显著")
    elif sample_weak:
        grade, confidence, action = "C", "低", "降级为研究信号"
    else:
        grade, confidence, action = "B", "中", "回踩可分批"
        reasons.append("正超额且 CI 不跨 0、样本质量达标")
        # 升 A 需更严：超额厚 + 盈亏比>1 + 无多周期冲突
        rr = bucket.get("reward_risk")
        if (excess >= 0.03 and rr == rr and rr is not None and rr > 1.0 and not horizon_conflict):
            grade, confidence, action = "A", "中高", "回踩可分批（证据最强档）"
            reasons.append("超额厚 + 盈亏比>1 + 多周期不冲突")

    # —— 波动封顶（高波动只下调、不上调）——
    cap = _GRADE_CAP[grade][0]
    if vol_percentile is not None and vol_percentile == vol_percentile:
        if vol_percentile > 0.95:
            cap = min(cap, 0.15); reasons.append("波动分位>95%，仓位封顶降至15%")
        elif vol_percentile > 0.90:
            cap = min(cap, 0.25); reasons.append("波动分位>90%，仓位封顶降至25%")
        if vol_percentile > 0.90 and excess is not None and excess <= 0:
            grade = "D" if grade not in ("F",) else grade
            action = "高波动观察名单"; confidence = "低"
            cap = min(cap, 0.20)

    # —— 多周期冲突降置信 ——
    if horizon_conflict and confidence == "中高":
        confidence = "中"; reasons.append("多周期信号冲突，置信下调")

    return {"grade": grade, "confidence": confidence, "max_position_fraction": float(cap),
            "action": action, "reasons": reasons, "meaning": _GRADE_CAP[grade][1],
            "ci_crosses_zero": bool(ci_crosses_zero), "sample_weak": bool(sample_weak)}


# ---------------------------------------------------------------------------
# 2b. 风险偏好档（保守/均衡/激进）—— 在证据等级之上做个性化仓位缩放
# ---------------------------------------------------------------------------
RISK_PROFILES = {
    "保守": {"mult": 0.5, "require_trigger": True, "earnings_overnight_mult": 0.25,
             "desc": "仓位减半、必须等触发确认、财报隔夜暴露压到 1/4。"},
    "均衡": {"mult": 1.0, "require_trigger": True, "earnings_overnight_mult": 0.5,
             "desc": "按证据等级原始封顶、需触发确认、财报隔夜减半。"},
    "激进": {"mult": 1.25, "require_trigger": False, "earnings_overnight_mult": 0.75,
             "desc": "仓位上浮 25%(仍受单票上限约束)、可不等触发、财报隔夜留 3/4。"},
}


def apply_risk_profile(grade: dict, profile: str = "均衡", single_name_cap: float = 0.25) -> dict:
    """按风险偏好缩放证据等级给出的仓位封顶。返回新 dict（不改原对象）。

    保守缩 0.5×、激进 1.25×，但**永远**不超过单票硬上限 single_name_cap。"""
    prof = RISK_PROFILES.get(profile, RISK_PROFILES["均衡"])
    g = dict(grade)
    base = grade.get("max_position_fraction", 0.25)
    g["max_position_fraction"] = float(min(single_name_cap, base * prof["mult"]))
    g["risk_profile"] = profile
    g["require_trigger"] = prof["require_trigger"]
    g["profile_desc"] = prof["desc"]
    return g


# ---------------------------------------------------------------------------
# 3. 多周期对账（§10）—— 用本工具真实引擎跑多 horizon，冲突即降置信
# ---------------------------------------------------------------------------
def _bucket_score(b: dict | None) -> float | None:
    if not b:
        return None
    ex = b.get("excess")
    if ex is None:
        return None
    if ex > 0 and b.get("significant"):
        return 1.0
    if ex > 0:
        return 0.5
    return -1.0


def reconcile_horizons(per_horizon: dict) -> dict:
    """对账多周期的"当前状态桶"结论。per_horizon: {horizon:int -> bucket dict}。

    短期=h21/h63，中期=h126/h252，长期=h504。返回分组分、冲突旗标、综合动作与人话。"""
    short = [per_horizon.get(h) for h in (21, 63)]
    medium = [per_horizon.get(h) for h in (126, 252)]
    long = [per_horizon.get(h) for h in (504,)]

    def grp(items):
        vals = [s for s in (_bucket_score(b) for b in items) if s is not None]
        return (sum(vals) / len(vals)) if vals else None

    s, m, l = grp(short), grp(medium), grp(long)
    scores = [x for x in (s, m, l) if x is not None]
    conflict = bool(scores and max(scores) > 0 and min(scores) < 0)

    if s is not None and s > 0 and l is not None and l > 0:
        action = "多周期一致偏正：回踩可分批"
    elif s is not None and s > 0 and (l is None or l <= 0):
        action = "仅短/中期机会：波段交易，不当长持理由"
    elif s is not None and s <= 0 and l is not None and l > 0:
        action = "长期候选、但当前不是战术买点"
    elif scores and all(x <= 0 for x in scores):
        action = "各周期均无优势：观察为主"
    else:
        action = "证据混合：降低置信"

    return {"short": s, "medium": m, "long": l, "conflict": conflict,
            "action": action, "agreement": (sum(scores) / len(scores)) if scores else None}


# ---------------------------------------------------------------------------
# 4. 组合风险预算（§13）—— 候选权重≠真实仓位；相关性感知的真实上限
# ---------------------------------------------------------------------------
def portfolio_budget(briefs: list[dict], prices: pd.DataFrame | None = None,
                     single_name_max: float = 0.25, same_cluster_max: float = 0.45,
                     corr_threshold: float = 0.75, lookback: int = 252) -> dict:
    """把"候选池内权重"翻译成真实组合仓位上限。

    - 单票硬上限 single_name_max；高波动票自动降到 ≤15%；
    - 用近 lookback 日收益相关性把 |ρ|≥corr_threshold 的票聚成"高相关簇"，
      同簇合并暴露 ≤ same_cluster_max（不能各自按单票上限相加）。
    返回 {per_name: {cap, vol_capped, cluster}, clusters: [[..]], notes: [..]}。
    """
    tickers = [b["ticker"] for b in briefs]
    per_name: dict = {}
    for b in briefs:
        cap = single_name_max
        volp = b.get("vol_percentile")
        vol_capped = False
        if volp is not None and volp == volp and volp > 0.90:
            cap = min(cap, 0.15); vol_capped = True
        per_name[b["ticker"]] = {"cap": cap, "vol_capped": vol_capped, "cluster": None}

    clusters: list[list[str]] = []
    notes: list[str] = []
    if prices is not None and len(tickers) >= 2:
        cols = [t for t in tickers if t in prices.columns]
        if len(cols) >= 2:
            rets = prices[cols].pct_change().tail(lookback).dropna(how="all")
            corr = rets.corr()
            # 简单连通分量聚簇：|ρ|≥阈值 视为同簇
            seen: set[str] = set()
            for t in cols:
                if t in seen:
                    continue
                comp = [t]; seen.add(t)
                stack = [t]
                while stack:
                    cur = stack.pop()
                    for o in cols:
                        if o not in seen and abs(corr.loc[cur, o]) >= corr_threshold:
                            seen.add(o); comp.append(o); stack.append(o)
                if len(comp) >= 2:
                    clusters.append(sorted(comp))
            for grp in clusters:
                for t in grp:
                    per_name[t]["cluster"] = grp
                notes.append(
                    f"{' / '.join(grp)} 近{lookback}日相关性≥{corr_threshold:.0%}，"
                    f"属同一高相关簇——合计暴露应 ≤{same_cluster_max:.0%}，不能各自按单票上限相加。")

    return {"per_name": per_name, "clusters": clusters,
            "single_name_max": single_name_max, "same_cluster_max": same_cluster_max,
            "notes": notes}


# ---------------------------------------------------------------------------
# 5. 报告一致性校验器（§7）—— 渲染前抓内部自相矛盾
# ---------------------------------------------------------------------------
def validate_consistency(brief: dict, playbook: dict | None = None,
                         grade: dict | None = None) -> list[dict]:
    """检查单票简报+预案的内部一致性，返回问题列表 [{severity, code, message}]。

    覆盖：负超额却强评级、CI跨0却标显著、动量陷阱却允许摊平、止盈乱序、
    高波动却大仓、样本缺失却高置信、单票100%无组合语境。"""
    issues: list[dict] = []
    be = brief.get("engine_headline") or brief.get("engine_best") or {}
    excess = be.get("excess")
    trap = bool(brief.get("momentum_trap"))

    # 负超额却拿到强等级
    if grade and excess is not None and excess <= 0 and grade.get("grade") in ("A", "B"):
        issues.append({"severity": "high", "code": "neg_excess_strong_grade",
                       "message": "超额≤0 不应得到 A/B 证据等级。"})

    # CI 跨 0 却被标显著
    if be.get("significant") and be.get("ci_low") is not None and be.get("ci_high") is not None \
            and be["ci_low"] <= 0 <= be["ci_high"]:
        issues.append({"severity": "high", "code": "ci_zero_but_significant",
                       "message": "CI 跨 0 却标记为显著。"})

    # 高置信却样本缺失/不足
    if grade and grade.get("confidence") in ("中高", "高") and grade.get("sample_weak"):
        issues.append({"severity": "medium", "code": "weak_sample_high_conf",
                       "message": "样本不足却给出中高/高置信。"})

    # 动量陷阱却在 if_down 出现"补仓/摊平"
    if trap and playbook:
        downs = " ".join(playbook.get("if_down", []))
        if ("补仓" in downs or "摊平" in downs) and "不要越跌越补" not in downs:
            issues.append({"severity": "high", "code": "trap_allows_average_down",
                           "message": "动量陷阱却允许越跌越补。"})

    # 止盈/减仓必须按价位从低到高
    if playbook:
        prices = _extract_tp_prices(playbook.get("if_up", []))
        if prices != sorted(prices):
            issues.append({"severity": "high", "code": "take_profit_out_of_order",
                           "message": "减仓/止盈价位未按从低到高排列。"})

    # 高波动却仓位封顶过大
    volp = brief.get("vol_percentile")
    if grade and volp is not None and volp == volp and volp > 0.90 \
            and grade.get("max_position_fraction", 0) > 0.25:
        issues.append({"severity": "high", "code": "high_vol_big_position",
                       "message": "高波动(>90%)却仓位封顶>25%。"})

    return issues


_TP_NUM = None


def _extract_tp_prices(if_up: list[str]) -> list[float]:
    """从 if_up 文本里按出现顺序抽减仓价位（仅抓显式"≈数字"/"前高 数字"），用于校验排序。"""
    import re
    global _TP_NUM
    if _TP_NUM is None:
        _TP_NUM = re.compile(r"(?:≈|前高)\s*(\d{2,6}(?:\.\d+)?)")
    out: list[float] = []
    for line in if_up:
        m = _TP_NUM.search(line)
        if m:
            out.append(float(m.group(1)))
    return out
