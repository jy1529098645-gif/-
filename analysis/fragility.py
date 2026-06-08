"""市场脆弱性预警 + 等/追操作指南（非主流信号，经网格迭代选定参数）。

核心信号：**宽度恶化**——一篮子股票中"在自身 200 日均线上方"的比例，跌到历史低分位时，
未来大回撤概率显著抬升。网格迭代(MA∈{50..200}×阈值∈{10%..25}×周期{42,63,126})结论：
**MA200 + 触发分位 15%** 跨周期最稳（最差 lift 1.90x、平均 2.39x、召回~32%、误报~60%）。
它比"指数跌破200线"更冷门、且常**领先于指数破位**（宽度先恶化）。

诚实边界：lift ~2.4x、误报 ~60%——这是**降仓开关**，不是崩盘预言；会频繁假警报。
"等/追"指南来自回测：高位附近"等浅回调"历史上更亏(回调70%不来)，应追/分批；
唯一该"等"的是深回撤区(指数 -20~30% 有 edge)。脆弱性高时一切偏防守。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# —— 迭代选定的默认参数 ——
MA_WINDOW = 200
PCT_WINDOW = 756        # 历史分位回看 ~3 年
FRAGILE_THRESH = 0.15   # 宽度分位 < 15% → 脆弱
# 验证统计（基率 17%；MA200/15%/H63）——展示用，源自 /tmp/grid.py 回测
VALIDATED = {"lift_h42": 2.83, "lift_h63": 2.45, "lift_h126": 1.90,
             "recall_h63": 0.32, "fp_h63": 0.60, "base_h63": 0.17}


def breadth_above_ma(panel: pd.DataFrame, ma: int = MA_WINDOW) -> pd.Series:
    """一篮子中'收盘在自身 ma 日均线上方'的比例（0–1）。panel: date×ticker 收盘价。"""
    p = panel.ffill()
    above = p > p.rolling(ma, min_periods=ma // 2).mean()
    return above.mean(axis=1)


def rolling_pct_rank(s: pd.Series, win: int = PCT_WINDOW) -> pd.Series:
    """当前值在过去 win 天里的分位（0–1，因果，无前视）。"""
    return s.rolling(win, min_periods=min(252, win // 2)).apply(lambda x: (x[-1] >= x).mean(), raw=True)


def fragility_frame(panel: pd.DataFrame, ma: int = MA_WINDOW, win: int = PCT_WINDOW,
                    thresh: float = FRAGILE_THRESH) -> pd.DataFrame:
    """返回逐日 {breadth(宽度%), pctile(宽度历史分位), fragile(是否脆弱)}。"""
    br = breadth_above_ma(panel, ma)
    pct = rolling_pct_rank(br, win)
    return pd.DataFrame({"breadth": br, "pctile": pct, "fragile": pct < thresh})


def evaluate_breadth_warning(panel: pd.DataFrame, index_price: pd.Series,
                             horizon: int = 63, dd_thresh: float = -0.10,
                             ma: int = MA_WINDOW, win: int = PCT_WINDOW,
                             thresh: float = FRAGILE_THRESH) -> dict:
    """实测宽度信号的预警力：触发后未来 horizon 日内出现 ≤dd_thresh 回撤的命中率/lift/召回/误报。"""
    ff = fragility_frame(panel, ma, win, thresh)
    idx = index_price.dropna()
    p = idx.to_numpy()
    fmdd = pd.Series(index=idx.index, dtype=float)
    for i in range(len(p) - 1):
        fmdd.iloc[i] = p[i:i + horizon + 1].min() / p[i] - 1.0
    target = (fmdd <= dd_thresh)
    sig = ff["fragile"].reindex(idx.index).fillna(False)
    v = target.notna() & sig.notna()
    s2, t2 = sig[v], target[v]
    base = float(t2.mean()) if len(t2) else float("nan")
    if s2.sum() < 20:
        return {"base": base, "n_signal": int(s2.sum()), "note": "触发样本不足"}
    cond = float(t2[s2].mean())
    return {"base": base, "cond": cond, "lift": cond / base if base else float("nan"),
            "recall": float(s2[t2].mean()), "fp": float((s2 & ~t2).sum() / max(1, s2.sum())),
            "n_signal": int(s2.sum()), "horizon": horizon}


def current_fragility(panel: pd.DataFrame, ma: int = MA_WINDOW, win: int = PCT_WINDOW,
                      thresh: float = FRAGILE_THRESH) -> dict:
    """当前市场脆弱性读数（宽度% / 历史分位 / 红绿灯）。"""
    ff = fragility_frame(panel, ma, win, thresh).dropna()
    if ff.empty:
        return {"available": False}
    last = ff.iloc[-1]
    return {"available": True, "breadth": float(last["breadth"]), "pctile": float(last["pctile"]),
            "fragile": bool(last["fragile"]), "thresh": thresh,
            "light": "🔴 脆弱(宜降仓)" if bool(last["fragile"]) else "🟢 正常",
            "date": str(ff.index[-1].date())}


# ---------------------------------------------------------------------------
# 等 / 追 操作指南（来自回测：高位别等浅回调；深回撤才是该等的有 edge 区）
# ---------------------------------------------------------------------------
def wait_or_chase(current_dd: float, fragile_now: bool = False, is_index: bool = False,
                  conviction: bool = True) -> dict:
    """给定当前距前高回撤，返回'等/追'操作指南。

    current_dd: ≤0，价格相对前高的回撤；fragile_now: 市场脆弱性是否触发；
    is_index: 是否指数(深回撤建仓 edge 更可靠)；conviction: 个股是否有'会回来'的把握。
    """
    dd = current_dd
    if fragile_now:
        head = "🔴 市场脆弱性已触发——一切偏防守"
    else:
        head = "🟢 市场脆弱性正常"

    if dd >= -0.05:
        action = "追 / 直接分批建仓"
        detail = ("价格在高位附近。**别等浅回调**——历史上想等的 −4% 回调约 70% 不会来，"
                  "等不到反而追更高。直接分批进场（在场>择时）。")
    elif dd > -0.15:
        action = "边追边备子弹（分档）"
        detail = ("浅~中度回撤，无统计 edge。按计划分批，留部分现金给更深的档，别一次性梭哈也别空等。")
    elif dd > -0.20:
        action = "接近建仓区，开始分批加重"
        detail = ("已进入较深回撤，接近历史有 edge 的建仓带。开始分批、逐步加重。")
    else:
        if is_index:
            action = "✅ 有 edge 的建仓区：分批加重"
            detail = ("指数深回撤(−20%+)是回测里少数真有正超额、胜率近100%的建仓带。分批加重、持有数月。")
        elif conviction:
            action = "深跌建仓区（仅在你确信会回来时）"
            detail = ("单票深跌：赢家历史反弹大，但靠'它会回来'。有基本面把握才加重，否则警惕价值陷阱。")
        else:
            action = "深跌但无把握 → 观望/极轻仓"
            detail = ("单票深跌且无把握会回来——可能是价值陷阱(输家池此档历史超额为负)。别越跌越补。")

    if fragile_now and dd >= -0.15:
        action = "降仓优先 / 建仓减半等确认"
        detail = "市场脆弱性触发(宽度恶化)，高位追势风险升高——建仓减半、等趋势确认或回到更深档。 " + detail

    return {"current_dd": dd, "headline": head, "action": action, "detail": detail,
            "fragile": fragile_now}
