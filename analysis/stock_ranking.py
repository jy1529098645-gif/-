"""选股榜：多因子横截面综合评分 → "最推荐买入"排名（专业量化口径）。

方法论(业界标准，见 Jegadeesh-Titman 动量 / Frazzini-Pedersen 低beta / Alphalens IC 验证)：
  多因子 z-score(winsorize) 加权综合分 → 横截面排名 → 分级，并用 IC/RankIC/ICIR + 分位价差验证预测力。

价格类因子(全部免费、**无前视**)：
  · risk_adj_mom  风险调整动量 = (12-1月动量)/已实现波动 —— 专业版动量，避开"最强动量=最高波动"的爆仓票
  · trend_quality 趋势健康 = f(站上200线, 200线斜率, 站上50线) —— v3 验证过的趋势参与核心
  · rel_strength  相对强度 = 近6月收益 − SPY 近6月收益 —— 跑赢大盘的成分
  · mom_12_1      经典 12-1 动量
  · low_vol       低波动(−已实现波动) —— 在科技池里与动量打架，故低权重

诚实铁律(与全工具一致)：
  · 这是**校准式筛选**(谁更符合"已验证的趋势参与策略")，**不是"这些会涨"的预测**；
  · 科技窄池里横截面多空 edge 弱(实测夏普≈0)，故榜单价值在"选健康趋势票"，不在"保证跑赢"；
  · 附 IC 验证历史预测力，且**实盘 IC 通常只有回测一半**(行业经验)；
  · 基本面/质量/价值因子需 PIT 付费数据(免费 yfinance 有前视)，故**不入评分**，仅可人读。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 评分权重(强调已验证维度：风险调整动量 + 趋势健康 + 相对强度)
DEFAULT_WEIGHTS = {"risk_adj_mom": 0.30, "trend_quality": 0.25, "rel_strength": 0.25,
                   "mom_12_1": 0.10, "low_vol": 0.10}


def _zwins(s: pd.Series, winsor: float = 3.0) -> pd.Series:
    """横截面 z-score + winsorize(±winsor σ)。"""
    s = s.astype(float)
    mu, sd = s.mean(), s.std()
    if not (sd == sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return ((s - mu) / sd).clip(-winsor, winsor)


def _factor_frame(prices: pd.DataFrame, spy: pd.Series) -> dict:
    """全历史因子面板(date×ticker)，无前视(全部用截至 t 的信息)。"""
    px = prices.dropna(how="all").ffill()
    rets = px.pct_change()
    vol = rets.rolling(63, min_periods=32).std() * np.sqrt(252)
    mom = px.shift(21) / px.shift(252) - 1.0                      # 12-1 动量
    radj = mom / vol.replace(0, np.nan)                           # 风险调整动量
    sma200 = px.rolling(200, min_periods=100).mean()
    sma50 = px.rolling(50, min_periods=25).mean()
    trend = (0.5 * (px > sma200) + 0.3 * (sma200 > sma200.shift(20)) + 0.2 * (px > sma50)).astype(float)
    spy6 = (spy / spy.shift(126) - 1.0).reindex(px.index).ffill()
    rel = (px / px.shift(126) - 1.0).sub(spy6, axis=0)
    return {"risk_adj_mom": radj, "trend_quality": trend, "rel_strength": rel,
            "mom_12_1": mom, "low_vol": -vol}


def _composite_at(panels: dict, date, weights: dict) -> pd.Series:
    """某日横截面综合分(各因子 z-score 加权和)。"""
    score = None
    for f, w in weights.items():
        row = panels[f].loc[date].dropna()
        z = _zwins(row) * w
        score = z if score is None else score.add(z, fill_value=0.0)
    return score.dropna().sort_values(ascending=False)


def rank_stocks(tickers: list[str], start: str = "2012-01-01", end: str | None = None,
                weights: dict | None = None, enrich_top: int = 12) -> dict:
    """生成"最推荐买入"排名(最新横截面)。

    返回 {asof, table(DataFrame: 排名/综合分/各因子z/趋势/tier/今日暴露), weights, note}。
    tier 同时看综合分与趋势健康——趋势破位的票即便动量分高也不进🟢(符合 v3 验证口径)。
    """
    from data import loader
    from analysis import position_guidance as pg

    weights = weights or DEFAULT_WEIGHTS
    tickers = list(dict.fromkeys([t.upper() for t in tickers]))
    px = loader.load_prices(tickers, start, end).dropna(how="all")
    spy = loader.load_prices(["SPY"], start, end)["SPY"]
    panels = _factor_frame(px, spy)
    date = px.dropna(how="all").index[-1]

    comp = _composite_at(panels, date, weights)
    rows = []
    for rank, (tk, sc) in enumerate(comp.items(), 1):
        p = px[tk].dropna()
        if p.shape[0] < 200:
            continue
        sma200 = p.rolling(200, min_periods=100).mean()
        above200 = bool(p.iloc[-1] > sma200.iloc[-1])
        slope = bool(sma200.iloc[-1] > sma200.iloc[-21]) if sma200.notna().iloc[-21] else True
        # tier：综合分 × 趋势健康
        if sc >= 0.6 and above200 and slope:
            tier = "🟢 强烈(趋势健康+综合分高)"
        elif sc >= 0.6 and not above200:
            tier = "🟡 分高但趋势破位(等站回200线)"
        elif sc >= -0.2 and above200:
            tier = "🟡 中性参与"
        elif not above200 and not slope:
            tier = "🔴 趋势破位·回避"
        else:
            tier = "🔴 弱/回避"
        rows.append({"排名": rank, "票": tk, "综合分": float(round(sc, 3)),
                     "风险调整动量z": float(round(_zwins(panels["risk_adj_mom"].loc[date].dropna()).get(tk, np.nan), 2)),
                     "趋势健康z": float(round(_zwins(panels["trend_quality"].loc[date].dropna()).get(tk, np.nan), 2)),
                     "相对强度z": float(round(_zwins(panels["rel_strength"].loc[date].dropna()).get(tk, np.nan), 2)),
                     "站上200线": above200, "200线斜率正": slope, "tier": tier})
    tab = pd.DataFrame(rows)

    # 仅对前 enrich_top 名补 v3 今日建议暴露(快)
    expo_map = {}
    for tk in tab["票"].head(enrich_top):
        try:
            g = pg.position_guidance(tk, start=start, end=end)
            expo_map[tk] = g["exposure"]["moderate"]["exposure_pct"]
        except Exception:  # noqa: BLE001
            expo_map[tk] = None
    tab["v3中性暴露%"] = tab["票"].map(expo_map)

    return {"asof": str(date.date()), "table": tab, "weights": weights,
            "note": ("多因子横截面综合评分(风险调整动量+趋势健康+相对强度为主)→排名。校准式筛选(谁更符合"
                     "已验证的趋势参与策略)，非'会涨'预测；科技窄池横截面edge弱，价值在选健康趋势票。"
                     "基本面/质量因子需PIT付费数据，未入评分。")}


def validate_ranking(tickers: list[str], start: str = "2013-01-01", end: str | None = None,
                     weights: dict | None = None, horizons=(21, 63), rebalance: int = 21) -> dict:
    """用 IC / RankIC / ICIR + 分位价差 + 安慰剂 验证综合分的历史预测力(诚实)。

    IC=Spearman(综合分, 未来h日收益) 的时序均值；ICIR=mean(IC)/std(IC)；
    分位价差=top1/3 − bottom1/3 的平均未来收益；安慰剂=随机打乱评分(应≈0)。
    """
    from data import loader
    from scipy.stats import spearmanr

    weights = weights or DEFAULT_WEIGHTS
    tickers = list(dict.fromkeys([t.upper() for t in tickers]))
    px = loader.load_prices(tickers, start, end).dropna(how="all").ffill()
    spy = loader.load_prices(["SPY"], start, end)["SPY"]
    panels = _factor_frame(px, spy)
    dates = px.index[252::rebalance]
    rng = np.random.default_rng(0)

    out = {}
    for h in horizons:
        fwd = px.shift(-h) / px - 1.0
        ics, qspreads, placebo = [], [], []
        for d in dates:
            sc = _composite_at(panels, d, weights)
            f = fwd.loc[d].dropna()
            common = sc.index.intersection(f.index)
            if len(common) < 6:
                continue
            s_, f_ = sc[common], f[common]
            ic = spearmanr(s_, f_).statistic
            if ic == ic:
                ics.append(ic)
                k = max(1, len(common) // 3)
                top = s_.nlargest(k).index
                bot = s_.nsmallest(k).index
                qspreads.append(float(f_[top].mean() - f_[bot].mean()))
                pf = f_.sample(frac=1.0, random_state=int(rng.integers(1e9)))
                pf.index = f_.index
                placebo.append(spearmanr(s_, pf.reindex(common)).statistic)
        ic_arr = np.array(ics)
        out[h] = {
            "ic_mean": float(np.nanmean(ic_arr)) if ic_arr.size else np.nan,
            "rank_ic_mean": float(np.nanmean(ic_arr)) if ic_arr.size else np.nan,  # 已用 spearman=RankIC
            "icir": float(np.nanmean(ic_arr) / np.nanstd(ic_arr)) if ic_arr.size and np.nanstd(ic_arr) > 0 else np.nan,
            "ic_positive_rate": float((ic_arr > 0).mean()) if ic_arr.size else np.nan,
            "quantile_spread_mean": float(np.nanmean(qspreads)) if qspreads else np.nan,
            "placebo_ic_mean": float(np.nanmean(placebo)) if placebo else np.nan,
            "n_periods": int(ic_arr.size),
        }
    out["_note"] = ("IC=综合分与未来收益的横截面RankIC(时序均值)；ICIR=mean/std(越稳越高，>0.5算可用)；"
                    "分位价差=top1/3−bottom1/3未来收益；安慰剂(随机分)应≈0。**实盘IC通常只有回测一半**(行业经验)。")
    return out


def ranking_verdict(validation: dict) -> dict:
    """把 IC 验证翻成诚实裁决：榜单到底能不能预测"谁跑赢"。"""
    h = 21 if 21 in validation else next(k for k in validation if isinstance(k, int))
    d = validation[h]
    ic, plac, icir = d["ic_mean"], d["placebo_ic_mean"], d["icir"]
    edge = ic - plac
    if icir > 0.5 and edge > 0.02:
        grade, verdict = "🟢 有可用横截面预测力", f"RankIC {ic:+.3f}(安慰剂 {plac:+.3f})、ICIR {icir:.2f}——排名能预测相对强弱。"
    elif edge > 0.01 and ic > plac:
        grade, verdict = "🟡 预测力弱(略胜随机)", f"RankIC {ic:+.3f} 略高于安慰剂 {plac:+.3f}、ICIR {icir:.2f}(<0.5)——有一点点信号但不稳，别据此重仓押单票。"
    else:
        grade, verdict = "🔴 无横截面预测力(≈随机)", f"RankIC {ic:+.3f} ≈ 安慰剂 {plac:+.3f}、ICIR {icir:.2f}——**排名预测不了谁跑赢**(科技窄池本就如此)。"
    return {"grade": grade, "verdict": verdict,
            "usage": ("正确用法：当**健康趋势票筛选器**——top名是趋势最健康/风险调整动量最好的票，"
                      "适合用 v3 策略去参与、并**回避**🔴趋势破位的；**别**当成'top1会跑赢top10'的预测。")}


def format_ranking(res: dict, top_n: int = 10) -> str:
    """渲染排名为 Markdown。"""
    tab = res["table"]
    L = [f"# 🏆 最推荐买入·选股榜  （{res['asof']}）", f"> {res['note']}", "",
         "| 排名 | 票 | 综合分 | 风险调整动量z | 趋势健康z | 相对强度z | v3暴露% | 评级 |",
         "|---|---|---|---|---|---|---|---|"]
    for _, r in tab.head(top_n).iterrows():
        ev = r.get("v3中性暴露%")
        L.append(f"| {int(r['排名'])} | **{r['票']}** | {r['综合分']:+.2f} | {r['风险调整动量z']:+.2f} | "
                 f"{r['趋势健康z']:+.2f} | {r['相对强度z']:+.2f} | {ev if ev is not None else '—'}% | {r['tier']} |")
    L += ["", "_校准式筛选、非买卖指令；横截面edge在科技窄池偏弱，榜单价值在'选健康趋势票'；附IC验证见验证区。_"]
    return "\n".join(L)
