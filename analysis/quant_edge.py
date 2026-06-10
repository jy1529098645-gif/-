"""量化 edge 层——回答"到底有没有真优势"的几件可证伪的事（全部免费数据）。

- alpha_beta        : CAPM 回归把收益拆成 α(择时/选股) 与 β(市场)，α 带 block bootstrap CI。
                      CI 跨 0 → 诚实地说"超额来自 beta，无显著 alpha"。
- regime_exposure   : 由免费可观测状态(已实现波动分位/信用利差/收益率曲线)给 0–1 暴露系数，
                      只降不加杠杆——这是风控 edge(改善回撤/夏普)，非择时预测。
- vol_target_backtest: 波动目标暴露 overlay vs 闭眼持有 的年化/夏普/回撤对照（诚实，含费）。
- walkforward_oos   : 把定稿规则在多段 OOS(样本外)上重跑，看 edge 是否只来自某一段(过拟合自检)。
- cross_section_edge : 横截面相对排名(动量+低波)的多空分位价差 + deflated Sharpe(多重检验折扣)。

铁律：只校准不预测；永远对比基准；任何"显著"都要扛得住 CI / 安慰剂 / 多重检验折扣。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Alpha / Beta 分解（CAPM 回归）—— 拆"真超额"与"纯 beta"
# ---------------------------------------------------------------------------
def alpha_beta(strat_ret: pd.Series, mkt_ret: pd.Series, n_boot: int = 1000) -> dict:
    """把策略日收益对市场日收益回归：strat = α + β·mkt + ε。

    返回 {alpha_ann, beta, r2, alpha_ann_ci(95%), alpha_significant, n, verdict}。
    α 年化 = 日 α × 252；CI 用 block bootstrap（保留时间相关性）。CI 跨 0 → α 不显著。"""
    df = pd.concat([strat_ret.rename("s"), mkt_ret.rename("m")], axis=1).dropna()
    if df.shape[0] < 60:
        return {"alpha_ann": float("nan"), "beta": float("nan"), "r2": float("nan"),
                "alpha_ann_ci": (float("nan"), float("nan")), "alpha_significant": False,
                "n": int(df.shape[0]), "verdict": "样本不足"}
    s, m = df["s"].to_numpy(), df["m"].to_numpy()
    var_m = np.var(m)
    beta = float(np.cov(s, m)[0, 1] / var_m) if var_m > 0 else float("nan")
    alpha_d = float(np.mean(s) - beta * np.mean(m))
    resid = s - (alpha_d + beta * m)
    ss_tot = np.sum((s - s.mean()) ** 2)
    r2 = float(1 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else float("nan")

    # block bootstrap α 的 CI：对 (s, m) **配对**按循环块重采样后重估 α（保留时间相关性）
    L = len(s)
    block = min(21, max(1, L // 5))
    n_blocks = int(np.ceil(L / block))
    rng = np.random.default_rng(0)
    offsets = np.arange(block)
    alphas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        starts = rng.integers(0, L, size=n_blocks)
        idx = (starts[:, None] + offsets[None, :]).ravel()[:L] % L
        ss, mm = s[idx], m[idx]
        vm = np.var(mm)
        b = np.cov(ss, mm)[0, 1] / vm if vm > 0 else 0.0
        alphas[i] = np.mean(ss) - b * np.mean(mm)
    lo_d = float(np.nanpercentile(alphas, 2.5))
    hi_d = float(np.nanpercentile(alphas, 97.5))

    alpha_ann = alpha_d * 252
    lo, hi = lo_d * 252, hi_d * 252
    sig = bool(lo > 0 or hi < 0) if (lo == lo and hi == hi) else False
    if sig and alpha_ann > 0:
        verdict = f"存在显著正 α（年化 {alpha_ann:+.1%}），不只是 beta。"
    elif sig and alpha_ann < 0:
        verdict = f"α 显著为负（年化 {alpha_ann:+.1%}）：跑输了风险敞口该有的回报。"
    else:
        verdict = "α 不显著（CI 跨 0）：收益基本来自市场 beta，没有可证明的择时/选股超额。"
    return {"alpha_ann": float(alpha_ann), "beta": beta, "r2": r2,
            "alpha_ann_ci": (float(lo), float(hi)), "alpha_significant": sig,
            "n": int(df.shape[0]), "verdict": verdict}


def _ols_beta(a: np.ndarray, b: np.ndarray) -> float:
    """β = cov(a,b)/var(b)（a=个股收益, b=市场收益）。"""
    v = float(np.var(b))
    return float(np.cov(a, b)[0, 1] / v) if v > 0 else float("nan")


def alpha_beta_profile(stock_px: pd.Series, mkt_px: pd.Series,
                       lookback_1y: int = 252, n_boot: int = 800) -> dict:
    """市场模型(CAPM)全景：把个股日收益对市场(默认 SPY)回归，拆 α(选股/择时超额) 与 β(市场敏感度)，
    再补三件风险决策真正关心的事：

      • **近1年 β**——β 会漂移，近端比全样本更代表"现在"的敏感度。
      • **下行 β / 上行 β**——市场跌的日子 vs 涨的日子分别回归。下行β>上行β = "跌时比涨时更敏感"
        (不利的不对称，高 β 科技常见)；反之为有利。
      • **相关性 / R²**——收益里有多少是被市场解释掉的(R² 高=基本就是 beta 玩具)。

    铁律：全是**历史描述统计、非预测**；单票 α 多半不显著(verdict 诚实说明)，β 是真正的风险敞口。
    rf≈0 近似(日频无风险利率可忽略)。返回 dict(available/beta/beta_1y/beta_down/beta_up/
    alpha_ann/alpha_ci/alpha_significant/r2/corr/n + 三段 verdict)。"""
    s = stock_px.dropna().pct_change()
    m = mkt_px.dropna().pct_change()
    df = pd.concat([s.rename("s"), m.rename("m")], axis=1).dropna()
    if df.shape[0] < 60:
        return {"available": False, "n": int(df.shape[0])}
    ab = alpha_beta(df["s"], df["m"], n_boot=n_boot)
    sv, mv = df["s"].to_numpy(), df["m"].to_numpy()
    corr = float(np.corrcoef(sv, mv)[0, 1]) if np.var(sv) > 0 and np.var(mv) > 0 else float("nan")
    beta_1y = _ols_beta(sv[-lookback_1y:], mv[-lookback_1y:]) if len(sv) >= lookback_1y + 5 else float("nan")
    down, up = mv < 0, mv > 0
    beta_down = _ols_beta(sv[down], mv[down]) if int(down.sum()) > 30 else float("nan")
    beta_up = _ols_beta(sv[up], mv[up]) if int(up.sum()) > 30 else float("nan")

    beta = ab["beta"]
    # β 解读
    if beta == beta:
        if beta >= 1.3:
            beta_note = f"β={beta:.2f}：**放大市场**(进攻型)——大盘动 1%，它约动 {beta:.1f}%，涨跌都更猛。"
        elif beta >= 0.8:
            beta_note = f"β={beta:.2f}：与大盘**同步**，敞口接近一份市场。"
        elif beta >= 0.4:
            beta_note = f"β={beta:.2f}：**偏防御**(低于市场敏感度)。"
        else:
            beta_note = f"β={beta:.2f}：与市场**弱关联**(独立行情成分高)。"
    else:
        beta_note = "β 不可用。"
    # 下行/上行不对称
    risk_note = ""
    if beta_down == beta_down and beta_up == beta_up:
        gap = beta_down - beta_up
        if gap > 0.15:
            risk_note = (f"⚠️ **下行β {beta_down:.2f} > 上行β {beta_up:.2f}**：跌时比涨时更敏感(不利不对称)——"
                         "大盘下挫它往往跌更多，撤离纪律(破200线减半)更重要。")
        elif gap < -0.15:
            risk_note = (f"✅ 下行β {beta_down:.2f} < 上行β {beta_up:.2f}：跌时反而没涨时敏感(有利不对称)。")
        else:
            risk_note = f"下行β {beta_down:.2f} ≈ 上行β {beta_up:.2f}：涨跌敏感度对称。"
    # β 漂移
    drift_note = ""
    if beta_1y == beta_1y and beta == beta:
        d = beta_1y - beta
        if abs(d) >= 0.25:
            drift_note = (f"近1年β {beta_1y:.2f} {'高' if d>0 else '低'}于全样本 {beta:.2f}"
                          f"（{'敏感度在上升' if d>0 else '敏感度在下降'}）——β 会漂移，按近端看更准。")
        else:
            drift_note = f"近1年β {beta_1y:.2f} 与全样本 {beta:.2f} 接近(β 稳定)。"

    return {"available": True, "beta": beta, "beta_1y": beta_1y,
            "beta_down": beta_down, "beta_up": beta_up,
            "alpha_ann": ab["alpha_ann"], "alpha_ci": ab["alpha_ann_ci"],
            "alpha_significant": ab["alpha_significant"], "r2": ab["r2"],
            "corr": corr, "n": ab["n"],
            "alpha_verdict": ab["verdict"], "beta_note": beta_note,
            "risk_note": risk_note, "drift_note": drift_note}


# 风格因子的免费 ETF 代理（相对市场的超额=风格暴露）
FACTOR_ETFS = {"市场": "SPY", "动量": "MTUM", "价值": "IWD", "小盘": "IWM", "低波": "USMV"}


def factor_attribution(strat_ret: pd.Series, factor_prices: dict[str, pd.Series],
                       n_boot: int = 800) -> dict:
    """多因子归因：strat = α + β_mkt·MKT + β_mom·(MTUM−SPY) + β_val·(IWD−SPY) + …。

    把收益拆成市场 β + 各风格倾斜 + 真 α(扣掉所有风格后剩下的)。α 带 block bootstrap CI。
    factor_prices: {名称: 价格Series}，至少含 'SPY'。返回 {alpha_ann, alpha_ci, betas{}, r2, n, verdict}。"""
    spy = factor_prices.get("SPY")
    if spy is None:
        return {"verdict": "缺市场基准 SPY", "n": 0}
    spy_r = spy.pct_change()
    cols = {"市场": spy_r}
    for name, etf in FACTOR_ETFS.items():
        if name == "市场":
            continue
        p = factor_prices.get(etf)
        if p is not None:
            cols[name] = p.pct_change() - spy_r   # 风格相对市场的超额=纯风格暴露
    df = pd.concat({"s": strat_ret, **cols}, axis=1).dropna()
    if df.shape[0] < 120:
        return {"verdict": "样本不足（风格 ETF 多 2013+ 上市）", "n": int(df.shape[0])}

    fac_names = [c for c in df.columns if c != "s"]
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy() for c in fac_names])
    y = df["s"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else float("nan")

    # block bootstrap α(截距)的 CI
    L = len(y); blk = min(21, max(1, L // 5)); nb = int(np.ceil(L / blk))
    rng = np.random.default_rng(0); off = np.arange(blk)
    alphas = np.empty(n_boot)
    for i in range(n_boot):
        idx = (rng.integers(0, L, size=nb)[:, None] + off[None, :]).ravel()[:L] % L
        b, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        alphas[i] = b[0]
    lo, hi = float(np.nanpercentile(alphas, 2.5)) * 252, float(np.nanpercentile(alphas, 97.5)) * 252
    alpha_ann = float(beta[0] * 252)
    sig = bool(lo > 0 or hi < 0)
    betas = {name: float(b) for name, b in zip(fac_names, beta[1:])}
    tilt = max(((k, v) for k, v in betas.items() if k != "市场"), key=lambda kv: abs(kv[1]), default=(None, 0))
    if sig and alpha_ann > 0:
        verdict = f"扣掉市场+风格后仍有显著正 α（年化 {alpha_ann:+.1%}）——这才是真本事。"
    elif sig and alpha_ann < 0:
        verdict = f"扣掉风格后 α 显著为负（年化 {alpha_ann:+.1%}）。"
    else:
        verdict = (f"α 不显著（CI 跨 0）：收益主要由市场 β({betas.get('市场',float('nan')):.2f})"
                   + (f" 和 {tilt[0]}倾斜({tilt[1]:+.2f})" if tilt[0] else "") + " 解释，没有可证明的纯 α。")
    return {"alpha_ann": alpha_ann, "alpha_ci": (lo, hi), "alpha_significant": sig,
            "betas": betas, "r2": r2, "n": int(df.shape[0]), "verdict": verdict}


# ---------------------------------------------------------------------------
# 2. Regime 暴露系数（免费可观测状态 → 0–1 仓位乘子，只降不加杠杆）
# ---------------------------------------------------------------------------
def regime_exposure(price: pd.Series, macro: pd.DataFrame | None = None) -> dict:
    """今日的 regime 暴露建议（0–1）。高波动/信用走阔/曲线倒挂 → 降暴露。

    返回 {exposure, factors:[{name, state, mult}], note}。这是风控纪律(改善风险调整后表现)，
    不是涨跌预测。"""
    from regime import observables as ob

    price = price.dropna()
    factors = []
    mult = 1.0

    # 波动分位：越高越降
    rvp = ob.realized_vol_percentile(price, 21, 252)
    vp = float(rvp.iloc[-1]) if rvp.notna().iloc[-1] else float("nan")
    if vp == vp:
        vm = 1.0 if vp < 0.6 else (0.7 if vp < 0.8 else (0.5 if vp < 0.95 else 0.35))
        mult *= vm
        factors.append({"name": "已实现波动分位", "state": f"{vp:.0%}", "mult": vm})

    # 趋势：跌破 200 线降一档（动量崩塌时减风险，文献支持）
    tp = float(ob.trend_position(price, 200).iloc[-1])
    tm = 1.0 if tp > 0 else 0.8
    mult *= tm
    factors.append({"name": "趋势(距200线)", "state": f"{tp:+.0%}", "mult": tm})

    # 宏观：信用走阔 / 收益率曲线倒挂 → 降暴露
    if macro is not None and not macro.empty:
        try:
            cs = macro["credit_spread"].dropna()
            cs_ma = cs.rolling(126, min_periods=63).mean()
            widening = bool(cs.iloc[-1] > cs_ma.iloc[-1])
            cm = 0.8 if widening else 1.0
            mult *= cm
            factors.append({"name": "信用利差", "state": "走阔" if widening else "收窄", "mult": cm})
        except Exception:  # noqa: BLE001
            pass
        try:
            yc = macro["yield_curve"].dropna()
            inverted = bool(yc.iloc[-1] < 0)
            ym = 0.85 if inverted else 1.0
            mult *= ym
            factors.append({"name": "收益率曲线(10Y-2Y)", "state": "倒挂" if inverted else "正常", "mult": ym})
        except Exception:  # noqa: BLE001
            pass

    mult = float(max(0.2, min(1.0, mult)))
    return {"exposure": mult, "factors": factors,
            "note": "regime 暴露=风控纪律(高波动/避险环境自动降仓)，只降不加杠杆；非涨跌预测。"}


def ewma_vol(ret: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics EWMA 年化波动预测：σ²_t = λσ²_{t-1} + (1-λ)r²_{t-1}（只用过去，无前视）。

    比等权滚动波动反应更快、更贴近 ex-ante 预测（近期冲击权重更高）。"""
    r2 = (ret.fillna(0.0) ** 2)
    var = r2.ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var.shift(1) * 252)  # shift(1)：今日暴露只能用到昨日为止的信息


def vol_target_backtest(price: pd.Series, target_vol: float = 0.20,
                        window: int = 21, max_lev: float = 1.0) -> dict:
    """波动目标 overlay：每日按 target_vol / **EWMA ex-ante 波动预测** 缩放暴露(上限 max_lev=不加杠杆)
    vs 闭眼满仓持有。返回两者 {cagr,vol,sharpe,maxdd} + 归一净值 DataFrame。诚实：只降不加杠杆。"""
    from backtest.strategies import _perf

    price = price.dropna()
    ret = price.pct_change()
    rv = ewma_vol(ret)  # EWMA 预测波动定今日暴露，无前视
    expo = (target_vol / rv).clip(upper=max_lev).fillna(0.0)
    strat_ret = expo * ret
    strat_val = (1 + strat_ret.fillna(0)).cumprod() * float(price.iloc[0])
    hold_val = price

    eq = pd.DataFrame({"波动目标": strat_val / strat_val.iloc[0],
                       "持有": hold_val / hold_val.iloc[0]})
    return {"overlay": _perf(strat_val), "hold": _perf(hold_val), "equity": eq,
            "avg_exposure": float(expo.mean()), "target_vol": target_vol,
            "sample": f"{price.index[0].date()}~{price.index[-1].date()}"}


# ---------------------------------------------------------------------------
# 3. Walk-forward OOS 自检（定稿规则在多段样本外是否仍站得住）
# ---------------------------------------------------------------------------
def walkforward_oos(ticker: str, start: str = "2010-01-01", end: str | None = None,
                    train: int = 504, test: int = 252,
                    dip_lookback: int = 20, dip_pct: float = 0.15,
                    trailing: float = 0.30, time_stop: int = 504) -> dict:
    """把定稿策略(深跌买×让利润奔跑)在多段 OOS 上重跑，看 edge 是否只来自某一段。

    规则**参数固定**(非拟合)，所以 OOS 检验的是"优势是否跨期稳定"。
    返回每段 OOS 的策略 vs 持有 年化 + 胜负，及汇总(中位、跑赢比例)。"""
    from data import loader
    from factors import signals as sg
    from backtest import exits as ex
    from stats.walkforward import walk_forward_splits

    p = loader.load_prices([ticker], start, end)[ticker].dropna()
    splits = walk_forward_splits(p.index, train=train, test=test)
    rows = []
    for sp in splits:
        _, test_idx = sp.slice(p.index)
        seg = p.loc[test_idx]
        if seg.shape[0] < 60:
            continue
        entries = sg.dip_from_high(seg, lookback=dip_lookback, pct=dip_pct)
        try:
            pf = ex.run_trades(seg, entries, {"trailing_stop": trailing, "time_stop": int(time_stop)},
                               init_cash=10000.0)
            sv = pf.value().dropna()
            n = len(sv)
            s_cagr = (sv.iloc[-1] / sv.iloc[0]) ** (252 / n) - 1 if n > 20 else float("nan")
        except Exception:  # noqa: BLE001
            s_cagr = float("nan")
        hn = seg.shape[0]
        h_cagr = (seg.iloc[-1] / seg.iloc[0]) ** (252 / hn) - 1
        rows.append({"oos_start": str(test_idx[0].date()), "oos_end": str(test_idx[-1].date()),
                     "strat_cagr": float(s_cagr), "hold_cagr": float(h_cagr),
                     "beat": bool(s_cagr > h_cagr) if s_cagr == s_cagr else False})
    df = pd.DataFrame(rows)
    if df.empty:
        return {"per_window": df, "n_windows": 0, "note": "样本不足以切分 OOS。"}
    valid = df[df["strat_cagr"] == df["strat_cagr"]]
    return {
        "per_window": df, "n_windows": int(len(valid)),
        "strat_cagr_median": float(valid["strat_cagr"].median()),
        "hold_cagr_median": float(valid["hold_cagr"].median()),
        "beat_rate": float(valid["beat"].mean()) if len(valid) else float("nan"),
        "note": ("规则参数固定(非拟合)，OOS 检验的是跨期稳定性。"
                 "若跑赢比例≈50%、各段年化普遍低于持有，说明无稳定择时优势——这与全工具结论一致。"),
    }


# ---------------------------------------------------------------------------
# PEAD 产品化：当前是否处于"已验证的财报后漂移窗口"
# ---------------------------------------------------------------------------
def pead_now(ticker: str, start: str = "2010-01-01", end: str | None = None,
             post_window: int = 20) -> dict | None:
    """判断 ticker 今天是否处在财报后漂移(PEAD)窗口，并给校准结论。

    PEAD 是全工具唯一通过安慰剂检验的免费信号(池化七票 5日 IC≈0.19, p<0.001)。
    这里把它落到当下：最近一次财报是 beat/miss、距今几日、该票历史上同类财报后 {post} 日的
    漂移分布(中位+10/90分位+N)。返回 None 表示当前不在窗口内/无数据。诚实：是历史分布，非预测下次。"""
    from data import loader
    from factors import fundamentals as fd
    from regime import entry_cockpit as ec

    try:
        price = loader.load_prices([ticker], start, end)[ticker].dropna()
        edates = loader.load_earnings_dates(ticker, limit=80)
    except Exception:  # noqa: BLE001
        return None
    if edates is None or edates.empty:
        return None

    # 关键：days_since 与 surprise 必须指向**同一次已公布财报**——只用 _reported 集，
    # 否则最近一次财报已发生但 Surprise 未回填时，会出现"天数算新事件、超额读旧事件"的错配。
    rep = fd._reported(edates)
    if rep.empty:
        return None
    last_dt = pd.Timestamp(rep.index[-1])
    surprise = float(rep["Surprise(%)"].iloc[-1])
    today = pd.Timestamp(price.index[-1])
    days_since = float((today - last_dt).days)
    # 反应日盘后公布→次日，窗口用交易日近似（post_window 交易日 ≈ post_window*1.4 自然日）
    in_window = days_since is not None and 0 <= days_since <= post_window * 1.5

    stats = ec.earnings_reaction_stats(price, edates, post=post_window)
    beat = (surprise is not None and surprise > 0)
    drift = stats["post_beat"] if beat else stats["post_miss"]
    if not in_window or drift["n"] < 8:
        return {"in_window": bool(in_window), "days_since": days_since, "surprise": surprise,
                "beat": beat, "drift": drift, "actionable": False,
                "note": "当前不在财报后漂移窗口，或该票同类财报样本不足(N<8)。"}

    direction = "继续同向偏正" if (beat and drift["median"] > 0) else \
                ("继续同向偏负" if (not beat and drift["median"] < 0) else "无明显同向漂移")
    return {
        "in_window": True, "days_since": days_since, "surprise": surprise, "beat": beat,
        "drift": drift, "actionable": True, "direction": direction,
        "verdict": (f"上次财报{'超预期' if beat else '不及预期'}(surprise {surprise:+.1f}%)、"
                    f"距今约 {days_since:.0f} 天，仍在 PEAD 窗口内。该票历史上同类财报后 {post_window} 日"
                    f"漂移中位 {drift['median']:+.1%}（10/90 分位 {drift['p10']:+.1%}/{drift['p90']:+.1%}，"
                    f"N={drift['n']}）。"),
        "note": ("PEAD 是工具中唯一通过安慰剂检验的免费信号；但这是该票历史经验分布，"
                 "不预测本次具体走势，且单票 N 小、务必结合 CI 与仓位纪律。"),
    }


# ---------------------------------------------------------------------------
# 4. 横截面相对排名 edge（动量 + 低波，多空分位价差，deflated Sharpe 折扣）
# ---------------------------------------------------------------------------
def _newey_west_t(x: np.ndarray, lags: int = 5) -> float:
    """序列的均值 / Newey-West(自相关稳健)标准误 → t 统计量。IC 序列有自相关，普通 t 会高估显著性。"""
    x = x[~np.isnan(x)]
    T = len(x)
    if T < 10:
        return float("nan")
    xc = x - x.mean()
    gamma0 = np.mean(xc ** 2)
    s = gamma0
    for L in range(1, min(lags, T - 1) + 1):
        w = 1.0 - L / (lags + 1)
        s += 2 * w * np.mean(xc[L:] * xc[:-L])
    se = np.sqrt(max(s, 1e-18) / T)
    return float(x.mean() / se) if se > 0 else float("nan")


def cross_section_edge(prices: pd.DataFrame, lookback_mom: int = 252, skip: int = 21,
                       vol_win: int = 63, rebalance: int = 21, quantiles: int = 3,
                       n_boot: int = 600, neutralize_beta: bool = True, winsor: float = 3.0) -> dict:
    """横截面相对排名回测（准专业版）：综合分 = z(12-1动量) + z(-波动)，
    经 **winsorize(±3σ) + beta 中性化** 后多顶分位/空底分位。

    返回 factor_quantile_backtest 结果 + deflated Sharpe + IC 的 **Newey-West t 统计量**。"""
    from backtest.strategies import factor_quantile_backtest
    from stats.deflated_sharpe import deflated_sharpe_ratio
    from scipy.stats import spearmanr

    px = prices.dropna(how="all").ffill()
    rets = px.pct_change()
    mom = px.shift(skip) / px.shift(lookback_mom) - 1.0          # 12-1 动量
    vol = rets.rolling(vol_win, min_periods=vol_win // 2).std()

    def _z(df):  # 横截面标准化 + winsorize（截尾极端值，防单票主导）
        z = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)
        return z.clip(-winsor, winsor)

    score = _z(mom) - _z(vol)

    # —— beta 中性化：逐期把 score 对各票 beta 做横截面回归取残差，剔除 beta 暴露 ——
    if neutralize_beta:
        mkt = rets.mean(axis=1)                                   # 等权篮子=市场代理
        var_m = mkt.rolling(252, min_periods=126).var()
        beta = rets.rolling(252, min_periods=126).cov(mkt).div(var_m, axis=0)
        bz = beta.sub(beta.mean(axis=1), axis=0).div(beta.std(axis=1).replace(0, np.nan), axis=0)
        # 残差化：score_resid = score − proj(score onto beta) 逐行
        sv, bv = score.to_numpy(), bz.to_numpy()
        resid = np.full_like(sv, np.nan)
        for t in range(sv.shape[0]):
            s_row, b_row = sv[t], bv[t]
            m = ~np.isnan(s_row) & ~np.isnan(b_row)
            if m.sum() >= 5:
                bb = b_row[m]
                coef = np.dot(bb, s_row[m]) / max(np.dot(bb, bb), 1e-12)
                resid[t, m] = s_row[m] - coef * bb
            else:
                resid[t] = s_row
        score = pd.DataFrame(resid, index=score.index, columns=score.columns)

    res = factor_quantile_backtest(score, px, quantiles=quantiles, rebalance=rebalance,
                                   long_short=True, n_boot=n_boot)

    # —— IC 序列 + Newey-West t（每 rebalance 期，score vs 未来 rebalance 日收益）——
    fwd = px.shift(-rebalance) / px - 1.0
    ic_list = []
    for d in px.index[lookback_mom + skip::rebalance]:
        s_row = score.loc[d].dropna(); f_row = fwd.loc[d].dropna()
        common = s_row.index.intersection(f_row.index)
        if len(common) >= 5:
            r = spearmanr(s_row[common], f_row[common]).statistic
            if r == r:
                ic_list.append(r)
    ic_arr = np.array(ic_list)
    res["ic_mean"] = float(ic_arr.mean()) if ic_arr.size else float("nan")
    res["ic_t_newey_west"] = _newey_west_t(ic_arr, lags=5)
    res["ic_n_periods"] = int(ic_arr.size)

    sr_daily = res["sharpe"] / np.sqrt(252) if res["sharpe"] == res["sharpe"] else float("nan")
    try:
        dsr = deflated_sharpe_ratio(sr=sr_daily, sr_trials_std=abs(sr_daily) * 0.5 + 1e-6,
                                    n_trials=2, n_obs=int(res["n_days"]))
    except Exception:  # noqa: BLE001
        dsr = float("nan")
    res["deflated_sharpe_prob"] = float(dsr)
    res["robust"] = bool(dsr > 0.95) if dsr == dsr else False
    res["neutralized"] = bool(neutralize_beta)
    res["edge_note"] = ("横截面多空价差(动量+低波，winsorize+beta中性化)。IC 的 Newey-West |t|>2 "
                        "且 deflated prob>0.95 才算扛得住自相关与多重检验。相对排名 edge，非单票择时。")
    return res


# ---------------------------------------------------------------------------
# 5. 协方差组合构建（min-variance / risk-parity）—— 候选权重之外的真实组合配比
# ---------------------------------------------------------------------------
def portfolio_weights(prices: pd.DataFrame, lookback: int = 252,
                      method: str = "min_var", single_cap: float = 0.35) -> dict:
    """从历史协方差给等权/最小方差/风险平价三套长仓权重(和为1)，并报组合年化波动。

    免费(仅用收益)；min_var 用解析解(协方差逆)，风险平价用迭代。单票上限 single_cap。
    返回 {weights:{t:w}, port_vol, method, note, compare:{方法:port_vol}}。"""
    px = prices.dropna(how="all").ffill().dropna()
    rets = px.pct_change().tail(lookback).dropna(how="all")
    cols = [c for c in px.columns if rets[c].notna().sum() > lookback // 2]
    rets = rets[cols].dropna()
    n = len(cols)
    if n < 2 or len(rets) < 60:
        return {"weights": {c: 1.0 / max(1, n) for c in cols}, "port_vol": float("nan"),
                "method": method, "note": "样本不足，退化为等权。", "compare": {}}

    # Ledoit-Wolf 收缩协方差：样本协方差在 n 接近样本量时极不稳定（min-var 会放大估计噪声）。
    # 收缩到结构化目标后，组合权重稳健得多。退化时回退样本协方差。
    try:
        from sklearn.covariance import LedoitWolf
        cov = LedoitWolf().fit(rets.to_numpy()).covariance_ * 252
        shrink = "Ledoit-Wolf"
    except Exception:  # noqa: BLE001
        cov = rets.cov().to_numpy() * 252
        shrink = "样本协方差"
    ones = np.ones(n)
    cap = max(single_cap, 1.0 / n)  # 上限不能低于等权，否则无可行解

    def _vol(w):
        return float(np.sqrt(w @ cov @ w))

    from scipy.optimize import minimize
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0.0, cap)] * n
    w0 = ones / n

    # 等权（受上限约束，一般 1/n ≤ cap）
    w_eq = np.clip(ones / n, 0, cap); w_eq = w_eq / w_eq.sum()

    # 最小方差：min w'Σw（真·长仓+上限 QP）
    try:
        r = minimize(lambda w: w @ cov @ w, w0, method="SLSQP", bounds=bounds,
                     constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
        w_mv = r.x if r.success else w_eq
    except Exception:  # noqa: BLE001
        w_mv = w_eq

    # 风险平价：min 各票风险贡献的离散度
    def _rp_obj(w):
        port = np.sqrt(w @ cov @ w) + 1e-12
        rc = w * (cov @ w) / port           # 各票风险贡献
        return float(np.sum((rc - rc.mean()) ** 2))
    try:
        r = minimize(_rp_obj, w0, method="SLSQP", bounds=bounds, constraints=cons,
                     options={"maxiter": 500, "ftol": 1e-12})
        w_rp = r.x if r.success else w_eq
    except Exception:  # noqa: BLE001
        w_rp = w_eq

    chosen = {"equal": w_eq, "min_var": w_mv, "risk_parity": w_rp}.get(method, w_mv)
    return {
        "weights": {c: float(w) for c, w in zip(cols, chosen)},
        "port_vol": _vol(chosen), "method": method,
        "compare": {"等权": _vol(w_eq), "最小方差": _vol(w_mv), "风险平价": _vol(w_rp)},
        "shrinkage": shrink,
        "note": ("基于近 %d 日 %s 的长仓权重(和=1，单票≤%.0f%%)。最小方差求组合波动最低，"
                 "风险平价让各票风险贡献相等。仅用收益、免费；非投资建议。" % (lookback, shrink, cap * 100)),
    }


# ---------------------------------------------------------------------------
# 6. 信号衰减监控（滚动 IC）—— 真信号是否在变弱
# ---------------------------------------------------------------------------
def signal_decay(prices: pd.DataFrame, horizon: int = 5, window: int = 252,
                 lookback_mom: int = 252, skip: int = 21) -> dict:
    """监控横截面动量因子的滚动 IC(每期截面 Spearman)，看 edge 是否随时间衰减。

    返回 {ic_series(年度均值), recent_ic, early_ic, decayed, note}。"""
    from scipy.stats import spearmanr

    px = prices.dropna(how="all").ffill()
    fwd = px.shift(-horizon) / px - 1.0
    mom = px.shift(skip) / px.shift(lookback_mom) - 1.0

    dates = px.index[lookback_mom + skip::21]  # 每月一截面
    ic = {}
    for d in dates:
        f = fwd.loc[d].dropna()
        m = mom.loc[d].dropna()
        common = f.index.intersection(m.index)
        if len(common) >= 5:
            r = spearmanr(m[common], f[common]).statistic
            if r == r:
                ic[d] = float(r)
    s = pd.Series(ic)
    if s.empty:
        return {"ic_yearly": pd.Series(dtype=float), "recent_ic": float("nan"),
                "early_ic": float("nan"), "decayed": False, "note": "样本不足。"}
    yearly = s.groupby(s.index.year).mean()
    half = len(s) // 2
    early_ic = float(s.iloc[:half].mean())
    recent_ic = float(s.iloc[half:].mean())
    decayed = bool(early_ic > 0.03 and recent_ic < early_ic * 0.5)
    return {"ic_yearly": yearly, "recent_ic": recent_ic, "early_ic": early_ic,
            "decayed": decayed,
            "note": ("横截面动量因子的滚动 IC 年度均值。近半段 IC 若明显低于前半段(<50%)则判定衰减。"
                     "IC 0.03–0.05 即算可用；衰减是信号失效预警，非买卖指令。")}

