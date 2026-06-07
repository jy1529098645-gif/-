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


def vol_target_backtest(price: pd.Series, target_vol: float = 0.20,
                        window: int = 21, max_lev: float = 1.0) -> dict:
    """波动目标 overlay：每日按 target_vol / 近 window 日已实现波动 缩放暴露(上限 max_lev=不加杠杆)
    vs 闭眼满仓持有。返回两者 {cagr,vol,sharpe,maxdd} + 归一净值 DataFrame。诚实：只降不加杠杆。"""
    from regime import observables as ob
    from backtest.strategies import _perf

    price = price.dropna()
    ret = price.pct_change()
    rv = ob.realized_vol(price, window).shift(1)  # 用昨日波动定今日暴露，无前视
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

    idx = price.index
    dsl = fd.days_since_last_earnings(idx, edates)
    ls = fd.last_surprise(idx, edates)
    days_since = float(dsl.iloc[-1]) if dsl.notna().iloc[-1] else None
    surprise = float(ls.iloc[-1]) if ls.notna().iloc[-1] else None
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
def cross_section_edge(prices: pd.DataFrame, lookback_mom: int = 252, skip: int = 21,
                       vol_win: int = 63, rebalance: int = 21, quantiles: int = 3,
                       n_boot: int = 600) -> dict:
    """一篮子标的的横截面相对排名回测：综合分 = z(12-1动量) + z(-波动)，多顶分位/空底分位。

    返回 factor_quantile_backtest 结果 + deflated Sharpe（按 2 个因子做多重检验折扣）。"""
    from backtest.strategies import factor_quantile_backtest
    from stats.deflated_sharpe import deflated_sharpe_ratio

    px = prices.dropna(how="all").ffill()
    rets = px.pct_change()
    # 12-1 动量（跳过最近 skip 日，避开短期反转）
    mom = px.shift(skip) / px.shift(lookback_mom) - 1.0
    vol = rets.rolling(vol_win, min_periods=vol_win // 2).std()

    def _z(df):
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)

    score = _z(mom) - _z(vol)  # 高动量 + 低波动 得分高
    res = factor_quantile_backtest(score, px, quantiles=quantiles, rebalance=rebalance,
                                   long_short=True, n_boot=n_boot)
    # deflated Sharpe：2 个因子搜索 → 折扣
    sr_daily = res["sharpe"] / np.sqrt(252) if res["sharpe"] == res["sharpe"] else float("nan")
    try:
        dsr = deflated_sharpe_ratio(sr=sr_daily, sr_trials_std=abs(sr_daily) * 0.5 + 1e-6,
                                    n_trials=2, n_obs=int(res["n_days"]))
    except Exception:  # noqa: BLE001
        dsr = float("nan")
    res["deflated_sharpe_prob"] = float(dsr)
    res["robust"] = bool(dsr > 0.95) if dsr == dsr else False
    res["edge_note"] = ("横截面多空价差的夏普；deflated prob>0.95 才算扛得住多重检验。"
                        "这是相对排名 edge(谁强于谁)，与单票择时是两回事。")
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

    cov = rets.cov().to_numpy() * 252
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
        "note": ("基于近 %d 日协方差的长仓权重(和=1，单票≤%.0f%%)。最小方差求组合波动最低，"
                 "风险平价让各票风险贡献相等。仅用收益、免费；非投资建议。" % (lookback, cap * 100)),
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

