"""前端交互图（plotly，暗色主题）。

定位铁律（补充规格 §6）：这些是"决策与验证"图，不是看盘终端——展示分布/CI/N，
强制标注样本数，不标"目标价/最佳买卖点"。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# 调色板（与暗色主题协调）
C = {
    "accent": "#7C5CFC",
    "accent2": "#00D4FF",
    "win": "#2BE6A8",
    "loss": "#FF5C7A",
    "baseline": "#8A93A6",
    "band": "rgba(124,92,252,0.18)",
    "band2": "rgba(0,212,255,0.12)",
    "grid": "rgba(255,255,255,0.07)",
    "text": "#E6E9EF",
}


def _style(fig: go.Figure, height: int = 380, title: str = "") -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C["text"], family="Inter, -apple-system, Segoe UI, sans-serif", size=13),
        margin=dict(l=10, r=10, t=46 if title else 16, b=10),
        height=height,
        title=dict(text=title, x=0.01, font=dict(size=15)),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.08, x=0),
        hoverlabel=dict(font_size=12),
        dragmode="pan", spikedistance=-1, hoverdistance=100,  # TradingView 手感：拖动平移
    )
    # TradingView 式十字光标（spike）：悬浮时轴上出现贴近光标的虚线，读价/读轴
    _spk = dict(showspikes=True, spikesnap="cursor", spikecolor="rgba(255,255,255,0.45)",
                spikethickness=1, spikedash="dot")
    fig.update_xaxes(gridcolor=C["grid"], zerolinecolor=C["grid"], spikemode="across", **_spk)
    fig.update_yaxes(gridcolor=C["grid"], zerolinecolor=C["grid"], **_spk)
    return fig


def return_hist(returns, median=None, baseline_median=None, n=None, n_eff=None, title="") -> go.Figure:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    fig = go.Figure()
    fig.add_histogram(x=r, nbinsx=min(50, max(10, len(r) // 3)), marker_color=C["accent"],
                      opacity=0.8, name="单笔收益")
    if median is None:
        median = float(np.median(r)) if r.size else 0.0
    fig.add_vline(x=median, line=dict(color=C["accent2"], width=2, dash="dash"),
                  annotation_text=f"中位 {median:+.1%}", annotation_position="top")
    if baseline_median is not None:
        fig.add_vline(x=baseline_median, line=dict(color=C["baseline"], width=2, dash="dot"),
                      annotation_text=f"基准 {baseline_median:+.1%}", annotation_position="bottom")
    fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.3)", width=1))
    sub = f"N={n}" + (f" · N_eff≈{n_eff:.0f}" if n_eff else "")
    return _style(fig, title=f"{title}　{sub}".strip())


def mae_hist(mae, n=None, title="途中最大浮亏 MAE") -> go.Figure:
    m = np.asarray(mae, dtype=float)
    m = m[~np.isnan(m)]
    fig = go.Figure()
    fig.add_histogram(x=m, nbinsx=min(50, max(10, len(m) // 3)), marker_color=C["loss"], opacity=0.75)
    if m.size:
        fig.add_vline(x=float(np.median(m)), line=dict(color=C["accent2"], width=2, dash="dash"),
                      annotation_text=f"中位 {np.median(m):+.1%}")
    return _style(fig, title=f"{title}　N={n or m.size}")


def corr_heatmap(corr: pd.DataFrame, title="标的相关矩阵") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index),
        colorscale="Tealrose", zmin=0, zmax=1, reversescale=True,
        text=np.round(corr.values, 2), texttemplate="%{text}", textfont=dict(size=11),
        colorbar=dict(thickness=12),
    ))
    return _style(fig, height=420, title=title)


def forward_cone(horizons, p10, p25, med, p75, p90, baseline_med=None, title="远期收益锥") -> go.Figure:
    x = list(horizons)
    fig = go.Figure()
    fig.add_scatter(x=x, y=p90, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=x, y=p10, mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor=C["band"], name="10–90 分位")
    fig.add_scatter(x=x, y=p75, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=x, y=p25, mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor=C["band2"], name="25–75 分位")
    fig.add_scatter(x=x, y=med, mode="lines+markers", line=dict(color=C["accent2"], width=3), name="中位")
    if baseline_med is not None:
        fig.add_scatter(x=x, y=baseline_med, mode="lines", line=dict(color=C["baseline"], width=2, dash="dot"),
                        name="无条件基准中位")
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.25)", width=1))
    fig.update_layout(yaxis_tickformat=".0%")
    fig.update_xaxes(title="远期天数")
    return _style(fig, title=title)


def event_study(study: dict, title="财报日历结构") -> go.Figure:
    off = list(study["offsets"])
    fig = go.Figure()
    fig.add_scatter(x=off, y=study["p90"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=off, y=study["p10"], mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor=C["band"], name="10–90 分位")
    fig.add_scatter(x=off, y=study["mean_car"], mode="lines", line=dict(color=C["accent2"], width=3),
                    name="平均 CAR")
    if study.get("mean_car_beat") is not None:
        fig.add_scatter(x=off, y=study["mean_car_beat"], mode="lines", line=dict(color=C["win"], width=2),
                        name=f"超预期 (N={study.get('n_beat','')})")
    if study.get("mean_car_miss") is not None:
        fig.add_scatter(x=off, y=study["mean_car_miss"], mode="lines", line=dict(color=C["loss"], width=2),
                        name=f"不及预期 (N={study.get('n_miss','')})")
    fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.4)", width=1.5, dash="dash"),
                  annotation_text="财报反应日")
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", width=1))
    fig.update_layout(yaxis_tickformat=".1%")
    fig.update_xaxes(title="相对财报反应日的交易日偏移")
    return _style(fig, height=440, title=f"{title}　N={study['n_events']} 事件")


def ic_bars(ic_table: pd.DataFrame, real_col="ic_real", title="PEAD 领先 IC") -> go.Figure:
    fig = go.Figure()
    colors = [C["win"] if v > 0 else C["loss"] for v in ic_table[real_col]]
    fig.add_bar(x=[f"{h}日" for h in ic_table["horizon"]], y=ic_table[real_col], marker_color=colors, name="真实 IC")
    if "ic_fake_mean" in ic_table.columns:
        fig.add_scatter(x=[f"{h}日" for h in ic_table["horizon"]], y=ic_table["ic_fake_mean"],
                        mode="markers", marker=dict(color=C["baseline"], size=9, symbol="x"),
                        name="假财报日对照(≈0)")
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.3)", width=1))
    return _style(fig, title=title)


def factor_ic_bars(ic_summary: pd.DataFrame, title="因子 IC（按远期周期）") -> go.Figure:
    fig = go.Figure()
    colors = [C["win"] if v > 0 else C["loss"] for v in ic_summary["IC_mean"]]
    fig.add_bar(x=list(ic_summary.index), y=ic_summary["IC_mean"], marker_color=colors,
                error_y=dict(type="data", array=ic_summary["IC_std"] / np.sqrt(ic_summary["n_days"].clip(lower=1))),
                name="IC 均值")
    fig.add_hline(y=0.03, line=dict(color=C["accent2"], width=1, dash="dot"), annotation_text="可用阈值 0.03")
    fig.add_hline(y=-0.03, line=dict(color=C["accent2"], width=1, dash="dot"))
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.3)", width=1))
    return _style(fig, title=title)


def strategy_compare(per_strategy: dict, title="建仓策略：资本回报中位 + 95% CI") -> go.Figure:
    names = list(per_strategy.keys())
    med = [per_strategy[k]["median"] for k in names]
    lo = [per_strategy[k]["median"] - per_strategy[k]["median_ci_low"] for k in names]
    hi = [per_strategy[k]["median_ci_high"] - per_strategy[k]["median"] for k in names]
    fig = go.Figure()
    fig.add_bar(x=names, y=med, marker_color=[C["accent"], C["accent2"], C["win"]][: len(names)],
                error_y=dict(type="data", symmetric=False, array=hi, arrayminus=lo))
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.3)", width=1))
    fig.update_layout(yaxis_tickformat=".0%")
    return _style(fig, title=title)


def ladder_risk_return(per_strategy: dict, title="建仓方案：回报 vs 建仓期痛感（越靠右下越好）") -> go.Figure:
    """三方案的风险-回报权衡散点：x=建仓期最深浮亏(痛感)，y=中位资本回报。右下角=高回报+低痛感。"""
    cn = {"lump_sum": "一次性全投", "dca": "定投(DCA)", "ladder": "越跌越补(阶梯)"}
    colors = {"lump_sum": C["accent"], "dca": C["accent2"], "ladder": C["win"]}
    fig = go.Figure()
    for k, s in per_strategy.items():
        pain = abs(s.get("mdd_median", float("nan")))
        fig.add_scatter(
            x=[pain], y=[s["median"]], mode="markers+text",
            marker=dict(size=22, color=colors.get(k, C["accent"]), line=dict(width=1, color="white")),
            text=[cn.get(k, k)], textposition="top center",
            name=cn.get(k, k),
            hovertemplate=(f"<b>{cn.get(k,k)}</b><br>中位回报 %{{y:.1%}}<br>建仓期最深浮亏 −%{{x:.1%}}<br>"
                          f"最坏窗口 {s.get('p5',float('nan')):.0%}<br>投满用时 {s.get('deploy_days_median',0):.0f} 天<extra></extra>"),
        )
    fig.update_xaxes(title="建仓期最深浮亏（越左越不痛）", tickformat=".0%", autorange="reversed")
    fig.update_yaxes(title="中位资本回报（越上越赚）", tickformat=".0%")
    return _style(fig, title=title)


def walk_forward(wf: pd.DataFrame, title="Walk-forward IS vs OOS") -> go.Figure:
    fig = go.Figure()
    x = [str(s) for s in wf["oos_start"]]
    fig.add_bar(x=x, y=wf["is_sharpe"], name="IS（样本内）", marker_color=C["baseline"])
    fig.add_bar(x=x, y=wf["oos_sharpe"], name="OOS（样本外）", marker_color=C["accent2"])
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.3)", width=1))
    fig.update_layout(barmode="group")
    return _style(fig, title=title)


# TradingView 式时间区间按钮
_RANGE_BUTTONS = dict(
    buttons=[
        dict(count=1, label="1M", step="month", stepmode="backward"),
        dict(count=3, label="3M", step="month", stepmode="backward"),
        dict(count=6, label="6M", step="month", stepmode="backward"),
        dict(count=1, label="YTD", step="year", stepmode="todate"),
        dict(count=1, label="1Y", step="year", stepmode="backward"),
        dict(count=3, label="3Y", step="year", stepmode="backward"),
        dict(step="all", label="全部"),
    ],
    bgcolor="rgba(124,92,252,0.14)", activecolor="#7C5CFC",
    font=dict(color="#E6E9EF", size=11), x=0, y=1.12,
)

# st.plotly_chart 用的交互配置（滚轮缩放 + 去掉画线/套索，守住"非画线终端"）
CHART_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "drawline", "drawopenpath",
                               "drawclosedpath", "drawcircle", "drawrect", "eraseshape"],
}

# TradingView 手感配置：滚轮缩放、双击复位、工具栏仅悬浮出现(平时干净)、只留平移/缩放/复位
TV_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "displayModeBar": "hover",
    "doubleClick": "reset",
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "zoomIn2d", "zoomOut2d", "toggleSpikelines",
                               "hoverClosestCartesian", "hoverCompareCartesian",
                               "drawline", "drawopenpath", "drawclosedpath", "drawcircle", "drawrect", "eraseshape"],
}


def _apply_tv(fig: go.Figure, height: int, title: str) -> go.Figure:
    """TradingView 式手感：暗色 + 十字光标(crosshair) + 拖动平移 + 统一悬浮。"""
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C["text"], family="Inter, Segoe UI, sans-serif", size=12),
        margin=dict(l=8, r=8, t=54, b=8), height=height,
        title=dict(text=title, x=0.01, font=dict(size=14)),
        hovermode="x unified", dragmode="pan",
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.06, x=0.18, font=dict(size=10)),
        spikedistance=-1, hoverdistance=1,
    )
    # TradingView 式十字光标：贴光标的虚线 + 轴上价/时标签（spikemode across+marker）
    fig.update_xaxes(gridcolor=C["grid"], showspikes=True, spikemode="across+marker",
                     spikesnap="cursor", spikecolor="rgba(0,212,255,0.7)", spikethickness=1, spikedash="dot")
    fig.update_yaxes(gridcolor=C["grid"], showspikes=True, spikemode="across+marker", spikesnap="cursor",
                     spikecolor="rgba(0,212,255,0.7)", spikethickness=1, spikedash="dot")
    return fig


def trade_map_candles(ohlcv: pd.DataFrame, trades: pd.DataFrame, title="进出场标记图", logy=True) -> go.Figure:
    """K 线 + 成交量副图 + 每笔入场▲/出场▼。TradingView 式缩放/平移/十字光标，但无画线工具。"""
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.78, 0.22])
    fig.add_trace(go.Candlestick(
        x=ohlcv.index, open=ohlcv["open"], high=ohlcv["high"], low=ohlcv["low"], close=ohlcv["close"],
        increasing_line_color="#2BE6A8", decreasing_line_color="#FF5C7A",
        increasing_line_width=0.7, decreasing_line_width=0.7, name="K线", showlegend=False,
    ), row=1, col=1)

    up = (ohlcv["close"] >= ohlcv["open"]).to_numpy()
    vcol = np.where(up, "rgba(43,230,168,0.45)", "rgba(255,92,122,0.45)")
    fig.add_trace(go.Bar(x=ohlcv.index, y=ohlcv["volume"], marker_color=vcol, name="成交量",
                         showlegend=False, hovertemplate="量 %{y:.3s}<extra></extra>"), row=2, col=1)

    if trades is not None and len(trades):
        wins = trades[trades["return"] > 0]
        loss = trades[trades["return"] <= 0]
        for grp, col, nm in [(wins, C["win"], "盈利"), (loss, C["loss"], "亏损")]:
            if len(grp):
                fig.add_trace(go.Scatter(x=grp["entry_date"], y=grp["entry_price"], mode="markers",
                              marker=dict(symbol="triangle-up", size=12, color=col, line=dict(color="white", width=0.7)),
                              name=f"入场·{nm}", hovertemplate="入场 %{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>"), row=1, col=1)
                fig.add_trace(go.Scatter(x=grp["exit_date"], y=grp["exit_price"], mode="markers",
                              marker=dict(symbol="triangle-down", size=12, color=col, line=dict(color="white", width=0.7)),
                              name=f"出场·{nm}", showlegend=False,
                              hovertemplate="出场 %{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>"), row=1, col=1)

    n = len(trades) if trades is not None else 0
    wr = (trades["return"] > 0).mean() if n else float("nan")
    _apply_tv(fig, 560, f"{title}　N={n} 笔 · 胜率 {wr:.0%}")
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_xaxes(rangeselector=_RANGE_BUTTONS, row=1, col=1)
    if logy:
        fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    return fig


def volume_profile_bars(vp: dict, title="Volume Profile（日线近似）") -> go.Figure:
    """按价位的成交量横向柱状 + POC + 价值区间。"""
    fig = go.Figure()
    poc = vp["poc"]
    va_lo, va_hi = vp["value_area"]
    colors = ["#7C5CFC" if (va_lo <= c <= va_hi) else "rgba(124,92,252,0.35)" for c in vp["centers"]]
    fig.add_bar(x=vp["volumes"], y=vp["centers"], orientation="h", marker_color=colors, name="成交量")
    fig.add_hline(y=poc, line=dict(color="#00D4FF", width=2, dash="dash"),
                  annotation_text=f"POC {poc:.1f}", annotation_position="right")
    fig.update_xaxes(title="成交量（近似）")
    fig.update_yaxes(title="价格")
    return _style(fig, height=460, title=title)


def panorama_price_chart(ohlcv: pd.DataFrame, zones=None, vp: dict | None = None,
                         best_entry: dict | None = None, show_best=True, show_zones=True,
                         show_vp=True, title="K线 + 图层", logy=True, best_endorsed=True) -> go.Figure:
    """全景页主图（Plotly）：K线 + 量副图，并按开关叠加：
      🎯最佳入场区(绿/黄阴影带 + 历史常驻价线) · 📊回撤价位带(蓝=当前/灰=其它) · 📦换手位(筹码柱+POC+价值区)。
    用 Plotly 画横线/阴影(streamlit_lightweight_charts 不支持 priceLine，故改用此图可靠渲染)。"""
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.80, 0.20])
    ymin = float(ohlcv["low"].min()); ymax = float(ohlcv["high"].max())
    x0, x1 = ohlcv.index[0], ohlcv.index[-1]

    # 叠加层一律用 go.Scatter **数据 trace**（而非 add_hline/add_hrect 的 layout shape）：
    # shape 在「子图 + 叠加副x轴(x3)」组合下浏览器里常常完全不渲染（横线/区域不显示的根因）；
    # 数据 trace 绑定坐标轴，必现。横线=两点连线，阴影带=自闭合多边形 fill。
    _lvls: list[float] = []   # 记录所有已画叠加层的价位，最后据此把 y 轴撑开，杜绝叠加层被裁出视野

    def _hline(level, color, width, dash, label=None, side="right"):
        _lvls.append(float(level))
        fig.add_trace(go.Scatter(x=[x0, x1], y=[level, level], mode="lines",
                                 line=dict(color=color, width=width, dash=dash),
                                 hoverinfo="skip", showlegend=False), row=1, col=1)
        if label:
            fig.add_annotation(x=(x1 if side == "right" else x0), y=level, text=label,
                               xanchor=("right" if side == "right" else "left"), yanchor="bottom",
                               showarrow=False, font=dict(color=color, size=10),
                               bgcolor="rgba(13,17,28,0.55)", row=1, col=1)

    def _band(y0, y1, fillcolor, label=None, lab_color="#E6E9EF"):
        _lvls.extend([float(y0), float(y1)])
        fig.add_trace(go.Scatter(x=[x0, x1, x1, x0, x0], y=[y0, y0, y1, y1, y0],
                                 fill="toself", fillcolor=fillcolor, line=dict(width=0),
                                 mode="lines", hoverinfo="skip", showlegend=False), row=1, col=1)
        if label:
            fig.add_annotation(x=x0, y=y1, text=label, xanchor="left", yanchor="bottom",
                               showarrow=False, font=dict(color=lab_color, size=11), row=1, col=1)

    # —— 📦 换手位筹码横柱（叠在价格轴；用副 x 轴 x3 控制柱长靠左）——
    if show_vp and vp is not None:
        centers = np.asarray(vp["centers"], dtype=float); vols = np.asarray(vp["volumes"], dtype=float)
        if centers.size and vols.max() > 0:
            bw = float(centers[1] - centers[0]) if centers.size > 1 else None
            va_lo, va_hi = vp["value_area"]
            bcol = ["rgba(255,159,69,0.55)" if (va_lo <= c <= va_hi) else "rgba(255,159,69,0.22)" for c in centers]
            fig.add_trace(go.Bar(x=vols, y=centers, orientation="h", width=bw, marker_color=bcol,
                                 marker_line_width=0, xaxis="x3", yaxis="y", showlegend=False, name="筹码",
                                 hovertemplate="价位 %{y:.1f}<br>筹码量 %{x:.3s}<extra></extra>"))

    # —— 阴影带先画（垫在 K 线下面，不挡）：🎯最佳入场区 + 📦价值区 ——
    # best_endorsed=False（引擎判定当前别建仓）时，带/线降级为灰色"历史常驻价(暂非买点)"，
    # 与顶部三联卡口径一致，杜绝"卡说暂非买点、图却画绿色最佳入场区"的冲突。
    if show_best and best_entry and best_entry.get("has_zone"):
        if not best_endorsed:
            gcol = "rgba(138,147,166,0.95)"; gfill = "rgba(138,147,166,0.12)"; _blab = "📍历史常驻价(暂非买点)"
        elif best_entry.get("tier") == "稳健最佳入场区":
            gcol = "#2BE6A8"; gfill = "rgba(43,230,168,0.15)"; _blab = "🎯最佳入场区"
        else:
            gcol = "#FFD166"; gfill = "rgba(255,209,102,0.15)"; _blab = "🎯最佳入场区"
        bd = best_entry.get("price_band") or (None, None)
        lo, hi = bd[0], bd[1]
        if hi is not None:
            _band(float(lo) if lo is not None else ymin, float(hi), gfill, _blab, gcol)
    if show_vp and vp is not None:
        _band(vp["value_area"][0], vp["value_area"][1], "rgba(255,159,69,0.08)")

    # —— K线 + 量副图 ——
    fig.add_trace(go.Candlestick(x=ohlcv.index, open=ohlcv["open"], high=ohlcv["high"], low=ohlcv["low"],
                  close=ohlcv["close"], increasing_line_color="#2BE6A8", decreasing_line_color="#FF5C7A",
                  increasing_line_width=0.7, decreasing_line_width=0.7, name="K线", showlegend=False), row=1, col=1)
    up = (ohlcv["close"] >= ohlcv["open"]).to_numpy()
    vcol = np.where(up, "rgba(43,230,168,0.45)", "rgba(255,92,122,0.45)")
    fig.add_trace(go.Bar(x=ohlcv.index, y=ohlcv["volume"], marker_color=vcol, showlegend=False,
                  hovertemplate="量 %{y:.3s}<extra></extra>"), row=2, col=1)

    # —— 横线叠在 K 线之上：📊回撤价位带 · 📦POC · 🎯历史常驻价 ——
    if show_zones and zones is not None and len(zones):
        zz = zones[zones["enough"]] if "enough" in zones.columns else zones
        for _, r in zz.iterrows():
            cur = bool(r.get("is_current", False))
            col = "#00D4FF" if cur else "rgba(138,147,166,0.55)"
            _hline(float(r["price_high"]), col, 1.4 if cur else 1.0, "dot",
                   ("▶ " if cur else "") + str(r["zone"]), side="left")
    if show_vp and vp is not None:
        _hline(vp["poc"], "#FF9F45", 1.6, "dash", f"POC {vp['poc']:.1f}", side="right")
    if show_best and best_entry and best_entry.get("has_zone"):
        if not best_endorsed:
            gcol = "rgba(138,147,166,0.95)"; _alab = "📍历史常驻价"
        else:
            gcol = "#2BE6A8" if best_entry.get("tier") == "稳健最佳入场区" else "#FFD166"; _alab = "🎯历史常驻价"
        anc = best_entry.get("anchor_price")
        if anc == anc and anc is not None:
            _hline(float(anc), gcol, 2.0, "solid", f"{_alab} {float(anc):.1f}", side="right")

    _apply_tv(fig, 560, title)
    _vmax = (np.asarray(vp["volumes"], float).max() if (show_vp and vp is not None and np.asarray(vp["volumes"], float).size) else 0)
    fig.update_layout(xaxis_rangeslider_visible=False, barmode="overlay",
                      xaxis3=dict(overlaying="x", side="top", range=[0, (_vmax * 4.5) if _vmax > 0 else 1],
                                  showgrid=False, showticklabels=False, zeroline=False, fixedrange=True))
    fig.update_xaxes(rangeselector=_RANGE_BUTTONS, row=1, col=1)
    # y 轴范围：把已画的叠加层一并纳入（否则深档价位带/历史常驻价会被裁出视野=用户说的"不显示"）；
    # 但对极端深档设下限，避免 K 线被过度压扁（叠加层最多把下界压到近端低点的 0.78×）。
    _lo = ymin; _hi = ymax
    if _lvls:
        _lo = min(_lo, max(min(_lvls), ymin * 0.78))
        _hi = max(_hi, min(max(_lvls), ymax * 1.22))
    if logy:
        # 显式给定 log 轴范围：注释/trace 不破坏 log 自动量程，按实际价格手动算 log10 范围。
        import math
        fig.update_yaxes(type="log", range=[math.log10(max(_lo * 0.96, 1e-6)), math.log10(_hi * 1.04)], row=1, col=1)
    else:
        fig.update_yaxes(range=[_lo * 0.97, _hi * 1.03], row=1, col=1)
    fig.update_yaxes(title_text="量", row=2, col=1)
    return fig


def candle_with_levels(ohlcv: pd.DataFrame, vp: dict, title="近一年 K线 + 筹码分布", logy=False) -> go.Figure:
    """K线 + 量副图，并把**每个价位的筹码量(Volume Profile)**作为横向柱直接叠加在K线价格轴上
    （靠左、半透明，不挡近端价格；价值区柱更亮）。POC=最高换手价虚线，价值区=70%成交带。"""
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.78, 0.22])
    poc = vp["poc"]; va_lo, va_hi = vp["value_area"]
    centers = np.asarray(vp["centers"], dtype=float)
    vols = np.asarray(vp["volumes"], dtype=float)

    # —— 筹码分布横向柱：叠加在 K线价格轴(y) 上，用独立 x 轴(x3) 控制柱长(靠左占约22%宽) ——
    if centers.size and vols.max() > 0:
        vmax = float(vols.max())
        bw = float(centers[1] - centers[0]) if centers.size > 1 else None
        bcol = ["rgba(124,92,252,0.60)" if (va_lo <= c <= va_hi) else "rgba(124,92,252,0.24)" for c in centers]
        fig.add_trace(go.Bar(x=vols, y=centers, orientation="h", width=bw, marker_color=bcol,
                             marker_line_width=0, xaxis="x3", yaxis="y", showlegend=False, name="筹码",
                             hovertemplate="价位 %{y:.1f}<br>筹码量 %{x:.3s}<extra></extra>"))

    fig.add_trace(go.Candlestick(x=ohlcv.index, open=ohlcv["open"], high=ohlcv["high"], low=ohlcv["low"],
                  close=ohlcv["close"], increasing_line_color="#2BE6A8", decreasing_line_color="#FF5C7A",
                  increasing_line_width=0.7, decreasing_line_width=0.7, name="K线", showlegend=False), row=1, col=1)
    up = (ohlcv["close"] >= ohlcv["open"]).to_numpy()
    vcol = np.where(up, "rgba(43,230,168,0.45)", "rgba(255,92,122,0.45)")
    fig.add_trace(go.Bar(x=ohlcv.index, y=ohlcv["volume"], marker_color=vcol, showlegend=False,
                  hovertemplate="量 %{y:.3s}<extra></extra>"), row=2, col=1)
    # POC/价值区用数据 trace（非 add_hline/add_hrect 的 layout shape）——子图+副x轴下 shape 常不渲染
    _x0, _x1 = ohlcv.index[0], ohlcv.index[-1]
    fig.add_trace(go.Scatter(x=[_x0, _x1, _x1, _x0, _x0], y=[va_lo, va_lo, va_hi, va_hi, va_lo],
                             fill="toself", fillcolor="rgba(124,92,252,0.10)", line=dict(width=0),
                             mode="lines", hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[_x0, _x1], y=[poc, poc], mode="lines",
                             line=dict(color="#00D4FF", width=1.5, dash="dash"),
                             hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_annotation(x=_x1, y=poc, text=f"POC {poc:.1f}", xanchor="right", yanchor="bottom",
                       showarrow=False, font=dict(color="#00D4FF", size=10),
                       bgcolor="rgba(13,17,28,0.55)", row=1, col=1)
    fig.add_annotation(x=_x0, y=va_hi, text="价值区(70%)", xanchor="left", yanchor="bottom",
                       showarrow=False, font=dict(color="#7C5CFC", size=11), row=1, col=1)
    _apply_tv(fig, 540, title)
    # x3 = 筹码柱专用轴：range 放大到 4.5×max → 柱只占左侧约 22% 宽，近端K线不被遮挡
    fig.update_layout(xaxis_rangeslider_visible=False, barmode="overlay",
                      xaxis3=dict(overlaying="x", side="top", range=[0, (vols.max() * 4.5) if (centers.size and vols.max() > 0) else 1],
                                  showgrid=False, showticklabels=False, zeroline=False, fixedrange=True))
    fig.update_xaxes(rangeselector=_RANGE_BUTTONS, row=1, col=1)
    if logy:
        fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    return fig


def options_oi(snap: dict, title="期权未平仓（OI）按行权价") -> go.Figure:
    fig = go.Figure()
    c, p = snap["calls"], snap["puts"]
    cg = c.groupby("strike")["openInterest"].sum()
    pg = p.groupby("strike")["openInterest"].sum()
    fig.add_bar(x=cg.index, y=cg.values, name="Call OI", marker_color="#2BE6A8", opacity=0.7)
    fig.add_bar(x=pg.index, y=pg.values, name="Put OI", marker_color="#FF5C7A", opacity=0.7)
    fig.add_vline(x=snap["spot"], line=dict(color="#00D4FF", width=2), annotation_text=f"现价 {snap['spot']:.1f}")
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="行权价")
    fig.update_yaxes(title="未平仓合约")
    return _style(fig, height=420, title=title)


def signal_excess(df: pd.DataFrame, title="信号远期超额（vs 各股基准，95% CI）") -> go.Figure:
    """各信号的远期超额横向柱 + bootstrap CI 误差棒；显著(CI不跨0)高亮。"""
    d = df.sort_values("excess")
    fig = go.Figure()
    colors = [C["win"] if s else C["baseline"] for s in d["sig_raw"]]
    fig.add_bar(y=d["signal"], x=d["excess"], orientation="h", marker_color=colors,
                error_x=dict(type="data", symmetric=False,
                             array=(d["ci_high"] - d["excess"]).clip(lower=0),
                             arrayminus=(d["excess"] - d["ci_low"]).clip(lower=0),
                             color="rgba(255,255,255,0.4)"),
                hovertemplate="%{y}: %{x:.1%}<extra></extra>")
    fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.45)", width=1))
    fig.update_layout(xaxis_tickformat=".0%")
    return _style(fig, height=max(360, 36 * len(d)), title=title)


def score_timeseries(df: pd.DataFrame, ticker="") -> go.Figure:
    """价格(对数) + 建仓分/风险分(0-100) 双面板时序。"""
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.58, 0.42])
    fig.add_trace(go.Scatter(x=df.index, y=df["price"], line=dict(color="#9aa7bd", width=1.2), name="价格"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["建仓分"], line=dict(color=C["win"], width=1.3), name="建仓分"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["风险分"], line=dict(color=C["loss"], width=1.3), name="风险分"), row=2, col=1)
    fig.add_hline(y=70, line=dict(color="rgba(43,230,168,0.4)", width=1, dash="dot"), row=2, col=1)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1, title_text="分位")
    _apply_tv(fig, 480, f"{ticker}　建仓分 / 风险分（0–100 历史分位，非预测）")
    fig.update_xaxes(rangeselector=_RANGE_BUTTONS, row=1, col=1)  # TradingView 时间区间按钮
    return fig


def event_timeline_chart(price: pd.Series, events: pd.DataFrame, title="事件时间线") -> go.Figure:
    """价格折线 + 事件标记（SEC/财报分色）。仅复盘对照，不作信号。"""
    fig = go.Figure()
    fig.add_scatter(x=price.index, y=price.values, mode="lines", line=dict(color="#34495E", width=1.2), name="价格", showlegend=False)
    pser = price.dropna()
    for typ, col in [("财报", "#2BE6A8"), ("SEC", "#00D4FF")]:
        sub = events[events["type"] == typ]
        if sub.empty:
            continue
        ys, xs, txt = [], [], []
        for _, r in sub.iterrows():
            d = pd.Timestamp(r["date"])
            near = pser.index[pser.index.get_indexer([d], method="nearest")[0]]
            xs.append(near); ys.append(float(pser.loc[near]))
            rr = r.get("reaction_5d", float("nan"))
            txt.append(f"{r['label']}<br>{d.date()}<br>5日反应 {rr:+.1%}" if rr == rr else f"{r['label']}<br>{d.date()}")
        fig.add_scatter(x=xs, y=ys, mode="markers", name=typ,
                        marker=dict(size=9, color=col, line=dict(color="white", width=0.5), symbol="diamond"),
                        text=txt, hoverinfo="text")
    _apply_tv(fig, 460, title)
    fig.update_xaxes(rangeselector=_RANGE_BUTTONS)
    fig.update_yaxes(type="log")
    return fig


def entry_zone_bars(zones: pd.DataFrame, horizon: int, title="条件价位带 · 远期收益分布") -> go.Figure:
    """各回撤价位带的远期收益中位(带 CI)横向柱 + 无条件基准线 + 标记今日所处带。

    只画有足够样本(enough)的带；颜色按盈亏比(reward_risk)着色，越绿越优。"""
    z = zones[zones["enough"]].copy()
    fig = go.Figure()
    if z.empty:
        return _style(fig, title=title + "（各带样本均不足）")
    labels = [f"{r['zone']}<br>≈{r['price_low']:.0f}-{r['price_high']:.0f}"
              + ("　◀今日" if r["is_current"] else "") for _, r in z.iterrows()]
    rr = z["reward_risk"].to_numpy()
    colors = [C["win"] if (v == v and v >= 1.0) else (C["accent"] if v == v else C["baseline"]) for v in rr]
    lo = (z["median"] - z["ci_low"]).clip(lower=0)
    hi = (z["ci_high"] - z["median"]).clip(lower=0)
    fig.add_bar(y=labels, x=z["median"], orientation="h", marker_color=colors,
                error_x=dict(type="data", symmetric=False, array=hi, arrayminus=lo, color="rgba(255,255,255,0.5)"),
                text=[f"中位{m:+.0%} · 盈亏比{(v if v==v else float('nan')):.2f} · N≈{int(n)}"
                      for m, v, n in zip(z["median"], rr, z["n_events"])],
                textposition="auto", name="条件中位收益")
    base = float(zones.attrs.get("baseline_median", 0.0))
    fig.add_vline(x=base, line=dict(color=C["baseline"], width=2, dash="dot"),
                  annotation_text=f"无条件基准 {base:+.0%}", annotation_position="top")
    fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.3)", width=1))
    fig.update_layout(xaxis_tickformat=".0%")
    fig.update_xaxes(title=f"{horizon} 日远期收益（中位 + 95% CI）")
    return _style(fig, height=max(360, 60 + 52 * len(z)), title=title)


def price_with_zones(price: pd.Series, zones: pd.DataFrame, lookback: int = 504,
                     title="价格 + 回撤价位带（建仓/补仓候选区）") -> go.Figure:
    """近 lookback 日价格折线 + 各价位带阈值横线（今日带高亮），直观看"现价处在哪一带、补仓位在哪"。"""
    p = price.dropna().iloc[-lookback:]
    fig = go.Figure()
    fig.add_scatter(x=p.index, y=p.values, mode="lines", line=dict(color=C["accent2"], width=1.6),
                    name="价格", showlegend=False)
    for _, r in zones.iterrows():
        lvl = r["price_high"]
        cur = r["is_current"]
        fig.add_hline(y=lvl, line=dict(color=C["win"] if cur else "rgba(255,255,255,0.18)",
                                       width=1.4 if cur else 0.8, dash="solid" if cur else "dot"),
                      annotation_text=f"{r['zone'].replace('距前高 ','-').replace('%','%')} ≈{lvl:.0f}"
                                      + (" ◀今日" if cur else ""),
                      annotation_position="right",
                      annotation_font=dict(size=9, color=C["win"] if cur else C["baseline"]))
    cur_px = float(zones.attrs.get("current_price", p.iloc[-1]))
    fig.add_hline(y=cur_px, line=dict(color="#FFD166", width=1.2),
                  annotation_text=f"现价 {cur_px:.0f}", annotation_position="left",
                  annotation_font=dict(size=10, color="#FFD166"))
    _apply_tv(fig, 440, title)
    fig.update_xaxes(rangeselector=_RANGE_BUTTONS)
    return fig


def equity_underwater(trades: pd.DataFrame, title="规则净值水下图") -> go.Figure:
    tr = trades.sort_values("exit_date")
    eq = (1.0 + tr["return"].fillna(0.0)).cumprod()
    dd = eq / eq.cummax() - 1.0
    x = pd.to_datetime(tr["exit_date"].values)
    fig = go.Figure()
    fig.add_scatter(x=x, y=dd, mode="lines", line=dict(color=C["loss"], width=1),
                    fill="tozeroy", fillcolor="rgba(255,92,122,0.25)")
    fig.update_layout(yaxis_tickformat=".0%")
    maxdd = float(dd.min()) if len(dd) else 0.0
    _apply_tv(fig, 380, f"{title}　最大回撤 {maxdd:.0%}")
    fig.update_xaxes(rangeselector=_RANGE_BUTTONS)
    return fig


def equity_compare(equity: pd.DataFrame, title="策略 vs 持有 · 净值(归一)") -> go.Figure:
    """归一净值对比（按列名画线，第一列实线高亮、其余虚线对照）。

    兼容 {策略,持有} 与 {波动目标,持有} 等任意两列。"""
    fig = go.Figure()
    _names = {"策略": "策略(深跌买×让利润奔跑)", "持有": "闭眼持有", "波动目标": "波动目标 overlay"}
    for i, col in enumerate(equity.columns):
        first = (i == 0)
        fig.add_scatter(x=equity.index, y=equity[col], mode="lines",
                        line=dict(color=C["accent"] if first else C["baseline"], width=2,
                                  dash=None if first else "dot"),
                        name=_names.get(col, col))
    fig.add_hline(y=1.0, line=dict(color="rgba(255,255,255,0.25)", width=1))
    fig.update_yaxes(type="log", title="净值(起点=1)")
    fig.update_xaxes(rangeselector=_RANGE_BUTTONS)  # 时间范围切换（1M/3M/.../全部）
    return _style(fig, height=420, title=title)
