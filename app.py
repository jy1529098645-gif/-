"""量化研究工具 · 本地交互前端（Streamlit）。

运行：  .venv/Scripts/streamlit run app.py     （或双击 run_app.bat）

定位铁律（贯穿全局）：校准而非预测 / 永远对比基准 / 反过拟合优先。
这是"决策与验证"仪表盘，不是看盘终端——展示分布 + 置信区间 + 样本数 N，
绝不输出单一目标价/最佳买卖点。
"""
from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import streamlit as st

# 把 Streamlit Cloud 的 secrets 桥接成环境变量（FRED key 等），供 data.loader 读取。
# 本地无 secrets.toml 时静默跳过；密钥永不进 git（见 .gitignore / config.yaml 注释）。
try:
    for _k in ("FRED_API_KEY",):
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:  # noqa: BLE001
    pass

import config
from frontend import charts as ch
from frontend import glossary as gl
from frontend import theme as tm

st.set_page_config(page_title="量化研究工具", page_icon="📊", layout="wide",
                   initial_sidebar_state="auto")  # auto：桌面展开 / 移动端自动收起，避免侧边栏全屏遮挡正文

CFG = config.load_config()

# 当前主题配色（shadcn slate 双主题）。整脚本每次交互都全量重跑，故 T 始终反映最新主题。
# app.py 内联 HTML / 原生控件统一用真实 hex（而非 CSS var），明暗下都正确。
T = tm.tokens()

# ---------------------------------------------------------------------------
# 右上角实时纽约时间（JS 跳秒，注入父文档；不依赖 Streamlit rerun）
# ---------------------------------------------------------------------------
def _ny_clock():
    import streamlit.components.v1 as _components
    # 纯 iframe 内渲染：零跨域依赖，云端/本地都必显示（不再注入父文档，避免被沙箱挡掉）。
    # 头部样式按主题填色（f-string，仅含成对花括号）；下方 JS 含大量单花括号，保持普通字符串后拼接。
    _head = (
        '<style>html,body{margin:0;padding:0;overflow:hidden;background:transparent}</style>'
        '<div id="nyc" style="display:inline-block;float:right;'
        'font-family:-apple-system,Segoe UI,Roboto,monospace;font-size:0.82rem;'
        f'color:{T["clock_text"]};background:{T["clock_bg"]};border:1px solid {T["border"]};'
        'border-radius:9px;padding:4px 11px;letter-spacing:0.3px">🗽 纽约 …</div>'
    )
    _open_col, _shut_col = T["good"], T["muted"]
    _js = """
        <script>
        (function(){
          const el=document.getElementById('nyc');
          function tick(){
            try{
              const now=new Date();
              const t=now.toLocaleString('en-US',{timeZone:'America/New_York',year:'numeric',
                month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',
                hour12:false,weekday:'short'});
              // 美股是否开盘（周一–五 9:30–16:00 ET）
              const parts=new Intl.DateTimeFormat('en-US',{timeZone:'America/New_York',
                hour:'2-digit',minute:'2-digit',hour12:false,weekday:'short'}).formatToParts(now);
              const g=k=>parts.find(p=>p.type===k)?.value;
              const wd=g('weekday'); const hh=parseInt(g('hour')); const mm=parseInt(g('minute'));
              const isWk=!['Sat','Sun'].includes(wd); const mins=hh*60+mm; const open=isWk&&mins>=570&&mins<960;
              el.innerHTML='🗽 纽约 '+t+'  '+(open?'<span style="color:__OPEN__">●开盘</span>'
                                               :'<span style="color:__SHUT__">●休市</span>');
            }catch(e){ el.textContent='🗽 '+new Date().toUTCString(); }
          }
          tick(); setInterval(tick,1000);
        })();
        </script>
    """.replace("__OPEN__", _open_col).replace("__SHUT__", _shut_col)
    _components.html(_head + _js, height=32)

_ny_clock()

# ---------------------------------------------------------------------------
# 主题样式（shadcn/ui slate · 明暗双主题，集中在 frontend/theme.py）
# ---------------------------------------------------------------------------
tm.inject(st)
st.markdown(gl.inject_css(), unsafe_allow_html=True)

def run_gate(key: str, params: dict, label: str = "🚀 运行回测", hint: str = "配置好参数后点击运行，期间不会自动重算"):
    """运行门控：改参数不重算，只有点「运行」才用快照参数计算。返回快照 dict 或 None。

    解决 Streamlit「改任何控件就立即重算」导致的卡顿——专业回测工具的习惯是「设好→运行→看结果」。
    """
    c1, c2 = st.columns([1, 3])
    if c1.button(label, type="primary", use_container_width=True, key=f"btn_{key}"):
        st.session_state[f"run_{key}"] = params
    run = st.session_state.get(f"run_{key}")
    if run is None:
        c2.info("👈 " + hint)
        return None
    if run != params:
        c2.warning("⚠️ 参数已修改——点「运行」刷新结果（当前显示的是上次运行）")
    return run

def _lazy_gate(key: str, label: str = "▶ 加载此分析（较重，按需运行）") -> bool:
    """惰性门控：Streamlit 的 Tab/expander 内容**每次都会执行**(折叠也算)，重计算会拖慢整页。
    用此门控让重模块只在点击后运行，且本会话内记住。返回 True=已激活。"""
    sk = f"_lazy_{key}"
    if st.session_state.get(sk):
        return True
    if st.button(label, key=f"btn_{sk}"):
        st.session_state[sk] = True
        return True
    return False

def _col_cfg(columns):
    """为数据表生成带悬浮释义的 column_config：列名照常显示，悬浮列头出含义。"""
    cfg = {}
    for c in columns:
        h = gl.col_help(c)
        if h:
            cfg[c] = st.column_config.Column(c, help=h)
    return cfg

def stat_card(label, value, sub="", color=T["text"], tip=None):
    """tip：术语 key，则标签变成可悬浮解释的术语。"""
    lab = gl.term(tip, label) if tip else label
    return (
        f'<div class="glass" style="text-align:left">'
        f'<div class="stat-label">{lab}</div>'
        f'<div class="stat-value" style="color:{color}">{value}</div>'
        f'<div class="hero-sub" style="font-size:.8rem">{sub}</div></div>'
    )


def _claude_deep_link(prompt: str) -> str:
    """生成一键打开 Claude 并预填提示词的链接（claude.ai 新对话·q 参数预填）。"""
    from urllib.parse import quote
    return "https://claude.ai/new?q=" + quote(prompt)


def claude_deep_button(label: str, prompt: str, key: str = "", hint: bool = True):
    """渲染"一键用 Claude 深度分析"按钮：新标签页打开 claude.ai 并预填提示词。

    hint=False 时省略下方说明（用于顶部等已在别处解释过的位置，避免重复字眼）。"""
    st.link_button("🚀 " + label, _claude_deep_link(prompt), use_container_width=True)
    if hint:
        st.caption("↑ 新标签页打开 Claude 并预填提示词，**回车即开始**联网读全文+深度推理"
                   "（需你的 Claude 已开启 quant-deep-brief / 深度分析能力）。")

# ---------------------------------------------------------------------------
# 图表周期切换：每个时序/K线图上方的「时间范围 + K线粒度」控件 + TV 渲染包装器
# ---------------------------------------------------------------------------
def _chart_period_controls(key: str, with_timeframe: bool = False,
                           default_period: str = "全部", default_tf: str = "日"):
    """渲染一行横向周期控件，返回 (period, timeframe)。key 用于隔离各图控件状态。"""
    from frontend import tvchart as _tv
    if with_timeframe:
        c1, c2 = st.columns([3, 2])
        period = c1.radio("时间范围", _tv.PERIODS, horizontal=True, label_visibility="collapsed",
                          index=_tv.PERIODS.index(default_period), key=f"per_{key}")
        tf = c2.radio("K线粒度", _tv.TIMEFRAMES, horizontal=True, label_visibility="collapsed",
                      index=_tv.TIMEFRAMES.index(default_tf), key=f"tf_{key}")
        return period, tf
    period = st.radio("时间范围", _tv.PERIODS, horizontal=True, label_visibility="collapsed",
                      index=_tv.PERIODS.index(default_period), key=f"per_{key}")
    return period, "日"

def render_tv_candles(ohlcv, trades=None, price_lines=None, key="tv", height=540, log=False,
                      with_timeframe=True, caption=None):
    """TradingView K 线 + 周期控件（时间范围 + 日/周/月）。聚合时把成交标记吸附到 bar。"""
    from frontend import tvchart as _tv
    period, tf = _chart_period_controls(key, with_timeframe=with_timeframe)
    df = _tv.resample_ohlcv(ohlcv, tf)
    df = _tv.slice_period(df, period)
    tr = _tv.snap_markers_to_bars(trades, df.index) if (trades is not None and tf != "日") else trades
    _tv.tv_candles(df, tr, price_lines=price_lines, key=key, height=height, log=log)
    if tf != "日":
        st.caption(f"⏱️ 当前 {tf}线 · 进出场标记按{tf}聚合周期对齐（非精确日）。")
    elif caption:
        st.caption(caption)

def render_tv_line(series, markers=None, price_lines=None, key="tvl", height=460,
                   color=T["primary"], log=True):
    """TradingView 折线 + 时间范围切换（事件时间线等）。"""
    from frontend import tvchart as _tv
    period, _ = _chart_period_controls(key, with_timeframe=False)
    s = _tv.slice_period(series.to_frame("v"), period)["v"]
    mk = markers
    if markers and period != "全部" and len(s):
        lo = s.index.min().strftime("%Y-%m-%d")
        mk = [m for m in markers if m.get("time", "") >= lo]
    _tv.tv_line(s, markers=mk, price_lines=price_lines, key=key, height=height, color=color, log=log)

def _chart_horizon(key: str, default_days: int, label: str = "持有期") -> int:
    """per-chart 持有期下拉（默认跟随侧边栏「分析周期」），返回天数。用于由 horizon 驱动的图。"""
    labels = list(_HZ)
    days = [_HZ[l] for l in labels]
    idx = days.index(default_days) if default_days in days else 0
    lab = st.selectbox(label, labels, index=idx, key=f"hz_{key}",
                       help="该图独立的远期收益持有期（默认跟随侧边栏「分析周期」）")
    return _HZ[lab]

# ---------------------------------------------------------------------------
# 缓存的数据 / 计算
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def c_prices(tickers: tuple, start: str, end: str | None):
    from data import loader
    return loader.load_prices(list(tickers), start, end)

@st.cache_data(show_spinner=False)
def c_macro(start: str, end: str | None):
    from data import loader
    return loader.load_macro(start, end)

@st.cache_data(show_spinner=False)
def c_earnings(ticker: str):
    from data import loader
    return loader.load_earnings_dates(ticker, limit=80)

@st.cache_data(show_spinner=False)
def c_regime(asset: str, start: str, end: str, horizons: tuple):
    from data import loader
    from regime import conditional_returns as cr
    px = loader.load_prices([asset], start, end)
    macro = loader.load_macro("1990-01-01", end)
    tab = cr.conditional_forward_returns(px, macro, asset=asset, horizons=horizons, n_boot=300)
    fp = cr.current_fingerprint(px, macro, asset=asset)
    return tab, fp

@st.cache_data(show_spinner=False)
def c_factor(factor_name: str, universe: str, start: str, end: str):
    from data import loader
    from factors import price_factors as pf
    from evaluation import factor_eval as fe
    tickers = loader.load_universe(universe)
    px = loader.load_prices(tickers, start, end)
    fac = pf.REGISTRY[factor_name](px)
    return fe.evaluate_factor(fac, px, quantiles=5, periods=(1, 5, 21, 63))

@st.cache_data(show_spinner=False)
def c_factor_decay(factor_name: str, universe: str, start: str, end: str):
    """滚动 IC + IC 衰减曲线（监控因子有效性随时间/持有期变化）。"""
    from data import loader
    from factors import price_factors as pf
    from evaluation import factor_eval as fe
    tickers = loader.load_universe(universe)
    px = loader.load_prices(tickers, start, end)
    fac = pf.REGISTRY[factor_name](px)
    decay = fe.ic_decay(fac, px, horizons=(1, 5, 21, 63, 126, 252))
    roll = fe.rolling_ic(fac, px, horizon=21, window=126)
    return {"decay": decay, "roll": roll, "verdict": fe.decay_verdict(decay)}

@st.cache_data(show_spinner=False)
def c_blend(factor_names: tuple, universe: str, start: str, end: str):
    """多因子组合（Phase 6）：z-score 等权混合 → IC + 多空分位回测（带 Sharpe CI）。"""
    from data import loader
    from factors import price_factors as pf
    from evaluation import factor_eval as fe
    from backtest import strategies as bt
    tickers = loader.load_universe(universe)
    px = loader.load_prices(tickers, start, end)
    panels = {n: pf.REGISTRY[n](px) for n in factor_names}
    composite = pf.blend(panels)
    ic = fe.evaluate_factor(composite, px, quantiles=5, periods=(1, 5, 21, 63))
    bt_res = bt.factor_quantile_backtest(composite, px, quantiles=5, long_short=True, n_boot=300)
    return ic, bt_res

def _exit_spec(trailing, tp, time_stop, ma_exit=0):
    spec = {"trailing_stop": trailing, "take_profit": tp, "time_stop": int(time_stop)}
    if ma_exit and int(ma_exit) > 0:
        spec["ma_exit"] = int(ma_exit)
    return spec

def _build_condition(cond_kind, cond_window, regime_kind):
    from evaluation import rule_eval as re
    parts = []
    if cond_kind != "none":
        parts.append(re.earnings_condition(cond_kind, window=int(cond_window)))
    if regime_kind != "none":
        parts.append(re.regime_condition(regime_kind))
    if not parts:
        return None
    return re.combine_conditions(*parts) if len(parts) > 1 else parts[0]

@st.cache_data(show_spinner=False)
def c_rule(specs: tuple, op, trailing, tp, time_stop, universe, start, end, cond_kind, cond_window, regime_kind, rule_name, ma_exit=0):
    from data import loader
    from factors import signals as sg
    from evaluation import rule_eval as re

    entry = sg.build_entry(list(specs), op)
    spec = _exit_spec(trailing, tp, time_stop, ma_exit)
    cond = _build_condition(cond_kind, cond_window, regime_kind)
    tickers = loader.load_universe(universe)
    res = re.evaluate_rule(entry, spec, tickers=tickers, start=start, end=end,
                           rule_name=rule_name, n_boot=250, condition_fn=cond)
    res["_verdict"] = re.format_rule_verdict(res)
    return res

@st.cache_data(show_spinner=False)
def c_gate(specs: tuple, op, trailing, tp, time_stop, universe, start, end, ma_exit=0,
           oos_sharpe_min: float = 1.0, max_dd_tol: float = 0.35):
    """验收闸门：跑赢基准 + OOS年化夏普≥阈值 + 最大回撤≤容忍。基于无条件基础规则。"""
    from factors import signals as sg
    from data import loader
    from evaluation import rule_eval as re, acceptance as acc
    entry = sg.build_entry(list(specs), op)
    spec = _exit_spec(trailing, tp, time_stop, ma_exit)
    tickers = loader.load_universe(universe)
    res = re.evaluate_rule(entry, spec, tickers=tickers, start=start, end=end,
                           rule_name="gate", n_boot=250)
    return acc.acceptance_gate(res, entry, spec, tickers, start=start, end=end,
                               oos_sharpe_min=oos_sharpe_min, max_dd_tol=max_dd_tol)

@st.cache_data(show_spinner=False)
def c_single_trades(specs: tuple, op, trailing, tp, time_stop, ticker, start, end, cond_kind, cond_window, regime_kind, ma_exit=0):
    """单票 OHLCV + 该规则逐笔交易，供 K 线标记图。"""
    from data import loader
    from factors import signals as sg
    from backtest import exits as ex

    ohlcv = loader.load_ohlcv(ticker, start, end)
    price = ohlcv["close"].dropna()
    entries = sg.build_entry(list(specs), op)(price)
    cond = _build_condition(cond_kind, cond_window, regime_kind)
    if cond is not None:
        entries = entries & cond(ticker, price).reindex(price.index).fillna(False).astype(bool)
    pf = ex.run_trades(price, entries, _exit_spec(trailing, tp, time_stop, ma_exit))
    trades = ex.extract_trades(pf, price)
    return ohlcv, trades

@st.cache_data(show_spinner=False)
def c_today_panel(asset: str, start: str, end: str):
    from data import loader
    from regime import observables as ob
    px = loader.load_prices([asset], start, end)[asset]
    return ob.today_panel(px)

@st.cache_data(show_spinner=False)
def c_perf(ticker: str, start: str, end: str):
    from backtest import strategies as bt
    return bt.strategy_vs_hold(ticker, start, end)

@st.cache_data(show_spinner=False)
def c_regime_overlay(ticker: str, start: str, end: str):
    from analysis import quant_edge as qe
    px = c_prices((ticker,), start, end)[ticker]
    macro = c_macro("1990-01-01", end)
    return {"exposure": qe.regime_exposure(px, macro), "overlay": qe.vol_target_backtest(px)}

@st.cache_data(show_spinner=False)
def c_alpha_beta(ticker: str, start: str, end: str, bench: str = "SPY"):
    from analysis import quant_edge as qe
    px = c_prices((ticker,), start, end)[ticker]
    bpx = c_prices((bench,), start, end)[bench]
    return qe.alpha_beta_profile(px, bpx)

@st.cache_data(show_spinner=False)
def c_pead(ticker: str, start: str, end: str):
    from analysis import quant_edge as qe
    return qe.pead_now(ticker, start, end)

@st.cache_data(show_spinner=False, ttl=3600)
def c_exposure_spectrum(ticker: str, start: str, end: str):
    from analysis import position_guidance as pg
    return pg.exposure_backtest_spectrum(ticker, start=start, end=end)

@st.cache_data(show_spinner=False)
def c_port_weights(tickers: tuple, start: str, end: str, method: str):
    from analysis import quant_edge as qe
    return qe.portfolio_weights(c_prices(tickers, start, end), method=method)

@st.cache_data(show_spinner=False)
def c_data_health(tickers: tuple, start: str, end: str):
    from analysis import data_quality as dq
    return dq.data_health(c_prices(tickers, start, end))

@st.cache_data(show_spinner=False, ttl=30)
def c_live_quote(ticker: str, _bucket: int = 0):
    """近实时现价快照（缓存 30 秒）。_bucket 让盘中按分钟桶强制刷新。

    单点兜底：任何失败——含部署/模块缓存错位导致 loader 暂无 live_quote、或取价异常——
    都返回 ok=False，由调用方回退到日线收盘，绝不让「今日状态/总览」整页崩溃。"""
    try:
        from data import loader
        fn = getattr(loader, "live_quote", None)
        if fn is not None:
            return fn(ticker)
    except Exception:  # noqa: BLE001
        pass
    return {"ticker": ticker, "price": float("nan"), "change": float("nan"),
            "change_pct": float("nan"), "ok": False, "delayed": True}

def is_market_open() -> bool:
    """美股常规时段是否开盘（周一–五 9:30–16:00 美东，不含节假日）。"""
    try:
        from zoneinfo import ZoneInfo
        import datetime as _d
        now = _d.datetime.now(ZoneInfo("America/New_York"))
        if now.weekday() >= 5:
            return False
        mins = now.hour * 60 + now.minute
        return 570 <= mins < 960
    except Exception:  # noqa: BLE001
        return False


def _vix_level(v: float):
    """VIX 数值 → (恐慌等级文案, 主题色 token, 一句解读)。阈值取市场惯用分档。"""
    if not (v == v):
        return ("数据缺失", "muted", "实时取价失败，回退最近收盘")
    if v < 15:
        return ("平静", "good", "波动极低，市场情绪乐观/自满")
    if v < 20:
        return ("正常", "info", "波动温和，常态区间")
    if v < 27:
        return ("警觉", "gold", "不安升温，注意风险")
    if v < 35:
        return ("担忧", "amber", "明显避险，波动放大")
    return ("恐慌", "bad", "极度避险，常见于急跌/危机")

@st.cache_data(show_spinner=False, ttl=3600)
def c_event_radar(ticker: str, today_iso: str, next_earnings: str | None, horizon: int = 45):
    """事件雷达（全网自动抓 IPO/经济日历 + 规则日历 + 手填）。缓存 1 小时，避免重复联网/限流。"""
    import datetime as _dtm
    from analysis import event_radar as _er
    today = _dtm.date.fromisoformat(today_iso)
    earn = [{"date": next_earnings, "ticker": ticker}] if next_earnings else None
    res = _er.upcoming(today, ticker=ticker, horizon_days=horizon, earnings=earn, include_web=True)
    try:
        res["news_leads"] = _er.fetch_event_news(ticker, limit=5)
    except Exception:  # noqa: BLE001
        res["news_leads"] = []
    return res

@st.cache_data(show_spinner=False)
def c_volume_profile(ticker: str, start: str, end: str, lookback: int):
    from data import loader
    from analysis import volume_profile as vpm
    ohlcv = loader.load_ohlcv(ticker, start, end)
    vp = vpm.volume_profile(ohlcv, bins=50, lookback=lookback)
    return ohlcv.tail(lookback), vp

@st.cache_data(show_spinner=False, ttl=900)
def c_options(ticker: str):
    from data import options
    return options.options_snapshot(ticker)

@st.cache_data(show_spinner=False)
def c_events(ticker: str, start: str, end: str, forms: tuple, include_earnings: bool):
    from data import loader, edgar
    price = loader.load_prices([ticker], start, end)[ticker]
    ev = edgar.event_timeline(ticker, price, forms=list(forms) if forms else None,
                              horizons=(1, 5), include_earnings=include_earnings)
    return price, ev

@st.cache_data(show_spinner=False)
def c_overfit_check(specs: tuple, op, trailing, tp, time_stop, start, end):
    """围绕**当前规则**做 deflated Sharpe + walk-forward（反过拟合体检）。"""
    from evaluation import rule_select as rs
    base_exit = _exit_spec(trailing, tp, time_stop)
    grid = rs.candidate_grid_from(list(specs), op, base_exit)
    deflated = rs.deflated_rule_sharpe(candidates=grid, start=start, end=end)
    wf = rs.walk_forward_rule(candidates=grid, start=start, end=end)
    return deflated, wf

@st.cache_data(show_spinner=False)
def c_single_equity(specs: tuple, op, trailing, tp, time_stop, ticker, start, end, cond_kind, cond_window, regime_kind):
    """单票规则的日度策略收益（喂 quantstats）。"""
    from data import loader
    from factors import signals as sg
    from backtest import exits as ex
    price = loader.load_ohlcv(ticker, start, end)["close"].dropna()
    entries = sg.build_entry(list(specs), op)(price)
    cond = _build_condition(cond_kind, cond_window, regime_kind)
    if cond is not None:
        entries = entries & cond(ticker, price).reindex(price.index).fillna(False).astype(bool)
    pf = ex.run_trades(price, entries, _exit_spec(trailing, tp, time_stop))
    rets = pf.returns()
    return rets

def export_quantstats(rets, ticker, rule_name):
    import quantstats as qs
    out = str(Path(config.get_path("reports")) / f"{rule_name}_{ticker}_quantstats.html".replace("/", "_"))
    qs.reports.html(rets, output=out, title=f"{ticker} · {rule_name}", download_filename=out)
    return out

@st.cache_data(show_spinner=False)
def c_earnings_study(universe: str, start: str, end: str):
    from data import loader
    from evaluation import earnings_eval as ee
    tickers = loader.load_universe(universe)
    prices = {t: loader.load_prices([t], start, end)[t].dropna() for t in tickers}
    edates = {t: loader.load_earnings_dates(t, limit=80) for t in tickers}
    study = ee.earnings_event_study(prices, edates, pre=10, post=20, by_beat=True)
    ic = ee.earnings_drift_ic(prices, edates, horizons=(1, 5, 21, 63), n_control=60)
    return study, ic

@st.cache_data(show_spinner=False)
def c_strategies(asset: str, start: str, end: str):
    from data import loader
    from backtest import strategies as bt
    px = loader.load_prices([asset], start, end)
    return bt.compare_entry_strategies(px, asset=asset, n_boot=400, start_step=63)

# ---- 建仓作战室（升级模块）----
@st.cache_data(show_spinner=False)
def c_zones(asset: str, start: str, end: str, horizon: int):
    from data import loader
    from regime import entry_cockpit as ec
    px = loader.load_prices([asset], start, end)
    return ec.entry_zones(px, asset=asset, horizon=horizon, n_boot=400)

@st.cache_data(show_spinner=False)
def c_regime_path(asset: str, start: str, end: str, window: int = 504):
    from data import loader
    from analysis import analogs as _an
    px = loader.load_prices([asset], start, end)[asset].dropna()
    return _an.format_regime_path(_an.regime_path_distribution(px, window=window))

@st.cache_data(show_spinner=False)
def c_best_entry_scan(asset: str, start: str, end: str):
    """跨持有期(21/63/126/252)择优：自动挑置信度最高的入场点，避免长周期低置信埋没好结果。"""
    from data import loader
    from regime import entry_cockpit as ec
    px = loader.load_prices([asset], start, end)[asset]
    return ec.best_entry_across_horizons(px, asset=asset, single_name=(asset != "SPY"), n_boot=350)

@st.cache_data(show_spinner=False)
def c_entry_confluence(asset: str, start: str, end: str,
                       warn_red: bool = False, warn_amber: bool = False, warn_label: str = ""):
    """合理入场位：regime/飞刀/离场预警门控 + 可执行回踩支撑（统计锚定价已降级）。"""
    from data import loader
    from regime import entry_cockpit as ec
    ohlcv = loader.load_ohlcv(asset, start, end).dropna()
    return ec.entry_confluence(ohlcv, asset=asset, warn_red=warn_red, warn_amber=warn_amber, warn_label=warn_label)

# 宽度信号用的大盘篮子（跨行业 ~40 只大盘，代表"全市场"宽度）
_BREADTH_BASKET = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "BAC", "V", "UNH",
                   "JNJ", "LLY", "XOM", "CVX", "WMT", "HD", "PG", "KO", "CAT", "BA",
                   "DIS", "NFLX", "INTC", "CSCO", "ORCL", "CRM", "PEP", "MCD", "NKE", "ABT",
                   "TMO", "COST", "AMD", "QCOM", "TXN", "HON", "UNP", "LOW", "GS", "MS"]

@st.cache_data(show_spinner=False)
def c_fragility(start: str, end: str, basket: tuple | None = None):
    """板块脆弱性(宽度恶化)：当前读数 + 实测预警力 + 历史序列。basket=None→全市场40大盘。"""
    from data import loader
    from analysis import fragility as fg
    names = list(basket) if basket else _BREADTH_BASKET
    panel = loader.load_prices(names, start, end)
    # 板块用等权指数作回撤目标；全市场用 SPY
    idx = (loader.load_prices(["SPY"], start, end)["SPY"] if basket is None
           else panel.ffill().pct_change().mean(axis=1).add(1).cumprod())
    return {"cur": fg.current_fragility(panel),
            "eval": {h: fg.evaluate_breadth_warning(panel, idx, horizon=h) for h in (42, 63, 126)},
            "frame": fg.fragility_frame(panel), "idx": idx}


@st.cache_data(show_spinner=False)
def c_overlay(asset: str, start: str, end: str):
    """风险管理叠加(已验证)回测 vs 闭眼持有。用全市场宽度脆弱性 + 该标的趋势/波动。"""
    from data import loader
    from analysis import overlay as ov
    px = loader.load_prices([asset], start, end)[asset]
    fr = c_fragility(start, end)["frame"]["fragile"]
    return ov.backtest_overlay(px, fr)


_FOCUS_UNIVERSE = ["QQQ", "SPY", "XLK", "SMH", "NVDA", "AAPL", "MSFT", "GOOGL",
                   "AMZN", "META", "AVGO", "AMD", "TSM", "CRM"]
# 稳健度档 → 风险叠加目标波动（保守=更稳更低仓，激进=更高仓）。贯穿决策卡仓位+稳定面板。
_PROFILE_VOL = {"保守": 0.10, "均衡": 0.15, "激进": 0.20}


_STABLE_UNIVERSES = {
    "仅指数(SPY/QQQ/DIA)": ["SPY", "QQQ", "DIA"],
    # 跨行业优质大盘篮子：科技/医药/金融/消费/能源/工业分散——直接对治"押单票易归零"的失败点。
    # 全是长历史优质龙头，单票崩了也不至于伤筋动骨(这正是'分散'的意义)。
    "跨行业优质篮子(分散)": ["AAPL", "MSFT", "GOOGL", "JNJ", "UNH", "JPM", "V",
                            "PG", "KO", "HD", "COST", "XOM"],
    "聚焦科技/半导体": _FOCUS_UNIVERSE,
    "七姐妹": ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"],
}


@st.cache_data(show_spinner=False)
def c_stability(uni_key: str, target_vol: float, start: str, end: str):
    """稳定配置回测：选定组合+目标波动，叠加 vs 持有 的稳定性画像 + 净值。"""
    from data import loader
    from analysis import overlay as ov
    prices = {}
    for t in _STABLE_UNIVERSES.get(uni_key, _FOCUS_UNIVERSE):
        try:
            prices[t] = loader.load_prices([t], start, end)[t]
        except Exception:  # noqa: BLE001
            pass
    fr = c_fragility(start, end)["frame"]["fragile"]
    bt = ov.backtest_portfolio(prices, fragile=fr, target_vol=target_vol)
    return {"overlay": ov.stability_stats(bt["ret_overlay"]),
            "hold": ov.stability_stats(bt["ret_hold"]), "equity": bt["equity"]}


@st.cache_data(show_spinner=False)
def c_product_bt(start: str, end: str):
    """产品级组合回测：聚焦组合(ETF+科技+半导体)应用风险叠加 vs 持有 vs SPY。"""
    _cache_ver = "v2-crisis-rollsharpe"  # 改此值可使缓存失效(已加 ret_overlay/ret_hold 字段)
    from data import loader
    from analysis import overlay as ov
    prices = {}
    for t in _FOCUS_UNIVERSE:
        try:
            prices[t] = loader.load_prices([t], start, end)[t]
        except Exception:  # noqa: BLE001
            pass
    fr = c_fragility(start, end)["frame"]["fragile"]
    spy = loader.load_prices(["SPY"], start, end)["SPY"]
    return ov.backtest_portfolio(prices, fragile=fr, benchmark=spy)

@st.cache_data(show_spinner=False)
def c_earnings_reaction(ticker: str, start: str, end: str):
    from data import loader
    from regime import entry_cockpit as ec
    from evaluation import earnings_eval as ee
    price = loader.load_prices([ticker], start, end)[ticker].dropna()
    edates = loader.load_earnings_dates(ticker, limit=80)
    stats = ec.earnings_reaction_stats(price, edates)
    upcoming = ec.upcoming_events(price, edates)
    study = ee.earnings_event_study({ticker: price}, {ticker: edates}, pre=10, post=20, by_beat=True)
    return stats, upcoming, study

@st.cache_data(show_spinner=False)
def c_ladder(asset: str, start: str, end: str, bands: tuple, budget: float = 10000.0):
    from data import loader
    from regime import entry_cockpit as ec
    px = loader.load_prices([asset], start, end)
    return ec.ladder_plan_backtest(px, asset=asset, bands=bands, budget=budget, n_boot=500)

@st.cache_data(show_spinner=False, ttl=1800)
def c_brief(ticker: str, horizon: int, end: str, broad: bool = False):
    """单票综合简报（含全网免费新闻，ttl 30 分钟）。broad=含 GDELT 全球外媒。"""
    from analysis import briefing as bf
    bstart = "1995-01-01" if ticker == "SPY" else "2008-01-01"
    sources = ("google", "yahoo", "gdelt") if broad else ("google", "yahoo")
    return bf.stock_brief(ticker, bstart, end, horizon=horizon, with_news=True, news_sources=sources)


def _fund_cache_save(ticker: str, data: dict) -> None:
    """把最近一次成功取到的基本面落盘，供日后限流时回退（避免硬报错）。"""
    import json as _j, sqlite3 as _sq
    import config as _cfg
    try:
        p = _cfg.user_db_path(); p.parent.mkdir(parents=True, exist_ok=True)
        with _sq.connect(p) as c:
            c.execute("CREATE TABLE IF NOT EXISTS fund_cache(ticker TEXT PRIMARY KEY, json TEXT, fetched_at TEXT)")
            c.execute("INSERT OR REPLACE INTO fund_cache(ticker,json,fetched_at) VALUES(?,?,datetime('now'))",
                      (ticker, _j.dumps(data, default=str)))
    except Exception:  # noqa: BLE001
        pass


def _fund_cache_load(ticker: str):
    """读取上次成功的基本面 (data, fetched_at)；无则 (None, None)。"""
    import json as _j, sqlite3 as _sq
    import config as _cfg
    try:
        with _sq.connect(_cfg.user_db_path()) as c:
            r = c.execute("SELECT json, fetched_at FROM fund_cache WHERE ticker=?", (ticker,)).fetchone()
        if r:
            return _j.loads(r[0]), r[1]
    except Exception:  # noqa: BLE001
        pass
    return None, None


@st.cache_resource(show_spinner=False)
def _fund_seed():
    """随仓库提交的基本面快照（关注清单），供云端限流且无本地缓存时兜底。返回 {ticker: fields}, gen_date。"""
    import json as _j
    from pathlib import Path as _P
    try:
        p = _P(__file__).resolve().parent / "data" / "fundamentals_seed.json"
        obj = _j.loads(p.read_text(encoding="utf-8"))
        return obj.get("data", {}), obj.get("_generated", "")
    except Exception:  # noqa: BLE001
        return {}, ""


def _ensure_ascii_ca():
    """修复 Windows 非 ASCII 路径(如中文目录)下 curl_cffi 读取证书失败(SSLError curl:77)：
    若 certifi 路径含非 ASCII，复制一份到 ASCII 临时路径并指向它。云端 Linux 不受影响、无副作用。"""
    import os
    if os.environ.get("CURL_CA_BUNDLE") and os.environ["CURL_CA_BUNDLE"].isascii():
        return
    try:
        import certifi
        ca = certifi.where()
        if ca.isascii():
            os.environ.setdefault("CURL_CA_BUNDLE", ca); os.environ.setdefault("SSL_CERT_FILE", ca)
            return
        import shutil, tempfile
        dst = os.path.join(tempfile.gettempdir(), "quantlab_cacert_ascii.pem")
        if not (os.path.exists(dst) and os.path.getsize(dst) > 1000):
            shutil.copy(ca, dst)
        os.environ["CURL_CA_BUNDLE"] = dst; os.environ["SSL_CERT_FILE"] = dst
    except Exception:  # noqa: BLE001
        pass


@st.cache_data(show_spinner=False, ttl=3600)
def c_fund_info(ticker: str):
    """取全面基本面字段(体检后)供分析师视角用。ttl 1 小时。

    指数退避重试防 yfinance 限流(429)；取不到则**回退到上次成功落盘的数据**(标注陈旧)，
    彻底没有才抛错——避免云端共享IP被限流时直接给用户硬报错。
    """
    import time as _t
    import yfinance as yf
    _ensure_ascii_ca()
    info = {}
    for _i in range(4):
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:  # noqa: BLE001
            info = {}
        if (info.get("trailingPE") is not None) or (len(info) > 30):
            break
        if _i < 3:
            _t.sleep(0.8 * (2 ** _i))   # 指数退避 0.8/1.6/3.2s，给 Yahoo 解除限流的时间
    if info and len(info) >= 20:
        from analysis.engine_discipline import sanity_check_fundamentals
        from analysis.analyst import FULL_FIELDS
        chk = sanity_check_fundamentals(info)
        merged = dict(info); merged.update(chk["clean"])
        for k in chk["suspicious"]:
            merged.pop(k, None)
        out = {k: merged.get(k) for k in FULL_FIELDS}
        _fund_cache_save(ticker, out)       # 成功 → 落盘
        out["_stale"] = None
        return out
    # 取不到 → ① 回退到本地上次成功；② 回退到仓库内置快照；③ 都没有才抛
    cached, ts = _fund_cache_load(ticker)
    if cached:
        cached["_stale"] = ts
        return cached
    seed, sdate = _fund_seed()
    if ticker in seed:
        out = dict(seed[ticker]); out["_stale"] = f"{sdate}·内置快照"
        return out
    raise RuntimeError(f"yfinance 暂未返回 {ticker} 基本面(限流)，且无历史/内置快照可回退，稍后重试")


@st.cache_data(show_spinner=False, ttl=1800)
def c_read_articles(ticker: str, broad: bool, end: str, limit: int = 5):
    """抓取该票前 limit 条新闻正文 + 抽关键句（慢，ttl 30 分钟）。"""
    from data import news as nws
    sources = ("google", "yahoo", "gdelt") if broad else ("google", "yahoo")
    df = nws.stock_news(ticker, limit=10, sources=sources)
    return nws.read_articles(df, limit=limit)

# ---------------------------------------------------------------------------
# 侧栏
# ---------------------------------------------------------------------------
MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
# 分组股票池（好找）
# 科技/半导体 3x 杠杆 ETF（按板块拆开；也并入对应单票组，浏览时一起看）
_TECH_3X = ["TQQQ", "TECL", "FNGU"]   # 3x 纳指 / 3x 科技 / 3x FANG+
_SEMI_3X = ["SOXL"]                    # 3x 半导体（市面唯一主流 3x 半导体多头）
_TICKER_GROUPS = {
    "🌟 七姐妹": ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"],
    "🖥️ 科技股": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "ORCL", "CRM", "ADBE", "NOW",
                "INTU", "AMD", "PLTR", "UBER", "NFLX", "QCOM", "TXN", "AVGO", "CSCO", "IBM"] + _TECH_3X,
    "🔌 半导体": ["NVDA", "AMD", "AVGO", "TSM", "ASML", "MU", "INTC", "QCOM", "TXN", "ARM", "SMCI", "MRVL"] + _SEMI_3X,
    "🎬 流媒体/其他大盘": ["NFLX", "DIS", "UBER", "PLTR"],
    "🚀 科技3x(杠杆)": _TECH_3X,
    "🔥 半导体3x(杠杆)": _SEMI_3X,
    "⚡ 宽基3x(杠杆)": ["UPRO", "TNA"],   # 3x 标普 / 3x 罗素2000（非科技/半导体，单列）
    "📈 指数ETF": ["SPY", "QQQ", "DIA", "IWM"],
    "🧩 行业ETF": ["XLK", "SMH", "XLC", "XLY", "XLF", "XLV", "XLE", "XLI", "XLP", "XLU", "XLB", "XLRE"],
}
# 脆弱性面板可选的板块宽度篮子（板块级 de-risk 信号；半导体实测 lift 2.85x@42日）
_FRAGILITY_BASKETS = {
    "全市场(40大盘)": None,  # None=用 _BREADTH_BASKET
    "🔌 半导体": ["NVDA", "AMD", "AVGO", "TSM", "ASML", "MU", "INTC", "QCOM", "TXN", "MRVL"],
    "🖥️ 科技股": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "ORCL", "CRM", "ADBE", "NOW",
                "INTU", "AMD", "QCOM", "TXN", "AVGO", "CSCO"],
    "🌟 七姐妹": ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"],
}
_ETF_SET = {"SPY", "QQQ", "DIA", "IWM", "XLK", "SMH", "SOXX", "XLC", "XLY", "XLF", "XLV", "XLE",
            "XLI", "XLP", "XLU", "XLB", "XLRE", "TQQQ", "SOXL", "UPRO", "TECL", "FNGU", "TNA",
            "AGG", "HYG", "IEF", "MTUM", "USMV", "IWD"}   # 无单公司财报的 ETF/基金（用于"下次财报"口径）
_ALL_TICKERS = list(dict.fromkeys(t for g in _TICKER_GROUPS.values() for t in g))
_SPY_FIRST = ["SPY"] + [t for t in _ALL_TICKERS if t != "SPY"]
# 任务导向导航（按"用户要做什么"组织，把核心工作流提到台面，每次只算一个页面）：
#   🎯 个股决策 = 一只票该不该买/在哪买/何时撤（全景图 + 作战卡=入场位/离场警示 + 快照/事件/财报）
#   🛡️ 组合配置 = 一篮子分散+长持（核心打法）   📋 多票简报 = 多票横向对比
#   🔬 研究台 = 选股/因子/regime/回测等研究验证   ℹ️ 关于 = 定位 & 术语
_JOBS = ["🎯 个股决策", "🛡️ 组合配置", "📋 多票简报", "🔬 研究台", "ℹ️ 关于"]
_STOCK_SUB_NAMES = ["📊 全景图（图+裁决）", "🎖️ 作战卡（入场位 / 离场警示）", "📈 当前快照",
                    "🗞️ 事件时间线", "📅 财报 PEAD"]
_RESEARCH_SUB_NAMES = ["🏆 最推荐买入（选股榜）", "🏭 行业动向（半导体/科技）", "🎯 进出场规则（回测器）",
                       "🔬 因子评估", "🌊 大盘 regime（SPY/宏观）", "💰 建仓策略对比"]
_HZ = {"3 个月 (63日)": 63, "6 个月 (126日)": 126, "12 个月 (252日)": 252, "24 个月 (504日)": 504}

with st.sidebar:
    st.markdown("### 📊 量化研究工具")
    st.caption("合理入场 · 离场警示 · 分散长持 — 校准非预测，不许诺跑赢市场")
    tm.toggle(st)   # 🌙/☀️ 明暗主题切换

    # —— 😱 恐慌指数 VIX（侧栏置顶·盘中高频自刷新）——
    # 每次整页重跑都重建 fragment，让 run_every 按当前开/收盘动态切换（盘中15秒、休市不自刷）。
    _vix_open = is_market_open()
    @st.fragment(run_every=("15s" if _vix_open else None))
    def _vix_panel():
        import datetime as _dvix
        _T = tm.tokens()
        # 15 秒桶：盘中每 15 秒强制取一次新报价（绕过 c_live_quote 的 30s 缓存键）
        q = c_live_quote("^VIX", int(_dvix.datetime.now().timestamp() // 15))
        v = q.get("price", float("nan"))
        chg = q.get("change_pct", float("nan"))
        lab, tok, why = _vix_level(v)
        col = _T.get(tok, _T["muted"])
        vtxt = f"{v:.2f}" if v == v else "—"
        if chg == chg:
            ccol = _T["bad"] if chg > 0 else _T["good"]   # VIX↑=恐慌升(红)，VIX↓=趋稳(绿)，与个股相反
            chtxt = f'<span style="font-size:0.8rem;color:{ccol}">{"▲" if chg>0 else "▼"} {chg:+.2%}</span>'
        else:
            chtxt = ""
        status = ("🟢 盘中 · ≈15min延迟 · 每15秒自动刷新" if _vix_open
                  else ("⚪ 休市 · 最近收盘值" if q.get("ok") else "⚪ 取价失败 · 回退收盘"))
        st.markdown(
            f'<div style="border-radius:10px;padding:9px 12px;margin:4px 0 8px;'
            f'background:{col}1f;border:1px solid {col}55;border-left:4px solid {col}">'
            f'<div style="font-size:0.72rem;color:var(--muted);letter-spacing:.4px">😱 恐慌指数 VIX · <b style="color:{col}">{lab}</b></div>'
            f'<div style="display:flex;align-items:baseline;gap:8px;margin-top:1px">'
            f'<span style="font-size:1.55rem;font-weight:800;color:{col};line-height:1.1">{vtxt}</span>{chtxt}</div>'
            f'<div style="font-size:0.66rem;color:var(--muted);margin-top:2px">{why}</div>'
            f'<div style="font-size:0.62rem;color:var(--muted);margin-top:1px">{status}</div>'
            f'</div>', unsafe_allow_html=True)
    _vix_panel()

    st.markdown("**📍 查询**")
    grp = st.selectbox("📂 板块", list(_TICKER_GROUPS), index=0)
    asset = st.selectbox("🎯 选择标的", _TICKER_GROUPS[grp], index=0)
    from analysis.decision import classify as _classify
    if _classify(asset) == "leveraged_etf":   # 3x ETF 出现在多个组里，按标的本身判定而非组名
        st.caption("⚠️ 3x 杠杆 ETF 有每日复利衰减、长持有失真，历史短；分析仅供参考——"
                   "工具已对其给杠杆专属口径（别摊平/只做短线波段/严设止损）。")
    st.markdown("**⚙️ 分析参数**")
    gl_horizon = _HZ[st.selectbox("分析周期", list(_HZ), index=0, help="远期收益/建仓校准的持有期")]
    gl_profile = st.selectbox("🧭 风险偏好", ["保守", "均衡", "激进"], index=1,
                              help="在证据等级给的仓位封顶上做个性化缩放：保守 0.5×、均衡 1×、激进 1.25×（仍受单票硬上限约束）。")
    _pm = {"保守": 0.5, "均衡": 1.0, "激进": 1.25}[gl_profile]
    st.caption(f"→ 当前档位仓位封顶 **×{_pm:g}**（在证据等级给的上限上缩放；单票硬上限不破）")
    if st.button("🔄 重新分析（刷新数据/重算）", use_container_width=True,
                 help="清空缓存并用最新数据重新计算当前页"):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.markdown("**🧭 你想做什么？**")
    job = st.radio("任务", _JOBS, index=0, label_visibility="collapsed")
    sub = None
    if job == "🎯 个股决策":
        sub = st.selectbox("看什么", _STOCK_SUB_NAMES, index=0,
                           help="全景图=图+裁决总览；作战卡=合理入场位+分级离场警示+复利谱")
    elif job == "🔬 研究台":
        sub = st.selectbox("研究工具", _RESEARCH_SUB_NAMES, index=0)
    st.divider()
    st.markdown("**🔧 数据 & 工具**")
    import datetime as _dt
    _today = _dt.date.today().isoformat()
    _presets = {"近 10 年": "2016-01-01", "近 15 年": "2010-01-01", "2005 至今": "2005-01-01", "自定义…": None}
    _choice = st.selectbox("📅 历史区间", list(_presets), index=1, help="回测/分析的历史样本区间")
    if _presets[_choice] is None:
        start = st.text_input("起始", value="2010-01-01"); end = st.text_input("结束", value=_today)
    else:
        start, end = _presets[_choice], _today
    st.caption(f"样本 {start} → {end}　·　yfinance + FRED + 财报，本地缓存")
    with st.expander("💾 我的数据（备份 / 恢复）"):
        from analysis import userdata as _ud
        st.caption("校准信号 / 手填事件 / 检验账本 / 保存的规则。云端重启会清空——**先导出备份**，需要时上传恢复。"
                   "（自建主机可设环境变量 QUANTLAB_DB_PATH 指向持久盘，免手动备份。）")
        try:
            _cnt = _ud.export_userdata().get("_counts", {})
            st.caption("当前：" + (" · ".join(f"{k}={v}" for k, v in _cnt.items()) or "空"))
            st.download_button("⬇️ 导出备份(JSON)", _ud.export_json(),
                               file_name=f"quantlab_backup_{end}.json", mime="application/json",
                               use_container_width=True)
        except Exception as _e:  # noqa: BLE001
            st.caption(f"导出失败：{_e}")
        _up = st.file_uploader("⬆️ 上传备份恢复", type=["json"], key="restore_up")
        if _up is not None:
            _mode = st.radio("恢复方式", ["merge", "replace"], horizontal=True,
                             format_func=lambda m: "合并(追加)" if m == "merge" else "替换(先清空)")
            if st.button("恢复数据", use_container_width=True):
                try:
                    w = _ud.import_userdata(_up.getvalue().decode("utf-8"), mode=_mode)
                    st.success("已恢复：" + (" · ".join(f"{k}+{v}" for k, v in w.items()) or "无"))
                except Exception as _e:  # noqa: BLE001
                    st.error(f"恢复失败：{_e}")

# ===========================================================================
# 页面：信号挖掘（Phase G）
# ===========================================================================
# ===========================================================================
# 页面：概览
# ===========================================================================
def page_overview():
    st.markdown('<div class="hero-title">量化研究工具</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">面向长期个人投资者的<b>规则校准器</b>——把"建仓/进出场/因子"从感觉变成带置信区间和样本数的经验分布。</div>', unsafe_allow_html=True)
    st.write("")
    # —— 产品定位（防误用：别当复利机器/市场跑赢器）——
    st.info(
        "🧭 **这个工具是什么（请先读）**：它是一个**合理入场 + 离场警示 + 分散长持**的"
        "**校准 / 风控**工具。\n\n"
        "- ✅ 它帮你：在技术支撑回踩处分批入场、避开破位飞刀、用分级预警提示减仓、"
        "用一篮子分散+轻保护把回撤压到能扛住（别在崩盘底部割肉）。\n"
        "- ❌ 它**不**做：预测涨跌、给目标价/单一概率、选出会暴涨的票（实测选股无可靠 edge）、"
        "**跑赢市场**——回测一致显示：减暴露必让出复利，绝对收益上没有任何配置跑赢闭眼长持。\n\n"
        "**一句话**：长期最优=买一篮子优质、长期持有；它只帮你**真能拿住**，不替你预测、不许诺超额。")
    st.write("")

    c1, c2, c3 = st.columns(3)
    c1.markdown(stat_card("铁律一", "校准而非预测", "永远输出 分布 + 置信区间 + 样本数 N", T["primary"]), unsafe_allow_html=True)
    c2.markdown(stat_card("铁律二", "永远对比基准", "条件结果必与无脑买入持有并排", T["info"]), unsafe_allow_html=True)
    c3.markdown(stat_card("铁律三", "反过拟合优先", "block bootstrap · walk-forward · deflated Sharpe · N_eff", T["good"]), unsafe_allow_html=True)
    st.write("")

    try:
        macro = c_macro("2000-01-01", end)
        src = macro.attrs.get("sources", {})
        ok = all("fred_api" in v for v in src.values())
        pill = '<span class="pill pill-good">FRED 官方 API 已激活</span>' if ok else '<span class="pill pill-warn">Yahoo 代理回退</span>'
    except Exception:
        pill = '<span class="pill pill-warn">宏观数据未就绪</span>'
        src = {}

    st.markdown("#### 数据源状态")
    st.markdown(pill + " ".join(f'<span class="pill pill-info">{k}: {v}</span>' for k, v in src.items()), unsafe_allow_html=True)

    st.write("")
    st.markdown("#### 模块导航")
    cols = st.columns(5)
    mods = [
        ("🌊 建仓概率引擎", "回撤×估值×信用分桶的条件远期收益 + 当前指纹"),
        ("🔬 因子评估", "价格因子 IC / 分位收益 / 随机因子健全性"),
        ("🎯 个股进出场规则", "七姐妹池化 + N_eff + 基准对比 + PEAD 条件"),
        ("📅 财报 PEAD", "财报日历事件研究 + 领先 IC + 假财报日对照"),
        ("💰 建仓策略对比", "lump/DCA/补仓 同预算滚动对比 + 置信区间"),
    ]
    for col, (t, d) in zip(cols, mods):
        col.markdown(f'<div class="glass" style="min-height:120px"><b>{t}</b><div class="hero-sub">{d}</div></div>', unsafe_allow_html=True)

    st.write("")
    st.caption("💡 全站专业名词都可**鼠标悬浮**看解释（带虚线下划线的词）。下面是完整术语表：")
    with st.expander("📖 术语表（点开查看全部解释）"):
        items = sorted(gl.GLOSSARY.items())
        gcols = st.columns(2)
        for i, (k, v) in enumerate(items):
            gcols[i % 2].markdown(f"**{k}** — {v}")

# ===========================================================================
# 页面：建仓概率引擎
# ===========================================================================
@st.fragment
def page_regime():
    st.markdown('<div class="hero-title">🌊 建仓概率引擎</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">当前点位该不该建仓？粗分桶给条件远期收益的<b>经验分布</b>——禁止光秃秃的"建仓概率73%"。</div>', unsafe_allow_html=True)
    st.write("")

    asset = st.selectbox("标的", _SPY_FIRST, index=0)

    # 今日状态面板（U2，免费可观测状态）
    tp_panel = c_today_panel(asset, "2005-01-01", end)
    st.markdown("##### 📍 今日状态快照（免费可观测，供对照不给结论）")
    pc = st.columns(4)
    vp = tp_panel["vol_percentile"]
    pc[0].markdown(stat_card("波动率状态", {"low_vol": "低波动", "mid_vol": "中波动", "high_vol": "高波动"}.get(tp_panel["vol_state"], "—"),
                             f"年化{tp_panel['realized_vol']:.0%} · 分位{vp:.0%}" if vp == vp else "—",
                             T["info"], tip="regime"), unsafe_allow_html=True)
    pc[1].markdown(stat_card("趋势位置", "均线上方" if tp_panel["trend_state"] == "up_trend" else "均线下方",
                             f"距200日线 {tp_panel['trend_position']:+.1%}", T["good"], tip="regime"), unsafe_allow_html=True)
    pc[2].markdown(stat_card("回撤状态", "回撤中" if tp_panel["drawdown_state"] == "in_drawdown" else "近高点",
                             f"距前高 {tp_panel['drawdown']:+.1%}", T["bad"], tip="回撤"), unsafe_allow_html=True)
    pc[3].markdown(stat_card("快照日期", str(tp_panel["date"].date()), "免费数据·不可预测未来", T["primary"]), unsafe_allow_html=True)
    st.write("")

    with st.spinner("计算条件远期收益…"):
        tab, fp = c_regime(asset, "1993-01-01", end, (21, 63, 126, 252, 504))

    groupings = [g for g in tab["grouping"].unique() if g != "__baseline__"]
    g = st.selectbox("状态维度", groupings, index=0, help=gl.help_for("regime"))
    sub = tab[tab["grouping"] == g]
    buckets = sub["bucket"].unique().tolist()
    b = st.selectbox("状态桶", buckets, index=0, help=gl.help_for("远期收益锥"))

    from regime import conditional_returns as cr
    rows252 = sub[(sub["bucket"] == b) & (sub["horizon"] == 252)]
    if not rows252.empty:
        st.markdown(f'<div class="verdict">{cr.format_bucket_verdict(rows252.iloc[0])}</div>', unsafe_allow_html=True)
    st.write("")

    # 远期收益锥
    bsub = sub[sub["bucket"] == b].sort_values("horizon")
    base = tab[tab["grouping"] == "__baseline__"].sort_values("horizon")
    fig = ch.forward_cone(bsub["horizon"], bsub["p10"], bsub["p25"], bsub["median"],
                          bsub["p75"], bsub["p90"], baseline_med=base["median"].values,
                          title=f"{asset}　状态=[{b}]　远期收益锥（vs 无条件基准）")
    st.plotly_chart(fig, use_container_width=True, config=ch.CHART_CONFIG)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("##### 各状态下 · 未来1年表现")
        show = sub[sub["horizon"] == 252][["bucket", "n_events", "win_rate", "median", "p10", "excess_median", "ci_low", "ci_high"]]
        show = show.rename(columns={"bucket": "状态", "n_events": "历史次数", "win_rate": "胜率",
                                    "median": "中位涨幅", "p10": "最差10%", "excess_median": "比基准多",
                                    "ci_low": "中位区间下", "ci_high": "中位区间上"})
        st.dataframe(show.style.format({"胜率": "{:.0%}", "中位涨幅": "{:+.1%}", "最差10%": "{:+.1%}",
                                        "比基准多": "{:+.1%}", "中位区间下": "{:+.1%}", "中位区间上": "{:+.1%}"}),
                     use_container_width=True, hide_index=True)
        st.caption("📖 **比基准多**=该状态中位收益比随便买入持有(无条件基准)多多少(点估计)；"
                   "**中位区间**=「中位涨幅」的95% bootstrap CI——跨 0 表示**该状态的绝对收益方向都不确定**"
                   "(注意：这是绝对收益的区间，**不是**'比基准多'的显著性检验)；**最差10%**=坏情形收益。")
    with c2:
        st.markdown("##### 今天 vs 历次大跌谷底（供对照）")
        _vmap = {"low": "低估", "mid": "中性", "high": "高估"}
        _cmap = {"widening": "走阔(紧)", "narrowing": "收窄(松)"}
        fp2 = fp[["date", "drawdown", "valuation_tercile", "credit_spread", "credit_trend", "yield_curve"]].rename(
            columns={"date": "日期", "drawdown": "距前高", "valuation_tercile": "估值",
                     "credit_spread": "信用利差", "credit_trend": "信用趋势", "yield_curve": "收益曲线"})
        fp2["估值"] = fp2["估值"].map(lambda v: _vmap.get(v, v))
        fp2["信用趋势"] = fp2["信用趋势"].map(lambda v: _cmap.get(v, v))
        st.dataframe(fp2.style.format({"距前高": "{:+.0%}"}), use_container_width=True)
        st.caption("把今天环境和历史几次大底当时并排，**供你对照**，不替你下结论。")

# ===========================================================================
# 页面：因子评估
# ===========================================================================
@st.fragment
def page_factor():
    st.markdown('<div class="hero-title">🔬 因子评估</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">价格因子对未来收益的领先 IC + 分位收益。随机因子作健全性检查（IC≈0）。</div>', unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns(2)
    factor = c1.selectbox("因子", ["momentum_12_1", "short_reversal", "low_volatility", "trend", "random_factor"],
                          help=gl.help_for("领先"))
    universe = c2.selectbox("股票池", ["spy_demo", "mag7"], index=0, help=gl.help_for("幸存者偏差"))

    run = run_gate("factor", {"factor": factor, "universe": universe}, label="🚀 运行因子评估",
                   hint="选好因子与股票池后点运行——alphalens 评估约 10 秒。")
    if run is None:
        return
    factor, universe = run["factor"], run["universe"]
    with st.spinner("alphalens 评估中…"):
        res = c_factor(factor, universe, start, end)

    ic = res["ic"]
    cols = st.columns(4)
    for col, (per, row) in zip(cols, ic.iterrows()):
        color = T["good"] if abs(row["IC_mean"]) >= 0.03 else T["muted"]
        col.markdown(stat_card(f"预测力 {per}", f"{row['IC_mean']:+.3f}",
                               f"风险调整后 {row['IR']:+.2f} · 样本{int(row['n_days'])}天", color, tip="IC"), unsafe_allow_html=True)
    # 白话强度判断
    best = ic["IC_mean"].abs().max()
    if best >= 0.05:
        lvl, col_ = "较强（少见，留意是否前视偏差）", T["gold"]
    elif best >= 0.03:
        lvl, col_ = "可用（因子本就很弱，靠广度取胜）", T["good"]
    elif best >= 0.015:
        lvl, col_ = "偏弱（单独用价值有限）", T["muted"]
    else:
        lvl, col_ = "≈0（几乎没有预测力）", T["bad"]
    st.markdown(f'<div class="verdict">这个因子对未来收益的<b>预测力：{lvl}</b>（最强周期 IC={best:.3f}）。</div>', unsafe_allow_html=True)
    st.write("")
    st.plotly_chart(ch.factor_ic_bars(ic), use_container_width=True, config=ch.CHART_CONFIG)
    st.caption("📖 看法：**预测力(IC)**=因子值与未来收益的相关系数，**0.03–0.05 就算可用**（因子天生很弱，靠数量多和一致性赚钱）；"
               "**柱子越高越好**，超过虚线(0.03)才算有用；IC 高得离谱(>0.2)反而要怀疑用了未来数据。")
    if factor == "random_factor":
        st.success("✅ 健全性检查：随机因子的预测力应≈0。若明显偏离 0，说明流程有前视偏差(用了未来信息)。")

    # 滚动 IC + 因子衰减（有效期监控）
    st.divider()
    st.markdown("##### 📉 因子衰减 & 滚动 IC（有效持有期 + 是否随时间失效）")
    with st.spinner("计算横截面 IC 衰减曲线…"):
        dec = c_factor_decay(factor, universe, start, end)
    st.markdown(f'<div class="verdict">{dec["verdict"]}</div>', unsafe_allow_html=True)
    dcol = st.columns(2)
    _dd = dec["decay"].dropna()
    if not _dd.empty:
        ddf = pd.DataFrame({"持有期(日)": _dd.index, "平均IC": _dd.values})
        dcol[0].markdown("**IC 随持有期衰减**")
        dcol[0].bar_chart(ddf.set_index("持有期(日)"), color=T["primary"])
    _roll = dec["roll"]
    if _roll is not None and not _roll.empty and _roll["roll_ic"].notna().any():
        dcol[1].markdown("**滚动 IC(126日窗·h21)**")
        dcol[1].line_chart(_roll[["roll_ic"]].dropna(), color=T["info"])
    st.caption("📖 IC 衰减曲线：因子在哪个持有期最强、多久失效（峰值=最佳持有期）。滚动 IC 跌到 0 附近=因子近年走弱。**横截面需多标的，建议用 mag7 池**。")

    # Phase 6（免费）：多因子组合
    st.divider()
    st.markdown("##### 🧬 多因子组合（Phase 6：z-score 等权混合 → IC + 多空回测）")
    blend_sel = st.multiselect("选择要组合的因子（≥2）", ["momentum_12_1", "short_reversal", "low_volatility", "trend"],
                               default=["momentum_12_1", "low_volatility"], help=gl.help_for("IC"))
    brun = run_gate("blend", {"sel": tuple(blend_sel), "u": universe}, label="🚀 运行组合回测",
                    hint="选≥2个因子后点运行。")
    if brun is not None and len(brun["sel"]) >= 2:
        with st.spinner("混合 + 评估 + 多空回测…"):
            bic, bbt = c_blend(brun["sel"], brun["u"], start, end)
        bc = st.columns(4)
        for col, (per, row) in zip(bc, bic["ic"].iterrows()):
            color = T["good"] if abs(row["IC_mean"]) >= 0.03 else T["muted"]
            col.markdown(stat_card(f"组合 IC {per}", f"{row['IC_mean']:+.3f}", f"IR {row['IR']:+.2f}", color, tip="IC"), unsafe_allow_html=True)
        sig = bbt["sharpe_ci_low"] > 0 or bbt["sharpe_ci_high"] < 0
        st.markdown(f'<div class="verdict">多空组合年化夏普 {bbt["sharpe"]:+.2f}（95% CI [{bbt["sharpe_ci_low"]:+.2f}, {bbt["sharpe_ci_high"]:+.2f}]，{"显著" if sig else "不显著（CI 跨 0）"}）·年化收益 {bbt["ann_return"]:+.1%}。{bbt["note"]}</div>', unsafe_allow_html=True)
        st.plotly_chart(ch.factor_ic_bars(bic["ic"], title="组合因子 IC"), use_container_width=True, config=ch.CHART_CONFIG)
    elif brun is not None and len(brun["sel"]) < 2:
        st.caption("至少选 2 个因子做组合。")

# ===========================================================================
# 页面：个股进出场规则
# ===========================================================================
def _load_saved_rule(name: str) -> bool:
    """把已保存规则写入各控件的 session_state（供 rerun 后回填）。返回是否成功。"""
    from frontend import store
    spec = store.get_rule(name)
    if not spec:
        return False
    specs = spec.get("specs", [])
    if any(s[0] == "vol_regime" for s in specs):
        st.warning("该规则含 vol_regime（含非数值参数），请手动设置后再回测。")
    ss = st.session_state
    ss["nsig"] = len(specs)
    ss["combop"] = spec.get("op", "and")
    for i, s in enumerate(specs[:3], start=1):
        nm = s[0]
        ss[f"sig{i}"] = nm
        if nm == "dip_from_high":
            ss[f"p1_{i}"] = float(s[1]); ss[f"p2_{i}"] = 0.0
        elif nm in ("rsi_oversold", "ma_cross"):
            ss[f"p1_{i}"] = int(s[1]); ss[f"p2_{i}"] = int(s[2])
    ss["trailing"] = float(spec.get("trailing", 0.20))
    ss["tp"] = float(spec.get("tp", 0.25))
    ss["ts"] = int(spec.get("time_stop", 63))
    ss["cond_kind"] = spec.get("cond_kind", "none")
    ss["regime_kind"] = spec.get("regime_kind", "none")
    ss["ma_exit"] = int(spec.get("ma_exit", 0))
    if spec.get("cond_kind", "none") != "none":
        ss["cond_window"] = int(spec.get("cond_window", 20))
    return True

def _signal_controls(slot: int, default: str):
    """渲染单个入场信号的控件，返回 (name, p1, p2)。"""
    sigmap = {"dip_from_high": "距高点回撤", "rsi_oversold": "RSI 超卖", "ma_cross": "均线金叉", "vol_regime": "波动率状态"}
    cc = st.columns([2, 2, 2])
    name = cc[0].selectbox(f"信号 {slot}", list(sigmap), index=list(sigmap).index(default),
                           format_func=lambda k: sigmap[k], key=f"sig{slot}")
    if name == "dip_from_high":
        p1 = cc[1].slider("回撤阈值", 0.05, 0.30, 0.15, 0.01, key=f"p1_{slot}", help=gl.help_for("回撤")); p2 = 0.0
    elif name == "rsi_oversold":
        p1 = cc[1].slider("RSI 窗口", 5, 30, 14, 1, key=f"p1_{slot}")
        p2 = cc[2].slider("超卖线", 10, 40, 30, 1, key=f"p2_{slot}")
    elif name == "ma_cross":
        p1 = cc[1].slider("快线", 10, 100, 50, 5, key=f"p1_{slot}")
        p2 = cc[2].slider("慢线", 100, 300, 200, 10, key=f"p2_{slot}")
    else:  # vol_regime
        p1 = cc[1].slider("窗口", 5, 60, 20, 1, key=f"p1_{slot}")
        p2 = 1.0 if cc[2].radio("状态", ["低波动", "高波动"], key=f"p2_{slot}") == "低波动" else 0.0
    return (name, float(p1), float(p2))

@st.fragment
def page_rule():
    st.markdown('<div class="hero-title">🎯 个股进出场规则</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">七姐妹<b>池化</b>逐笔评估：N_eff 折算高相关证据 · 入场×出场成对 · 对比随机基准。禁止"最佳买点/目标价"。</div>', unsafe_allow_html=True)
    st.write("")

    # 📥 载入已保存规则（完成保存→复用闭环）
    from frontend import store as _store
    _saved = _store.list_rules()
    if _saved:
        lc = st.columns([3, 1])
        _names = [r["名称"] for r in _saved]
        pick = lc[0].selectbox("📥 载入已保存规则", ["—"] + _names)
        if lc[1].button("载入", use_container_width=True) and pick != "—":
            if _load_saved_rule(pick):
                st.rerun()

    with st.container(border=True):
        st.markdown("**1️⃣ 入场信号**（可组合）")
        cc = st.columns([1, 1])
        n_sig = cc[0].radio("信号数量", [1, 2, 3], horizontal=True, key="nsig")
        op = cc[1].radio("组合方式", ["and", "or"], horizontal=True,
                         format_func=lambda k: "全部满足 AND" if k == "and" else "任一满足 OR", key="combop")
        defaults = ["dip_from_high", "rsi_oversold", "ma_cross"]
        specs = tuple(_signal_controls(i + 1, defaults[i]) for i in range(int(n_sig)))

        st.markdown("**2️⃣ 出场规则**（与入场成对）")
        c4, c5, c6, c7 = st.columns(4)
        trailing = c4.slider("移动止损", 0.05, 0.40, 0.20, 0.01, help=gl.help_for("移动止损"), key="trailing")
        tp = c5.slider("止盈", 0.05, 0.60, 0.25, 0.01, help=gl.help_for("止盈"), key="tp")
        time_stop = c6.slider("时间止损(日)", 10, 252, 63, 1, help=gl.help_for("时间止损"), key="ts")
        ma_exit = c7.selectbox("趋势跌破出场", [0, 50, 100, 200], index=0,
                               format_func=lambda w: "无" if w == 0 else f"跌破{w}日线",
                               help="收盘跌破 N 日均线才出场——让利润奔跑、破势才走（趋势股推荐）", key="ma_exit")

        with st.expander("🧭 条件门（可选）— 仅在某财报窗口 / 市场状态下启用规则"):
            cg1, cg2 = st.columns(2)
            cond_kind = cg1.selectbox("财报条件（PEAD）",
                                      ["none", "post_beat", "post_earnings", "pre_earnings", "away_from_earnings"],
                                      format_func=lambda k: {"none": "无", "post_beat": "财报后超预期窗口",
                                                             "post_earnings": "财报后窗口", "pre_earnings": "财报前窗口",
                                                             "away_from_earnings": "远离财报"}[k],
                                      help=gl.help_for("PEAD"), key="cond_kind")
            regime_kind = cg2.selectbox("市场状态条件",
                                        ["none", "up_trend", "down_trend", "low_vol", "high_vol", "in_drawdown", "near_high"],
                                        format_func=lambda k: {"none": "无", "up_trend": "仅上升趋势", "down_trend": "仅下降趋势",
                                                               "low_vol": "仅低波动", "high_vol": "仅高波动",
                                                               "in_drawdown": "仅回撤中", "near_high": "仅近高点"}[k],
                                        help=gl.help_for("regime"), key="regime_kind")
            cond_window = st.slider("财报窗口(日)", 5, 40, 20, 1, key="cond_window") if cond_kind != "none" else 20

    rule_name = ("+".join(s[0] for s in specs) + f"_{op}"
                 + (f"_{cond_kind}" if cond_kind != "none" else "")
                 + (f"_{regime_kind}" if regime_kind != "none" else ""))
    params = dict(specs=specs, op=op, trailing=trailing, tp=tp, time_stop=int(time_stop),
                  cond_kind=cond_kind, cond_window=int(cond_window), regime_kind=regime_kind,
                  ma_exit=int(ma_exit), rule_name=rule_name)

    st.divider()
    run = run_gate("rule", params, label="🚀 运行池化回测",
                   hint="设好入场/出场/条件后点运行——七票逐笔回测约 8 秒，期间拖动滑块不会卡。")
    if run is None:
        return
    specs, op = run["specs"], run["op"]
    trailing, tp, time_stop = run["trailing"], run["tp"], run["time_stop"]
    cond_kind, cond_window, regime_kind, rule_name = run["cond_kind"], run["cond_window"], run["regime_kind"], run["rule_name"]
    ma_exit = run.get("ma_exit", 0)
    st.session_state["_last_rule"] = run

    with st.spinner("七票逐笔回测 + 池化 + bootstrap…"):
        res = c_rule(specs, op, trailing, tp, int(time_stop), "mag7", start, end, cond_kind, int(cond_window), regime_kind, rule_name, int(ma_exit))
    p = res["pooled"]

    st.markdown("#### 📊 结果")
    st.markdown(f'<div class="verdict">{res["_verdict"]}</div>', unsafe_allow_html=True)
    st.write("")
    cols = st.columns(5)
    cols[0].markdown(stat_card("交易笔数 / N_eff", f"{p['n_trades']}", f"N_eff≈{p['n_eff']:.0f} · ρ̄={p['rho_bar']:.2f}", T["primary"], tip="N_eff"), unsafe_allow_html=True)
    cols[1].markdown(stat_card("胜率", f"{p['win_rate']:.0%}", "", T["info"], tip="胜率"), unsafe_allow_html=True)
    cols[2].markdown(stat_card("收益中位", f"{p['median_return']:+.1%}", f"5分位 {p['p5_return']:+.1%}", T["good"], tip="远期收益"), unsafe_allow_html=True)
    cols[3].markdown(stat_card("MAE 中位", f"{p['median_mae']:+.1%}", f"最长连亏 {p['longest_losing_streak']}", T["bad"], tip="MAE"), unsafe_allow_html=True)
    excol = T["good"] if p["excess_significant"] else T["muted"]
    cols[4].markdown(stat_card("超额(vs随机基准)", f"{p['excess_median']:+.1%}", f"CI[{p['excess_ci_low']:+.0%},{p['excess_ci_high']:+.0%}] {'显著' if p['excess_significant'] else '不显著'}", excol, tip="超额"), unsafe_allow_html=True)
    if len(specs) > 1:
        st.caption(f"⚠️ 多重检验：{len(specs)} 信号组合，批量调参时看下方「反过拟合体检」。")
    st.write("")

    # 🚦 验收闸门：跑赢基准 + OOS夏普≥1 + 回撤可承受（合一 go/no-go）
    with st.container(border=True):
        gc1, gc2, gc3, gc4 = st.columns([1.4, 1, 1, 1])
        gc1.markdown("**🚦 验收闸门**")
        g_uni = gc2.selectbox("票池", ["mag7", "diversified"], key="gate_uni",
                              format_func=lambda k: "七姐妹" if k == "mag7" else "降偏差宽池",
                              help="选『降偏差宽池』看选股偏差被剥掉后规则还成不成立")
        g_sh = gc3.slider("OOS夏普门槛", 0.0, 2.0, 1.0, 0.1, key="gate_sh")
        g_dd = gc4.slider("回撤容忍", 0.10, 0.60, 0.35, 0.05, key="gate_dd")
        if st.button("运行验收闸门（OOS 滚动夏普 + 回撤，约 10–25 秒）", key="run_gate_btn"):
            with st.spinner("跑 OOS 日度夏普 + 最大回撤…"):
                gate = c_gate(specs, op, trailing, tp, int(time_stop), g_uni, start, end,
                              int(ma_exit), float(g_sh), float(g_dd))
            head = "🟢 PASS · 过反过拟合及格线（非买入信号）" if gate["overall"] else "🔴 FAIL · 未达标，别信"
            st.markdown(f'<div class="verdict">{head}</div>', unsafe_allow_html=True)
            cb = gate["criteria"]
            gk = st.columns(3)
            b1 = cb["beat_baseline"]
            gk[0].markdown(stat_card(("✅ " if b1["pass"] else "❌ ") + "跑赢随机基准",
                                     f"{b1['excess_median']:+.1%}",
                                     f"CI[{b1['ci'][0]:+.0%},{b1['ci'][1]:+.0%}]{'显著' if b1['significant'] else '不显著'}·N_eff≈{b1['n_eff']:.0f}",
                                     T["good"] if b1["pass"] else T["bad"], tip="超额"), unsafe_allow_html=True)
            b2 = cb["oos_sharpe"]
            gk[1].markdown(stat_card(("✅ " if b2["pass"] else "❌ ") + f"OOS夏普≥{b2['threshold']}",
                                     f"{b2['value']:.2f}" if b2["value"] == b2["value"] else "NA",
                                     f"OOS {b2['n_oos_days']} 日", T["good"] if b2["pass"] else T["bad"], tip="Sharpe"), unsafe_allow_html=True)
            b3 = cb["drawdown"]
            gk[2].markdown(stat_card(("✅ " if b3["pass"] else "❌ ") + f"最大回撤≤{abs(b3['tolerance']):.0%}",
                                     f"{b3['max_drawdown']:.0%}", "池化日度净值", T["good"] if b3["pass"] else T["bad"], tip="水下图"), unsafe_allow_html=True)
            st.caption("三条全过才 PASS。PASS≠买入信号，只代表过了反过拟合及格线；选『降偏差宽池』常会把七姐妹上的假优势打回原形。")
    st.write("")

    trades = res["trades"]
    t0, t1, t2, t3 = st.tabs(["📈 K线进出场标记", "收益分布", "MAE / 水下图", "相关矩阵 + 单票"])
    with t0:
        tk = st.selectbox("查看单只标的（仅复核规则在买卖什么，池化结论仍基于七票）", _ALL_TICKERS, index=0)
        with st.spinner(f"{tk} K线 + 逐笔标记…"):
            ohlcv, str_trades = c_single_trades(specs, op, trailing, tp, int(time_stop), tk, start, end, cond_kind, int(cond_window), regime_kind, int(ma_exit))
        try:
            render_tv_candles(ohlcv, str_trades, key=f"tv_rule_{tk}", height=540, log=True)
        except Exception:  # noqa: BLE001
            st.plotly_chart(ch.trade_map_candles(ohlcv, str_trades, title=f"{tk}　{rule_name}"), use_container_width=True, config=ch.CHART_CONFIG)
        st.caption("🖱️ TradingView 手感：滚轮缩放 · 拖动平移 · 十字光标(轴上显示价/时间) · 缩放自动适配 y。▲买 ▼卖（按盈亏着色）。仅复核规则，无画线工具/不标买卖点。")
    with t1:
        st.plotly_chart(ch.return_hist(trades["return"], median=p["median_return"],
                                       baseline_median=p["baseline_median"], n=p["n_trades"],
                                       n_eff=p["n_eff"], title="单笔收益分布"), use_container_width=True, config=ch.CHART_CONFIG)
    with t2:
        a, b = st.columns(2)
        a.plotly_chart(ch.mae_hist(trades["mae"], n=p["n_trades"]), use_container_width=True)
        b.plotly_chart(ch.equity_underwater(trades), use_container_width=True)
    with t3:
        from data import loader
        px = loader.load_prices(MAG7, start, end)
        corr = px.pct_change().corr()
        st.plotly_chart(ch.corr_heatmap(corr), use_container_width=True, config=ch.CHART_CONFIG)
        st.caption("📖 相关矩阵：越红=两只票越同涨同跌。七姐妹普遍 0.4–0.6，所以 7 只≈不到 3 个独立赌注(N_eff)，证据要打折。")
        pt = pd.DataFrame(res["per_ticker"]).T.rename(
            columns={"n_trades": "笔数", "win_rate": "胜率", "median_return": "收益中位"})
        st.markdown("**各票单独表现（仅展示，结论仍以池化为准）**")
        st.dataframe(pt.style.format({"胜率": "{:.0%}", "收益中位": "{:+.1%}"}), use_container_width=True)
    st.caption("仅复核规则与理解分布，非买卖点建议。")

    # 🛠️ 更多工具（保存/导出/体检/绩效，全部收进折叠以保持清爽）
    st.write("")
    with st.expander("🛠️ 更多工具：保存 · 导出报告 · 反过拟合体检 · quantstats 绩效"):
        tA, tB, tC = st.tabs(["💾 保存 / 导出", "🛡️ 反过拟合体检", "📑 quantstats"])
        with tA:
            from frontend import store
            sc1, sc2, sc3 = st.columns([2, 1, 1])
            save_name = sc1.text_input("规则名称", value=rule_name, key="save_name")
            if sc2.button("💾 保存", use_container_width=True):
                store.save_rule(save_name, st.session_state["_last_rule"])
                st.success(f"已保存：{save_name}")
            if sc3.button("📄 导出HTML", use_container_width=True):
                from frontend import report as rep
                path = rep.export_rule_report(save_name, res, specs, op, _exit_spec(trailing, tp, int(time_stop)))
                st.success(f"已导出：{path}")
            saved = store.list_rules()
            if saved:
                st.dataframe(pd.DataFrame(saved), use_container_width=True, hide_index=True)
        with tB:
            st.caption("围绕当前规则缩放首信号参数×移动止损生成 3×3 候选：IS 选优 / OOS 报 + deflated Sharpe。缺口大、OOS 差 = 过拟合。")
            if st.button("运行体检（约 20–40 秒）"):
                with st.spinner("walk-forward + deflated Sharpe…"):
                    deflated, wf = c_overfit_check(specs, op, trailing, tp, int(time_stop), start, end)
                dc = st.columns(3)
                dc[0].markdown(stat_card("最优候选", deflated["best_name"], f"per-trade Sharpe {deflated['best_sharpe']:.2f}", T["primary"], tip="Sharpe"), unsafe_allow_html=True)
                dc[1].markdown(stat_card("Deflated Sharpe 概率", f"{deflated['deflated_sharpe_prob']:.2f}", f"{deflated['n_trials']} 候选 · {'稳健' if deflated['robust'] else '存疑'}", T["good"] if deflated["robust"] else T["bad"], tip="deflated Sharpe"), unsafe_allow_html=True)
                s = wf["summary"]
                dc[2].markdown(stat_card("IS vs OOS Sharpe", f"{s['mean_is_sharpe']:.2f} / {s['mean_oos_sharpe']:.2f}", f"过拟合缺口 {s['overfit_gap']:+.2f}", T["info"], tip="walk-forward"), unsafe_allow_html=True)
                st.markdown(f'<div class="verdict">{deflated["note"]}<br>{wf["note"]}</div>', unsafe_allow_html=True)
                wfv = wf["table"].dropna(subset=["oos_sharpe"])
                if not wfv.empty:
                    st.plotly_chart(ch.walk_forward(wfv), use_container_width=True, config=ch.CHART_CONFIG)
        with tC:
            qtk = st.selectbox("标的", _ALL_TICKERS, index=0, key="qs_tk")
            if st.button("生成 quantstats 报告"):
                try:
                    with st.spinner("生成 tear sheet…"):
                        rets = c_single_equity(specs, op, trailing, tp, int(time_stop), qtk, start, end, cond_kind, int(cond_window), regime_kind)
                        path = export_quantstats(rets, qtk, rule_name)
                    st.success(f"已生成：{path}")
                    st.caption("用浏览器打开该 HTML 看完整水下图/回撤/月度收益。")
                except Exception as e:  # noqa: BLE001
                    st.warning(f"生成失败：{type(e).__name__}: {e}")

# ===========================================================================
# 页面：当前快照（Volume Profile + 期权，免费、仅展示、不可回测）
# ===========================================================================
@st.fragment
def page_snapshot():
    st.markdown('<div class="hero-title">📈 当前快照</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">免费日线 Volume Profile（筹码密集区/POC）+ 期权当前快照。<b>仅当前状态、不可回测</b>，作支撑/压力的可视化辅助，非信号。</div>', unsafe_allow_html=True)
    st.write("")

    tk = st.selectbox("标的", _ALL_TICKERS, index=0)
    lookback = st.slider("回看交易日", 120, 756, 252, 21)

    with st.spinner("计算 Volume Profile…"):
        ohlcv, vp = c_volume_profile(tk, "2015-01-01", end, int(lookback))
    pc = st.columns(3)
    pc[0].markdown(stat_card("POC（成交最密价位）", f"{vp['poc']:.1f}", "强支撑/压力参考", T["info"], tip="POC"), unsafe_allow_html=True)
    pc[1].markdown(stat_card("价值区(70%)", f"{vp['value_area'][0]:.0f} – {vp['value_area'][1]:.0f}", "成交集中区间", T["primary"], tip="Volume Profile"), unsafe_allow_html=True)
    # 现价接入盘中近实时（休市/取价失败回退日线收盘）
    _q = c_live_quote(tk, int(__import__("datetime").datetime.now().timestamp() // 30))
    if _q.get("ok") and _q["price"] == _q["price"]:
        _ch = _q.get("change_pct")
        _arr = ("▲" if (_ch == _ch and _ch >= 0) else "▼") if _ch == _ch else ""
        _sub = (f"{_arr} {_ch:+.2%} · {'🟢盘中' if is_market_open() else '⚪休市'}" if _ch == _ch else "近实时")
        _col = T["good"] if (_ch == _ch and _ch >= 0) else T["bad"]
        pc[2].markdown(stat_card("现价(≈15min延迟)", f"{_q['price']:.2f}", _sub, _col), unsafe_allow_html=True)
    else:
        pc[2].markdown(stat_card("现价", f"{ohlcv['close'].iloc[-1]:.1f}", str(ohlcv.index[-1].date()), T["good"]), unsafe_allow_html=True)
    st.write("")
    a, b = st.columns([3, 1])
    with a:
        # 用 Plotly candle_with_levels：POC/价值区横线与筹码柱均为数据 trace，浏览器必现
        # （lightweight-charts 的 priceLine 在本环境里横线不渲染，与全景主图同因，故统一改用此图）
        st.plotly_chart(ch.candle_with_levels(ohlcv, vp, title=f"{tk}　近{lookback}日 K线 + 筹码位"),
                        use_container_width=True, config=ch.TV_CONFIG)
        st.caption("🖱️ 滚轮缩放·拖动平移·十字光标读价（双击复位）。蓝虚线=POC 最高换手价；紫带=价值区(70%成交)；左侧紫柱=各价位筹码量。")
    b.plotly_chart(ch.volume_profile_bars(vp, title="筹码分布"), use_container_width=True)

    st.divider()
    st.markdown("##### 🎰 期权当前快照（不可回测）")
    if st.button("加载期权链快照", help="实时拉取 yfinance 期权链；仅当前，不可回测"):
        try:
            with st.spinner("拉取期权链…"):
                snap = c_options(tk)
            oc = st.columns(4)
            ivc, ivp = snap["atm_iv_call"], snap["atm_iv_put"]
            oc[0].markdown(stat_card("ATM IV(Call)", f"{ivc:.0%}" if ivc == ivc else "—", "隐含波动率", T["info"], tip="IV"), unsafe_allow_html=True)
            oc[1].markdown(stat_card("ATM IV(Put)", f"{ivp:.0%}" if ivp == ivp else "—", "隐含波动率", T["primary"], tip="IV"), unsafe_allow_html=True)
            oc[2].markdown(stat_card("Put/Call (OI)", f"{snap['put_call_oi_ratio']:.2f}", ">1 偏空对冲", T["bad"]), unsafe_allow_html=True)
            oc[3].markdown(stat_card("最大OI磁吸位", f"P{snap['max_oi_put_strike']:.0f} / C{snap['max_oi_call_strike']:.0f}", "下方支撑/上方压力", T["good"]), unsafe_allow_html=True)
            st.plotly_chart(ch.options_oi(snap), use_container_width=True, config=ch.CHART_CONFIG)
            st.caption("⚠️ 期权数据仅当前快照，历史不可得（需付费）；IV 类 regime 条件请用现货历史波动率 proxy。")
        except Exception as e:  # noqa: BLE001
            st.warning(f"期权链拉取失败（可能无期权或网络限制）：{e}")

# ===========================================================================
# 页面：财报 PEAD
# ===========================================================================
@st.fragment
def page_earnings():
    st.markdown('<div class="hero-title">📅 财报 PEAD</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">免费财报日历因子：盈利超预期后漂移。只测对<b>未来</b>收益的领先力 + 蒙特卡洛假财报日对照(IC≈0)。</div>', unsafe_allow_html=True)
    st.write("")

    with st.spinner("事件研究 + PEAD 领先 IC（蒙特卡洛对照）…"):
        study, ic = c_earnings_study("mag7", start, end)

    tab = ic["ic_table"]
    cols = st.columns(4)
    for col, (_, row) in zip(cols, tab.iterrows()):
        sig = row["significant"]
        col.markdown(stat_card(f"领先 IC {int(row['horizon'])}日", f"{row['ic_real']:+.3f}",
                               f"p={row['perm_pvalue']:.2f} {'显著' if sig else '不显著'} · 假对照{row['ic_fake_mean']:+.3f}",
                               T["good"] if sig else T["muted"], tip="PEAD"), unsafe_allow_html=True)
    # 记入全局多重检验账本
    try:
        from analysis import mt_ledger as _mt
        for _, row in tab.iterrows():
            _mt.log_test("PEAD", f"领先IC_h{int(row['horizon'])}", float(row["perm_pvalue"]), stat=float(row["ic_real"]))
    except Exception:  # noqa: BLE001
        pass
    st.write("")
    a, b = st.columns([3, 2])
    a.plotly_chart(ch.event_study(study), use_container_width=True)
    b.plotly_chart(ch.ic_bars(tab), use_container_width=True)
    st.markdown(f'<div class="verdict">{ic["note"]}</div>', unsafe_allow_html=True)
    st.caption("📖 看法：**领先 IC**=「上次财报超预期幅度」与「公布后 N 日涨幅」的相关性，正且显著=超预期后真有继续上涨的漂移；"
               "**p<0.05 才算显著**；**假对照≈0** 证明不是巧合。左图绿线(超预期)在财报日后高于红线(不及预期)=漂移存在。")

# ===========================================================================
# 页面：事件时间线（SEC EDGAR + 财报，免费，仅复盘）
# ===========================================================================
@st.fragment
def page_events():
    st.markdown('<div class="hero-title">🗞️ 事件时间线</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">SEC EDGAR filings（10-K/10-Q/8-K…）+ 财报，自动对齐<b>客观价格反应</b>。<b>产出供人复盘，不作信号</b>，主观判断禁止流入回测。</div>', unsafe_allow_html=True)
    st.write("")

    ec = st.columns([2, 3, 1])
    tk = ec[0].selectbox("标的", _ALL_TICKERS, index=0)
    from data.edgar import MATERIAL_FORMS
    forms = ec[1].multiselect("filing 类型", MATERIAL_FORMS, default=["10-K", "10-Q", "8-K"])
    inc_earn = ec[2].checkbox("含财报", value=True)
    with st.spinner("拉取 SEC EDGAR + 财报，对齐价格反应…"):
        try:
            price, ev = c_events(tk, "2015-01-01", None, tuple(forms), inc_earn)
        except Exception as e:  # noqa: BLE001
            st.warning(f"SEC EDGAR 拉取失败（网络限制？）：{type(e).__name__}: {e}")
            return
    if ev.empty:
        st.info("当前筛选无事件，请放宽 filing 类型或勾选含财报。")
        return

    # 各类事件的平均 5 日反应
    summ = ev.dropna(subset=["reaction_5d"]).groupby("label")["reaction_5d"].agg(["count", "mean"])
    if not summ.empty:
        top = summ.sort_values("mean", ascending=False).head(6)
        st.caption("各事件类型平均 5 日反应（客观，仅复盘）：" +
                   " · ".join(f"{i}({int(r['count'])}): {r['mean']:+.1%}" for i, r in top.iterrows()))

    try:
        pser = price.dropna()
        mk = []
        for _, r in ev.iterrows():
            d = pd.Timestamp(r["date"])
            near = pser.index[pser.index.get_indexer([d], method="nearest")[0]]
            col = T["good"] if r["type"] == "财报" else T["info"]
            rr = r.get("reaction_5d", float("nan"))
            txt = f"{r['label']}" + (f" {rr:+.0%}" if rr == rr else "")
            mk.append({"time": near.strftime("%Y-%m-%d"), "position": "aboveBar",
                       "color": col, "shape": "circle", "text": txt})
        render_tv_line(pser, markers=mk, key=f"tvev_{tk}", height=440, color=T["muted"], log=True)
        st.caption("🖱️ TradingView 操作：滚轮缩放 · 拖动平移 · 十字光标。圆点=事件(绿=财报/蓝=SEC)，鼠标悬浮看 5 日反应。")
    except Exception:  # noqa: BLE001
        st.plotly_chart(ch.event_timeline_chart(price, ev, title=f"{tk}　事件时间线（客观价格反应）"), use_container_width=True, config=ch.CHART_CONFIG)
    st.markdown("##### 近期事件 + 客观反应（次个交易日起）")
    show = ev[["date", "type", "label", "reaction_1d", "reaction_5d"]].head(40)
    st.dataframe(
        show.style.format({"reaction_1d": "{:+.1%}", "reaction_5d": "{:+.1%}"}),
        use_container_width=True, hide_index=True,
    )
    st.caption("仅客观字段（事件类型/日期/价格反应）；事件的重要性/性质属主观判断，不入量化、不作买卖依据。")

# ===========================================================================
# 页面：建仓策略对比
# ===========================================================================
@st.fragment
def page_strategies():
    st.markdown('<div class="hero-title">💰 建仓策略对比</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">同一预算、同一持有窗口，lump_sum / DCA / 补仓 谁更优？滚动多窗口 + block bootstrap 置信区间。</div>', unsafe_allow_html=True)
    st.write("")

    asset = st.selectbox("标的", _SPY_FIRST, index=0)
    with st.spinner("滚动窗口模拟 + bootstrap…"):
        res = c_strategies(asset, "1995-01-01" if asset == "SPY" else start, end)

    st.markdown(f'<div class="hero-sub">{res["note"]}</div>', unsafe_allow_html=True)
    st.write("")
    st.plotly_chart(ch.strategy_compare(res["per_strategy"]), use_container_width=True, config=ch.CHART_CONFIG)

    _nm = {"lump_sum": "一次性买入", "dca": "定投", "average_down": "越跌越补"}
    st.markdown("##### 分批 vs 一次性买入：到底差多少")
    rows = []
    for k, v in res["vs_lump_sum"].items():
        rows.append({"策略": _nm.get(k, k), "比一次性多/少": v["median_diff"],
                     "区间下": v["ci_low"], "区间上": v["ci_high"],
                     "跑赢一次性的比例": v["beats_lump_rate"],
                     "差异是否确凿": "✅ 确凿" if v["significant"] else "— 看不出"})
    df = pd.DataFrame(rows)
    st.dataframe(df.style.format({"比一次性多/少": "{:+.1%}", "区间下": "{:+.1%}", "区间上": "{:+.1%}", "跑赢一次性的比例": "{:.0%}"}),
                 use_container_width=True, hide_index=True)
    st.caption("📖 **比一次性多/少**=该分批法相对「一把全买」的资本回报差；**区间**跨 0=差异看不出。"
               "结论通常是：长期上涨的票里，**一次性买入往往胜过分批**(分批让闲钱空等)。")

# ===========================================================================
# 页面：建仓作战室（升级模块）—— 校准式：价位带分布 + 盈亏比/期望值 + 事件日程 + 阶梯布局
# ===========================================================================
_HORIZON_OPTS = {"3 个月 (63日)": 63, "6 个月 (126日)": 126, "12 个月 (252日)": 252, "24 个月 (504日)": 504}

# ===========================================================================
# 页面：多票作战简报（综合层）—— 一屏总览 + 每票建仓档 + 引擎桶 + 财报 + 免费新闻 + 自动权重
# ===========================================================================
def _brief_overview_row(b: dict, live: dict | None = None) -> dict:
    be = b.get("engine_headline")
    trap = "⚠️" if b.get("momentum_trap") else ""
    sig = "✅" if (be and be["significant"] and be["excess"] > 0) else ""
    bk = f"{be['bucket']} {be['median']:+.1%}(超额{be['excess']:+.1%}){sig}{trap}" if be else "—"
    volp = f"{b['vol_percentile']:.0%}" if b["vol_percentile"] == b["vol_percentile"] else "—"
    # 现价：有实时报价用实时（带今日涨跌），否则回退日线收盘
    if live and live.get("ok") and live["price"] == live["price"]:
        px = f"{live['price']:.2f}"
        _c = live.get("change_pct")
        chg = (f"{'▲' if _c >= 0 else '▼'} {_c:+.2%}") if _c == _c else "—"
    else:
        px, chg = f"{b['price']:.1f}", "—"
    return {
        "标的": b["ticker"], "现价": px, "今日涨跌": chg,
        "趋势/距历史高": f"{b['trend']} / {b['drawdown']:.1%}", "波动分位": volp,
        "引擎最优桶": bk,
        "盈亏比": f"{be['reward_risk']:.2f}" if (be and be["reward_risk"] == be["reward_risk"]) else "—",
        "胜率": f"{be['win_rate']:.0%}" if be else "—",
        "下次财报": b.get("next_earnings") or "—",
    }

@st.fragment
def page_briefing():
    st.markdown('<div class="hero-title">📋 多票作战简报</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">多只一屏总览 + 每票<b>建仓档(技术位共振→目标/止损/盈亏比)</b>'
                ' + 引擎当前状态桶 + 财报日程/drift + 免费新闻 + <b>自动权重</b>。可导出 Markdown。</div>',
                unsafe_allow_html=True)
    st.warning("⚠️ 可计算层(表/档位/RR/引擎结论)来自价格与引擎、诚实可复现；**新闻/基本面仅供人读、不入量化、含前视风险**；"
               "目标/止损是按技术位规则推导的风险参考(供算盈亏比)，**非预测点位**；权重为机械规则、非投资建议。")

    cc = st.columns([3, 2, 2])
    sel = cc[0].multiselect("标的（七姐妹 + SPY）", _ALL_TICKERS, default=["GOOGL", "NVDA", "MSFT"])
    horizon = _HORIZON_OPTS[cc[1].selectbox("引擎周期", list(_HORIZON_OPTS), index=0,
                                            help=gl.help_for("远期收益"))]
    broad = cc[2].checkbox("🌐 含全球外媒(GDELT)", value=False,
                           help="新闻默认全网检索(Google News 上千家媒体+Yahoo)；勾选再并入 GDELT 全球新闻库(英文过滤)，更广但更慢。")
    run = run_gate("brief", {"sel": tuple(sel), "h": horizon, "broad": broad}, label="🚀 生成作战简报",
                   hint="选好标的后点生成——每票引擎分桶+技术位+财报+全网新闻，约每票 4–8 秒。")
    if run is None or not run["sel"]:
        if run is not None:
            st.caption("至少选 1 只标的。")
        return

    from analysis import briefing as bf
    briefs = []
    prog = st.progress(0.0, text="合成简报…")
    for i, tk in enumerate(run["sel"], 1):
        prog.progress(i / len(run["sel"]), text=f"合成 {tk}（全网新闻检索中）…")
        briefs.append(c_brief(tk, run["h"], end, run.get("broad", False)))
    prog.empty()
    weights = bf.auto_weights(briefs)

    # 一屏总览（现价接入盘中近实时，开盘时每 30 秒自动刷新）
    st.markdown("#### 🗒️ 一屏总览")
    _mkt_open = is_market_open()

    @st.fragment(run_every=("30s" if _mkt_open else None))
    def _overview_table():
        import datetime as _d
        bucket = int(_d.datetime.now().timestamp() // 30)
        rows = []
        for b in briefs:
            try:
                lq = c_live_quote(b["ticker"], bucket)
            except Exception:  # noqa: BLE001
                lq = None
            rows.append(_brief_overview_row(b, lq))
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(("🟢 盘中 · 现价为近实时(≈15min延迟)，每30秒自动刷新" if _mkt_open
                    else "⚪ 休市 · 现价为最近报价") + " · 仅展示、不入量化")

    _overview_table()

    # 自动权重
    st.markdown("#### ⚖️ 候选池内排序权重（机械规则：引擎超额×显著性折扣×反波动率 · 非投资建议）")
    st.caption("⚠️ 这是**候选池内的相对排序权重**，不是真实组合仓位。真实仓位上限见下方「组合风险预算」。")
    wc = st.columns(len(briefs)) if briefs else [st]
    for col, (tk, w) in zip(wc, sorted(weights.items(), key=lambda kv: kv[1]["weight"], reverse=True)):
        col.markdown(stat_card(tk, f"{w['weight']:.0f}%", w["role"], T["primary"]), unsafe_allow_html=True)
    st.write("")

    # 组合风险预算（相关性感知的真实仓位上限）
    if len(briefs) >= 2:
        from analysis import engine_discipline as ed
        pxw = c_prices(tuple(b["ticker"] for b in briefs), "2021-01-01", end)
        budg = ed.portfolio_budget(briefs, pxw)
        st.markdown("#### 🧮 组合风险预算（候选权重≠真实仓位 · 相关性感知）")
        bc = st.columns(len(briefs))
        for col, b in zip(bc, briefs):
            pn = budg["per_name"][b["ticker"]]
            sub = "高波动已降档" if pn["vol_capped"] else "单票上限"
            col.markdown(stat_card(f"{b['ticker']} 真实仓位上限", f"≤{pn['cap']:.0%}", sub, T["gold"]), unsafe_allow_html=True)
        for note in budg["notes"]:
            st.markdown(f'<div class="verdict">🔗 {note}</div>', unsafe_allow_html=True)
        if not budg["notes"]:
            st.caption("当前所选标的两两相关性均低于阈值，可各自按单票上限独立计仓。")

        # 协方差组合构建（min-var / risk-parity / 等权）
        st.markdown("##### 📐 协方差组合配比（最小方差 / 风险平价 / 等权）")
        method = st.radio("配比方法", ["min_var", "risk_parity", "equal"], horizontal=True,
                          format_func=lambda m: {"min_var": "最小方差", "risk_parity": "风险平价", "equal": "等权"}[m],
                          key="portmethod")
        pw = c_port_weights(tuple(b["ticker"] for b in briefs), "2021-01-01", end, method)
        wcol = st.columns(len(briefs))
        for col, b in zip(wcol, briefs):
            w = pw["weights"].get(b["ticker"], 0.0)
            col.markdown(stat_card(b["ticker"], f"{w:.0%}", "组合权重", T["good"]), unsafe_allow_html=True)
        cmp = pw.get("compare", {})
        if cmp:
            st.caption("组合年化波动对照：" + " · ".join(f"{k} {v:.1%}" for k, v in cmp.items())
                       + f"　|　{pw['note']}")
    st.write("")

    # 每票详情
    st.markdown("#### 🔍 单票详情")
    for b in briefs:
        be = b.get("engine_headline")
        head = f"{b['ticker']} {b['price']:.1f} · {b['trend']}/{b['drawdown']:.1%}"
        head += f" · {be['bucket']} 超额{be['excess']:+.1%}" if be else ""
        head += "  ⚠️动量陷阱" if b.get("momentum_trap") else ""
        with st.expander(head, expanded=(b is briefs[0])):
            # 一致性 / 数据质量告警（置顶）
            for c in (b.get("consistency") or []):
                st.markdown(f'<div class="verdict">🚧 <b>一致性告警</b>：{c["message"]}</div>', unsafe_allow_html=True)
            _dq = (b.get("fundamentals") or {}).get("_data_quality")
            if _dq:
                st.markdown(f'<div class="verdict">⚠️ <b>数据质量</b>：{_dq}</div>', unsafe_allow_html=True)
            # 证据等级 + 多周期对账
            g = b.get("grade")
            if g:
                _gc = {"A": T["good"], "B": T["good2"], "C": T["gold"], "D": T["amber"], "F": T["bad"]}.get(g["grade"], T["muted"])
                st.markdown(stat_card(f"证据等级 {g['grade']}", f"封顶 {g['max_position_fraction']:.0%}",
                                      f"{g['confidence']}置信 · {g['action']}", _gc), unsafe_allow_html=True)
                st.caption(f"📐 证据等级=历史证据强度(非涨跌预测)：{g['meaning']}。" + ("依据：" + "；".join(g["reasons"]) if g.get("reasons") else ""))
            hz = b.get("horizons")
            if hz and hz.get("action"):
                cf = " ⚠️多周期冲突" if hz.get("conflict") else ""
                st.caption(f"🕐 多周期对账{cf}：短期{bf._score_cn(hz.get('short'))} / 中期{bf._score_cn(hz.get('medium'))} / "
                           f"长期{bf._score_cn(hz.get('long'))} → {hz['action']}")
            es, ev = b.get("engine_state"), b.get("engine_value")
            kc = st.columns(4)
            kc[0].markdown(stat_card("波动分位", f"{b['vol_percentile']:.0%}" if b['vol_percentile'] == b['vol_percentile'] else "—",
                                     {"low_vol": "低", "mid_vol": "中", "high_vol": "高"}.get(b["vol_state"], ""), T["info"], tip="regime"), unsafe_allow_html=True)
            if es:
                c = T["good"] if es["excess"] > 0 else T["bad"]
                kc[1].markdown(stat_card(f"当前状态桶·{b['current_state_bucket']}", f"{es['median']:+.1%}",
                                         f"超额{es['excess']:+.1%}·胜率{es['win_rate']:.0%}{'·显著' if es['significant'] else ''}", c, tip="超额"), unsafe_allow_html=True)
            if ev:
                c = T["good"] if ev["excess"] > 0 else T["muted"]
                kc[2].markdown(stat_card("估值低位桶(买便宜)", f"{ev['median']:+.1%}",
                                         f"超额{ev['excess']:+.1%}·RR{ev['reward_risk']:.2f}{'·显著' if ev['significant'] else ''}", c, tip="超额"), unsafe_allow_html=True)
            if b.get("next_earnings"):
                kc[3].markdown(stat_card("下次财报", b["next_earnings"], f"{b['days_to_earnings']} 天后", T["gold"]), unsafe_allow_html=True)
            if b.get("momentum_trap"):
                st.markdown('<div class="verdict">⚠️ <b>动量陷阱</b>：当前在回撤中，但回撤桶超额≤0——历史上逢跌买并不优于随机进场。正确做法是等趋势确认/波动回落，而非抢这段回撤。</div>', unsafe_allow_html=True)

            # 技术参考档（候选执行·非买点）
            if b["tranches"]:
                st.markdown("**技术参考档（共振支撑位 · 候选执行档 · 非买点信号；过裁决才升级建仓档）**")
                tr = pd.DataFrame([{
                    "档": t["tier"], "价位": f"{t['price']:.1f}", "是什么": t["what"],
                    "目标": f"{t['target']:.0f}({t['to_target_pct']:+.0%})",
                    "止损": f"{t['stop']:.0f}({t['to_stop_pct']:+.0%})",
                    "盈亏比RR": f"{t['rr']:.2f}" if t["rr"] == t["rr"] else "—",
                    "引擎胜率": f"{t['engine_win_rate']:.0%}" if t["engine_win_rate"] == t["engine_win_rate"] else "—",
                } for t in b["tranches"]])
                st.dataframe(tr, use_container_width=True, hide_index=True)

            # 财报反应
            esr = b.get("earnings_stats")
            if esr and esr.get("n_events"):
                st.caption(f"📅 财报反应(历史{esr['n_events']}次)：财报日典型波动 ±{esr['day_abs_move']['median']:.1%}；"
                           f"财报前{esr['pre']}日 drift 中位 {esr['pre_drift']['median']:+.1%}(市场提前消化)；"
                           f"财报后{esr['post']}日 超预期 {esr['post_beat']['median']:+.1%} / 不及 {esr['post_miss']['median']:+.1%}。")

            # 财报要点（结构化）+ 基本面快照
            hl = b.get("highlights") or {}
            if hl:
                kv = " · ".join(f"**{k}** {v}" for k, v in hl.items() if not k.startswith("_"))
                st.markdown(f"💵 财报要点(免费结构化·仅供人读)：{kv}")
            nf = b.get("fundamentals") or {}
            if nf:
                kv = " · ".join(f"{k} {v}" for k, v in nf.items() if not k.startswith("_"))
                st.caption(f"📊 基本面快照(仅供人读，含前视)：{kv}")

            # 新闻启发式推理（情绪+主题 × 引擎，非信号）
            nr = b.get("news_reason")
            if nr:
                st.markdown(f'<div class="verdict">🧠 <b>新闻启发式推理</b>（非 LLM、非买卖信号）<br>{nr}</div>', unsafe_allow_html=True)
            # 免费新闻（全网多源聚合，带情绪/主题标注）
            nw = b.get("news")
            if nw is not None and not nw.empty:
                na = b.get("news_analysis") or {}
                tag = {it["title"]: it for it in na.get("items", [])}
                _emoji = {"正面": "🟢", "负面": "🔴", "中性": "⚪"}
                from analysis.news_reason import _THEME_CN
                st.markdown(f"**🗞️ 全网新闻（{nw['provider'].nunique()} 家媒体 · 情绪 {na.get('tone_label','')} 净{na.get('net_tone',0):+.0%} · 仅线索 · 不入量化）**")
                for _, r in nw.head(8).iterrows():
                    it = tag.get(r["title"], {})
                    em = _emoji.get(it.get("sentiment"), "⚪")
                    th = " ".join(f"`{_THEME_CN.get(t, t)}`" for t in it.get("themes", [])[:3])
                    title = f"[{r['title']}]({r['url']})" if r["url"] else r["title"]
                    st.markdown(f"- {em} `{r['date']}` {title} — *{r['provider']}* {th}")

                # 📖 读正文要点（按需，慢）
                if st.button(f"📖 读{b['ticker']}正文要点（抓全文+抽关键句，约 10–20 秒）", key=f"read_{b['ticker']}"):
                    with st.spinner(f"抓取 {b['ticker']} 新闻正文 + 抽取关键句…"):
                        arts = c_read_articles(b["ticker"], run.get("broad", False), end, limit=5)
                    if not arts:
                        st.caption("正文抽取失败（多为 Google News 重定向/付费墙），可勾选 GDELT 或换标的重试。")
                    for a in arts:
                        t = f"[{a['title']}]({a['url']})" if a["url"] else a["title"]
                        st.markdown(f"**{t}** — *{a['provider']}* `{a['date']}` · 正文 {a['n_chars']} 字")
                        for s in a["excerpts"]:
                            st.markdown(f"  > {s}")
                    if arts:
                        st.caption("以上为原文抽取的含数字/事件关键句，**仅供人读、未做事实校验、不入量化**。"
                                   "要全面读各种新闻并深度推理，请在 Claude 对话里触发 `quant-deep-brief` skill。")

    # 🚀 一键转 Claude 深度分析这几只（联网读全文 + 跨票对比推理）
    _tks = [b.get("ticker", "") for b in briefs if b.get("ticker")]
    if _tks:
        st.divider()
        claude_deep_button(
            f"用 Claude 深度分析这 {len(_tks)} 只（{', '.join(_tks[:6])}{'…' if len(_tks) > 6 else ''}）",
            f"深度分析这几只：{', '.join(_tks)}。逐只读全网新闻全文 + 结合作战简报，给每只一句话判断、"
            f"为什么是这些价位、催化剂、已 price in 什么；再做**跨票对比**(谁更值得、相关性/集中度风险)，"
            f"给一个组合层面的中长期定位。用 quant-deep-brief 口径：校准而非预测、给情景分布不拍单点。",
            key="cl_brief")

    # 导出
    st.divider()
    md = bf.render_markdown(briefs, weights, run["h"])
    st.download_button("📄 导出 Markdown 简报", md,
                       file_name=f"briefing_{end}.md", mime="text/markdown")
    with st.expander("📖 预览 Markdown"):
        st.code(md, language="markdown")

# ===========================================================================
# 页面：个股全景分析（主页 —— 选股自动出全套）
# ===========================================================================
def page_panorama():
    a = asset                      # 侧栏全局选中的标的
    horizon = gl_horizon
    zstart = "1995-01-01" if a == "SPY" else "2008-01-01"

    st.markdown(f'<div class="hero-title">📊 {a} · 现在怎么做</div>', unsafe_allow_html=True)
    st.caption("⚡ **顶部 = 现在该做什么**（行动 · 建议仓位 · 入场价 · 操作预案）；往下才是当前盘面、建仓价位与深入分析。"
               "只校准不预测，给的是历史分布+区间+概率，**非目标价/买卖指令**。卡片标题/列名可**鼠标悬浮**看名词含义。")
    with st.expander("ℹ️ 30 秒上手 / 这页怎么看", expanded=False):
        st.markdown(
            "1. **🎯 行动面板**（最上）= 一眼结论：现在该 **追/等/建仓/防守** + **建议仓位%** + 历史常驻价 + 持有周期。\n"
            "2. **📋 操作预案 / 📌 已建仓怎么办** = 具体怎么建仓、涨了/跌了/到时间/触发风控；已持仓则守/加/减/离。\n"
            "3. **🎯 该在哪建仓** = 最佳入场区(历史常驻价) + K线(蓝线=回撤档·橙线=换手位) + 建仓档。\n"
            "4. **🧭 当前状态** = 证据等级 + 多周期 + 今日价格/趋势/波动 + 未来事件雷达。\n"
            "5. **📂 深入分析(Tab)** = 想细看才点：基本面&情景 / 价位&方案 / 风险&事件 / 工具。\n"
            "> 侧栏 **🧭 风险偏好** 决定稳健度(影响建议仓位)；要**稳定收益**去『🛡️ 稳定配置 & 风险』调目标波动。"
            "重计算项默认收起、点『▶』才算。一切是历史校准+概率，非买卖指令。")
    st.write("")

    with st.spinner(f"正在生成 {a} 全景分析（首次约 10 秒，之后秒开）…"):
        b = c_brief(a, horizon, end)
        z = c_zones(a, zstart, end, horizon)
        from data import loader as _ld
        price = _ld.load_prices([a], zstart, end)[a]

    # ===== 💲 现价大字（最显眼）：盘中近实时(≈15min延迟)，开盘每 30 秒自动刷新；仅展示、不入量化 =====
    _mkt_open_top = is_market_open()

    @st.fragment(run_every=("30s" if _mkt_open_top else None))
    def _hero_price(ticker=a, fb=b):
        import datetime as _d2
        q = c_live_quote(ticker, int(_d2.datetime.now().timestamp() // 30))
        px = q["price"] if (q.get("ok") and q["price"] == q["price"]) else fb["price"]
        chg = q.get("change_pct"); chv = q.get("change")
        if q.get("ok") and chg == chg:
            col = T["good"] if chg >= 0 else T["bad"]
            arr = "▲" if chg >= 0 else "▼"
            sub = f'<span style="color:{col};font-weight:700">{arr} {chv:+.2f} ({chg:+.2%})</span> · {"🟢 盘中" if _mkt_open_top else "⚪ 休市"}{" · ≈15min延迟" if q.get("delayed") else ""}'
        else:
            col = T["text"]; sub = f'收盘 {fb["date"]} · 休市'
        st.markdown(
            f'<div style="display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;'
            f'border-radius:14px;padding:12px 20px;margin:2px 0 10px;background:var(--card);border:1px solid var(--border)">'
            f'<span style="font-size:1.05rem;color:var(--muted);font-weight:700">{ticker} 现价</span>'
            f'<span style="font-size:2.6rem;font-weight:800;color:{col};line-height:1">{px:.2f}</span>'
            f'<span style="font-size:0.95rem;color:var(--muted)">{sub}</span>'
            f'<span style="font-size:0.8rem;color:var(--muted);margin-left:auto">仅展示·不入量化</span>'
            f'</div>', unsafe_allow_html=True)
    _hero_price()

    # ===== 👤 你是新建仓 还是 已持仓？—— 决定重点看哪些区块（深跌破位类票两视角口径不同，先分流）=====
    _persona = st.radio("👤 你的情况", ["🆕 还没建仓 / 想建仓", "📦 已经持有这只票"],
                        horizontal=True, key=f"persona_{a}", label_visibility="collapsed")
    _is_holder = _persona.startswith("📦")
    if _is_holder:
        st.markdown(
            '<div style="border-radius:12px;padding:10px 16px;margin:2px 0 10px;'
            'background:var(--primary-weak);border:1px solid var(--primary-border);border-left:6px solid var(--primary);color:var(--text);font-size:0.86rem">'
            '📦 <b>已持仓视角</b>：重点看下方 <b>🚨 撤离预警 + 📌 已建仓怎么办（守/加/减/离）</b>。'
            '顶部"现在该建多少新仓 / 入场参考价 / 踏空风险"是给<b>新买家</b>的——你不用纠结那些，盯**撤离信号**与止盈止损纪律。</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="border-radius:12px;padding:10px 16px;margin:2px 0 10px;'
            'background:var(--good-weak);border:1px solid var(--good-border);border-left:6px solid var(--good);color:var(--text);font-size:0.86rem">'
            '🆕 <b>新建仓视角</b>：重点看 <b>🎯 现在怎么做（裁决/仓位/入场价/踏空） → 📋 操作预案 → 🎯 该在哪建仓</b>。'
            '下方"🚨 撤离预警 / 📌 已建仓"是给<b>已持仓者</b>的，可先略过。<br>'
            '⚠️ 注意：深跌且跌破200线的票，"新建仓(逆势价值·慢建严止损)"与"持仓者(减仓防守)"口径**本就不同**，按你的身份看对应区块。</div>',
            unsafe_allow_html=True)

    # ===== 🎯 决策卡（合成：现在该做什么 / 入场点 / 入场后 / 离场）—— 页面第一眼 =====
    _card = None
    _bezc = None
    # 入场门控状态（函数级初始化，确保下方"候选执行区间"总裁决横幅也能引用，保持全页同口径）
    _efc = None; _enter_ok = True; _warn_red = False; _warn_amber = False; _no_stat_edge = False
    try:
        from analysis import decision as _dec
        _bezc = c_best_entry_scan(a, zstart, end)
        _mkt = c_fragility(zstart, end).get("cur", {})
        # 引擎纪律入参：动量陷阱 / 证据等级 / 无稳健入场区 → 决策卡自动转防守，避免与下方引擎层矛盾
        _card = _dec.decision_card(a, price, _bezc, _mkt.get("fragile", False), _mkt.get("light", ""),
                                   momentum_trap=bool(b.get("momentum_trap")), grade=b.get("grade"))
        _cc = tm.remap(_card["color"])
        _e = _card.get("entry")
        # 当前建议仓位%（风险管理叠加规则给出"现在该持多少"）—— 显示在下方大数字三联里
        _posnow = None
        try:
            from analysis import overlay as _ov2
            _tvol = _PROFILE_VOL.get(gl_profile, 0.15)
            _posnow = float(_ov2.risk_managed_position(price, c_fragility(zstart, end)["frame"]["fragile"],
                                                       target_vol=_tvol).iloc[-1])
        except Exception:  # noqa: BLE001
            pass
        st.markdown(
            f'<div style="border-radius:16px;padding:20px 24px 16px;margin:2px 0 12px;'
            f'background:{_cc}1f;border:1px solid {_cc}55;border-left:6px solid {_cc}">'
            f'<div style="font-size:0.78rem;color:var(--muted);letter-spacing:1px">🎯 现在怎么做 · {a}（距1年高 {_card["drawdown"]:+.1%}）· 市场 {_card.get("market_light","")}</div>'
            f'<div style="font-size:1.6rem;font-weight:800;color:var(--heading);line-height:1.35;margin-top:6px">{_card["state"]}　▸　{_card["action"]}</div>'
            f'</div>', unsafe_allow_html=True)
        # 三个最该看的数字：建议仓位 / 入场参考价 / 持有周期 —— 放大、最醒目
        # —— 离场预警先算一次（两身份共用），并据此门控新建仓：避免"可分批建仓"与"撤离黄灯"自相矛盾 ——
        try:
            _eft0 = c_fragility(zstart, end).get("cur", {})
            _ewt = _dec.exit_warning(price, _eft0.get("fragile", False), _eft0.get("pctile"))
        except Exception:  # noqa: BLE001
            _ewt = {"red": False, "amber": False, "level": "", "color": "#8A93A6", "action": "", "dist_ma200": float("nan")}
        _warn_red = bool(_ewt.get("red"))
        _warn_amber = bool(_ewt.get("amber"))
        # 入场以 entry_confluence(regime + 技术支撑) 为准，与作战卡同口径——只有飞刀/红灯=🔴 才真不建新仓
        # (已回测：黄灯远期不更差、只小仓即可)。decision_card 的"动量陷阱/无稳健档"(基于噪声回撤桶)
        # 降级为"可买但此跌无统计优势"的提示，不再硬挡，杜绝"panorama暂不建仓 vs 作战卡可分批"的跨页矛盾。
        try:
            _efc = c_entry_confluence(a, zstart, end, _warn_red, _warn_amber, _ewt.get("level", ""))
        except Exception:  # noqa: BLE001
            _efc = None
        _enter_ok = (_efc is None) or (_efc.get("grade_tag") != "🔴")
        _no_stat_edge = bool(_enter_ok and not _card.get("enter_ok", True))  # 技术面允许、但统计层想挡
        _mt = st.columns(3)
        if _is_holder:
            # 📦 持仓者口径：撤离状态 / 距撤离线 / 现在该怎么办（与身份 toggle 联动）
            try:
                _holdt = _dec.holding_advice(_card, b, _bezc)
                _mt[0].markdown(stat_card("🚨 撤离状态", _ewt["level"], _ewt["action"][:26], tm.remap(_ewt["color"]), tip="脆弱"), unsafe_allow_html=True)
                _d2 = _ewt.get("dist_ma200", float("nan"))
                _dcol = T["bad"] if (_d2 == _d2 and _d2 < 0.04) else T["good"]
                _dsub = ("已跌破200线→减半仓" if (_d2 == _d2 and _d2 < 0) else "跌破即趋势破位→减半仓")
                _mt[1].markdown(stat_card("距撤离线(200日线)", f"{_d2:+.0%}" if _d2 == _d2 else "—", _dsub, _dcol, tip="regime"), unsafe_allow_html=True)
                _mt[2].markdown(stat_card("📦 现在该怎么办", _holdt["stance"], "守/加/减/离·详见下方", tm.remap(_holdt["color"]), tip=None), unsafe_allow_html=True)
            except Exception:  # noqa: BLE001
                _mt[0].markdown(stat_card("📦 已持仓", "见下方", "🚨撤离预警 / 守加减离", T["primary"]), unsafe_allow_html=True)
        else:
            # 🆕 新建仓口径
            if not _enter_ok:
                _sub0 = (f"引擎不建议在此建新仓（见上方裁决）；若已持仓，风控暴露上限≈{_posnow:.0%}，见下方『已建仓』"
                         if _posnow is not None else "引擎不建议在此建新仓（见上方裁决）；已持仓者见下方『已建仓』")
                _mt[0].markdown(stat_card("📊 现在该建多少新仓", "暂不建仓", _sub0, T["amber"]), unsafe_allow_html=True)
                if _e and _e.get("anchor") == _e.get("anchor"):
                    _mt[1].markdown(stat_card("🎯 历史常驻价（暂非买点）", f"{_e['anchor']:.1f}",
                                              f"{_e['zone']}·{'稳健档但裁决未背书' if _e.get('confident') else '低置信·引擎未背书此处建仓'}", T["muted"]), unsafe_allow_html=True)
                else:
                    _mt[1].markdown(stat_card("🎯 历史常驻价", "无", "当前无统计稳健点位", T["muted"]), unsafe_allow_html=True)
                _mt[2].markdown(stat_card("⏳ 触发条件", "等更深/确认", "见上方裁决：等深跌或趋势/宽度确认再议", T["muted"]), unsafe_allow_html=True)
            elif _posnow is not None:
                _pos_col = T["good"] if _posnow >= 0.66 else (T["gold"] if _posnow >= 0.33 else T["amber"])
                _mt[0].markdown(stat_card("📊 现在该持多少仓", f"{_posnow:.0%}",
                                          f"{gl_profile}档·目标波动{_PROFILE_VOL.get(gl_profile,0.15):.0%}·这只票满仓=100%(风控暴露·非占组合比例)", _pos_col), unsafe_allow_html=True)
            else:
                _mt[0].markdown(stat_card("📊 现在该持多少仓", "—", "暂不可用", T["muted"]), unsafe_allow_html=True)
            if _enter_ok and _efc is not None:
                # 入场位：用 entry_confluence 的**可执行回踩支撑**(替代旧噪声锚定价)，与作战卡口径一致
                _gcol = T["good"] if _efc["grade_tag"] == "🟢" else (T["gold"] if _efc["grade_tag"] == "🟡" else T["muted"])
                _near = _efc.get("supports_near_now") or []
                _below = _efc.get("supports_below") or []
                if _efc.get("at_support_now") and _near:
                    _mt[1].markdown(stat_card("🎯 入场位·可分批", f"现价 {_efc['current_price']:.1f}",
                                              f"在支撑共振区（{_near[0]['label']}）", _gcol), unsafe_allow_html=True)
                elif _below:
                    _s0 = _below[0]
                    _mt[1].markdown(stat_card("🎯 回踩分批价", f"{_s0['price']:.1f}",
                                              f"{_s0['label']}（{_s0['dist_pct']:+.0%}）·回踩到此分批", _gcol), unsafe_allow_html=True)
                else:
                    _mt[1].markdown(stat_card("🎯 入场位", "小仓/等回踩", "现价下方近处无明显支撑", _gcol), unsafe_allow_html=True)
                _mt[2].markdown(stat_card("⏳ 入场判断", _efc["grade_tag"], _efc["grade"][:16], _gcol), unsafe_allow_html=True)
            elif _enter_ok:
                _mt[1].markdown(stat_card("🎯 入场位", "回踩支撑分批", "见下方候选区间", T["muted"]), unsafe_allow_html=True)
                _mt[2].markdown(stat_card("⏳ 建议持有", "~按周期", "见下方", T["primary"]), unsafe_allow_html=True)
        # 黄灯(非红)：不暂停建仓、只提示小仓（回测：黄灯远期不更差、仅 MAE 略深）
        if (not _is_holder) and _enter_ok and _warn_amber:
            st.caption("⚠️ 当前有**离场黄灯**（波动飙升/高位拉伸/临近撤离线）——回测显示远期收益并不更差、"
                       "只是进场后浮亏略深，故建议**小仓 / 分批**降暴露，不必完全暂停。")
        # 动量陷阱/无稳健档(统计层想挡、但技术面允许)：降级为诚实提示，不再硬"暂不建仓"
        if (not _is_holder) and _no_stat_edge:
            st.caption("ℹ️ 注意：此回撤档**历史上没有抄底超额**（动量陷阱/单票样本不足）——"
                       "意思是**别指望「逢跌买」的统计 alpha**；若你看好基本面想建仓，按上方**技术支撑回踩分批**即可，别越跌越重仓。")
        # 🪂 踏空风险：死等深档却没等到的概率 + 机会成本 + Plan B（直接挂在入场卡下，反"光等抄底"）
        # 仅在引擎建议建仓(enter_ok)时显示——若裁决已是"暂不建仓/别越跌越补"，踏空讨论无意义且会与之打架。
        try:
            from regime import entry_cockpit as _ecmr
            # 踏空风险只对"想新建仓"有意义；已持仓者不显示（与身份 toggle 联动）
            _mrline = _ecmr.format_miss_risk(_ecmr.entry_miss_risk(price, _bezc)) if (_enter_ok and not _is_holder) else None
            if _mrline:
                _mrcol = T["amber"] if "踏空风险高" in _mrline else (
                    T["gold"] if ("踏空风险中" in _mrline or "要等很久" in _mrline) else T["good"])
                st.markdown(f'<div style="border-radius:12px;padding:9px 15px;margin:6px 0 2px;'
                            f'background:{_mrcol}1c;border:1px solid {_mrcol}55;border-left:5px solid {_mrcol};'
                            f'color:var(--text);font-size:0.84rem">🪂 {_mrline}</div>', unsafe_allow_html=True)
        except Exception:  # noqa: BLE001
            pass
        # 入场后 / 离场 一行操作摘要（细节见下方📋操作预案）
        st.markdown(
            f'<div style="border-radius:12px;padding:10px 16px;margin:8px 0 2px;background:var(--card);border:1px solid var(--border)">'
            f'<span style="color:var(--muted);font-size:0.84rem">📈 <b>入场后</b>：{_card["post_entry"]["add"]}；涨了 {_card["post_entry"]["trim"]}；{_card["post_entry"]["stop"]}</span><br>'
            f'<span style="color:var(--muted);font-size:0.84rem">🚪 <b>离场</b>：{"；".join(_card["exit_rules"])}</span>'
            f'</div>', unsafe_allow_html=True)
        if _is_holder:
            st.caption("📦 已持仓口径：上方=撤离状态 / 距撤离线 / 现在该怎么办；详细守/加/减/离 + 触发式止盈止损见下方『📌 已经持有』。")
        elif _enter_ok:
            st.caption("📍 说明：上方『历史常驻价』=**统计最佳入场区里历史价格最常驻处**（远期收益最佳档）；下方『📋操作预案/建仓档』=**技术支撑位**"
                       "（MA/POC 等）用于分批。两者是不同视角、互为补充，合起来当一个入场区间用，不是互相矛盾。")
        else:
            st.caption("📍 说明：裁决判定**当前不是建新仓的统计买点**，故上方不给建仓仓位/入场价，只留『历史常驻价』供你了解历史最优档位置。"
                       "想建仓请按裁决条件（等更深回撤 / 趋势或宽度确认）；已持仓者直接看下方『📌 已建仓怎么办』。")
        # 桥接两个仓位口径：杜绝"现在该持多少仓 75%"与下方"证据等级封顶 20%"看似打架
        try:
            from analysis import engine_discipline as _edx
            _graw = b.get("grade")
            if _posnow is not None and _graw and not _is_holder:   # 桥接说明只对新建仓视角；持仓者上方已是撤离口径
                _gcap = _edx.apply_risk_profile(_graw, gl_profile).get("max_position_fraction")
                if _gcap is not None and _gcap == _gcap:
                    if _enter_ok:
                        st.caption(
                            f"📊 **两个仓位口径别混**：上方『现在该持多少仓 {_posnow:.0%}』=这只票的**风控暴露**"
                            f"(满仓=100%·按波动/趋势降仓·管\"该不该在场、在场多少\")；下方『证据等级封顶 {_gcap:.0%}』"
                            f"=这只票**最多占组合多少**(跨票分配)。两者相乘 ≈ **实际下单 {_posnow*_gcap:.0%} 占组合**"
                            "——一个管何时在场、一个管占组合多大，不是互相矛盾。")
                    else:
                        st.caption(
                            f"📊 **仓位口径**：引擎当前**不建议新建仓**。若你**已持仓**——风控暴露上限≈{_posnow:.0%}、"
                            f"单票最多占组合{_gcap:.0%}（相乘≈{_posnow*_gcap:.0%}），减仓/止盈止损口径见下方『📌 已建仓怎么办』。")
        except Exception:  # noqa: BLE001
            pass
        # 🗂️ 自动留痕：每天打开该标的就静默记录当日指导(按 标的+日期+周期 去重·一天一条)，
        # 供「校准追踪」长期回填真实结果、检验工具有效性。
        try:
            from analysis import journal as _jn
            _logged = _jn.log_from_brief(b, extra={
                "decision_state": _card.get("state"), "decision_action": _card.get("action"),
                "rec_position": (round(_posnow, 3) if _posnow is not None else None),
                "entry_anchor": (_card.get("entry") or {}).get("anchor")})
            st.caption("🗂️ 已自动留痕今日指导（决策状态/建议仓位/历史常驻价/引擎预期）→ "
                       "「📂深入分析 → 🧰工具&校准 → 🎯校准追踪」看历史有效性。" if _logged
                       else "🗂️ 今日该标的指导已留痕（去重）。")
        except Exception:  # noqa: BLE001
            pass
    except Exception as _e:  # noqa: BLE001
        st.caption(f"决策卡暂不可用（{type(_e).__name__}）——其余分析照常。")

    # ---- 📍 关键状态一行（紧跟裁决：把"现在在哪"看全；现价见顶部大字条）----
    _oc = st.columns(4)
    _oc[0].markdown(stat_card("距历史高", f"{b['drawdown']:+.0%}", "全期峰值回撤·引擎分桶口径", T["bad"], tip="回撤"), unsafe_allow_html=True)
    _up0 = b.get("trend_position", 0) > 0
    _oc[1].markdown(stat_card("趋势", "均线上方" if _up0 else "均线下方",
                              f"距200线 {b['trend_position']:+.0%}", T["good"] if _up0 else T["bad"], tip="regime"), unsafe_allow_html=True)
    _vp0 = b["vol_percentile"]
    _oc[2].markdown(stat_card("波动状态", {"low_vol": "低", "mid_vol": "中", "high_vol": "高"}.get(b["vol_state"], "—") + "波动",
                              f"分位 {_vp0:.0%}" if _vp0 == _vp0 else "—", T["info"], tip="regime"), unsafe_allow_html=True)
    if b.get("next_earnings"):
        _oc[3].markdown(stat_card("下次财报", b["next_earnings"], f"{b['days_to_earnings']} 天后", T["primary"]), unsafe_allow_html=True)
    elif a in _ETF_SET:
        _oc[3].markdown(stat_card("下次财报", "—", "ETF·无单公司财报", T["muted"]), unsafe_allow_html=True)
    else:
        _oc[3].markdown(stat_card("下次财报", "—", "未取到(限流/暂无日程)·稍后刷新", T["muted"]), unsafe_allow_html=True)
    st.write("")

    # 🚀 一键转 Claude 联网读全文+深度推理（最显眼处；app 给校准数字，Claude 补读网推理）
    _clc = st.columns([2, 1, 2])
    with _clc[1]:
        claude_deep_button(f"用 Claude 深度分析 {a}",
                           f"深度分析 {a}：读全网新闻全文 + 结合作战简报，给一句话判断、为什么是这些价位、"
                           f"已发生了什么、未来催化剂、市场已 price in 什么、中长期定位。用 quant-deep-brief 口径"
                           f"（校准而非预测、给情景分布不拍单点）。", key=f"cl_top_{a}", hint=False)

    # ========== 📋 操作预案（🆕新建仓） & 📌已建仓（📦持仓者）—— 随身份 toggle 展开/折叠联动 ==========
    from analysis.playbook import build_playbook
    _eok = bool(_card.get("enter_ok", True)) if _card else True   # 与决策卡同口径：别建仓时预案转防守
    pbk = build_playbook(b, enter_ok=_eok)

    def _pcard(col, title, accent, items):
        body = "".join(
            f'<div style="font-size:.85rem;line-height:1.55;color:var(--text);margin:5px 0;'
            f'padding-left:12px;border-left:2px solid {accent}55">{x}</div>'
            for x in (items or ["—"]))
        col.markdown(
            f'<div class="glass" style="min-height:172px;padding:14px 16px">'
            f'<div style="font-weight:700;color:{accent};font-size:.95rem;margin-bottom:6px">{title}</div>'
            f'{body}</div>', unsafe_allow_html=True)

    # 📋 操作预案（新建仓用）：新建仓身份默认展开，持仓者自动折叠（点开仍可看）
    _pe = st.expander("📋 操作预案（🆕 新建仓怎么操作）", expanded=not _is_holder)
    r1 = _pe.columns(3)
    _pcard(r1[0], "🎯 建仓", T["gold"], pbk.get("entry"))
    _pcard(r1[1], "📈 涨了怎么操作", T["good"], pbk.get("if_up"))
    _pcard(r1[2], "📉 跌了怎么操作", T["bad"], pbk.get("if_down"))
    r2 = _pe.columns(2)
    _pcard(r2[0], "⏱️ 时间 / 事件", T["info"], pbk.get("time_event"))
    _pcard(r2[1], "🛡️ 风控", T["primary"], pbk.get("risk"))
    _pe.caption("⚠️ 机械 if-then 预案：价位是「**若到达就行动**」的区间(非预测)，**非买卖指令**；动量陷阱/未过闸门时自动转防守口径。")

    # 📌 已建仓 + 🚨撤离预警（持仓者用）：持仓身份默认展开，新建仓自动折叠（点开仍可看）
    if _card is not None:
        try:
            _hold = _dec.holding_advice(_card, b, _bezc)
            _hcol = tm.remap(_hold["color"])
            _ph = st.expander("📌 已经持有？守/加/减/离 + 🚨 撤离预警", expanded=_is_holder)
            try:
                _ef = c_fragility(zstart, end).get("cur", {})
                _ew = _dec.exit_warning(price, _ef.get("fragile", False), _ef.get("pctile"))
                _ewc = tm.remap(_ew["color"])
                _chips = "".join(
                    f'<span style="display:inline-block;margin:3px 6px 3px 0;padding:2px 9px;border-radius:7px;'
                    f'background:var(--card);border:1px solid var(--border);font-size:0.78rem;color:var(--text)">{s["state"]}</span>'
                    for s in _ew.get("signals", []))
                _ph.markdown(
                    f'<div style="border-radius:14px;padding:13px 18px;margin:2px 0 10px;'
                    f'background:{_ewc}1f;border:1px solid {_ewc}55;border-left:6px solid {_ewc}">'
                    f'<div style="font-size:1.12rem;font-weight:800;color:var(--heading)">🚨 撤离预警 · {_ew["level"]}</div>'
                    f'<div style="margin:6px 0 4px">{_chips}</div>'
                    f'<div style="color:var(--text);font-size:0.86rem">▸ {_ew["action"]}</div>'
                    f'</div>', unsafe_allow_html=True)
                # 撤离明细（四信号）—— 内联（避免 expander 套 expander）
                for s in _ew.get("signals", []):
                    _ph.markdown(f"- **{s['name']}**：{s['state']} —— {s['detail']}")
                _ph.caption("📖 红=触发**已验证的减半仓规则**（跌破200线/宽度恶化，回测砍回撤约40%·样本外稳健）；"
                            "黄=**提前预警**（接近撤离线/高位拉伸/波动飙升/宽度转弱 → 收紧止损·分批止盈·降仓）；绿=无撤离信号。"
                            "宽度信号略领先指数但误报约60%，是**降仓开关非崩盘预言**。全部历史校准、非预测。")
                _ph.caption("🧭 **撤离口径(大规模回测后)**：离场**换不来超额、只为压回撤**(科技/半导体最大回撤~64%→~50%)，别指望靠离场多赚；"
                            "**减半仓 > 清仓**(清仓少赚太多、夏普更低)；只认**破200线+宽度/波动恶化**这条已验证触发。"
                            "**别用**跌破MA50(过敏·反复被甩、年化腰斩)、固定−20%移动止损(卖低踩不回)、过热止盈(对回撤几乎0保护)当离场alpha——"
                            "后两者只当**锁浮盈的纪律**。")
            except Exception:  # noqa: BLE001
                pass
            _ph.markdown(
                f'<div style="border-radius:14px;padding:14px 18px;margin:6px 0 8px;'
                f'background:{_hcol}1f;border:1px solid {_hcol}55;border-left:6px solid {_hcol}">'
                f'<div style="font-size:1.18rem;font-weight:800;color:var(--heading)">{_hold["stance"]}</div>'
                f'<div style="color:var(--text);font-size:0.9rem;margin-top:5px">{_hold["headline"]}</div>'
                f'</div>', unsafe_allow_html=True)
            _ha = _ph.columns(2)
            _pcard(_ha[0], "🔧 现在的动作", T["good"], _hold["actions"])
            _pcard(_ha[1], "🎯 触发式止盈 / 止损（到价才动）", T["amber"], _hold["triggers"])
            _ph.caption("⚠️ 已建仓视角与上方建仓视角**同源**(回撤/趋势/脆弱/动量陷阱/证据等级)、口径一致；价位是「到了才行动」的触发区间。")
        except Exception as _e:  # noqa: BLE001
            st.caption(f"已建仓建议暂不可用（{type(_e).__name__}）——其余分析照常。")

    st.divider()
    # ---- 候选执行区间：TradingView K线 + 价位带横线 + 候选执行档 ----
    st.markdown("#### 🎯 在什么价位/状态执行（**候选区间** · 过裁决才升级为建仓区）")
    # 总裁决横幅与上方三联卡**同口径**(entry_confluence)：只有飞刀/红灯=🔴 才"暂不建仓"；
    # 动量陷阱/无稳健档(统计层)降级为"可小仓·但此跌无统计alpha"，不再硬挡(杜绝同页两处裁决打架)。
    _eok2 = _enter_ok
    _conf2 = bool((_bezc or {}).get("confident")) if _bezc else False
    _knife_red = bool(_efc and _efc.get("grade_tag") == "🔴")
    _principle = ('下面的历史常驻价/价位带/档位是 <b>候选执行区间</b>（技术共振支撑 / 历史相对较优档）——'
                  '<b>价格"到了"不等于"无脑买"</b>：先看上方<b>能不能建仓</b>，再按这些**技术支撑回踩分批**。')
    if not _eok2:
        st.markdown(
            '<div style="border-radius:12px;padding:11px 16px;margin:2px 0 8px;'
            'background:var(--bad-weak);border:1px solid var(--bad-border);border-left:6px solid var(--bad);color:var(--text);font-size:0.86rem">'
            f'🔴 <b>当前总裁决：暂不建新仓</b>（{(_efc or {}).get("grade","破位/离场红灯")}）。{_principle} '
            '故下面只当"历史相对较优档/支撑在哪"看，<b>不是买入信号</b>；等站回200线 / 企稳 / 预警解除再议。</div>',
            unsafe_allow_html=True)
    elif _no_stat_edge:
        st.markdown(
            '<div style="border-radius:12px;padding:11px 16px;margin:2px 0 8px;'
            'background:var(--amber-weak);border:1px solid var(--amber-border);border-left:6px solid var(--amber);color:var(--text);font-size:0.86rem">'
            '🟡 <b>当前总裁决：可小仓建仓（但此跌无统计 alpha）</b>。技术面趋势健康、可在下方**支撑回踩分批**；'
            '但该回撤档**历史上没有抄底超额**（动量陷阱/单票样本不足）——'
            f'{_principle} <b>别指望"逢跌买"的超额、别越跌越重仓</b>，看好基本面才小仓参与。</div>',
            unsafe_allow_html=True)
    elif _conf2:
        st.markdown(
            '<div style="border-radius:12px;padding:11px 16px;margin:2px 0 8px;'
            'background:var(--good-weak);border:1px solid var(--good-border);border-left:6px solid var(--good);color:var(--text);font-size:0.86rem">'
            f'✅ <b>当前总裁决：可建仓 · 且有"稳健入场区"</b>（过了 DSR≥0.95 / CI下界>基准 等闸门）。{_principle} '
            '这是<b>极少数</b>闸门全过的情况（回测显示个股/科技半导体历史上几乎不出现），'
            '下面价位可<b>按档分批执行</b>（仍按置信度控仓）。</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="border-radius:12px;padding:11px 16px;margin:2px 0 8px;'
            'background:var(--good-weak);border:1px solid var(--good-border);border-left:6px solid var(--good);color:var(--text);font-size:0.86rem">'
            '🟢 <b>当前总裁决：可分批建仓</b>（温和正期望 · 择时低置信）。'
            '回测（<b>PIT 无前视</b>）显示这类「正期望回撤档」分批进<b>历史上跑赢「随便买」</b>'
            '（浅跌 0–10% 档尤佳、胜率约 70–82%），只是没到 DSR≥0.95 那条「高置信精准买点」线。'
            '<b>这就是常态可买信号</b>——按「<b>分批 + 控仓 + 认尾部</b>」执行，'
            '别空等几乎从不出现的「稳健档」（严口径下个股历史上基本不触发）。'
            '下方 K 线价位带是技术共振支撑，用于<b>分档挂单</b>。</div>', unsafe_allow_html=True)
    # per-chart 持有期：本段及下方价位带/状态扫描/评分时序统一按此持有期校准（默认=侧边栏分析周期）
    horizon = _chart_horizon(f"pan_{a}", horizon, label="建仓校准持有期")
    z = c_zones(a, zstart, end, horizon)
    if horizon != gl_horizon:
        st.caption(f"ℹ️ 此持有期（~{int(round(horizon/21))}个月）**仅重算下方价位带 / 状态扫描 / 各价位带明细**；"
                   f"决策卡 / 操作预案 / 证据等级（页面其余处）用侧栏「分析周期」（~{int(round(gl_horizon/21))}个月）。"
                   "要让全页统一，请改侧栏分析周期。")

    # —— 🎯 最佳入场区 + 历史常驻价（跨持有期择优：自动挑置信度最高的周期，避免长周期低置信埋没好结果）——
    # 整块容错：任何失败（含云端旧构建/数据缺失）只降级为提示，绝不崩整页。
    from regime import entry_cockpit as ec
    try:
        bez = c_best_entry_scan(a, zstart, end)
        if bez.get("has_zone"):
            _band = bez["price_band"]
            _anc = "触发价" if _band[0] is None else "历史常驻价"
            _bs = f"≤ {_band[1]:.1f}" if _band[0] is None else f"{_band[0]:.1f}–{_band[1]:.1f}"
            _hm = int(round(bez.get("horizon", horizon) / 21))
            # 命名随"闸门状态"升降，杜绝把候选区间当买点：
            if not _eok2:
                _tc = T["muted"]; _t0 = f"📍 相对较优档·{_anc}（暂非买点）"; _t1 = "📍 历史相对较优档（暂非买点）"
            elif _conf2:
                _tc = T["good"]; _t0 = f"✅ 入场区·{_anc}（可执行）"; _t1 = "✅ 稳健入场区（已过闸门）"
            else:
                _tc = "var(--good)"; _t0 = f"🟢 可分批建仓·{_anc}（低置信·正期望）"; _t1 = "🟢 可分批建仓区（低置信）"
            _bc = st.columns(4)
            _d = bez.get("anchor_distance", float("nan"))
            _bc[0].markdown(stat_card(_t0, f"{bez['anchor_price']:.1f}",
                                      f"距现价 {_d:+.1%}·持有~{_hm}月" if _d == _d else bez["zone_label"], _tc), unsafe_allow_html=True)
            _bc[1].markdown(stat_card(_t1, _bs, f"{bez['zone_label']}·持有~{_hm}月", T["gold"] if _eok2 else T["muted"]), unsafe_allow_html=True)
            _bc[2].markdown(stat_card("历史超基准", f"{bez['excess_median']:+.1%}",
                                      f"盈亏比 {bez['reward_risk']:.1f}·胜率 {bez['win_rate']:.0%}", T["primary"], tip="远期收益"), unsafe_allow_html=True)
            _ci = bez["ci"]
            _dsr = bez.get("dsr", float("nan"))
            _dsr_s = f"·DSR{_dsr:.2f}" if _dsr == _dsr else ""
            _bc[3].markdown(stat_card("置信", bez["tier"],
                                      f"有效窗口≈{bez.get('n_independent','?')}·CI[{_ci[0]:+.0%},{_ci[1]:+.0%}]{_dsr_s}",
                                      _tc, tip="CI"), unsafe_allow_html=True)
        st.markdown(f'<div class="verdict">{ec.format_best_entry(bez)}</div>', unsafe_allow_html=True)
        # 各持有期对比（看不同周期的置信与历史常驻价，哪个周期最该信）
        _scan = bez.get("horizon_scan") or []
        if _scan:
            _rows = [{"持有期": f"{int(s['horizon']/21)}个月({s['horizon']}日)" if s.get("horizon") else "—",
                      "结论档": s.get("zone_label") or ("防守" if not s.get("has_zone") else "—"),
                      "历史常驻价": (f"{s['anchor']:.1f}" if s.get("anchor") == s.get("anchor") and s.get("anchor") is not None else "—"),
                      "超基准": (f"{s['excess']:+.1%}" if s.get("excess") == s.get("excess") and s.get("excess") is not None else "—"),
                      "有效N": s.get("n_independent", "—"),
                      "DSR": (f"{s['dsr']:.2f}" if s.get("dsr") == s.get("dsr") and s.get("dsr") is not None else "—"),
                      "置信": "✅稳健" if s.get("confident") else (s.get("tier") or "—")}
                     for s in _scan]
            with st.expander("🔭 各持有期对比（自动已选其中置信最高者为上方推荐）"):
                st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
                st.caption("📖 同一只票在不同持有期的入场区/置信不同。工具自动挑 DSR/置信最高的那个周期作为顶部推荐；"
                           "全为低置信=该票当前没有统计稳健的最佳入场点，按区间分批+认低置信。")
    except Exception as _e:  # noqa: BLE001
        st.caption(f"🎯 最佳入场区暂不可用（{type(_e).__name__}）——其余分析不受影响。若刚更新代码，请在 Streamlit 后台 Reboot 清缓存。")

    # 整合：市场环境(脆弱性) + 等/追操作（一行看清"现在该追/等/防守"）
    try:
        from analysis import fragility as fg
        _curfz = c_fragility(zstart, end).get("cur", {})
        _cd = float(price.iloc[-1] / price.rolling(252, min_periods=120).max().iloc[-1] - 1.0) if len(price) > 120 else 0.0
        _wc = fg.wait_or_chase(_cd, fragile_now=_curfz.get("fragile", False), is_index=(a in ("SPY", "QQQ", "DIA", "IWM")),
                               momentum_trap=bool(b.get("momentum_trap")))
        _mc = st.columns([1, 3])
        _mc[0].markdown(stat_card("市场环境", _curfz.get("light", "—"),
                                  f"宽度分位 {_curfz.get('pctile', float('nan')):.0%}" if _curfz.get("pctile") == _curfz.get("pctile") else "—",
                                  T["bad"] if _curfz.get("fragile") else T["good"], tip="脆弱"), unsafe_allow_html=True)
        _mc[1].markdown(f'<div class="verdict">🧭 没到位怎么办（现价距1年高 {_cd:+.1%}）：'
                        f'<b>{_wc["action"]}</b> — {_wc["detail"]}</div>', unsafe_allow_html=True)
    except Exception:  # noqa: BLE001
        pass
    st.write("")

    # —— 图层开关：控制 K 线上叠加什么（默认都开；嫌乱可关）——
    _ovc = st.columns([1.2, 1, 1.2, 1.4, 1.6])
    _show_best = _ovc[0].toggle("🎯 最佳入场区", value=True, key=f"ovbest_{a}",
                                help="在K线上用绿/黄线标出工具推荐的最佳入场价位区（历史常驻价 + 区间上下沿）")
    _show_zones = _ovc[1].toggle("📊 价位带", value=True, key=f"ovzone_{a}", help="历史回撤价位带（蓝=当前·灰=其它）")
    _show_vp = _ovc[2].toggle("📦 换手位", value=True, key=f"ovvp_{a}", help="POC 最高换手价 / 价值区（橙）")
    _show_rsi = _ovc[3].toggle("📉 超买超卖", value=False, key=f"ovrsi_{a}",
                               help="价下加一个 Wilder RSI(14) 振荡子图：>70 超买带(红)、<30 超卖带(绿)、50 中轴。"
                                    "校准口径——超卖≠买点、超买≠卖点，只标动量极值，须与裁决/价位带合看")

    cL, cR = st.columns([3, 2])
    with cL:
        # 最佳入场区数据（开关开时才取；失败不影响其它图层）
        _bz_chart = None
        if _show_best:
            try:
                _bz_chart = c_best_entry_scan(a, zstart, end)
            except Exception:  # noqa: BLE001
                _bz_chart = None
        # 换手位数据（开关开时才取）
        _vp_chart = None
        if _show_vp:
            try:
                _, _vp_chart = c_volume_profile(a, "2015-01-01", end, 252)
            except Exception:  # noqa: BLE001
                _vp_chart = None
        # Plotly 主图（TradingView 手感：滚轮缩放/拖动平移/十字光标贴价）—— 可靠绘制阴影/横线
        try:
            ohlcv_pan = _ld.load_ohlcv(a, zstart, end).tail(504)
            _endorse = bool((_card or {}).get("enter_ok", True))   # 与顶部三联卡同步：别建仓时入场区带降级为灰色"历史常驻价"
            st.plotly_chart(
                ch.panorama_price_chart(ohlcv_pan, zones=z, vp=_vp_chart, best_entry=_bz_chart,
                                        show_best=_show_best, show_zones=_show_zones, show_vp=_show_vp,
                                        show_rsi=_show_rsi,
                                        title=f"{a} · K线 + 图层", logy=False, best_endorsed=_endorse),
                use_container_width=True, config=ch.TV_CONFIG)
            st.caption("🖱️ **滚轮缩放·拖动平移·十字光标贴价读数**（双击复位；顶部按钮切时间范围）。"
                       + ("**绿/黄阴影带=🎯推荐最佳入场区(+历史常驻价线)**；" if _endorse else "**灰阴影带=📍历史常驻价(裁决判定暂非买点)**；")
                       + "蓝/灰虚线=回撤价位带(▶=当前)；橙=POC换手密集价/价值区+左侧筹码柱。"
                       + ("**下方 RSI 子图：>70 超买带/红、<30 超卖带/绿、50 中轴**——只标动量极值，超卖≠买点、须与裁决合看。" if _show_rsi else "")
                       + "用上方开关增删图层；完整筹码分布见『📦 筹码分布』。")
        except Exception as _ce:  # noqa: BLE001
            st.plotly_chart(ch.price_with_zones(price, z), use_container_width=True, config=ch.CHART_CONFIG)
            st.caption(f"主图降级（{type(_ce).__name__}）。")
    cR.plotly_chart(ch.entry_zone_bars(z, horizon), use_container_width=True)
    from regime import entry_cockpit as ec
    cur = z[z["is_current"] & z["enough"]] if "is_current" in z.columns else z.iloc[0:0]
    if not cur.empty:
        st.markdown(f'<div class="verdict">当前价位带：{ec.format_zone_verdict(cur.iloc[0], horizon)}</div>', unsafe_allow_html=True)
    if b.get("tranches"):
        st.markdown("**技术参考档（共振支撑位 · 到价分批的候选执行档 · 非买点信号）**　💡 列名可悬浮看含义"
                    + ("　——✅裁决已可建仓，可按此分批" if _eok2 else "　——⚠️裁决暂不建仓，仅作参考、别照此抄底"))
        _tdf = pd.DataFrame([{
            "档": t["tier"], "价位": f"{t['price']:.1f}", "依据": t["what"],
            "回前高目标": f"{t['target']:.0f} ({t['to_target_pct']:+.0%})",
            "技术止损": f"{t['stop']:.0f} ({t['to_stop_pct']:+.0%})",
            "盈亏比": f"{t['rr']:.2f}" if t["rr"] == t["rr"] else "—",
            "引擎胜率": f"{t['engine_win_rate']:.0%}" if t["engine_win_rate"] == t["engine_win_rate"] else "—",
        } for t in b["tranches"]])
        st.dataframe(_tdf, use_container_width=True, hide_index=True, column_config=_col_cfg(_tdf.columns))
        st.caption("📖 「回前高目标」=前期高点(约52周高)，只用来和技术止损算盈亏比，**非预测目标价**。"
                   "盈亏比按**未四舍五入的精确价**计算，与表中取整价手算会有小数差。")
    st.write("")

    st.divider()
    # ========== 🧭 当前状态（证据等级 + 事件雷达 + 今日盘面）==========
    st.markdown("#### 🧭 当前状态")
    # ---- 证据等级 + 多周期对账 + 一致性/数据质量告警 ----
    for c in (b.get("consistency") or []):
        st.markdown(f'<div class="verdict">🚧 <b>一致性告警</b>：{c["message"]}</div>', unsafe_allow_html=True)
    _dq = (b.get("fundamentals") or {}).get("_data_quality")
    if _dq:
        st.markdown(f'<div class="verdict">⚠️ <b>数据质量</b>：{_dq}</div>', unsafe_allow_html=True)
    _g = b.get("grade")
    if _g:
        from analysis import engine_discipline as _ed
        _g = _ed.apply_risk_profile(_g, gl_profile)   # 按侧栏风险偏好缩放仓位封顶
    _hz = b.get("horizons")
    if _g or _hz:
        gcol = st.columns([1, 3])
        if _g:
            _gc = {"A": T["good"], "B": T["good2"], "C": T["gold"], "D": T["amber"], "F": T["bad"]}.get(_g["grade"], T["muted"])
            gcol[0].markdown(stat_card(f"证据等级 {_g['grade']}", f"封顶 {_g['max_position_fraction']:.0%}",
                                       f"{_g['confidence']}置信 · {gl_profile}", _gc), unsafe_allow_html=True)
        with gcol[1]:
            if _g:
                st.caption(f"📐 **证据等级=历史证据强度，非涨跌预测**：{_g['meaning']}。"
                           f"（{gl_profile}档：{_g.get('profile_desc','')}）"
                           + ("依据：" + "；".join(_g["reasons"]) if _g.get("reasons") else ""))
            if _hz and _hz.get("action"):
                cf = " ⚠️**多周期冲突**" if _hz.get("conflict") else ""
                from analysis import briefing as _bf
                st.caption(f"🕐 多周期对账{cf}：短期{_bf._score_cn(_hz.get('short'))} / 中期{_bf._score_cn(_hz.get('medium'))} / "
                           f"长期{_bf._score_cn(_hz.get('long'))} → {_hz['action']}")
        # 证据等级(基于"估值低位/买便宜"论据)与 当前状态桶/多周期 打架时醒目降级——杜绝"A级却观察为主"误导
        _es = b.get("engine_state") or {}
        _state_neg = _es.get("excess") is not None and _es["excess"] <= 0
        _hz_weak = bool(_hz and ("观察" in (_hz.get("action") or "") or (_hz.get("agreement") is not None and _hz.get("agreement") <= 0)))
        if _g and _g.get("grade") in ("A", "B") and (_state_neg or _hz_weak):
            _bits = []
            if _state_neg: _bits.append(f"当前状态桶超额 {_es['excess']:+.1%}(≤0)")
            if _hz_weak: _bits.append(f"多周期「{_hz.get('action','')}」")
            st.markdown(
                '<div style="border-radius:12px;padding:10px 16px;margin:2px 0 4px;'
                'background:var(--gold-weak);border:1px solid var(--gold);border-left:6px solid var(--gold);color:var(--text);font-size:0.85rem">'
                f'⚠️ <b>证据等级 {_g["grade"]} 偏乐观、请降级看</b>：这个等级主要来自「估值低位(买便宜)」论据，'
                f'但 {"、".join(_bits)} → **即时择时上并无优势**。'
                '把"等级 A/B"理解成"长期估值/赔率不差"，<b>不等于"现在就该重仓建"</b>；以裁决/撤离与多周期为准。</div>',
                unsafe_allow_html=True)
    st.write("")

    # ---- 📈📉 趋势全程分布（像今天这状态，历史后来再跌多深/反弹多高/见底多久/回到前高多久）----
    try:
        _rp = c_regime_path(a, zstart, end)
        if _rp:
            st.markdown("##### 📈📉 趋势全程分布（历史类比 · 分布非预测）")
            st.markdown(
                f'<div style="border-radius:12px;padding:12px 16px;margin:2px 0 6px;'
                f'background:var(--primary-weak);border:1px solid var(--primary-border);border-left:5px solid var(--primary)">'
                f'<div style="color:var(--text);font-size:0.84rem;margin-bottom:4px">{_rp["headline"]}</div>'
                + "".join(f'<div style="color:var(--text);font-size:0.85rem;line-height:1.6">{l}</div>' for l in _rp["lines"])
                + f'<div style="color:var(--muted);font-size:0.84rem;margin-top:5px">📏 {_rp["price_range"]}</div>'
                f'</div>', unsafe_allow_html=True)
            st.caption("📖 它回答的是「**历史上像今天这种深度的状态，后来全程怎么走**」"
                       "（再跌/反弹/见底时长/收复时长的分布）。" + _rp["caveat"])
    except Exception:  # noqa: BLE001
        pass

    # ---- 🛰️ 事件雷达（全网自动抓取，display-only 风险提醒，绝不入量化）----
    import datetime as _dtm
    from analysis import event_radar as _er
    _today = _dtm.date.today()
    with st.spinner("事件雷达：全网查询 IPO/经济日历…"):
        radar = c_event_radar(a, _today.isoformat(), b.get("next_earnings"), 45)
    _src_cn = {"auto": "规则", "web": "全网", "manual": "手填"}
    _sevcol = {"高": T["bad"], "中": T["gold"], "低": T["muted"]}
    _rc = T["bad"] if radar["n_high"] else T["primary"]
    chips = "".join(
        f'<span style="display:inline-block;margin:3px 6px 3px 0;padding:3px 9px;border-radius:8px;'
        f'background:{_sevcol.get(e["severity"],T["muted"])}22;border:1px solid {_sevcol.get(e["severity"],T["muted"])}66;'
        f'font-size:0.8rem;color:var(--text)">{e["date"].strftime("%m-%d")}·{e["days_ahead"]}天后 '
        f'{e["category"]}{("·"+e["title"]) if e.get("title") else ""}</span>'
        for e in radar["events"][:8])
    st.markdown(
        f'<div style="border-radius:14px;padding:13px 18px;margin:2px 0 6px;'
        f'background:{_rc}1f;border:1px solid {_rc}44;border-left:6px solid {_rc}">'
        f'<div style="font-size:0.8rem;color:var(--muted);letter-spacing:0.5px">🛰️ 事件雷达 · 未来45天 '
        f'（{radar["n_high"]} 项高风险 · {radar.get("n_web",0)} 项全网自动抓取）· <b>仅提醒、不入量化</b></div>'
        f'<div style="margin-top:7px">{chips or "<span style=\'color:var(--muted)\'>无登记事件</span>"}</div>'
        f'</div>', unsafe_allow_html=True)
    with st.expander("🛰️ 事件雷达明细（全网自动：IPO 日历 + 经济日历）+ 新闻线索"):
        st.caption("⚠️ 特大/一次性事件（巨型IPO抽流动性、并购、监管判决…）**无法回测、绝不进量化结果**；"
                   "工具自动从 NASDAQ IPO 日历 + 经济日历抓取，列成提醒，请你人工纳入仓位与风险判断。")
        if radar["events"]:
            _rows = [{"日期": e["date"].isoformat(), "距今": f"{e['days_ahead']}天", "范围": e["scope"],
                      "类别": e["category"], "事件": e.get("title", "") or "—", "严重度": e["severity"],
                      "盯什么/为什么重要": e["watch"], "来源": _src_cn.get(e["source"], e["source"])}
                     for e in radar["events"]]
            st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
        # 新闻里的前瞻事件线索（未核实，不进时间线）
        leads = radar.get("news_leads") or []
        if leads:
            st.markdown("**📰 新闻里的事件线索（未核实，仅供顺藤摸瓜，不入量化）**")
            for ld in leads:
                t = f"[{ld['title']}]({ld['url']})" if ld.get("url") else ld.get("title", "")
                st.markdown(f"- `{ld.get('date') or '—'}` {t} — *{ld.get('provider','')}*")
        with st.form(f"addevt_{a}", clear_on_submit=True):
            st.markdown("**➕ 补充手填事件（自动没抓到的，可手动加）**")
            fc = st.columns([2, 3, 2, 2])
            ev_date = fc[0].date_input("日期", value=_today + _dtm.timedelta(days=7))
            ev_title = fc[1].text_input("事件", placeholder="例：SpaceX 上市")
            ev_scope = fc[2].selectbox("范围", ["全市场", a])
            ev_sev = fc[3].selectbox("严重度", ["高", "中", "低"])
            ev_cat = fc[0].text_input("类别", value="大型IPO")
            ev_impact = fc[1].text_input("影响/盯什么", placeholder="例：抽走二级市场流动性、短期压制风险偏好")
            if st.form_submit_button("保存事件"):
                if ev_title.strip():
                    _er.add_event(ev_date.isoformat(), ev_title.strip(), scope=ev_scope,
                                  category=ev_cat.strip() or "特大事件", impact=ev_impact.strip(), severity=ev_sev)
                    st.success(f"已添加：{ev_date} · {ev_title}。刷新即出现在雷达。")
                else:
                    st.warning("请填事件名。")
        _man = _er.manual_events()
        if _man:
            st.caption("已手填事件（可删）：")
            for e in _man:
                dc = st.columns([6, 1])
                dc[0].markdown(f"- `{e['date']}` **{e['title']}**（{e['scope']}·{e['category']}·{e['severity']}）{('— '+e['watch']) if e['watch'] else ''}")
                if dc[1].button("删除", key=f"delevt_{e['id']}"):
                    _er.delete_event(e["id"]); st.rerun()

    # ========== 第四区：深入分析（全部收进 Tab，保持主区简洁）==========
    from regime import entry_cockpit as ec
    st.divider()
    st.markdown("#### 📂 深入分析（按需展开）")
    _T_FUND, _T_ZONE, _T_RISK, _T_TOOL = st.tabs(
        ["📊 基本面 & 情景", "💠 价位 & 方案", "🛡️ 风险 & 事件", "🧰 工具 & 校准"])
    _T_EVENT = _T_RISK      # 新闻/事件归入「风险 & 事件」
    _T_HIST = _T_TOOL       # 校准追踪归入「工具 & 校准」

    # ── 📊 基本面 & 情景：分析师视角 + 多引擎 + 财报反应 ──
    with _T_FUND.expander("📝 分析师视角（全面基本面 · 量化情景 · 催化剂 · 失效条件）", expanded=True):
        if _lazy_gate(f"analyst_{a}", "▶ 生成分析师报告（取实时基本面）"):
            try:
                from analysis import analyst as _an
                _info = dict(c_fund_info(a))
                _stale = _info.pop("_stale", None)
                if _stale:
                    st.caption(f"⏳ 实时基本面取数受限（yfinance 限流），下面显示**最近一次成功获取**（{_stale} UTC）。"
                               "稍后刷新可更新；价格/量化结论始终是实时的，不受影响。")
                _rep = _an.analyst_report(a, price, _info, b, horizon)
                st.markdown(_an.format_report(_rep))
                st.caption("💡 想要**联网读真实新闻全文 + LLM 深度推理**的分析师评论（工具内只用自身数据，不联网）：")
                claude_deep_button(
                    f"用 Claude 深度分析 {a}（读全文+推理）",
                    f"深度分析 {a}：读全网新闻全文、结合作战简报，给一句话判断、为什么是这些价位、"
                    f"已发生了什么、未来催化剂、市场已price in 什么、中长期定位。用 quant-deep-brief 口径。",
                    key=f"cl_pan_{a}")
            except Exception as _e:  # noqa: BLE001
                st.warning(f"基本面暂未取到（yfinance 限流且无历史缓存）：{_e}。请**再点一次**按钮重试，"
                           "或稍后再来——成功取到一次后即会落盘，之后限流也能显示最近数据。")
    with _T_FUND.expander("🧠 多引擎校准（不同视角的历史倾斜）", expanded=True):
        ec_cols = st.columns(3)
        for col, key, label in [(ec_cols[0], "engine_state", "当前状态桶"),
                                (ec_cols[1], "engine_value", "估值低位桶(买便宜)"),
                                (ec_cols[2], "engine_best", "最优倾斜桶")]:
            e = b.get(key)
            if e:
                c = T["good"] if e["excess"] > 0 else T["muted"]
                sub = f"超额{e['excess']:+.1%}·胜率{e['win_rate']:.0%}{'·显著' if e.get('significant') else ''}"
                col.markdown(stat_card(f"{label}", f"{e['median']:+.1%}", sub, c, tip="状态桶"), unsafe_allow_html=True)
    esr = b.get("earnings_stats")
    if esr and esr.get("n_events"):
        with _T_FUND.expander("📅 财报反应（历史）+ PEAD", expanded=True):
            st.caption(f"历史 {esr['n_events']} 次：财报日典型波动 ±{esr['day_abs_move']['median']:.1%}；"
                       f"财报前 {esr['pre']} 日 drift 中位 {esr['pre_drift']['median']:+.1%}（市场提前消化）；"
                       f"财报后 {esr['post']} 日 超预期 {esr['post_beat']['median']:+.1%} / 不及 {esr['post_miss']['median']:+.1%}。")
            up = b.get("upcoming")
            if up is not None and len(up):
                st.dataframe(up, use_container_width=True, hide_index=True, column_config=_col_cfg(up.columns))
            try:
                pd_now = c_pead(a, "2010-01-01", end)
            except Exception:  # noqa: BLE001
                pd_now = None
            if pd_now and pd_now.get("actionable"):
                st.markdown(f'<div class="verdict">📈 <b>财报后漂移(PEAD · 工具唯一通过安慰剂检验的免费信号)</b><br>'
                            f'{pd_now["verdict"]}<br><span style="color:var(--muted);font-size:0.82rem">{pd_now["note"]}</span></div>',
                            unsafe_allow_html=True)
            elif pd_now is not None:
                st.caption("📈 PEAD：当前不在财报后漂移窗口内（或该票同类财报样本不足）。")

    with _T_HIST.expander("🎯 校准追踪（记录此刻信号 · 事后比对'说的 vs 做到的'）"):
        from analysis import journal as jn
        st.caption("🗂️ **已自动留痕**：每天打开标的就记录当日指导；走完 horizon 后用真实价格回填，"
                   "**标注每条判断对/错**，长期检验工具准不准（自验证闭环）。想全自动(不打开也记)见下方说明。")
        try:
            sig_df = jn.load_signals()
            ev = jn.evaluate(sig_df) if len(sig_df) else sig_df
            cal = jn.calibration_summary(ev) if len(sig_df) else {"n_total": 0, "n_matured": 0}
            cc = st.columns(4)
            _acc = cal.get("accuracy", float("nan"))
            cc[0].markdown(stat_card("📊 判断准确率", f"{_acc:.0%}" if _acc == _acc else "—",
                                     f"{cal.get('n_judged',0)} 条可判·累计 {cal.get('n_total',0)}",
                                     T["good"] if (_acc == _acc and _acc >= 0.5) else T["gold"], tip="校准"), unsafe_allow_html=True)
            cc[1].markdown(stat_card("已成熟", f"{cal.get('n_matured',0)}", "走完horizon可评", T["info"]), unsafe_allow_html=True)
            if cal.get("n_matured"):
                rh, pw = cal.get("realized_hit", float("nan")), cal.get("pred_win_rate_mean", float("nan"))
                cc[2].markdown(stat_card("实现命中率", f"{rh:.0%}", f"引擎预测 {pw:.0%}",
                                         T["good"] if (rh == rh and pw == pw and abs(rh - pw) < 0.12) else T["gold"]), unsafe_allow_html=True)
                cc[3].markdown(stat_card("实现超额(均)", f"{cal.get('realized_excess_mean',float('nan')):+.1%}", f"Brier {cal.get('brier',float('nan')):.2f}",
                                         T["good"] if cal.get("realized_excess_mean", 0) > 0 else T["bad"]), unsafe_allow_html=True)
                # 逐条近期已成熟信号 + 判断对/错标注
                _m = ev[ev["matured"] == True].copy() if "matured" in ev.columns else pd.DataFrame()  # noqa: E712
                if not _m.empty:
                    _m["判断"] = _m["correct"].map({1.0: "✅对", 0.0: "❌错"}).fillna("—")
                    _disp = _m.sort_values("signal_date", ascending=False).head(12)[
                        ["ticker", "signal_date", "horizon", "grade", "pred_excess", "realized_excess", "判断"]].rename(
                        columns={"ticker": "标的", "signal_date": "信号日", "horizon": "周期", "grade": "等级",
                                 "pred_excess": "预测超额", "realized_excess": "实现超额"})
                    _disp["信号日"] = pd.to_datetime(_disp["信号日"]).dt.date.astype(str)
                    st.dataframe(_disp.style.format({"预测超额": "{:+.1%}", "实现超额": "{:+.1%}"}, na_rep="—"),
                                 use_container_width=True, hide_index=True)
                st.caption(f"📖 {cal.get('note','')}")
            else:
                st.caption("尚无成熟信号（需等 horizon 走完，信号才能评对错）——留痕会持续积累。")
            with st.expander("⚙️ 开启全自动每日追踪（不打开 app 也记录）"):
                st.markdown("跑一次或挂 Windows 定时任务（美股收盘后）：\n"
                            "```\n.venv\\Scripts\\python -m scripts.daily_track\n```\n"
                            "或双击 **run_daily_track.bat**；注册每日任务见 README『每日自动追踪』。"
                            "它会给关注清单留痕 + 回填评估历史准确率。")
        except Exception as _e:  # noqa: BLE001
            st.caption(f"校准库读取失败：{_e}")

    with _T_TOOL.expander("📐 方法与口径披露（样本/N/CI/DSR/费用/偏差 —— 怎么算的、哪里不可全信）"):
        st.markdown(
            "- **样本窗口**：SPY 自 1995、其余自 2008，到**上一交易日收盘**(yfinance+FRED+财报，本地缓存)。"
            "顶部『现价』是盘中≈15min延迟，仅展示；**距高/回撤桶/最佳入场区/盈亏比一律按上一收盘计**，故与盘中价会有小差。\n"
            "- **分桶**：距前高各回撤带(按 252 日滚动高折算价位)；引擎「回撤桶/in_drawdown」按**全历史峰值(cummax)**，"
            "决策卡/择时的「距1年高」按**近 252 日高**——两者刻意不同(深价值 vs 战术择时)，已分别标注。\n"
            "- **N vs 有效独立 N**：远期窗口高度重叠，名义 N(天数)**不是**独立交易机会数；有效独立窗口 ≈ N天/持有期。"
            "结论可信度看**有效独立 N**，不是大 N。\n"
            "- **CI**：中位数的 **block bootstrap**(块长≈持有期，吸收重叠)；'稳健最佳入场区'要求 CI 下界 > 无条件基准(即超额显著)。\n"
            "- **多重检验**：从多个回撤档选'最佳'用 **deflated Sharpe(DSR)** 折扣，**DSR<0.95** 即标'存疑(可能是运气)'。\n"
            f"- **费用/滑点**：回测含单边 手续费 {0.0005:.2%} + 滑点 {0.0005:.2%}(共 {0.001:.1%}/边)；杠杆 ETF 复利衰减不可长持。\n"
            "- **已知偏差/边界**：单票深档有**幸存者偏差**(只统计到活下来的)；**未做**退市/成分变更回填；"
            "基本面/新闻是 yfinance **当前快照**(含前视/重述，**仅供人读、不入量化**)；估值非 point-in-time。\n"
            "- **唯一 OOS 验证过的可交易规则**：跌破200线/宽度恶化→减半仓(ETF夏普7/7改善、回撤砍约40%、2015+样本外稳健)；"
            "其余(最佳入场区/因子/桶)是**校准**，非择时 alpha。\n"
            "- **「稳健 vs 低置信」大规模回测(科技17只/半导体13只·2008–2026)**："
            "严口径『稳健入场区』(DSR≥0.95)在科技/半导体乃至 SPY/防御股**历史上一次都没触发**(单票择时 DSR 上限≈0.78–0.92，过不了多重检验线)——"
            "所以它是个**实际不可达**的标准；现实里你看到的几乎都是『**可分批建仓·低置信**』。"
            "后者经 **PIT 走查(扩张窗口·无前视)** 仍**跑赢「随便买」**：按回撤档分，"
            "**浅跌 0–10% 档**最优(科技6月超基准+3.6%/12月+9.7%、半导体+7.9%/+12.2%，胜率72–82%)，"
            "中档(10–20%)边际、**深档(>20%)绝对收益高但尾部 p10 −18~−30%、胜率更低**；"
            "频率上『低置信』每股每年约 9–11 次、『稳健』0 次。半导体 edge 比科技大但尾部更深→**更该分批+严控仓**。"
            "（口径：PIT 只用当日之前数据判该档是否正期望，再看真未来收益；故这是扣掉 look-ahead 的**保守**估计。）\n"
            "- **「撤离/离场信号」大规模回测(同宇宙·叠加策略 vs 一直持有·含成本)**："
            "结论先行——**择时离场换不来超额**，每条离场规则都让年化≤一直持有(科技基准夏普≈0.62、半导体≈0.65)；"
            "离场的**唯一价值是压回撤**(把最大回撤从~64% 降到~50%)、让你扛得住别被深套割肉，不是为多赚。"
            "**有效**：跌破200线/宽度恶化→**减半仓**(夏普基本持平、回撤−13pt)；200线+波动高→空仓(回撤压到~−45%、夏普还略升但少赚更多)；"
            "大盘(SPY)破200线→半仓(对半导体性价比最高、少赚最少)。"
            "**有害·别当离场信号**：跌破MA50→空仓(过敏·反复被甩、年化腰斩，科技16%→5.6%)、"
            "固定−20%移动止损→空仓(卖在低点又踩不回、夏普 0.62→0.50)、"
            "**过热乖离止盈**(对回撤几乎0保护、略拖累——过热的票通常更热)。"
            "**口径**：①离场=控回撤非择时alpha；②**减半 > 清仓**(清仓少赚太多、夏普更低)；③只认**破200线+宽度/波动恶化**这一已验证触发，"
            "移动止损/过热止盈只当**锁浮盈的纪律**，不是护回撤的alpha。")
        st.caption("一句话：本工具给**历史条件分布 + 区间 + 有效样本 + 多重检验折扣**，是研究校准、**非投资建议**；"
                   "醒目结论(等级/仓位/最佳入场区)都需结合 CI、有效 N、裁决与本页口径一起看。")
    with _T_TOOL.expander("🩺 数据质量体检（新鲜度 / 缺口 / 异常跳空）"):
      if _lazy_gate(f"dq_{a}"):
        peers_dq = tuple(dict.fromkeys([a] + [t for t in _TICKER_GROUPS[grp] if t != "SPY"]))[:12]
        dh = c_data_health(peers_dq, "2015-01-01", end)
        st.markdown(f'<div class="verdict">{dh["summary"]}</div>', unsafe_allow_html=True)
        st.dataframe(dh["table"], use_container_width=True, hide_index=True)
        st.caption("📖 免费数据(yfinance)可能停更/缺口/未除权跳空——陈旧🔴或带⚠️的标的，其分析结论要打折看。仅体检、不改数据。")

    with _T_ZONE.expander("💠 各价位带明细（盈亏比 / 期望值 / 超额）"):
        enough = z[z["enough"]] if "enough" in z.columns else z.iloc[0:0]
        if not enough.empty:
            zsel = st.selectbox("看某价位带的校准结论", enough["zone"].tolist(), key=f"zsel_{a}")
            row = enough[enough["zone"] == zsel].iloc[0]
            st.markdown(f'<div class="verdict">{ec.format_zone_verdict(row, horizon)}</div>', unsafe_allow_html=True)
            zc = st.columns(4)
            rr = row["reward_risk"]
            zc[0].markdown(stat_card("远期收益中位", f"{row['median']:+.1%}", f"基准 {row['baseline_median']:+.1%}", T["good"], tip="远期收益"), unsafe_allow_html=True)
            zc[1].markdown(stat_card("盈亏比", f"{rr:.2f}" if rr == rr else "—", "赚的÷要忍的浮亏", T["primary"], tip="盈亏比"), unsafe_allow_html=True)
            zc[2].markdown(stat_card("期望值", f"{row['expectancy']:+.1%}", f"胜率 {row['win_rate']:.0%}", T["good"] if row["expectancy"] > 0 else T["bad"], tip="期望值"), unsafe_allow_html=True)
            xcol = T["good"] if (row["ci_low"] > 0 or row["ci_high"] < 0) else T["muted"]
            zc[3].markdown(stat_card("比基准多", f"{row['excess_median']:+.1%}", f"N≈{int(row['n_events'])}·中位95%CI[{row['ci_low']:+.0%},{row['ci_high']:+.0%}]", xcol, tip="超额"), unsafe_allow_html=True)
        show = z[["zone", "price_low", "price_high", "n_events", "win_rate", "median", "reward_risk", "expectancy", "excess_median"]].rename(
            columns={"zone": "价位带(回撤)", "price_low": "价位低", "price_high": "价位高", "n_events": "历史次数",
                     "win_rate": "胜率", "median": "中位涨幅", "reward_risk": "盈亏比", "expectancy": "期望值", "excess_median": "比基准多"})
        st.dataframe(show.style.format({"价位低": "{:.0f}", "价位高": "{:.0f}", "胜率": "{:.0%}", "中位涨幅": "{:+.1%}",
                                        "盈亏比": "{:.2f}", "期望值": "{:+.1%}", "比基准多": "{:+.1%}"}, na_rep="样本不足"),
                     use_container_width=True, hide_index=True, column_config=_col_cfg(show.columns))
        st.caption("📖 列名可悬浮看含义。每个回撤区间对应一段价位，给该状态历史远期收益分布；区间+分布，不是目标价。"
                   "「比基准多」是中位相对无条件基准的**点估计**；上方卡片的绿色/「中位95%CI」是**中位涨幅(绝对收益)**的区间，"
                   "不跨0只说明该状态绝对收益方向较确定，**并非**'比基准多'本身的显著性检验。")

    with _T_ZONE.expander("📦 筹码分布 / 换手位（Volume Profile · POC 高换手价 · 价值区）"):
        try:
            _ohlcv_vp, _vp = c_volume_profile(a, "2015-01-01", end, 252)
            _cur = float(price.iloc[-1]); _poc = float(_vp["poc"])
            _rel = "上方（POC=支撑）" if _cur >= _poc else "下方（POC=压力）"
            cvp = st.columns(3)
            cvp[0].markdown(stat_card("POC 最高换手价", f"{_poc:.1f}", f"现价在其{_rel}", T["info"], tip="POC"), unsafe_allow_html=True)
            cvp[1].markdown(stat_card("高换手价值区(70%)", f"{_vp['value_area'][0]:.0f}–{_vp['value_area'][1]:.0f}", "成交最集中带", T["primary"], tip="Volume Profile"), unsafe_allow_html=True)
            cvp[2].markdown(stat_card("现价 vs POC", f"{_cur:.1f}", f"距POC {(_cur/_poc-1):+.1%}", T["gold"]), unsafe_allow_html=True)
            st.plotly_chart(ch.candle_with_levels(_ohlcv_vp, _vp, title=f"{a} K线 + 筹码分布（近1年日线）", logy=True),
                            use_container_width=True, config=ch.CHART_CONFIG)
            st.caption("📖 **左侧紫色横柱=每个价位累计成交量(筹码)**，直接叠在K线上：柱越长该价位换手越密、支撑/压力越强；"
                       "亮柱=价值区(70%成交带)。**POC=换手最密价位**(蓝色虚线·强磁吸)。现价上穿POC→其转支撑；下破→转压力。"
                       "日线均摊近似·仅辅助·非信号·不入量化。")
        except Exception as _e:  # noqa: BLE001
            st.caption(f"筹码分布暂不可用（{type(_e).__name__}）——可能该标的 OHLCV 数据不足。")

    with _T_ZONE.expander("🪜 建仓方案模拟器：一次性 vs 定投 vs 越跌越补（该怎么把钱投进去）"):
        st.caption("回答一个具体问题：**我要把一笔钱投进这只票，是一次性、还是分批/越跌越补更好？** "
                   "用历史滚动窗口对比三种方案的**回报**与**建仓期最深浮亏(痛感)**，给可执行结论。")
        mc = st.columns([2, 2, 3])
        budget_k = mc[0].number_input("投入金额（美元）", min_value=1000, max_value=10_000_000,
                                      value=25000, step=5000, key=f"budget_{a}")
        bands_pick = mc[1].multiselect("越跌越补触发档（距前高回撤）", ["5%", "10%", "15%", "20%", "25%", "30%"],
                                       default=["10%", "20%", "30%"], key=f"bands_{a}")
        if mc[2].button("▶ 运行建仓方案对比", key=f"runladder_{a}", use_container_width=True) and bands_pick:
            bands = tuple(int(x.rstrip("%")) / 100 for x in bands_pick)
            with st.spinner("滚动窗口模拟 一次性/定投/越跌越补 + bootstrap…"):
                lr = c_ladder(a, zstart, end, bands, float(budget_k))
            # 白话裁决置顶
            st.markdown(f'<div class="verdict" style="font-size:1.02rem">{lr["verdict"]}</div>', unsafe_allow_html=True)
            # 风险-回报权衡图（右下=高回报低痛感）
            st.plotly_chart(ch.ladder_risk_return(lr["per_strategy"]), use_container_width=True, config=ch.CHART_CONFIG)
            # 三方案明细表（回报 + 痛感 + 最坏 + 到位时间 + 跑赢一次性）
            cn = {"lump_sum": "一次性全投", "dca": "定投(DCA)", "ladder": "越跌越补(阶梯)"}
            rows = []
            for k, s in lr["per_strategy"].items():
                vl = lr["vs_lump_sum"].get(k, {})
                rows.append({
                    "方案": cn.get(k, k),
                    f"投${int(budget_k/1000)}k→期末": f"${budget_k*(1+s['median'])/1000:.0f}k ({s['median']:+.0%})",
                    "中位回报": s["median"], "差(p10~p90)": f"{s['p10']:+.0%} ~ {s['p90']:+.0%}",
                    "建仓期最深浮亏": s["mdd_median"], "最坏窗口(p5)": s["p5"],
                    "投满用时(天)": f"{s['deploy_days_median']:.0f}",
                    "跑赢一次性": "—" if k == "lump_sum" else f"{vl.get('beats_lump_rate',float('nan')):.0%}",
                })
            tdf = pd.DataFrame(rows)
            st.dataframe(tdf.style.format({"中位回报": "{:+.0%}", "建仓期最深浮亏": "{:+.0%}", "最坏窗口(p5)": "{:+.0%}"}),
                         use_container_width=True, hide_index=True)
            st.caption(f"📖 {lr['note']}　**怎么读**：右下角的方案=高回报+低痛感最优；"
                       "“建仓期最深浮亏”是投钱过程中净值相对自身峰值的最深回撤（你要扛的痛）；"
                       "“跑赢一次性”是历史上该方案期末回报高于一次性的窗口比例。**非预测、非投资建议**。")

    with _T_RISK.expander("📐 Alpha / Beta（市场模型 · 拆「选股超额」与「市场敞口」）", expanded=False):
      if _lazy_gate(f"ab_{a}"):
        _bench = "SPY"
        _ab = c_alpha_beta(a, "2014-01-01", end, _bench)
        if not _ab.get("available"):
            st.caption("样本不足，Alpha/Beta 暂不可用。")
        else:
            _abc = st.columns(4)
            _bt = _ab["beta"]
            _btcol = T["amber"] if (_bt == _bt and _bt >= 1.3) else (T["good"] if (_bt == _bt and _bt < 0.8) else T["primary"])
            _abc[0].markdown(stat_card(f"β（对 {_bench}）", f"{_bt:.2f}" if _bt == _bt else "—",
                                       f"近1年 {_ab['beta_1y']:.2f}" if _ab['beta_1y'] == _ab['beta_1y'] else "市场敏感度", _btcol, tip="β"), unsafe_allow_html=True)
            _aa = _ab["alpha_ann"]; _aci = _ab["alpha_ci"]; _asig = _ab["alpha_significant"]
            _acol = (T["good"] if (_asig and _aa > 0) else (T["bad"] if (_asig and _aa < 0) else T["muted"]))
            _abc[1].markdown(stat_card("α（年化）", f"{_aa:+.1%}" if _aa == _aa else "—",
                                       ("显著" if _asig else "**不显著**(CI跨0)") + f"·CI[{_aci[0]:+.0%},{_aci[1]:+.0%}]", _acol, tip="α"), unsafe_allow_html=True)
            _bd, _bu = _ab["beta_down"], _ab["beta_up"]
            _bdcol = T["bad"] if (_bd == _bd and _bu == _bu and _bd - _bu > 0.15) else T["primary"]
            _abc[2].markdown(stat_card("下行β / 上行β", f"{_bd:.2f} / {_bu:.2f}" if (_bd == _bd and _bu == _bu) else "—",
                                       "跌时vs涨时敏感度", _bdcol, tip="β"), unsafe_allow_html=True)
            _r2 = _ab["r2"]
            _abc[3].markdown(stat_card("R² / 相关", f"{_r2:.0%} / {_ab['corr']:.2f}" if _r2 == _r2 else "—",
                                       "收益被市场解释的比例", T["info"], tip="R²"), unsafe_allow_html=True)
            _ablines = "<br>".join(x for x in [_ab.get("beta_note"), _ab.get("drift_note"),
                                               _ab.get("risk_note"), "🎯 " + _ab.get("alpha_verdict", "")] if x)
            st.markdown(f'<div class="verdict">{_ablines}</div>', unsafe_allow_html=True)
            st.caption(f"📖 市场模型：个股日收益 对 {_bench} 日收益回归(2014至今·rf≈0近似)。"
                       "**β=市场敞口**(放大/防御)，**α=扣掉β后的超额**(年化·带block bootstrap CI)。"
                       "单票 α **通常不显著**——这与本工具一贯结论一致：可证明的择时/选股超额很罕见，你拿的多半是 beta。"
                       "**下行β>上行β** 说明跌时更敏感(不利不对称)→撤离纪律更重要。全是**历史描述、非预测**(β会漂移，已附近1年值)。")

    with _T_RISK.expander("🛡️ Regime 风险加权暴露（高波动/避险环境自动降仓 · 改善回撤）"):
      if _lazy_gate(f"regime_{a}"):
        ro = c_regime_overlay(a, "2014-01-01", end)
        ex, ov = ro["exposure"], ro["overlay"]
        st.markdown(stat_card("今日建议暴露", f"{ex['exposure']:.0%}", "满仓的百分比(只降不加杠杆)",
                              T["good"] if ex["exposure"] >= 0.8 else (T["gold"] if ex["exposure"] >= 0.5 else T["bad"])),
                    unsafe_allow_html=True)
        if ex["factors"]:
            fdf = pd.DataFrame([{"状态因子": f["name"], "当前": f["state"], "暴露乘子": f"×{f['mult']:.2f}"} for f in ex["factors"]])
            st.dataframe(fdf, use_container_width=True, hide_index=True)
        st.caption(f"📖 {ex['note']}")
        st.markdown("**波动目标 overlay vs 闭眼持有（年化 / 夏普 / 回撤 · 只降不加杠杆）**")
        oc = st.columns(4)
        o_, hh_ = ov["overlay"], ov["hold"]
        oc[0].markdown(stat_card("overlay 年化", f"{o_['cagr']:+.0%}", f"持有 {hh_['cagr']:+.0%}", T["primary"]), unsafe_allow_html=True)
        oc[1].markdown(stat_card("overlay 夏普", f"{o_['sharpe']:.2f}", f"持有 {hh_['sharpe']:.2f}",
                                 T["good"] if o_["sharpe"] >= hh_["sharpe"] else T["muted"], tip="Sharpe"), unsafe_allow_html=True)
        oc[2].markdown(stat_card("overlay 最大回撤", f"{o_['maxdd']:.0%}", f"持有 {hh_['maxdd']:.0%}",
                                 T["good"] if o_["maxdd"] >= hh_["maxdd"] else T["bad"], tip="回撤"), unsafe_allow_html=True)
        oc[3].markdown(stat_card("平均暴露", f"{ov['avg_exposure']:.0%}", f"目标波动 {ov['target_vol']:.0%}", T["info"]), unsafe_allow_html=True)
        st.plotly_chart(ch.equity_compare(ov["equity"], title="波动目标 overlay vs 持有"), use_container_width=True, config=ch.CHART_CONFIG)
        st.caption("📖 波动目标=按近期波动反比缩放仓位(平静加、动荡减，上限不加杠杆)。常以**更低回撤换更稳夏普**，"
                   "年化可能略低——这是风控 edge，不是择时预测。")

    nw = b.get("news")
    if nw is not None and len(nw):
        with _T_EVENT.expander("🗞️ 最近新闻（免费 · 仅线索 · 不入量化）"):
            for _, r in nw.head(6).iterrows():
                title = f"[{r['title']}]({r['url']})" if r.get("url") else r["title"]
                st.markdown(f"- `{r['date']}` {title} — *{r.get('provider','')}*")

    st.divider()
    from analysis import briefing as bf
    try:
        b["regime_path"] = c_regime_path(a, zstart, end)   # 与页面显示同源，报告含趋势全程分布
        b["enter_ok"] = bool(_card.get("enter_ok", True)) if _card else True  # 报告里的预案也随裁决转防守
    except Exception:  # noqa: BLE001
        pass
    md = bf.render_markdown([b], bf.auto_weights([b]), horizon)
    st.download_button("📄 导出该股分析(Markdown)", md, file_name=f"{a}_analysis_{end}.md", mime="text/markdown")

# ---------------------------------------------------------------------------
# 路由（三视图）
# ---------------------------------------------------------------------------
def page_fragility():
    from analysis import fragility as fg
    st.markdown('<div class="hero-title">🛡️ 一篮子分散 + 长持（核心打法）</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">长期最优解=<b>买一篮子优质资产、长期持有</b>。它只会输给两件事：'
                '<b>崩盘时恐慌割肉</b> 与 <b>押单票却没活下来</b>。这页就是对治这两点——'
                '①<b>分散</b>(跨行业篮子，单票归零也伤不到筋骨)；②<b>轻保护</b>(目标波动把回撤压到能扛住、'
                '让你别在底部跳车)。<b>它不跑赢市场，只帮你真能"拿住"。</b></div>', unsafe_allow_html=True)
    st.info("💡 **怎么用**：选一个**分散**的篮子（推荐「跨行业优质篮子」或「仅指数」）→ 调「稳健度」到你**夜里睡得着**的回撤水平 "
            "→ 看它 vs 闭眼持有：复利略让、但回撤腰斩。剩下的就是**拿着、别看盘、别因恐慌割肉**。")

    # ===== 🛡️ 稳定配置（可调目标波动，按你的稳健度）=====
    sc1, sc2 = st.columns([2, 3])
    _uni = sc1.selectbox("组合", list(_STABLE_UNIVERSES), index=0,
                         help="仅指数最稳；聚焦科技/半导体收益高些波动大些")
    _defv = int(_PROFILE_VOL.get(gl_profile, 0.15) * 100)
    _tv = sc2.slider("稳健度 = 目标年化波动（越低越稳，仓位越保守）", 8, 25, _defv, 1,
                     help=f"侧栏稳健度『{gl_profile}』默认 {_defv}%；这里可临时覆盖") / 100.0
    with st.spinner("回测稳定配置…"):
        stab = c_stability(_uni, _tv, "2010-01-01", end)
    o, h = stab["overlay"], stab["hold"]
    if o:
        mc = st.columns(5)
        mc[0].markdown(stat_card("年化", f"{o['cagr']:+.0%}", f"持有 {h['cagr']:+.0%}", T["primary"]), unsafe_allow_html=True)
        mc[1].markdown(stat_card("波动", f"{o['vol']:.0%}", f"持有 {h['vol']:.0%}", T["info"], tip="波动"), unsafe_allow_html=True)
        mc[2].markdown(stat_card("最大回撤", f"{o['maxdd']:.0%}", f"持有 {h['maxdd']:.0%}",
                                 T["good"] if o["maxdd"] >= h["maxdd"] else T["bad"], tip="回撤"), unsafe_allow_html=True)
        mc[3].markdown(stat_card("最差年", f"{o['worst_year']:+.0%}", f"持有 {h['worst_year']:+.0%}", T["gold"]), unsafe_allow_html=True)
        mc[4].markdown(stat_card("滚动1年正收益", f"{o['pos_roll1y']:.0%}", "历史滚动(回看)1年收益为正的占比", T["good"]), unsafe_allow_html=True)
        st.line_chart(stab["equity"], color=[T["good"], T["muted"]])
        st.caption(f"📖 {_uni} · 目标波动{_tv:.0%}：绿=稳定配置(风险叠加)，灰=闭眼持有。"
                   f"最长水下 {o['longest_underwater_m']} 个月、{o['pos_months']:.0%} 的月份为正。"
                   "调低目标波动→更稳、回撤更浅、收益略低。**这是为稳定收益设计的核心配置**。非投资建议。")
    st.divider()
    st.markdown("##### 🩸 市场脆弱性预警（宽度恶化·降仓开关）")
    bk_name = st.selectbox("板块宽度", list(_FRAGILITY_BASKETS), index=0,
                           help="选板块看其专属宽度脆弱性——半导体/科技板块的 de-risk 信号比全市场更贴该板块")
    bk = _FRAGILITY_BASKETS[bk_name]
    with st.spinner(f"计算{bk_name}宽度信号…"):
        fzz = c_fragility("2005-01-01", end, tuple(bk) if bk else None)
    cur, ev = fzz["cur"], fzz["eval"]
    if not cur.get("available"):
        st.warning("数据不足，无法计算。"); return
    cc = st.columns(4)
    col = T["bad"] if cur["fragile"] else T["good"]
    cc[0].markdown(stat_card("当前状态", cur["light"], f"截至 {cur['date']}", col), unsafe_allow_html=True)
    cc[1].markdown(stat_card("市场宽度", f"{cur['breadth']:.0%}", "个股在自身200线上方", T["info"]), unsafe_allow_html=True)
    cc[2].markdown(stat_card("宽度历史分位", f"{cur['pctile']:.0%}", f"<{cur['thresh']:.0%} 触发预警", col, tip="分位"), unsafe_allow_html=True)
    e63 = ev.get(63, {})
    cc[3].markdown(stat_card("信号预警力", f"{e63.get('lift', float('nan')):.2f}x" if e63.get("lift") == e63.get("lift") else "—",
                             f"未来63日大跌概率(误报{e63.get('fp', 0):.0%})", T["primary"], tip="lift"), unsafe_allow_html=True)
    st.markdown(f'<div class="verdict">{"🔴 宽度恶化已触发——历史上未来3个月大跌概率约为平时的 "+format(e63.get("lift",0),".1f")+" 倍，建议系统性降仓（注意：约 60% 是假警报，这是降仓开关非崩盘预言）。" if cur["fragile"] else "🟢 市场宽度健康，无脆弱性预警——不必因此降仓。"}</div>',
                unsafe_allow_html=True)
    # 实测预警力表
    st.markdown("##### 📊 该信号的实测预警力（历史回测，目标=未来H日内跌≥10%）")
    rows = []
    for h, r in ev.items():
        if "lift" in r:
            rows.append({"持有期": f"{h}日", "基率": f"{r['base']:.0%}", "触发后命中": f"{r['cond']:.0%}",
                         "提升(lift)": f"{r['lift']:.2f}x", "召回": f"{r['recall']:.0%}", "误报率": f"{r['fp']:.0%}"})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("📖 lift>1=有预警价值；误报率高=会频繁假警报（用作降仓开关，接受假警报换'漏不掉真回撤'）。")
    # 宽度分位 vs SPY 历史
    fr_df = fzz["frame"].dropna()
    if not fr_df.empty:
        st.markdown("##### 宽度历史分位走势（越低越脆弱）")
        st.line_chart(fr_df[["pctile"]].rename(columns={"pctile": "宽度分位"}), color=T["amber"])

    # 等 / 追 指南
    st.divider()
    st.markdown("##### 🧭 等还是追？（输入标的，按当前回撤 + 市场脆弱性给操作指南）")
    a2 = st.selectbox("标的", _SPY_FIRST, index=0, key="wc_asset")
    from data import loader as _ld
    px2 = _ld.load_prices([a2], "2010-01-01", end)[a2].dropna()
    hi = px2.rolling(252, min_periods=120).max()
    cur_dd = float(px2.iloc[-1] / hi.iloc[-1] - 1.0)
    g = fg.wait_or_chase(cur_dd, fragile_now=cur["fragile"], is_index=(a2 in ("SPY", "QQQ", "DIA", "IWM")))
    gc = st.columns(3)
    gc[0].markdown(stat_card("现价距1年高", f"{cur_dd:+.1%}", a2, T["gold"]), unsafe_allow_html=True)
    gc[1].markdown(stat_card("操作", g["action"], g["headline"], T["bad"] if g["fragile"] else T["good"]), unsafe_allow_html=True)
    gc[2].markdown(stat_card("市场环境", cur["light"], f"宽度分位 {cur['pctile']:.0%}", col), unsafe_allow_html=True)
    st.markdown(f'<div class="verdict">{g["detail"]}</div>', unsafe_allow_html=True)
    st.caption("📖 回测结论：高位附近'等浅回调'历史上更亏(回调70%不来)→应追/分批；唯一该'等'的是深回撤区(指数−20~30%有edge)；脆弱性触发→一切偏防守。非投资建议。")

    # 📉 风险管理叠加（已端到端验证·工具唯一OOS可部署规则）
    st.divider()
    st.markdown("##### 📉 风险管理叠加（已验证·可部署）：减半仓+按波动定仓 vs 闭眼持有")
    st.caption("规则=0.5×波动目标仓 + 0.5×趋势/宽度半仓floor。验证：ETF/个股、样本内外均升夏普、回撤砍~40%、"
               "对参数不敏感。这是改善夏普/砍回撤的**风险管理**，非择时alpha（牛市CAGR略低，换更稳）。")
    a3 = st.selectbox("回测标的", _SPY_FIRST, index=0, key="ov_asset")
    with st.spinner("回测风险管理叠加…"):
        bt = c_overlay(a3, "2008-01-01", end)
    s_, h_ = bt["strategy"], bt["hold"]
    oc = st.columns(4)
    oc[0].markdown(stat_card("夏普", f"{s_['sharpe']:.2f}", f"持有 {h_['sharpe']:.2f}",
                             T["good"] if s_["sharpe"] >= h_["sharpe"] else T["gold"], tip="Sharpe"), unsafe_allow_html=True)
    oc[1].markdown(stat_card("最大回撤", f"{s_['maxdd']:.0%}", f"持有 {h_['maxdd']:.0%}",
                             T["good"] if s_["maxdd"] >= h_["maxdd"] else T["bad"], tip="回撤"), unsafe_allow_html=True)
    oc[2].markdown(stat_card("年化", f"{s_['cagr']:+.0%}", f"持有 {h_['cagr']:+.0%}", T["primary"]), unsafe_allow_html=True)
    oc[3].markdown(stat_card("当前建议仓位", f"{bt['current_position']:.0%}", f"平均 {bt['avg_position']:.0%}", T["info"]), unsafe_allow_html=True)
    eq = bt["equity"]
    if eq is not None and not eq.empty:
        st.line_chart(eq, color=[T["good"], T["muted"]])
    from analysis import overlay as _ov
    st.markdown(f'<div class="verdict">{_ov.verdict(bt)}</div>', unsafe_allow_html=True)
    st.caption(f"🧭 板块适用性：{_ov.sector_effectiveness(a3)}")
    st.caption("📖 绿=风险管理叠加净值，灰=闭眼持有。叠加在ETF上夏普7/7改善、个股7/9、回撤显著更浅；"
               "大规模分板块测试：高beta/周期/ETF升夏普，能源/防御板块主要砍回撤。非投资建议。")

    # 📊 产品级组合回测（整个产品系统的端到端验证·机构指标）
    st.divider()
    st.markdown("##### 📊 产品级组合回测（聚焦组合 应用风险叠加 vs 闭眼持有 vs SPY · 端到端验证）")
    st.caption(f"组合={'/'.join(_FOCUS_UNIVERSE[:8])}… 等权，逐标的应用风险管理叠加。这是『整个产品系统』的端到端回测。")
    with st.spinner("回测整个产品组合…"):
        pbt = c_product_bt("2010-01-01", end)
    if pbt.get("available"):
        rows = []
        for k, lab in [("overlay", "聚焦组合+风险叠加"), ("hold", "聚焦组合 闭眼持有"), ("benchmark", "SPY 基准")]:
            m = pbt.get(k)
            if m:
                rows.append({"方案": lab, "年化": f"{m['cagr']:+.1%}", "波动": f"{m['vol']:.0%}",
                             "夏普": f"{m['sharpe']:.2f}", "索提诺": f"{m['sortino']:.2f}",
                             "卡玛": f"{m['calmar']:.2f}", "最大回撤": f"{m['maxdd']:.0%}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if not pbt["equity"].empty:
            st.line_chart(pbt["equity"], color=[T["good"], T["muted"], T["primary"]][:pbt["equity"].shape[1]])
        ov_m, ho_m = pbt["overlay"], pbt["hold"]
        win = (ov_m["sharpe"] >= ho_m["sharpe"] and ov_m["maxdd"] >= ho_m["maxdd"])
        st.markdown(f'<div class="verdict">{"✅" if win else "🟡"} 产品系统 vs 闭眼持有：'
                    f'夏普 {ov_m["sharpe"]:.2f}/{ho_m["sharpe"]:.2f}、索提诺 {ov_m["sortino"]:.2f}/{ho_m["sortino"]:.2f}、'
                    f'回撤 {ov_m["maxdd"]:.0%}/{ho_m["maxdd"]:.0%}——'
                    f'{"全面更优(风险调整后)，产品达可用专业级" if win else "风险调整后占优、收益略让(牛市现金拖累)"}。</div>',
                    unsafe_allow_html=True)
        # 危机压力测试：风控在关键时刻是否真管用
        cs = _ov.crisis_stress(pbt["equity"])
        if cs:
            st.markdown("**🔥 危机压力测试**（历次崩盘窗口内的区间收益——叠加是否真的少跌）")
            csrows = [{"崩盘事件": c["crisis"], "组合+风险叠加": f"{c.get('组合+风险叠加', float('nan')):+.0%}",
                       "组合闭眼持有": f"{c.get('组合闭眼持有', float('nan')):+.0%}",
                       "SPY": (f"{c.get('基准', float('nan')):+.0%}" if '基准' in c else "—")} for c in cs]
            st.dataframe(pd.DataFrame(csrows), use_container_width=True, hide_index=True)
            st.caption("📖 叠加在崩盘中应明显少跌(数字更接近0或更高)——这是风险管理的核心价值。")
        # 滚动夏普：edge 是否随时间衰减
        rs_o = _ov.rolling_sharpe(pbt["ret_overlay"]) if "ret_overlay" in pbt else pd.Series(dtype=float)
        rs_h = _ov.rolling_sharpe(pbt["ret_hold"]) if "ret_hold" in pbt else pd.Series(dtype=float)
        if len(rs_o) > 10:
            rsdf = pd.DataFrame({"叠加": rs_o, "持有": rs_h}).dropna()
            st.markdown("**📈 滚动 1 年夏普**（监控 edge 是否衰减——叠加线应多数时间≥持有）")
            st.line_chart(rsdf, color=[T["good"], T["muted"]])
    st.caption("📖 这是把工具的可部署规则用到整个聚焦组合的真实回测。结论：风险调整指标(夏普/索提诺/卡玛/回撤)优于持有与SPY。非投资建议。")

def page_industry():
    from analysis import industry as ind
    st.markdown('<div class="hero-title">🏭 行业动向</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">半导体 / 科技板块的<b>真实可观测动向</b>（宽度·相对强度·内部相关性·成长/估值·龙头落后）'
                '+ <b>历史类比情景分布</b>——只给"可能出现哪几种情况 + 各自历史频率/区间"，<b>不判牛熊、不给点位、不预测拐点</b>。</div>',
                unsafe_allow_html=True)
    st.write("")
    sec = st.selectbox("板块", list(ind.SECTORS), index=0)
    tks = ind.SECTORS[sec]
    with st.spinner(f"聚合 {sec} 板块动向…"):
        d = ind.sector_dashboard(tks, "2008-01-01", end)
    if not d.get("available"):
        st.warning(f"板块数据不足：{d.get('reason','')}"); return

    st.markdown(f'<div class="verdict">🧭 <b>{sec} 当前动向</b>（{d["n"]}只·截至{d["asof"]}）：{ind.sector_state_label(d)}</div>',
                unsafe_allow_html=True)
    c = st.columns(4)
    _bc = T["bad"] if (d["breadth_pctile"] == d["breadth_pctile"] and d["breadth_pctile"] < 0.2) else T["good"]
    c[0].markdown(stat_card("宽度(在200线上)", f"{d['breadth']:.0%}",
                            f"历史分位 {d['breadth_pctile']:.0%}" if d["breadth_pctile"] == d["breadth_pctile"] else "—", _bc, tip="脆弱"), unsafe_allow_html=True)
    _rc = T["good"] if d["rs_252"] > 0 else T["bad"]
    c[1].markdown(stat_card("相对大盘(近1年)", f"{d['rs_252']:+.0%}", f"近3月 {d['rs_63']:+.0%}", _rc), unsafe_allow_html=True)
    c[2].markdown(stat_card("内部相关性", f"{d['corr']:.2f}" if d["corr"] == d["corr"] else "—",
                            "高=β主导/低=个股分化", T["info"], tip="相关"), unsafe_allow_html=True)
    c[3].markdown(stat_card("成分分散度(近3月)", f"{d['dispersion']:.0%}", "越大=分化越剧烈", T["primary"]), unsafe_allow_html=True)

    lc = st.columns(2)
    lc[0].markdown("**🏆 龙头(近3月)**：" + "　".join(f"{k} {v:+.0%}" for k, v in d["leaders"]))
    lc[1].markdown("**🐌 落后(近3月)**：" + "　".join(f"{k} {v:+.0%}" for k, v in d["laggards"]))

    # 基本面横截面（仅供人读·不入量化）
    try:
        f = ind.sector_fundamentals(tks)
        if f["n_growth"] or f["n_pe"]:
            st.caption(f"📊 成分基本面中位（{f['n']}只·仅供人读·含前视/陈旧）：营收同比 "
                       + (f"{f['rev_growth_med']:+.0%}" if f['rev_growth_med'] == f['rev_growth_med'] else "—")
                       + "、盈利同比 " + (f"{f['earn_growth_med']:+.0%}" if f['earn_growth_med'] == f['earn_growth_med'] else "—")
                       + "、TTM PE " + (f"{f['pe_med']:.0f}" if f['pe_med'] == f['pe_med'] else "—") + "。")
    except Exception:  # noqa: BLE001
        pass

    # 板块级历史类比情景分布
    st.markdown("##### 📈📉 板块情景分布（历史类比 · 分布非预测）")
    sc = ind.sector_scenarios(d)
    if sc:
        _sc_html = "".join(f'<div style="color:var(--text);font-size:0.85rem;line-height:1.6">{l}</div>' for l in sc["lines"])
        st.markdown(
            f'<div style="border-radius:12px;padding:12px 16px;background:var(--primary-weak);border:1px solid var(--primary-border);border-left:5px solid var(--primary)">'
            f'<div style="color:var(--text);font-size:0.84rem;margin-bottom:4px">{sc["headline"]}</div>{_sc_html}'
            f'<div style="color:var(--muted);font-size:0.84rem;margin-top:5px">📏 {sc["price_range"]}（板块等权指数口径）</div></div>',
            unsafe_allow_html=True)
        st.caption("📖 " + sc["caveat"])
    else:
        st.caption("板块情景分布：等权指数历史不足，暂不可用。")

    # 成分明细表
    with st.expander("📋 成分明细（距200线 / 距1年高 / 波动分位 / 近3月 / 相对板块）", expanded=True):
        t = d["table"].copy()
        st.dataframe(t.style.format({"距200线": "{:+.0%}", "距1年高": "{:+.0%}", "波动分位": "{:.0%}",
                                     "近63日": "{:+.0%}", "vs板块": "{:+.0%}"}, na_rep="—"),
                     use_container_width=True, hide_index=True)

    # 🌐 联网深读（按钮触发·慢·含 GDELT 全球外媒·板块主题聚合·仅线索不入量化）
    st.divider()
    st.markdown("##### 🌐 联网深读（板块龙头新闻 · 含 GDELT 全球外媒 · 主题/情绪聚合 · 仅线索）")
    if st.button("🌐 拉取板块全球新闻动向（联网·约 20–40 秒）", key=f"webnews_{sec}"):
        with st.spinner("联网拉取板块龙头新闻（google + yahoo + GDELT 全球外媒）…"):
            try:
                nz = ind.sector_news_themes(tks, sources=("google", "yahoo", "gdelt"))
            except Exception as _e:  # noqa: BLE001
                nz = None; st.warning(f"联网失败（{type(_e).__name__}）——可能限流，稍后再试。")
        if nz and nz["n"]:
            st.session_state[f"_news_{sec}"] = nz
        elif nz is not None:
            st.caption("暂无拉到新闻（可能限流）。")
    _nz = st.session_state.get(f"_news_{sec}")
    if _nz and _nz.get("n"):
        st.markdown(f"**板块主导主题**：" + ("、".join(f"{t}({c})" for t, c in _nz["top_themes"]) or "无明显主题")
                    + f"　|　情绪天平 利好 {_nz['pos']} / 利空 {_nz['neg']}（{_nz['n']}条·{_nz['providers']}家媒体·启发式非信号）")
        if _nz["hot_pos"] or _nz["hot_neg"]:
            st.caption("情绪偏多成分：" + ("、".join(f"{k}(+{v})" for k, v in _nz["hot_pos"]) or "—")
                       + "　|　情绪偏空成分：" + ("、".join(f"{k}({v})" for k, v in _nz["hot_neg"]) or "—"))
        for it in _nz["items"][:20]:
            t = f"[{it['title']}]({it['url']})" if it.get("url") else it["title"]
            th = ("·".join(it["themes"][:2])) if it.get("themes") else ""
            st.markdown(f"- `{it['date'][:10]}` **{it['ticker']}** {('['+it['sentiment']+'] ') if it['sentiment'] else ''}{t}"
                        + (f"　_{th}_" if th else ""))
        st.caption("⚠️ 新闻仅供顺藤摸瓜、含前视、不入任何量化结果。")
        # 🚀 一键转 Claude 做全文读网 + 深度推理（app 内只聚合标题/情绪，全文推理在 Claude）
        claude_deep_button(f"用 Claude 深度分析 {sec.split()[-1]}板块（读全文+推理）",
                           f"深度分析 {sec} 板块（{', '.join(tks[:8])} 等）：读全网新闻全文，"
                           f"结合行业动向（宽度/相对强度/成长/估值/供应链）推理可能出现的情景与各自概率，"
                           f"给中长期定位。用 quant-deep-brief 口径：校准而非预测、给情景分布不拍单点。",
                           key=f"cl_{sec}")

    st.caption("⚠️ 全页为**历史可观测动向 + 历史类比分布**，非预测、不判牛熊、不给点位；新闻/基本面仅供人读、不入量化。")


def page_position_card():
    st.markdown('<div class="hero-title">🎖️ 建仓 / 撤离作战卡</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">把<b>全套回测验证过</b>的逻辑落成一张卡：今天该建多少仓、什么时候撤。'
                'v3 连续定仓(DSR 0.999)+回撤桶+趋势死亡触发——<b>证伪的逻辑(固定止损/快速二元出场)一律不进</b>。</div>',
                unsafe_allow_html=True)
    st.write("")
    from analysis import position_guidance as pg

    with st.spinner(f"生成 {asset} 作战卡…"):
        try:
            g = pg.position_guidance(asset, start=start, end=end, horizon=gl_horizon)
        except Exception as e:  # noqa: BLE001
            st.error(f"作战卡生成失败：{type(e).__name__}: {e}")
            return
    r, b, x = g["regime"], g["build"], g["exit"]
    st.info(g["headline"])

    # —— 今日四档暴露(含🔥杠杆进取) ——
    st.markdown("##### 📊 今日建议暴露（v3 连续定仓·已验证；非投资建议）")
    ec = st.columns(4)
    _pcol = {"conservative": T["info"], "moderate": T["good"], "aggressive": T["amber"], "leveraged": T["bad"]}
    for i, k in enumerate(["conservative", "moderate", "aggressive", "leveraged"]):
        ev = g["exposure"][k]
        _lev = f"·上限{ev['max_lev']:g}×" if ev.get("leveraged") else ""
        ec[i].markdown(stat_card(f"{pg.PROFILE_ZH[k]}档（波动{ev['target_vol']:.0%}{_lev}）",
                                 f"{ev['exposure_pct']}%", "建议暴露", _pcol[k]), unsafe_allow_html=True)
    st.caption(f"当前触发：{x['active_trigger']}")
    if g["exposure"]["leveraged"].get("lev_gated_off"):
        st.info("🔒 低波动门：当前波动分位偏高(>50%) → **🔥杠杆档今日不上杠杆**(封顶100%)。只在低波动+确认趋势时才放大。")
    if x.get("leverage_warning"):
        st.error(x["leverage_warning"])
    st.write("")

    # —— 暴露历史图（中性档）——
    hist = g.get("exposure_history")
    if hist is not None and not hist.empty:
        st.markdown("##### 📈 历史暴露轨迹（中性档·复核引擎在崩盘年是否真的减仓）")
        norm = (hist["price"] / hist["price"].iloc[0])
        chart_df = pd.DataFrame({"价格(归一)": norm, "建议暴露(0-1)": hist["exposure"]})
        st.line_chart(chart_df, color=[T["muted"], T["good"]])
        st.caption("绿线=v3 建议暴露(0-1)，灰线=价格归一。崩盘段绿线应明显下沉(自动减仓)，平时贴近满仓。")
    st.write("")

    # —— 🪜 保护↔复利谱（本票历史回测，v3.1 软地板）——
    st.markdown("##### 🪜 保护 ↔ 复利谱（**本票历史回测** · 四档暴露 vs 闭眼持有）")
    try:
        spec = c_exposure_spectrum(asset, start, end)
        _meta = spec.get("_meta", {})
        order = [("conservative", "稳健·最强保险"), ("moderate", "中性(默认)"),
                 ("aggressive", "进取"), ("leveraged", "🔥杠杆"), ("hold", "闭眼持有")]
        rows = []
        for k, zh in order:
            m = spec.get(k) or {}
            if not m:
                continue
            rows.append({"档位": zh, "年化": m["cagr"], "终值×": m["mult"], "夏普": m["sharpe"],
                         "最大回撤": m["mdd"], "Calmar": m["calmar"]})
        sdf = pd.DataFrame(rows).set_index("档位")
        st.dataframe(
            sdf.style.format({"年化": "{:+.0%}", "终值×": "{:.1f}x", "夏普": "{:.2f}",
                              "最大回撤": "{:+.0%}", "Calmar": "{:.2f}"}),
            use_container_width=True)
        # 一句话点评：相对持有，谱上各档的取舍
        _h = spec.get("hold", {})
        if _h:
            _mod = spec.get("moderate", {})
            st.caption(
                f"样本：{_meta.get('ticker','')} {_meta.get('start','')}→{_meta.get('end','')}"
                f"（约{_meta.get('years','?')}年）。**越往下复利越高、回撤越深**——你在谱上选位置即选"
                f"『保护 vs 复利』。默认中性档：年化 {_mod.get('cagr',float('nan')):+.0%}、回撤 "
                f"{_mod.get('mdd',float('nan')):+.0%}（持有 {_h.get('cagr',float('nan')):+.0%}/"
                f"{_h.get('mdd',float('nan')):+.0%}）——复利让一截、回撤砍一半。")
        st.caption("⚠️ 校准非预测：这是**这只票的历史**经验(含已知幸存者光环)，不代表未来；撤离=崩盘保险，"
                   "绝对收益上没有任何档跑赢长牛持有。已过 walk-forward 前向滚动 + Deflated Sharpe 多重检验。")
    except Exception as _e:  # noqa: BLE001
        st.caption(f"谱回测暂不可用：{type(_e).__name__}")
    st.write("")

    cL, cR = st.columns(2)
    with cL:
        # —— 🎯 合理入场位（先答能不能碰=regime/飞刀/预警 → 再给可执行回踩支撑；统计锚定价已降级）——
        try:
            # 传入离场预警，使入场判断与撤离一致（避免"可建仓"与"撤离黄灯"自相矛盾）
            _ew_for_entry = None
            try:
                from analysis import decision as _dec2
                _eft2 = c_fragility(start, end).get("cur", {})
                _ew_for_entry = _dec2.exit_warning(c_prices((asset,), start, end)[asset],
                                                   _eft2.get("fragile", False), _eft2.get("pctile"))
            except Exception:  # noqa: BLE001
                pass
            _wred = bool(_ew_for_entry and _ew_for_entry.get("red"))
            _wamb = bool(_ew_for_entry and _ew_for_entry.get("amber"))
            _wl = (_ew_for_entry or {}).get("level", "")
            _ef = c_entry_confluence(asset, start, end, _wred, _wamb, _wl)
            _cur = _ef["current_price"]
            _ecol = (T["bad"] if _ef["grade_tag"] == "🔴" else
                     T["good"] if _ef["grade_tag"] == "🟢" else
                     T["gold"] if _ef["grade_tag"] == "🟡" else T["muted"])
            _below = _ef.get("supports_below") or []
            _near = _ef.get("supports_near_now") or []
            # 主行：评级（能不能碰）
            st.markdown(
                f'<div style="border-radius:12px;padding:11px 15px;margin:0 0 8px;'
                f'background:{_ecol}1f;border:1px solid {_ecol}55;border-left:5px solid {_ecol}">'
                f'<div style="font-size:0.74rem;color:var(--muted);letter-spacing:.4px">🎯 合理入场位 · 现价 {_cur:.1f}</div>'
                f'<div style="font-size:1.05rem;font-weight:800;color:{_ecol};margin-top:2px">{_ef["grade_tag"]} {_ef["grade"]}</div>'
                f'</div>', unsafe_allow_html=True)
            # 可执行回踩分批区（现价下方最近支撑）—— 这是真正能用的"在哪买"
            if _ef.get("at_support_now") and _near:
                st.markdown("**现价即在支撑共振区**：" + "".join(
                    f'<span style="display:inline-block;margin:2px 5px 2px 0;padding:2px 8px;border-radius:6px;'
                    f'background:var(--good-weak);border:1px solid var(--good-border);font-size:0.72rem;color:var(--text)">'
                    f'{s["label"]} {s["price"]:.1f}</span>' for s in _near[:5]), unsafe_allow_html=True)
            if _below:
                st.markdown("**📉 回踩分批区（现价下方最近支撑，若到达分批）**：" + "".join(
                    f'<span style="display:inline-block;margin:2px 5px 2px 0;padding:2px 8px;border-radius:6px;'
                    f'background:var(--card2);border:1px solid var(--border);font-size:0.72rem;color:var(--text)">'
                    f'{s["label"]} <b>{s["price"]:.1f}</b>({s["dist_pct"]:+.0%})</span>' for s in _below[:4]),
                    unsafe_allow_html=True)
            st.caption(_ef["note"])
            if _ef.get("stat_note"):
                st.caption(_ef["stat_note"])
        except Exception as _e:  # noqa: BLE001
            st.caption(f"合理入场位暂不可用：{type(_e).__name__}")

        st.markdown(f"##### 🎯 建仓：{b['stance']}　`{b['grade']}`")
        st.caption(b["why"])
        bk = b["drawdown_bucket"]
        if bk.get("available"):
            tag = "⚠️动量陷阱(超额≤0)" if bk["momentum_trap"] else ("🟡正倾斜·suggestive" if bk["excess"] > 0 else "证据弱")
            st.markdown(f"- 回撤桶证据（{tag}）：回撤中 {bk['horizon']}日 **超额(vs随机买入) {bk['excess']:+.1%}**，"
                        f"N独立={bk['n_independent']}")
            st.caption(f"桶内绝对收益中位 {bk['median']:+.1%}、CI[{bk['ci'][0]:+.1%},{bk['ci'][1]:+.1%}]"
                       "——此CI是绝对收益非超额，牛股恒正≠择时有优势。")
        if b["zones"]:
            st.markdown("- 候选建仓区间（**若到达就行动**，非预测点位）：")
            for z in b["zones"]:
                mark = " ✅已到达" if z["reached"] else ""
                st.markdown(f"    - {z['tier']}（{z['band']}）：≈ **{z['price_low']:.1f}–{z['price_high']:.1f}**，"
                            f"{z['size_hint']}{mark}")
            st.caption(b["sizing_rule"] + " " + b["anti_chase"])
        if b["pead"]:
            p = b["pead"]
            st.markdown(f"- 📅 PEAD窗口：上次{'超预期' if p['beat'] else '不及'}(surprise {p['surprise']:+.1f}%)、"
                        f"距今{p['days_since']:.0f}天，历史同类漂移中位 {p['drift_median']:+.1%}(N={p['n']})")
    with cR:
        st.markdown("##### 📉 撤离：触发条件（已验证·崩盘保险口径）")
        for t in x["triggers"]:
            st.markdown(f"- {t}")
        st.warning(x["honesty"])

    st.caption("_" + g["disclaimer"] + "_")
    st.divider()
    st.download_button("⬇️ 导出作战卡(Markdown)", pg.format_guidance(g),
                       file_name=f"{asset}_作战卡_{r['asof']}.md", mime="text/markdown")


def page_stock_ranking():
    st.markdown('<div class="hero-title">🏆 最推荐买入 · 选股榜</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">多因子横截面综合评分(风险调整动量+趋势健康+相对强度)→排名，'
                '<b>带 Alphalens 口径的 IC 验证</b>。诚实：科技窄池横截面预测力弱——这是<b>健康趋势票筛选器</b>，'
                '不是"top1会跑赢"的预测。</div>', unsafe_allow_html=True)
    st.write("")
    from analysis import stock_ranking as sr

    _groups = {"科技+半导体(默认)": ["NVDA", "AMD", "AVGO", "TSM", "ASML", "QCOM", "MU", "TXN", "INTC", "AMAT",
                                  "LRCX", "KLAC", "ADI", "MRVL", "ON", "AAPL", "MSFT", "GOOGL", "AMZN", "META",
                                  "TSLA", "ORCL", "CRM", "ADBE", "NFLX"],
               "纯半导体": ["NVDA", "AMD", "AVGO", "TSM", "ASML", "QCOM", "MU", "TXN", "INTC", "AMAT", "LRCX",
                          "KLAC", "ADI", "MRVL", "ON"],
               "Mag7+": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "NFLX"]}
    gsel = st.selectbox("票池", list(_groups), index=0)
    tickers = _groups[gsel]

    if st.button("🚀 生成选股榜 + IC 验证", use_container_width=True):
        st.session_state["_rank_go"] = True
    if not st.session_state.get("_rank_go"):
        st.info("选好票池 → 点🚀。约 10–20s(含 IC 验证 + 前名 v3 暴露)。")
        return

    with st.spinner("计算多因子综合分 + IC 验证…"):
        res = sr.rank_stocks(tickers, start=str(start))
        val = sr.validate_ranking(tickers, start="2013-01-01")
        verd = sr.ranking_verdict(val)

    # —— 诚实裁决横幅 ——
    if verd["grade"].startswith("🔴"):
        st.error(f"**{verd['grade']}**　{verd['verdict']}")
    elif verd["grade"].startswith("🟡"):
        st.warning(f"**{verd['grade']}**　{verd['verdict']}")
    else:
        st.success(f"**{verd['grade']}**　{verd['verdict']}")
    st.caption("✅ 正确用法：" + verd["usage"])
    st.write("")

    # —— 排名表 ——
    st.markdown(f"##### 🏆 选股榜（{res['asof']}）")
    tab = res["table"].copy()
    show = tab[["排名", "票", "综合分", "风险调整动量z", "趋势健康z", "相对强度z", "站上200线", "v3中性暴露%", "tier"]]
    st.dataframe(show, use_container_width=True, hide_index=True,
                 column_config={"综合分": st.column_config.NumberColumn(format="%+.2f"),
                                "v3中性暴露%": st.column_config.NumberColumn("v3暴露%", format="%d%%")})
    st.caption(res["note"])

    # —— IC 验证表 ——
    st.markdown("##### 🔬 IC 验证（Alphalens 口径·诚实检验预测力）")
    vr = []
    for h in (21, 63):
        d = val[h]
        vr.append({"持有期": f"{h}日", "RankIC均": d["ic_mean"], "ICIR": d["icir"],
                   "IC>0占比": d["ic_positive_rate"], "分位价差(top-bot)": d["quantile_spread_mean"],
                   "安慰剂IC": d["placebo_ic_mean"], "期数N": d["n_periods"]})
    st.dataframe(pd.DataFrame(vr), use_container_width=True, hide_index=True,
                 column_config={"RankIC均": st.column_config.NumberColumn(format="%+.3f"),
                                "ICIR": st.column_config.NumberColumn(format="%.2f"),
                                "IC>0占比": st.column_config.NumberColumn(format="%.0f%%"),
                                "分位价差(top-bot)": st.column_config.NumberColumn(format="%+.2f%%"),
                                "安慰剂IC": st.column_config.NumberColumn(format="%+.3f")})
    st.caption(val["_note"])
    st.download_button("⬇️ 导出选股榜(Markdown)", sr.format_ranking(res, top_n=len(tab)),
                       file_name=f"选股榜_{gsel}_{res['asof']}.md", mime="text/markdown")


# 任务 → 页面映射（在所有 page_* 定义之后构建，引用函数对象）
_STOCK_SUB = dict(zip(_STOCK_SUB_NAMES,
                      [page_panorama, page_position_card, page_snapshot, page_events, page_earnings]))
_RESEARCH_SUB = dict(zip(_RESEARCH_SUB_NAMES,
                         [page_stock_ranking, page_industry, page_rule, page_factor, page_regime, page_strategies]))
try:
    if job == "🎯 个股决策":
        _STOCK_SUB.get(sub, page_panorama)()
    elif job == "🛡️ 组合配置":
        page_fragility()
    elif job == "📋 多票简报":
        page_briefing()
    elif job == "🔬 研究台":
        _RESEARCH_SUB.get(sub, page_stock_ranking)()
    else:
        page_overview()
except Exception as e:  # noqa: BLE001
    st.error(f"出错了：{type(e).__name__}: {e}")
    st.exception(e)
