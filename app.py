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

st.set_page_config(page_title="量化研究工具", page_icon="📊", layout="wide",
                   initial_sidebar_state="auto")  # auto：桌面展开 / 移动端自动收起，避免侧边栏全屏遮挡正文

CFG = config.load_config()


# ---------------------------------------------------------------------------
# 右上角实时纽约时间（JS 跳秒，注入父文档；不依赖 Streamlit rerun）
# ---------------------------------------------------------------------------
def _ny_clock():
    import streamlit.components.v1 as _components
    _components.html(
        """
        <script>
        (function(){
          const doc = window.parent.document;
          let el = doc.getElementById('ny-clock');
          if(!el){
            el = doc.createElement('div');
            el.id = 'ny-clock';
            el.style.cssText = 'position:fixed;top:8px;right:16px;z-index:100000;'
              + 'font-family:-apple-system,Segoe UI,Roboto,monospace;font-size:0.82rem;'
              + 'color:#BFD8FF;background:rgba(15,20,34,0.78);border:1px solid rgba(124,92,252,0.45);'
              + 'border-radius:9px;padding:5px 11px;letter-spacing:0.3px;pointer-events:none;'
              + 'box-shadow:0 4px 14px rgba(0,0,0,0.4)';
            doc.body.appendChild(el);
          }
          function tick(){
            try{
              const now = new Date();
              const t = now.toLocaleString('en-US',{timeZone:'America/New_York',
                year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',
                second:'2-digit',hour12:false,weekday:'short'});
              // 判断美股是否开盘（周一–五 9:30–16:00 ET）
              const parts = new Intl.DateTimeFormat('en-US',{timeZone:'America/New_York',
                hour:'2-digit',minute:'2-digit',hour12:false,weekday:'short'}).formatToParts(now);
              const get=k=>parts.find(p=>p.type===k)?.value;
              const wd=get('weekday'); const hh=parseInt(get('hour')); const mm=parseInt(get('minute'));
              const isWk=!['Sat','Sun'].includes(wd);
              const mins=hh*60+mm; const open=isWk&&mins>=570&&mins<960;
              el.innerHTML = '🗽 纽约 '+t+'  '+(open?'<span style="color:#2BE6A8">●开盘</span>':'<span style="color:#8A93A6">●休市</span>');
            }catch(e){}
          }
          if(window.parent.__nyClockTimer) clearInterval(window.parent.__nyClockTimer);
          tick(); window.parent.__nyClockTimer = setInterval(tick,1000);
        })();
        </script>
        """,
        height=0,
    )


_ny_clock()

# ---------------------------------------------------------------------------
# 炫酷暗色样式
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* 用系统字体栈，避免每次渲染阻塞式拉取远程 Google Fonts（卡顿主因之一） */
    html, body, [class*="css"] { font-family: -apple-system, 'Segoe UI', 'Microsoft YaHei', Roboto, sans-serif; }
    .stApp {
        background:
          radial-gradient(1200px 600px at 10% -10%, rgba(124,92,252,0.18), transparent 60%),
          radial-gradient(1000px 500px at 110% 10%, rgba(0,212,255,0.12), transparent 55%),
          #0B0E14;
    }
    .hero-title {
        font-size: 2.5rem; font-weight: 800; letter-spacing:-1px; margin-bottom:0;
        background: linear-gradient(92deg, #7C5CFC 0%, #00D4FF 60%, #2BE6A8 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-sub { color:#8A93A6; font-size:0.98rem; margin-top:2px; }
    /* 去掉 backdrop-filter:blur —— 多张玻璃卡叠加 blur 是滚动/悬浮重绘卡顿的元凶；
       改用稍实的半透明底色保持质感，GPU 开销趋近于零 */
    .glass {
        background: rgba(26,31,46,0.55); border:1px solid rgba(255,255,255,0.08);
        border-radius:16px; padding:18px 20px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.22);
    }
    .stat-label { color:#8A93A6; font-size:0.8rem; text-transform:uppercase; letter-spacing:.5px; }
    .stat-value { font-size:1.7rem; font-weight:800; line-height:1.2; }
    .pill {
        display:inline-block; padding:3px 11px; border-radius:999px; font-size:.74rem;
        font-weight:600; margin-right:6px;
    }
    .pill-good { background:rgba(43,230,168,0.15); color:#2BE6A8; border:1px solid rgba(43,230,168,0.35);}
    .pill-warn { background:rgba(255,92,122,0.15); color:#FF5C7A; border:1px solid rgba(255,92,122,0.35);}
    .pill-info { background:rgba(0,212,255,0.13); color:#00D4FF; border:1px solid rgba(0,212,255,0.3);}
    .verdict {
        border-left:4px solid #7C5CFC; padding:12px 16px; border-radius:8px;
        background:rgba(124,92,252,0.08); font-size:0.97rem; line-height:1.6;
    }
    section[data-testid="stSidebar"] { background:#0c1119; border-right:1px solid rgba(255,255,255,0.06);}
    #MainMenu, footer {visibility:hidden;}
    .stTabs [data-baseweb="tab-list"] { gap:4px; }

    /* ===================== 移动端适配 ===================== */
    /* 平板 / 大屏手机：≤768px —— 收紧留白、多列换行堆叠、标题与卡片缩放 */
    @media (max-width: 768px) {
        /* 主内容区留白收紧（wide 布局默认两侧留白在手机上过大）+ 顶部留位给 header/时钟 */
        [data-testid="stMainBlockContainer"], .block-container {
            padding-left: 0.7rem !important; padding-right: 0.7rem !important;
            padding-top: 3.2rem !important;
        }
        /* 关键：横向列组在窄屏换行堆叠，而不是被挤成细长条 */
        [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: 0.55rem !important; }
        [data-testid="stColumn"] {
            min-width: calc(50% - 0.55rem) !important;
            flex: 1 1 calc(50% - 0.55rem) !important;
        }
        /* 标题 / 文案 / 统计卡 缩放 */
        .hero-title { font-size: 1.55rem !important; letter-spacing:-0.4px !important; }
        .hero-sub   { font-size: 0.84rem !important; }
        .stat-value { font-size: 1.25rem !important; }
        .stat-label { font-size: 0.68rem !important; }
        .glass      { padding: 12px 13px !important; border-radius: 13px !important; }
        .verdict    { font-size: 0.88rem !important; padding: 10px 12px !important; }
        /* Tab 列表横向滚动，避免标签过多换行错乱 */
        .stTabs [data-baseweb="tab-list"] { overflow-x: auto !important; flex-wrap: nowrap !important; }
        .stTabs [data-baseweb="tab"] { white-space: nowrap !important; }
        /* 数据表 / 表格 横向可滚动 */
        [data-testid="stDataFrame"], [data-testid="stTable"] { overflow-x: auto !important; }
        [data-testid="stMetricValue"] { font-size: 1.3rem !important; }
        /* 右上角纽约时钟：缩小、避免遮挡 header，允许换行不溢出 */
        #ny-clock {
            font-size: 0.64rem !important; padding: 3px 7px !important;
            top: 6px !important; right: 8px !important; max-width: 66vw !important;
            line-height: 1.25 !important; white-space: normal !important;
        }
    }
    /* 小屏手机：≤480px —— 列全部单列堆叠、标题进一步缩小 */
    @media (max-width: 480px) {
        [data-testid="stColumn"] { min-width: 100% !important; flex: 1 1 100% !important; }
        .hero-title { font-size: 1.32rem !important; }
        .hero-sub   { font-size: 0.8rem !important; }
        [data-testid="stMainBlockContainer"], .block-container {
            padding-left: 0.5rem !important; padding-right: 0.5rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
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


def _col_cfg(columns):
    """为数据表生成带悬浮释义的 column_config：列名照常显示，悬浮列头出含义。"""
    cfg = {}
    for c in columns:
        h = gl.col_help(c)
        if h:
            cfg[c] = st.column_config.Column(c, help=h)
    return cfg


def stat_card(label, value, sub="", color="#E6E9EF", tip=None):
    """tip：术语 key，则标签变成可悬浮解释的术语。"""
    lab = gl.term(tip, label) if tip else label
    return (
        f'<div class="glass" style="text-align:left">'
        f'<div class="stat-label">{lab}</div>'
        f'<div class="stat-value" style="color:{color}">{value}</div>'
        f'<div class="hero-sub" style="font-size:.8rem">{sub}</div></div>'
    )


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
                   color="#7C5CFC", log=True):
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
def c_scan(tickers: tuple, start: str, end: str, horizon: int):
    """信号扫描 + 由扫描结果推出的建仓/风险权重 + 七股今日综合。"""
    from analysis import signal_scan as ss
    df = ss.scan(list(tickers), start=start, end=end, horizon=horizon, n_boot=400)
    ew = {r["signal"]: max(0.0, r["excess"]) for _, r in df[df.kind == "entry"].iterrows()}
    rw = {r["signal"]: abs(min(0.0, r["excess"])) + abs(r["fwd_drawdown"]) for _, r in df[df.kind == "risk"].iterrows()}
    cs = ss.cross_section_today(list(tickers), start, end, ew, rw)
    return df, cs, tuple(sorted(ew.items())), tuple(sorted(rw.items()))


@st.cache_data(show_spinner=False)
def c_score_series(ticker: str, start: str, end: str, ew_items: tuple, rw_items: tuple):
    from analysis import signal_scan as ss
    return ss.score_series(ticker, start, end, dict(ew_items), dict(rw_items))


@st.cache_data(show_spinner=False)
def c_perf(ticker: str, start: str, end: str):
    from backtest import strategies as bt
    return bt.strategy_vs_hold(ticker, start, end)


@st.cache_data(show_spinner=False)
def c_alpha_beta(ticker: str, start: str, end: str):
    """策略 与 持有 各自相对 SPY 的 α/β 分解。"""
    from analysis import quant_edge as qe
    pv = c_perf(ticker, start, end)
    spy = c_prices(("SPY",), start, end)["SPY"].pct_change()
    strat_ret = pv["equity"]["策略"].pct_change()
    hold_ret = pv["equity"]["持有"].pct_change()
    return {"strategy": qe.alpha_beta(strat_ret, spy), "hold": qe.alpha_beta(hold_ret, spy)}


@st.cache_data(show_spinner=False)
def c_factor_attr(ticker: str, start: str, end: str):
    """策略收益的多因子(市场/动量/价值/小盘/低波)归因。"""
    from analysis import quant_edge as qe
    pv = c_perf(ticker, start, end)
    strat_ret = pv["equity"]["策略"].pct_change()
    etfs = tuple(dict.fromkeys(qe.FACTOR_ETFS.values()))
    pxe = c_prices(etfs, start, end)
    fp = {e: pxe[e] for e in etfs if e in pxe.columns}
    return qe.factor_attribution(strat_ret, fp)


@st.cache_data(show_spinner=False)
def c_regime_overlay(ticker: str, start: str, end: str):
    from analysis import quant_edge as qe
    px = c_prices((ticker,), start, end)[ticker]
    macro = c_macro("1990-01-01", end)
    return {"exposure": qe.regime_exposure(px, macro), "overlay": qe.vol_target_backtest(px)}


@st.cache_data(show_spinner=False)
def c_walkforward(ticker: str, start: str, end: str):
    from analysis import quant_edge as qe
    return qe.walkforward_oos(ticker, start, end)


@st.cache_data(show_spinner=False)
def c_pead(ticker: str, start: str, end: str):
    from analysis import quant_edge as qe
    return qe.pead_now(ticker, start, end)


@st.cache_data(show_spinner=False)
def c_cross_section(tickers: tuple, start: str, end: str):
    from analysis import quant_edge as qe
    px = c_prices(tickers, start, end)
    return qe.cross_section_edge(px)


@st.cache_data(show_spinner=False)
def c_analogs(ticker: str, start: str, end: str, horizon: int):
    from analysis import analogs as ag
    px = c_prices((ticker,), start, end)[ticker]
    macro = c_macro("1990-01-01", end)
    return ag.historical_analogs(px, macro, ticker, horizon=horizon)


@st.cache_data(show_spinner=False)
def c_port_weights(tickers: tuple, start: str, end: str, method: str):
    from analysis import quant_edge as qe
    return qe.portfolio_weights(c_prices(tickers, start, end), method=method)


@st.cache_data(show_spinner=False)
def c_signal_decay(tickers: tuple, start: str, end: str):
    from analysis import quant_edge as qe
    return qe.signal_decay(c_prices(tickers, start, end))


@st.cache_data(show_spinner=False)
def c_purged_cv(tickers: tuple, start: str, end: str, horizon: int = 21):
    """横截面动量因子的 purged+embargo CV 无泄漏 OOS IC。"""
    from stats import purged_cv as pcv
    px = c_prices(tickers, start, end).dropna(how="all").ffill()
    score = px.pct_change(252).shift(21)  # 12-1 动量
    return pcv.purged_cv_ic(score, px, horizon=horizon, n_splits=6, embargo=0.02)


@st.cache_data(show_spinner=False)
def c_data_health(tickers: tuple, start: str, end: str):
    from analysis import data_quality as dq
    return dq.data_health(c_prices(tickers, start, end))


@st.cache_data(show_spinner=False, ttl=30)
def c_live_quote(ticker: str, _bucket: int = 0):
    """近实时现价快照（缓存 30 秒）。_bucket 让盘中按分钟桶强制刷新。"""
    from data import loader
    return loader.live_quote(ticker)


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
_TICKER_GROUPS = {
    "🌟 七姐妹": ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"],
    "🎬 流媒体/其他大盘": ["NFLX", "DIS", "UBER", "PLTR"],
    "🔌 半导体": ["NVDA", "AMD", "AVGO", "TSM", "ASML", "MU", "INTC", "QCOM", "TXN", "ARM", "SMCI", "MRVL"],
    "⚡ 杠杆ETF(3x)": ["TQQQ", "SOXL", "UPRO", "TECL", "FNGU", "TNA"],
    "📈 指数ETF": ["SPY", "QQQ", "DIA", "IWM"],
}
_ALL_TICKERS = list(dict.fromkeys(t for g in _TICKER_GROUPS.values() for t in g))
_SPY_FIRST = ["SPY"] + [t for t in _ALL_TICKERS if t != "SPY"]
_VIEWS = ["📊 个股分析", "📋 多票对比", "🧪 高级研究"]
_ADV_NAMES = ["🎯 个股进出场规则(回测器)", "🔬 因子评估", "🌊 大盘 regime(SPY/宏观)",
              "📈 当前快照", "🗞️ 事件时间线", "📅 财报 PEAD", "💰 建仓策略对比", "ℹ️ 关于/术语"]
_HZ = {"3 个月 (63日)": 63, "6 个月 (126日)": 126, "12 个月 (252日)": 252, "24 个月 (504日)": 504}

with st.sidebar:
    st.markdown("### 📊 量化研究工具")
    st.caption("选股 → 自动全景分析 · 校准而非预测")
    grp = st.selectbox("📂 板块", list(_TICKER_GROUPS), index=0)
    asset = st.selectbox("🎯 选择标的", _TICKER_GROUPS[grp], index=0)
    gl_horizon = _HZ[st.selectbox("分析周期", list(_HZ), index=0, help="远期收益/建仓校准的持有期")]
    gl_profile = st.selectbox("🧭 风险偏好", ["保守", "均衡", "激进"], index=1,
                              help="在证据等级给的仓位封顶上做个性化缩放：保守 0.5×、均衡 1×、激进 1.25×（仍受单票硬上限约束）。")
    if st.button("🔄 重新分析（刷新数据/重算）", use_container_width=True,
                 help="清空缓存并用最新数据重新计算当前页"):
        st.cache_data.clear()
        st.rerun()
    if grp == "⚡ 杠杆ETF(3x)":
        st.caption("⚠️ 3x 杠杆 ETF 有每日复利衰减、长持有失真，历史短；分析仅供参考。")
    st.divider()
    view = st.radio("视图", _VIEWS, index=0, label_visibility="collapsed")
    adv = st.selectbox("研究工具", _ADV_NAMES) if view == "🧪 高级研究" else None
    st.divider()
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
def page_signals():
    st.markdown('<div class="hero-title">🔎 信号挖掘</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">量化对比各<b>建仓/止盈状态</b>的远期超额——带 N、block bootstrap CI、'
                '和<b>多重检验 FDR 校正</b>。<b>只校准不预测</b>，分数是历史分位倾斜，非买卖指令。</div>', unsafe_allow_html=True)
    st.write("")

    uni = st.radio("票池", ["mag7", "spy_demo", "diversified", "sp500"], horizontal=True,
                   format_func=lambda u: {"mag7": "七姐妹(样本大)", "spy_demo": "SPY+成分股",
                                          "diversified": "宽池(降选股偏差)", "sp500": "标普500(全)"}.get(u, u))
    horizon = _HORIZON_OPTS[st.selectbox("远期收益周期", list(_HORIZON_OPTS), index=2)]
    run = run_gate("signals", {"uni": uni, "h": horizon}, label="🚀 运行信号扫描",
                   hint="扫描约 20–40 秒：跨票池化 + bootstrap CI + FDR 校正。")
    if run is None:
        return

    from data import loader
    tickers = tuple(loader.load_universe(run["uni"]))
    with st.spinner("扫描信号 + bootstrap + FDR…"):
        scan_df, cs, ew_items, rw_items = c_scan(tickers, "2012-01-01", end, int(run["h"]))

    # 七股今日综合（G6）
    st.markdown("#### 🛰️ 今日综合（建仓分 / 风险分，0–100 历史分位）")
    st.dataframe(cs.style.background_gradient(subset=["建仓分"], cmap="Greens")
                 .background_gradient(subset=["风险分"], cmap="Reds")
                 .format({"建仓分": "{:.0f}", "风险分": "{:.0f}", "现价": "{:.1f}"}),
                 use_container_width=True, hide_index=True)
    st.caption("建仓分高+风险分低 = 历史上相对更值得关注的建仓时机倾斜。**非买卖指令**。")

    # 信号扫描表（人话）+ 超额图（G1/G5）
    st.markdown("#### 📡 各状态下建仓/风险的历史表现")
    from analysis import signal_scan as _ssm
    disp, summary = _ssm.humanize_scan(scan_df)
    st.markdown(f'<div class="verdict">{summary}</div>', unsafe_allow_html=True)
    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.caption("📖 看法：**比闭眼买多/少**=该状态进场比随便买入持有多赚(或少赚)；"
               "**95%可信区间**跨 0=统计上分不清、别当真；**结论** ✅稳健>🟡边缘>⚪不显著；"
               "**进场后典型浮亏**=进场后短期常见最深浮亏，越深越颠簸。")
    st.plotly_chart(ch.signal_excess(scan_df), use_container_width=True, config=ch.CHART_CONFIG)

    # 单股建仓分/风险分时序（G2/G3）
    st.markdown("#### 📈 单股建仓分 / 风险分 时序")
    tk = st.selectbox("标的", list(tickers), index=min(4, len(tickers) - 1))
    sdf = c_score_series(tk, "2012-01-01", end, ew_items, rw_items).dropna(subset=["建仓分", "风险分"])
    if not sdf.empty:
        last = sdf.iloc[-1]
        sc = st.columns(3)
        bcol = "#2BE6A8" if last["建仓分"] >= 60 else "#8A93A6"
        rcol = "#FF5C7A" if last["风险分"] >= 60 else "#8A93A6"
        sc[0].markdown(stat_card("今日建仓分", f"{last['建仓分']:.0f}", "0–100 历史分位", bcol), unsafe_allow_html=True)
        sc[1].markdown(stat_card("今日风险分", f"{last['风险分']:.0f}", "近期下行风险环境", rcol), unsafe_allow_html=True)
        sc[2].markdown(stat_card("现价", f"{last['price']:.1f}", str(sdf.index[-1].date()), "#7C5CFC"), unsafe_allow_html=True)
        st.plotly_chart(ch.score_timeseries(sdf, tk), use_container_width=True, config=ch.CHART_CONFIG)


# ===========================================================================
# 页面：概览
# ===========================================================================
def page_overview():
    st.markdown('<div class="hero-title">量化研究工具</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">面向长期个人投资者的<b>规则校准器</b>——把"建仓/进出场/因子"从感觉变成带置信区间和样本数的经验分布。</div>', unsafe_allow_html=True)
    st.write("")

    c1, c2, c3 = st.columns(3)
    c1.markdown(stat_card("铁律一", "校准而非预测", "永远输出 分布 + 置信区间 + 样本数 N", "#7C5CFC"), unsafe_allow_html=True)
    c2.markdown(stat_card("铁律二", "永远对比基准", "条件结果必与无脑买入持有并排", "#00D4FF"), unsafe_allow_html=True)
    c3.markdown(stat_card("铁律三", "反过拟合优先", "block bootstrap · walk-forward · deflated Sharpe · N_eff", "#2BE6A8"), unsafe_allow_html=True)
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
                             "#00D4FF", tip="regime"), unsafe_allow_html=True)
    pc[1].markdown(stat_card("趋势位置", "均线上方" if tp_panel["trend_state"] == "up_trend" else "均线下方",
                             f"距200日线 {tp_panel['trend_position']:+.1%}", "#2BE6A8", tip="regime"), unsafe_allow_html=True)
    pc[2].markdown(stat_card("回撤状态", "回撤中" if tp_panel["drawdown_state"] == "in_drawdown" else "近高点",
                             f"距前高 {tp_panel['drawdown']:+.1%}", "#FF5C7A", tip="回撤"), unsafe_allow_html=True)
    pc[3].markdown(stat_card("快照日期", str(tp_panel["date"].date()), "免费数据·不可预测未来", "#7C5CFC"), unsafe_allow_html=True)
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
                                    "ci_low": "区间下", "ci_high": "区间上"})
        st.dataframe(show.style.format({"胜率": "{:.0%}", "中位涨幅": "{:+.1%}", "最差10%": "{:+.1%}",
                                        "比基准多": "{:+.1%}", "区间下": "{:+.1%}", "区间上": "{:+.1%}"}),
                     use_container_width=True, hide_index=True)
        st.caption("📖 **比基准多**=该状态进场比随便买入持有多赚多少；**区间**跨 0 即不显著；**最差10%**=坏情形收益。")
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
        color = "#2BE6A8" if abs(row["IC_mean"]) >= 0.03 else "#8A93A6"
        col.markdown(stat_card(f"预测力 {per}", f"{row['IC_mean']:+.3f}",
                               f"风险调整后 {row['IR']:+.2f} · 样本{int(row['n_days'])}天", color, tip="IC"), unsafe_allow_html=True)
    # 白话强度判断
    best = ic["IC_mean"].abs().max()
    if best >= 0.05:
        lvl, col_ = "较强（少见，留意是否前视偏差）", "#FFD166"
    elif best >= 0.03:
        lvl, col_ = "可用（因子本就很弱，靠广度取胜）", "#2BE6A8"
    elif best >= 0.015:
        lvl, col_ = "偏弱（单独用价值有限）", "#8A93A6"
    else:
        lvl, col_ = "≈0（几乎没有预测力）", "#FF5C7A"
    st.markdown(f'<div class="verdict">这个因子对未来收益的<b>预测力：{lvl}</b>（最强周期 IC={best:.3f}）。</div>', unsafe_allow_html=True)
    st.write("")
    st.plotly_chart(ch.factor_ic_bars(ic), use_container_width=True, config=ch.CHART_CONFIG)
    st.caption("📖 看法：**预测力(IC)**=因子值与未来收益的相关系数，**0.03–0.05 就算可用**（因子天生很弱，靠数量多和一致性赚钱）；"
               "**柱子越高越好**，超过虚线(0.03)才算有用；IC 高得离谱(>0.2)反而要怀疑用了未来数据。")
    if factor == "random_factor":
        st.success("✅ 健全性检查：随机因子的预测力应≈0。若明显偏离 0，说明流程有前视偏差(用了未来信息)。")

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
            color = "#2BE6A8" if abs(row["IC_mean"]) >= 0.03 else "#8A93A6"
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
    cols[0].markdown(stat_card("交易笔数 / N_eff", f"{p['n_trades']}", f"N_eff≈{p['n_eff']:.0f} · ρ̄={p['rho_bar']:.2f}", "#7C5CFC", tip="N_eff"), unsafe_allow_html=True)
    cols[1].markdown(stat_card("胜率", f"{p['win_rate']:.0%}", "", "#00D4FF", tip="胜率"), unsafe_allow_html=True)
    cols[2].markdown(stat_card("收益中位", f"{p['median_return']:+.1%}", f"5分位 {p['p5_return']:+.1%}", "#2BE6A8", tip="远期收益"), unsafe_allow_html=True)
    cols[3].markdown(stat_card("MAE 中位", f"{p['median_mae']:+.1%}", f"最长连亏 {p['longest_losing_streak']}", "#FF5C7A", tip="MAE"), unsafe_allow_html=True)
    excol = "#2BE6A8" if p["excess_significant"] else "#8A93A6"
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
                                     "#2BE6A8" if b1["pass"] else "#FF5C7A", tip="超额"), unsafe_allow_html=True)
            b2 = cb["oos_sharpe"]
            gk[1].markdown(stat_card(("✅ " if b2["pass"] else "❌ ") + f"OOS夏普≥{b2['threshold']}",
                                     f"{b2['value']:.2f}" if b2["value"] == b2["value"] else "NA",
                                     f"OOS {b2['n_oos_days']} 日", "#2BE6A8" if b2["pass"] else "#FF5C7A", tip="Sharpe"), unsafe_allow_html=True)
            b3 = cb["drawdown"]
            gk[2].markdown(stat_card(("✅ " if b3["pass"] else "❌ ") + f"最大回撤≤{abs(b3['tolerance']):.0%}",
                                     f"{b3['max_drawdown']:.0%}", "池化日度净值", "#2BE6A8" if b3["pass"] else "#FF5C7A", tip="水下图"), unsafe_allow_html=True)
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
                dc[0].markdown(stat_card("最优候选", deflated["best_name"], f"per-trade Sharpe {deflated['best_sharpe']:.2f}", "#7C5CFC", tip="Sharpe"), unsafe_allow_html=True)
                dc[1].markdown(stat_card("Deflated Sharpe 概率", f"{deflated['deflated_sharpe_prob']:.2f}", f"{deflated['n_trials']} 候选 · {'稳健' if deflated['robust'] else '存疑'}", "#2BE6A8" if deflated["robust"] else "#FF5C7A", tip="deflated Sharpe"), unsafe_allow_html=True)
                s = wf["summary"]
                dc[2].markdown(stat_card("IS vs OOS Sharpe", f"{s['mean_is_sharpe']:.2f} / {s['mean_oos_sharpe']:.2f}", f"过拟合缺口 {s['overfit_gap']:+.2f}", "#00D4FF", tip="walk-forward"), unsafe_allow_html=True)
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
    pc[0].markdown(stat_card("POC（成交最密价位）", f"{vp['poc']:.1f}", "强支撑/压力参考", "#00D4FF", tip="POC"), unsafe_allow_html=True)
    pc[1].markdown(stat_card("价值区(70%)", f"{vp['value_area'][0]:.0f} – {vp['value_area'][1]:.0f}", "成交集中区间", "#7C5CFC", tip="Volume Profile"), unsafe_allow_html=True)
    pc[2].markdown(stat_card("现价", f"{ohlcv['close'].iloc[-1]:.1f}", str(ohlcv.index[-1].date()), "#2BE6A8"), unsafe_allow_html=True)
    st.write("")
    a, b = st.columns([3, 1])
    with a:
        try:
            pls = [{"price": vp["poc"], "color": "#00D4FF", "title": "POC"},
                   {"price": vp["value_area"][0], "color": "#7C5CFC", "title": "VA低"},
                   {"price": vp["value_area"][1], "color": "#7C5CFC", "title": "VA高"}]
            render_tv_candles(ohlcv, None, price_lines=pls, key=f"tv_snap_{tk}", height=480, log=False,
                              caption="🖱️ TradingView 手感：滚轮缩放/拖动/十字光标。横线=POC 与价值区(70%)。")
        except Exception:  # noqa: BLE001
            st.plotly_chart(ch.candle_with_levels(ohlcv, vp, title=f"{tk}　近{lookback}日 K线 + 筹码位"), use_container_width=True, config=ch.CHART_CONFIG)
    b.plotly_chart(ch.volume_profile_bars(vp, title="筹码分布"), use_container_width=True)

    st.divider()
    st.markdown("##### 🎰 期权当前快照（不可回测）")
    if st.button("加载期权链快照", help="实时拉取 yfinance 期权链；仅当前，不可回测"):
        try:
            with st.spinner("拉取期权链…"):
                snap = c_options(tk)
            oc = st.columns(4)
            ivc, ivp = snap["atm_iv_call"], snap["atm_iv_put"]
            oc[0].markdown(stat_card("ATM IV(Call)", f"{ivc:.0%}" if ivc == ivc else "—", "隐含波动率", "#00D4FF", tip="IV"), unsafe_allow_html=True)
            oc[1].markdown(stat_card("ATM IV(Put)", f"{ivp:.0%}" if ivp == ivp else "—", "隐含波动率", "#7C5CFC", tip="IV"), unsafe_allow_html=True)
            oc[2].markdown(stat_card("Put/Call (OI)", f"{snap['put_call_oi_ratio']:.2f}", ">1 偏空对冲", "#FF5C7A"), unsafe_allow_html=True)
            oc[3].markdown(stat_card("最大OI磁吸位", f"P{snap['max_oi_put_strike']:.0f} / C{snap['max_oi_call_strike']:.0f}", "下方支撑/上方压力", "#2BE6A8"), unsafe_allow_html=True)
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
                               "#2BE6A8" if sig else "#8A93A6", tip="PEAD"), unsafe_allow_html=True)
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
            col = "#2BE6A8" if r["type"] == "财报" else "#00D4FF"
            rr = r.get("reaction_5d", float("nan"))
            txt = f"{r['label']}" + (f" {rr:+.0%}" if rr == rr else "")
            mk.append({"time": near.strftime("%Y-%m-%d"), "position": "aboveBar",
                       "color": col, "shape": "circle", "text": txt})
        render_tv_line(pser, markers=mk, key=f"tvev_{tk}", height=440, color="#9aa7bd", log=True)
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


@st.fragment
def page_cockpit():
    st.markdown('<div class="hero-title">🛰️ 建仓作战室</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">最近几个月该在哪建仓/补仓？给的是<b>条件价位带的经验分布</b>'
                '——价位区间 + 盈亏比 + 期望值 + N + CI + 超基准，外加<b>未来客观日程</b>与历史反应。'
                '<b>不给目标价、不给单一概率</b>，校准而非预测。</div>', unsafe_allow_html=True)
    st.warning("⚠️ 价位带是**区间+分布**不是买点；盈亏比/期望值基于历史独立事件，未来非平稳。"
               "事件日程是**日历事实**，不预测财报好坏。本页只校准预期、不构成买卖建议。")

    cc = st.columns([2, 3])
    asset = cc[0].selectbox("标的", _SPY_FIRST, index=0)
    horizon = _HORIZON_OPTS[cc[1].selectbox("持有/校准周期", list(_HORIZON_OPTS), index=2,
                                            help=gl.help_for("远期收益"))]
    zstart = "1995-01-01" if asset == "SPY" else "2008-01-01"

    # 今日状态
    tp = c_today_panel(asset, zstart, end)
    pc = st.columns(4)
    pc[0].markdown(stat_card("现价", f"{tp['drawdown']:.0%} 距前高",
                             f"{tp['date'].date()}", "#FFD166", tip="回撤"), unsafe_allow_html=True)
    pc[1].markdown(stat_card("趋势位置", "均线上方" if tp["trend_state"] == "up_trend" else "均线下方",
                             f"距200日线 {tp['trend_position']:+.1%}", "#2BE6A8", tip="regime"), unsafe_allow_html=True)
    vp = tp["vol_percentile"]
    pc[2].markdown(stat_card("波动状态", {"low_vol": "低波动", "mid_vol": "中波动", "high_vol": "高波动"}.get(tp["vol_state"], "—"),
                             f"年化{tp['realized_vol']:.0%}·分位{vp:.0%}" if vp == vp else "—", "#00D4FF", tip="regime"), unsafe_allow_html=True)
    pc[3].markdown(stat_card("回撤状态", "回撤中" if tp["drawdown_state"] == "in_drawdown" else "近高点",
                             "免费数据·不可预测未来", "#FF5C7A", tip="回撤"), unsafe_allow_html=True)
    st.write("")

    tabZ, tabE, tabL = st.tabs(["💠 条件价位带 + 盈亏比", "🗓️ 未来事件 + 提前消化", "🪜 阶梯式建仓布局"])

    # ---- 块1：条件价位带 ----
    with tabZ:
        hz = _chart_horizon(f"zone_{asset}", horizon)
        with st.spinner("分桶各回撤价位带的历史远期收益分布…"):
            z = c_zones(asset, zstart, end, hz)
        from regime import entry_cockpit as ec
        from data import loader as _ld
        price = _ld.load_prices([asset], zstart, end)[asset]

        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(ch.price_with_zones(price, z), use_container_width=True, config=ch.CHART_CONFIG)
        c2.plotly_chart(ch.entry_zone_bars(z, hz), use_container_width=True)

        # 选一个价位带看校准结论 + 盈亏比/期望值卡片
        enough = z[z["enough"]]
        if not enough.empty:
            cur_idx = z.index[z["is_current"]]
            default_zone = z.loc[cur_idx[0], "zone"] if len(cur_idx) and z.loc[cur_idx[0], "enough"] else enough.iloc[0]["zone"]
            zpick = st.selectbox("查看某价位带的校准结论", enough["zone"].tolist(),
                                 index=enough["zone"].tolist().index(default_zone) if default_zone in enough["zone"].tolist() else 0)
            row = enough[enough["zone"] == zpick].iloc[0]
            st.markdown(f'<div class="verdict">{ec.format_zone_verdict(row, hz)}</div>', unsafe_allow_html=True)
            kc = st.columns(5)
            kc[0].markdown(stat_card("价位带", f"{row['price_low']:.0f}–{row['price_high']:.0f}", row["zone"], "#FFD166"), unsafe_allow_html=True)
            kc[1].markdown(stat_card("远期收益中位", f"{row['median']:+.1%}", f"基准 {row['baseline_median']:+.1%}", "#2BE6A8", tip="远期收益"), unsafe_allow_html=True)
            rr = row["reward_risk"]
            kc[2].markdown(stat_card("盈亏比", f"{rr:.2f}" if rr == rr else "—", "中位收益 / 中位浮亏", "#7C5CFC", tip="盈亏比"), unsafe_allow_html=True)
            ecol = "#2BE6A8" if row["expectancy"] > 0 else "#FF5C7A"
            kc[3].markdown(stat_card("期望值", f"{row['expectancy']:+.1%}", f"胜率 {row['win_rate']:.0%}", ecol, tip="期望值"), unsafe_allow_html=True)
            xcol = "#2BE6A8" if (row["ci_low"] > 0 or row["ci_high"] < 0) else "#8A93A6"
            kc[4].markdown(stat_card("超额(vs基准)", f"{row['excess_median']:+.1%}", f"N≈{int(row['n_events'])}·CI[{row['ci_low']:+.0%},{row['ci_high']:+.0%}]", xcol, tip="超额"), unsafe_allow_html=True)
        st.write("")
        st.markdown("##### 各价位带明细")
        show = z[["zone", "price_low", "price_high", "n_events", "win_rate", "median",
                  "p10", "reward_risk", "expectancy", "excess_median", "ci_low", "ci_high"]].copy()
        show = show.rename(columns={"zone": "价位带", "price_low": "价位低", "price_high": "价位高",
                                    "n_events": "N", "win_rate": "胜率", "median": "中位", "p10": "10分位",
                                    "reward_risk": "盈亏比", "expectancy": "期望值", "excess_median": "超额",
                                    "ci_low": "CI下", "ci_high": "CI上"})
        st.dataframe(show.style.format({"价位低": "{:.0f}", "价位高": "{:.0f}", "胜率": "{:.0%}",
                                        "中位": "{:+.1%}", "10分位": "{:+.1%}", "盈亏比": "{:.2f}",
                                        "期望值": "{:+.1%}", "超额": "{:+.1%}", "CI下": "{:+.1%}", "CI上": "{:+.1%}"},
                                       na_rep="样本不足"), use_container_width=True, hide_index=True)
        st.caption(f"样本期 {z.attrs['sample_start']}~{z.attrs['sample_end']}。{z.attrs['disclaimer']}")

    # ---- 块2：未来事件 + 提前消化 ----
    with tabE:
        if asset == "SPY":
            st.info("SPY 为 ETF，无公司财报事件。个股(七姐妹)才有财报日程与 PEAD 反应分布。")
        else:
            with st.spinner("拉取财报日程 + 历史财报反应分布…"):
                estats, upcoming, study = c_earnings_reaction(asset, zstart, end)
            st.markdown(f"##### 🗓️ 未来客观日程（截至 {upcoming.attrs.get('as_of','')}，日历事实非预测）")
            st.dataframe(upcoming, use_container_width=True, hide_index=True)
            st.write("")
            st.markdown("##### 📊 历史财报反应分布（不预测下次好坏）")
            ec_ = st.columns(4)
            dm = estats["day_abs_move"]
            ec_[0].markdown(stat_card("财报日典型波动", f"±{dm['median']:.1%}", f"90分位 ±{dm['p90']:.1%}·N={dm['n']}", "#FF5C7A"), unsafe_allow_html=True)
            pdft = estats["pre_drift"]
            ec_[1].markdown(stat_card("财报前 drift(提前消化)", f"{pdft['median']:+.1%}", f"{estats['pre']}日·[{pdft['p10']:+.0%},{pdft['p90']:+.0%}]", "#00D4FF", tip="提前消化"), unsafe_allow_html=True)
            pb = estats["post_beat"]
            ec_[2].markdown(stat_card("财报后·超预期", f"{pb['median']:+.1%}", f"{estats['post']}日·N={pb['n']}", "#2BE6A8", tip="PEAD"), unsafe_allow_html=True)
            pm = estats["post_miss"]
            ec_[3].markdown(stat_card("财报后·不及预期", f"{pm['median']:+.1%}", f"{estats['post']}日·N={pm['n']}", "#FF5C7A", tip="PEAD"), unsafe_allow_html=True)
            st.plotly_chart(ch.event_study(study, title=f"{asset} 财报前后累计收益结构"), use_container_width=True, config=ch.CHART_CONFIG)
            st.caption(estats["note"] + " 「财报前 drift」即量化你说的『市场提前消化基本面』——但它是历史分布，非对下次的判断。")

    # ---- 块3：阶梯式建仓布局 ----
    with tabL:
        st.markdown("中长期布局：把预算分成若干档，**距前高每跌一档补一档**，未触发的在窗口末补齐。"
                    "历史滚动回测对比一次性(lump)/定投(DCA)，看哪种**布局**的资本回报分布更优。")
        bands_pick = st.multiselect("补仓触发档（距前高回撤）",
                                    ["5%", "10%", "15%", "20%", "25%", "30%"],
                                    default=["10%", "20%", "30%"])
        bands = tuple(int(b.rstrip("%")) / 100 for b in bands_pick)
        run = run_gate("ladder", {"asset": asset, "bands": bands}, label="🚀 运行阶梯布局回测",
                       hint="选好补仓档后点运行——滚动多窗口 + bootstrap 约 10–20 秒。")
        if run is not None and bands:
            with st.spinner("滚动窗口模拟 阶梯 vs lump vs DCA + bootstrap…"):
                res = c_ladder(run["asset"], "1995-01-01" if run["asset"] == "SPY" else "2008-01-01", end, run["bands"])
            st.markdown(f'<div class="hero-sub">{res["note"]}</div>', unsafe_allow_html=True)
            st.plotly_chart(ch.strategy_compare(res["per_strategy"], title="布局资本回报中位 + 95% CI"), use_container_width=True, config=ch.CHART_CONFIG)
            rows = []
            for k, v in res["vs_lump_sum"].items():
                rows.append({"布局": {"dca": "定投 DCA", "ladder": "阶梯补仓"}.get(k, k),
                             "相对一次性中位差": v["median_diff"], "CI下": v["ci_low"], "CI上": v["ci_high"],
                             "跑赢一次性比例": v["beats_lump_rate"], "显著": "✅" if v["significant"] else "—"})
            st.dataframe(pd.DataFrame(rows).style.format({"相对一次性中位差": "{:+.1%}", "CI下": "{:+.1%}",
                                                          "CI上": "{:+.1%}", "跑赢一次性比例": "{:.0%}"}),
                         use_container_width=True, hide_index=True)
            st.caption("CI 不跨 0 才算显著差异。注意：长期上行市场里一次性投入常胜过分批——阶梯/定投的价值更多在**降低择时风险与浮亏**，不一定提高期望收益。")
        elif run is not None and not bands:
            st.caption("至少选一个补仓档。")


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
        chg = f"{live['change_pct']:+.2%}" if live.get("change_pct") == live.get("change_pct") else "—"
    else:
        px, chg = f"{b['price']:.1f}", "—"
    return {
        "标的": b["ticker"], "现价": px, "今日涨跌": chg,
        "趋势/距前高": f"{b['trend']} / {b['drawdown']:.1%}", "波动分位": volp,
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
        col.markdown(stat_card(tk, f"{w['weight']:.0f}%", w["role"], "#7C5CFC"), unsafe_allow_html=True)
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
            col.markdown(stat_card(f"{b['ticker']} 真实仓位上限", f"≤{pn['cap']:.0%}", sub, "#FFD166"), unsafe_allow_html=True)
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
            col.markdown(stat_card(b["ticker"], f"{w:.0%}", "组合权重", "#2BE6A8"), unsafe_allow_html=True)
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
                _gc = {"A": "#2BE6A8", "B": "#7CFC9E", "C": "#FFD166", "D": "#FF9F45", "F": "#FF5C7A"}.get(g["grade"], "#8A93A6")
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
                                     {"low_vol": "低", "mid_vol": "中", "high_vol": "高"}.get(b["vol_state"], ""), "#00D4FF", tip="regime"), unsafe_allow_html=True)
            if es:
                c = "#2BE6A8" if es["excess"] > 0 else "#FF5C7A"
                kc[1].markdown(stat_card(f"当前状态桶·{b['current_state_bucket']}", f"{es['median']:+.1%}",
                                         f"超额{es['excess']:+.1%}·胜率{es['win_rate']:.0%}{'·显著' if es['significant'] else ''}", c, tip="超额"), unsafe_allow_html=True)
            if ev:
                c = "#2BE6A8" if ev["excess"] > 0 else "#8A93A6"
                kc[2].markdown(stat_card("估值低位桶(买便宜)", f"{ev['median']:+.1%}",
                                         f"超额{ev['excess']:+.1%}·RR{ev['reward_risk']:.2f}{'·显著' if ev['significant'] else ''}", c, tip="超额"), unsafe_allow_html=True)
            if b.get("next_earnings"):
                kc[3].markdown(stat_card("下次财报", b["next_earnings"], f"{b['days_to_earnings']} 天后", "#FFD166"), unsafe_allow_html=True)
            if b.get("momentum_trap"):
                st.markdown('<div class="verdict">⚠️ <b>动量陷阱</b>：当前在回撤中，但回撤桶超额≤0——历史上逢跌买并不优于随机进场。正确做法是等趋势确认/波动回落，而非抢这段回撤。</div>', unsafe_allow_html=True)

            # 建仓档
            if b["tranches"]:
                st.markdown("**建仓档（技术位共振 → 目标/止损/盈亏比）**")
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

    st.markdown(f'<div class="hero-title">📊 {a} · 全景分析</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">选股即自动生成：今日状态 · 该在哪建仓(价位带/建仓档) · 引擎结论 · 财报日程与反应 · 信号。'
                '<b>只校准不预测，给的是历史分布+区间+CI，不是目标价/买卖指令。</b></div>', unsafe_allow_html=True)
    st.caption("💡 阅读提示：卡片标题与表格列名带虚线下划线/可**鼠标悬浮**看专业名词含义；数值照常显示。")
    st.write("")

    with st.spinner(f"正在生成 {a} 全景分析（首次约 10 秒，之后秒开）…"):
        b = c_brief(a, horizon, end)
        z = c_zones(a, zstart, end, horizon)
        from data import loader as _ld
        price = _ld.load_prices([a], zstart, end)[a]

    # ---- 今日速读（醒目横幅）：与下方「操作预案」同源，避免重复结论 ----
    from analysis.playbook import build_playbook
    pbk = build_playbook(b)
    _conv = pbk["conviction"]                       # 如 "🔴 低（动量陷阱）"
    _col = next((v for k, v in {"🔴": "#FF5C7A", "🟢": "#2BE6A8", "🟡": "#FFD166",
                                "⚪": "#8A93A6"}.items() if _conv.startswith(k)), "#8A93A6")
    st.markdown(
        f'<div style="border-radius:16px;padding:18px 22px;margin:2px 0 16px;'
        f'background:linear-gradient(92deg,{_col}26,{_col}08);border:1px solid {_col}55;border-left:7px solid {_col}">'
        f'<div style="font-size:0.78rem;color:#8A93A6;letter-spacing:1px">今日速读 · 把握度 {_conv}</div>'
        f'<div style="font-size:1.25rem;font-weight:800;color:#E6E9EF;line-height:1.45;margin-top:4px">{pbk["headline"]}</div>'
        f'<div style="color:#8A93A6;font-size:0.82rem;margin-top:6px">依据：{pbk["conviction_basis"]}</div>'
        f'<div style="color:#8A93A6;font-size:0.76rem;margin-top:5px">⚠️ <b>历史倾斜的校准</b>，不是预测、不是买卖指令；详细操作见下方「📋 操作预案」。</div>'
        f'</div>', unsafe_allow_html=True)

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
            _gc = {"A": "#2BE6A8", "B": "#7CFC9E", "C": "#FFD166", "D": "#FF9F45", "F": "#FF5C7A"}.get(_g["grade"], "#8A93A6")
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
    st.write("")

    # ---- 🛰️ 事件雷达（全网自动抓取，display-only 风险提醒，绝不入量化）----
    import datetime as _dtm
    from analysis import event_radar as _er
    _today = _dtm.date.today()
    with st.spinner("事件雷达：全网查询 IPO/经济日历…"):
        radar = c_event_radar(a, _today.isoformat(), b.get("next_earnings"), 45)
    _src_cn = {"auto": "规则", "web": "全网", "manual": "手填"}
    _sevcol = {"高": "#FF5C7A", "中": "#FFD166", "低": "#8A93A6"}
    _rc = "#FF5C7A" if radar["n_high"] else "#7C5CFC"
    chips = "".join(
        f'<span style="display:inline-block;margin:3px 6px 3px 0;padding:3px 9px;border-radius:8px;'
        f'background:{_sevcol.get(e["severity"],"#8A93A6")}22;border:1px solid {_sevcol.get(e["severity"],"#8A93A6")}66;'
        f'font-size:0.8rem;color:#E6E9EF">{e["date"].strftime("%m-%d")}·{e["days_ahead"]}天后 '
        f'{e["category"]}{("·"+e["title"]) if e.get("title") else ""}</span>'
        for e in radar["events"][:8])
    st.markdown(
        f'<div style="border-radius:14px;padding:13px 18px;margin:2px 0 6px;'
        f'background:linear-gradient(92deg,{_rc}1f,{_rc}08);border:1px solid {_rc}44;border-left:6px solid {_rc}">'
        f'<div style="font-size:0.8rem;color:#8A93A6;letter-spacing:0.5px">🛰️ 事件雷达 · 未来45天 '
        f'（{radar["n_high"]} 项高风险 · {radar.get("n_web",0)} 项全网自动抓取）· <b>仅提醒、不入量化</b></div>'
        f'<div style="margin-top:7px">{chips or "<span style=\'color:#8A93A6\'>无登记事件</span>"}</div>'
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

    # ---- 今日状态 ----
    st.markdown("#### 📍 今日状态")
    hc = st.columns(5)
    # 现价：盘中近实时(yfinance≈15min延迟)，开盘时每 ~30 秒自动刷新；仅展示、不入量化
    _mkt_open = is_market_open()

    @st.fragment(run_every=("30s" if _mkt_open else None))
    def _live_price(ticker=a, fallback=b):
        import datetime as _d
        bucket = int(_d.datetime.now().timestamp() // 30)
        q = c_live_quote(ticker, bucket)
        price = q["price"] if (q.get("ok") and q["price"] == q["price"]) else fallback["price"]
        chg = q.get("change_pct")
        if q.get("ok") and chg == chg:
            col = "#2BE6A8" if chg >= 0 else "#FF5C7A"
            sub = f"{q['change']:+.2f} ({chg:+.2%}) · {'🟢盘中' if _mkt_open else '⚪休市'}"
        else:
            col = "#FFD166"; sub = fallback["date"]
        st.markdown(stat_card("现价", f"{price:.2f}", sub, col), unsafe_allow_html=True)
        if q.get("ok") and q.get("delayed"):
            st.caption("≈15min延迟·不入量化")

    with hc[0]:
        _live_price()
    hc[1].markdown(stat_card("距前高", f"{b['drawdown']:+.0%}", "回撤深度", "#FF5C7A", tip="回撤"), unsafe_allow_html=True)
    hc[2].markdown(stat_card("趋势", "均线上方" if b["trend"] == "up_trend" else "均线下方",
                             f"距200线 {b['trend_position']:+.0%}", "#2BE6A8", tip="regime"), unsafe_allow_html=True)
    vp_ = b["vol_percentile"]
    hc[3].markdown(stat_card("波动状态", {"low_vol": "低", "mid_vol": "中", "high_vol": "高"}.get(b["vol_state"], "—") + "波动",
                             f"分位 {vp_:.0%}" if vp_ == vp_ else "—", "#00D4FF", tip="regime"), unsafe_allow_html=True)
    if b.get("next_earnings"):
        hc[4].markdown(stat_card("下次财报", b["next_earnings"], f"{b['days_to_earnings']} 天后", "#7C5CFC"), unsafe_allow_html=True)
    st.write("")

    # ---- 该在哪建仓：TradingView K线 + 价位带横线 + 建仓档 ----
    st.markdown("#### 🎯 该在什么价位/状态建仓")
    # per-chart 持有期：本段及下方价位带/状态扫描/评分时序统一按此持有期校准（默认=侧边栏分析周期）
    horizon = _chart_horizon(f"pan_{a}", horizon, label="建仓校准持有期")
    z = c_zones(a, zstart, end, horizon)
    cL, cR = st.columns([3, 2])
    with cL:
        from frontend import tvchart
        zlines = []
        zz = z[z["enough"]] if "enough" in z.columns else z
        for _, r in zz.iterrows():
            iscur = bool(r.get("is_current", False))
            zlines.append({"price": float(r["price_high"]),
                           "color": "#00D4FF" if iscur else "rgba(138,147,166,0.6)",
                           "title": ("▶ " if iscur else "") + str(r["zone"])})
        try:
            ohlcv_pan = _ld.load_ohlcv(a, zstart, end)
            render_tv_candles(ohlcv_pan, None, price_lines=zlines, key=f"tvpan_{a}", height=460, log=True,
                              caption="🖱️ TradingView 操作：滚轮缩放 · 拖动平移 · 十字光标读价。横线=各回撤价位带上沿(▶=当前所处带)。")
        except Exception:  # noqa: BLE001
            st.plotly_chart(ch.price_with_zones(price, z), use_container_width=True, config=ch.CHART_CONFIG)
    cR.plotly_chart(ch.entry_zone_bars(z, horizon), use_container_width=True)
    from regime import entry_cockpit as ec
    cur = z[z["is_current"] & z["enough"]] if "is_current" in z.columns else z.iloc[0:0]
    if not cur.empty:
        st.markdown(f'<div class="verdict">当前价位带：{ec.format_zone_verdict(cur.iloc[0], horizon)}</div>', unsafe_allow_html=True)
    if b.get("tranches"):
        st.markdown("**建仓档（技术位共振 → 盈亏比；非预测点位）**　💡 列名可悬浮看含义")
        _tdf = pd.DataFrame([{
            "档": t["tier"], "价位": f"{t['price']:.1f}", "依据": t["what"],
            "回前高目标": f"{t['target']:.0f} ({t['to_target_pct']:+.0%})",
            "技术止损": f"{t['stop']:.0f} ({t['to_stop_pct']:+.0%})",
            "盈亏比": f"{t['rr']:.2f}" if t["rr"] == t["rr"] else "—",
            "引擎胜率": f"{t['engine_win_rate']:.0%}" if t["engine_win_rate"] == t["engine_win_rate"] else "—",
        } for t in b["tranches"]])
        st.dataframe(_tdf, use_container_width=True, hide_index=True, column_config=_col_cfg(_tdf.columns))
        st.caption("📖 「回前高目标」=前期高点(约52周高)，只用来和技术止损算盈亏比，**非预测目标价**。")
    st.write("")

    # ---- 📋 操作预案（if-then 条件指导；把握度见顶部速读，此处不重复结论）----
    st.markdown("#### 📋 操作预案（怎么建仓 · 涨了跌了怎么办）")

    def _pcard(col, title, accent, items):
        body = "".join(
            f'<div style="font-size:.85rem;line-height:1.55;color:#D2D8E3;margin:5px 0;'
            f'padding-left:12px;border-left:2px solid {accent}55">{x}</div>'
            for x in (items or ["—"]))
        col.markdown(
            f'<div class="glass" style="min-height:172px;padding:14px 16px">'
            f'<div style="font-weight:700;color:{accent};font-size:.95rem;margin-bottom:6px">{title}</div>'
            f'{body}</div>', unsafe_allow_html=True)

    r1 = st.columns(3)
    _pcard(r1[0], "🎯 建仓", "#FFD166", pbk.get("entry"))
    _pcard(r1[1], "📈 涨了怎么操作", "#2BE6A8", pbk.get("if_up"))
    _pcard(r1[2], "📉 跌了怎么操作", "#FF5C7A", pbk.get("if_down"))
    st.write("")
    r2 = st.columns(2)
    _pcard(r2[0], "⏱️ 时间 / 事件", "#00D4FF", pbk.get("time_event"))
    _pcard(r2[1], "🛡️ 风控", "#7C5CFC", pbk.get("risk"))
    st.caption("⚠️ 机械 if-then 预案：价位是「**若到达就行动**」的区间(非预测)，**非买卖指令**；动量陷阱/未过闸门时自动转防守口径。")
    st.write("")

    # ---- 多引擎结论 ----
    st.markdown("#### 🧠 多引擎校准（不同视角的历史倾斜）　💡 标题/数值可悬浮看含义")
    ec_cols = st.columns(3)
    for col, key, label in [(ec_cols[0], "engine_state", "当前状态桶"),
                            (ec_cols[1], "engine_value", "估值低位桶(买便宜)"),
                            (ec_cols[2], "engine_best", "最优倾斜桶")]:
        e = b.get(key)
        if e:
            c = "#2BE6A8" if e["excess"] > 0 else "#8A93A6"
            sub = f"超额{e['excess']:+.1%}·胜率{e['win_rate']:.0%}{'·显著' if e.get('significant') else ''}"
            col.markdown(stat_card(f"{label}", f"{e['median']:+.1%}", sub, c, tip="状态桶"), unsafe_allow_html=True)
    st.write("")

    # ---- 财报反应 ----
    esr = b.get("earnings_stats")
    if esr and esr.get("n_events"):
        st.markdown("#### 📅 财报反应（历史）")
        st.caption(f"历史 {esr['n_events']} 次：财报日典型波动 ±{esr['day_abs_move']['median']:.1%}；"
                   f"财报前 {esr['pre']} 日 drift 中位 {esr['pre_drift']['median']:+.1%}（市场提前消化）；"
                   f"财报后 {esr['post']} 日 超预期 {esr['post_beat']['median']:+.1%} / 不及 {esr['post_miss']['median']:+.1%}。")
        up = b.get("upcoming")
        if up is not None and len(up):
            st.dataframe(up, use_container_width=True, hide_index=True, column_config=_col_cfg(up.columns))
        # PEAD：当前是否处在"已验证的财报后漂移窗口"
        try:
            pd_now = c_pead(a, "2010-01-01", end)
        except Exception:  # noqa: BLE001
            pd_now = None
        if pd_now and pd_now.get("actionable"):
            st.markdown(f'<div class="verdict">📈 <b>财报后漂移(PEAD · 工具唯一通过安慰剂检验的免费信号)</b><br>'
                        f'{pd_now["verdict"]}<br><span style="color:#8A93A6;font-size:0.82rem">{pd_now["note"]}</span></div>',
                        unsafe_allow_html=True)
        elif pd_now is not None:
            st.caption("📈 PEAD：当前不在财报后漂移窗口内（或该票同类财报样本不足）。")

    # ---- 收起的扩展（吸收原"建仓作战室/信号挖掘"的单股模块，三页合一）----
    from regime import entry_cockpit as ec

    with st.expander("🔬 历史相似案例（当前状态在历史上的真实实例 · 可核对）"):
        ana = c_analogs(a, "2008-01-01" if a != "SPY" else "1995-01-01", end, int(horizon))
        st.markdown(f'<div class="verdict">{ana["summary"]}</div>', unsafe_allow_html=True)
        cs = ana["cases"]
        if cs is not None and len(cs):
            disp = cs.copy()
            disp["date"] = disp["date"].dt.date.astype(str)
            disp = disp.rename(columns={"date": "历史日期", "price": "当时价",
                                        "fwd_return": f"往后{horizon}日实现", "max_drawdown": "途中最大浮亏"})
            st.dataframe(disp.style.format({"当时价": "{:.1f}", f"往后{horizon}日实现": "{:+.1%}", "途中最大浮亏": "{:+.1%}"}),
                         use_container_width=True, hide_index=True)
            st.caption("📖 这些是构成上方引擎分布的**真实历史日期**；是样本陈列，不代表'现在更像哪一次'。")

    with st.expander("🎯 校准追踪（记录此刻信号 · 事后比对'说的 vs 做到的'）"):
        from analysis import journal as jn
        st.caption("把当前状态/引擎预期落库，待 horizon 走完后用真实价格回填，检验工具到底准不准（自验证闭环）。")
        if st.button("📝 记录当前信号到校准库", key=f"logsig_{a}"):
            ok = jn.log_from_brief(b)
            st.success("已记录。" if ok else "今天该标的/周期已记录过（去重）。")
        try:
            sig_df = jn.load_signals()
            ev = jn.evaluate(sig_df) if len(sig_df) else sig_df
            cal = jn.calibration_summary(ev) if len(sig_df) else {"n_total": 0, "n_matured": 0}
            cc = st.columns(4)
            cc[0].markdown(stat_card("累计信号", f"{cal.get('n_total',0)}", "已记录", "#7C5CFC"), unsafe_allow_html=True)
            cc[1].markdown(stat_card("已成熟", f"{cal.get('n_matured',0)}", "走完horizon可评", "#00D4FF"), unsafe_allow_html=True)
            if cal.get("n_matured"):
                rh, pw = cal.get("realized_hit", float("nan")), cal.get("pred_win_rate_mean", float("nan"))
                cc[2].markdown(stat_card("实现命中率", f"{rh:.0%}", f"引擎预测 {pw:.0%}",
                                         "#2BE6A8" if (rh == rh and pw == pw and abs(rh - pw) < 0.12) else "#FFD166"), unsafe_allow_html=True)
                cc[3].markdown(stat_card("实现超额(均)", f"{cal.get('realized_excess_mean',float('nan')):+.1%}", f"Brier {cal.get('brier',float('nan')):.2f}",
                                         "#2BE6A8" if cal.get("realized_excess_mean", 0) > 0 else "#FF5C7A"), unsafe_allow_html=True)
                if cal.get("by_grade"):
                    st.dataframe(pd.DataFrame(cal["by_grade"]).rename(columns={
                        "grade": "证据等级", "n": "样本", "realized_hit": "实现命中率", "realized_excess": "实现超额"})
                        .style.format({"实现命中率": "{:.0%}", "实现超额": "{:+.1%}"}, na_rep="—"),
                        use_container_width=True, hide_index=True)
                st.caption(f"📖 {cal.get('note','')}")
            else:
                st.caption("尚无成熟信号（需等 horizon 走完）。先点上面按钮持续记录，工具会越用越知道自己准不准。")
        except Exception as _e:  # noqa: BLE001
            st.caption(f"校准库读取失败：{_e}")

    with st.expander("🩺 数据质量体检（新鲜度 / 缺口 / 异常跳空）"):
        peers_dq = tuple(dict.fromkeys([a] + [t for t in _TICKER_GROUPS[grp] if t != "SPY"]))[:12]
        dh = c_data_health(peers_dq, "2015-01-01", end)
        st.markdown(f'<div class="verdict">{dh["summary"]}</div>', unsafe_allow_html=True)
        st.dataframe(dh["table"], use_container_width=True, hide_index=True)
        st.caption("📖 免费数据(yfinance)可能停更/缺口/未除权跳空——陈旧🔴或带⚠️的标的，其分析结论要打折看。仅体检、不改数据。")

    with st.expander("🌐 全局多重检验账本（扣除挖掘后，还剩几个真显著）"):
        from analysis import mt_ledger as _mt
        rep = _mt.fdr_report(alpha=0.10)
        if rep["n_tests"] == 0:
            st.caption("账本为空。跑「📅 财报 PEAD」或本页「🏁 横截面相对排名」后，其 p 值会自动累计到这里做全局 FDR。")
        else:
            mc = st.columns(3)
            mc[0].markdown(stat_card("累计检验数", f"{rep['n_tests']}", "跨页跨标的", "#7C5CFC", tip="数据窥探"), unsafe_allow_html=True)
            mc[1].markdown(stat_card("未校正显著", f"{rep['n_sig_raw']}", "p<0.05", "#8A93A6"), unsafe_allow_html=True)
            mc[2].markdown(stat_card("BH-FDR 后存活", f"{rep['n_sig_bh']}", f"阈值 p*≤{rep['p_star']:.3f}",
                                     "#2BE6A8" if rep["n_sig_bh"] > 0 else "#FF5C7A", tip="FDR"), unsafe_allow_html=True)
            st.caption(f"📖 {rep['note']}")
            tb = rep["table"][["family", "name", "p_value", "显著_未校正", "显著_BH"]].rename(
                columns={"family": "类别", "name": "检验", "p_value": "p值", "显著_未校正": "未校正", "显著_BH": "BH存活"})
            st.dataframe(tb.style.format({"p值": "{:.4f}"}), use_container_width=True, hide_index=True)

    with st.expander("💠 各价位带明细（盈亏比 / 期望值 / 超额）"):
        enough = z[z["enough"]] if "enough" in z.columns else z.iloc[0:0]
        if not enough.empty:
            zsel = st.selectbox("看某价位带的校准结论", enough["zone"].tolist(), key=f"zsel_{a}")
            row = enough[enough["zone"] == zsel].iloc[0]
            st.markdown(f'<div class="verdict">{ec.format_zone_verdict(row, horizon)}</div>', unsafe_allow_html=True)
            zc = st.columns(4)
            rr = row["reward_risk"]
            zc[0].markdown(stat_card("远期收益中位", f"{row['median']:+.1%}", f"基准 {row['baseline_median']:+.1%}", "#2BE6A8", tip="远期收益"), unsafe_allow_html=True)
            zc[1].markdown(stat_card("盈亏比", f"{rr:.2f}" if rr == rr else "—", "赚的÷要忍的浮亏", "#7C5CFC", tip="盈亏比"), unsafe_allow_html=True)
            zc[2].markdown(stat_card("期望值", f"{row['expectancy']:+.1%}", f"胜率 {row['win_rate']:.0%}", "#2BE6A8" if row["expectancy"] > 0 else "#FF5C7A", tip="期望值"), unsafe_allow_html=True)
            xcol = "#2BE6A8" if (row["ci_low"] > 0 or row["ci_high"] < 0) else "#8A93A6"
            zc[3].markdown(stat_card("比基准多", f"{row['excess_median']:+.1%}", f"N≈{int(row['n_events'])}·区间[{row['ci_low']:+.0%},{row['ci_high']:+.0%}]", xcol, tip="超额"), unsafe_allow_html=True)
        show = z[["zone", "price_low", "price_high", "n_events", "win_rate", "median", "reward_risk", "expectancy", "excess_median"]].rename(
            columns={"zone": "价位带(回撤)", "price_low": "价位低", "price_high": "价位高", "n_events": "历史次数",
                     "win_rate": "胜率", "median": "中位涨幅", "reward_risk": "盈亏比", "expectancy": "期望值", "excess_median": "比基准多"})
        st.dataframe(show.style.format({"价位低": "{:.0f}", "价位高": "{:.0f}", "胜率": "{:.0%}", "中位涨幅": "{:+.1%}",
                                        "盈亏比": "{:.2f}", "期望值": "{:+.1%}", "比基准多": "{:+.1%}"}, na_rep="样本不足"),
                     use_container_width=True, hide_index=True, column_config=_col_cfg(show.columns))
        st.caption("📖 列名可悬浮看含义。每个回撤区间对应一段价位，给该状态历史远期收益分布；区间+分布，不是目标价。")

    with st.expander("🪜 建仓方案模拟器：一次性 vs 定投 vs 越跌越补（该怎么把钱投进去）"):
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

    with st.expander("🔎 各状态下建仓/风险的历史表现（哪种状态进场更好）"):
        from analysis import signal_scan as _ssm
        sc_df, _cs, ew_items, rw_items = c_scan((a,), "2012-01-01", end, int(horizon))
        disp, summary = _ssm.humanize_scan(sc_df)
        st.markdown(f'<div class="verdict">{summary}</div>', unsafe_allow_html=True)
        st.dataframe(disp, use_container_width=True, hide_index=True, column_config=_col_cfg(disp.columns))
        st.caption("📖 列名可鼠标悬浮看含义（比闭眼买多/少 · 95%可信区间 · 结论 · 进场后典型浮亏）。")
        sdf = c_score_series(a, "2012-01-01", end, ew_items, rw_items).dropna(subset=["建仓分", "风险分"])
        if not sdf.empty:
            st.markdown("**建仓分 / 风险分 走势（0–100 历史分位，非预测）**")
            st.plotly_chart(ch.score_timeseries(sdf, a), use_container_width=True, config=ch.CHART_CONFIG)

    with st.expander("📊 策略 vs 持有 · 年化 / 夏普 / 回撤对照"):
        pv = c_perf(a, "2014-01-01", end)
        s_, h_ = pv["strategy"], pv["hold"]
        pc = st.columns(4)
        dlt = s_["cagr"] - h_["cagr"]
        pc[0].markdown(stat_card("策略年化", f"{s_['cagr']:+.0%}", f"持有 {h_['cagr']:+.0%}（差 {dlt:+.0%}）",
                                 "#2BE6A8" if dlt >= 0 else "#FFD166"), unsafe_allow_html=True)
        pc[1].markdown(stat_card("策略夏普", f"{s_['sharpe']:.2f}", f"持有 {h_['sharpe']:.2f}",
                                 "#2BE6A8" if s_["sharpe"] >= h_["sharpe"] else "#8A93A6", tip="Sharpe"), unsafe_allow_html=True)
        pc[2].markdown(stat_card("策略最大回撤", f"{s_['maxdd']:.0%}", f"持有 {h_['maxdd']:.0%}",
                                 "#2BE6A8" if s_["maxdd"] >= h_["maxdd"] else "#FF5C7A", tip="回撤"), unsafe_allow_html=True)
        im = s_.get("in_market", float("nan"))
        pc[3].markdown(stat_card("在场时间", f"{im:.0%}" if im == im else "—", "其余时间持现金", "#7C5CFC"), unsafe_allow_html=True)
        st.plotly_chart(ch.equity_compare(pv["equity"]), use_container_width=True, config=ch.CHART_CONFIG)
        verdict = ("策略年化更高" if dlt > 0.01 else ("两者相当" if abs(dlt) <= 0.01 else "策略年化更低(空仓拖累)"))
        risk = "回撤更小、拿得更稳" if s_["maxdd"] > h_["maxdd"] else "回撤未改善"
        st.caption(f"📖 样本 {pv['sample']}（含费用）。本策略=深跌买入×让利润奔跑，仅约 {im:.0%} 时间在场。"
                   f"结论：{verdict}，但通常**{risk}**。这是后视镜回测(选股偏差)，**非预测**；我们已验证它**未显著跑赢持有**。")

        # α/β 分解：到底有没有真超额，还是纯 beta
        st.markdown("**🔬 Alpha / Beta 分解（收益里有多少是真本事、多少是市场给的）**")
        ab = c_alpha_beta(a, "2014-01-01", end)
        sab, hab = ab["strategy"], ab["hold"]
        abc = st.columns(3)
        _ac = "#2BE6A8" if sab["alpha_significant"] and sab["alpha_ann"] > 0 else "#FFD166"
        abc[0].markdown(stat_card("策略年化α", f"{sab['alpha_ann']:+.1%}",
                                  f"CI[{sab['alpha_ann_ci'][0]:+.0%},{sab['alpha_ann_ci'][1]:+.0%}]"
                                  + ("·显著" if sab["alpha_significant"] else "·跨0"), _ac), unsafe_allow_html=True)
        abc[1].markdown(stat_card("策略β", f"{sab['beta']:.2f}", f"R²={sab['r2']:.0%}", "#00D4FF"), unsafe_allow_html=True)
        abc[2].markdown(stat_card("持有年化α", f"{hab['alpha_ann']:+.1%}",
                                  "显著" if hab["alpha_significant"] else "跨0(纯beta)", "#8A93A6"), unsafe_allow_html=True)
        st.caption(f"📖 把每日收益对 SPY 回归：α=扣掉市场后的真超额、β=市场敏感度。**策略**：{sab['verdict']}")

        # 多因子归因（市场 + 动量/价值/小盘/低波 风格）
        fa = c_factor_attr(a, "2014-01-01", end)
        if fa.get("n", 0) >= 120:
            st.markdown("**🧬 多因子归因（收益拆成 市场β + 风格倾斜 + 真α）**")
            bt = fa["betas"]
            fac_order = [k for k in ("市场", "动量", "价值", "小盘", "低波") if k in bt]
            fc = st.columns(len(fac_order) + 1)
            _ac = "#2BE6A8" if fa["alpha_significant"] and fa["alpha_ann"] > 0 else "#FFD166"
            fc[0].markdown(stat_card("年化α(扣风格)", f"{fa['alpha_ann']:+.1%}",
                                     ("显著" if fa["alpha_significant"] else "跨0") + f"·R²{fa['r2']:.0%}", _ac), unsafe_allow_html=True)
            for col, k in zip(fc[1:], fac_order):
                col.markdown(stat_card(f"{k}β", f"{bt[k]:+.2f}", "暴露", "#00D4FF" if k == "市场" else "#7C5CFC"), unsafe_allow_html=True)
            st.caption(f"📖 {fa['verdict']}（风格用免费 ETF 代理：MTUM/IWD/IWM/USMV 相对 SPY 的超额；多 2013+ 上市）")
        else:
            st.caption(f"🧬 多因子归因：{fa.get('verdict','样本不足')}")

        # Walk-forward OOS：edge 是否跨期稳定（过拟合自检）
        wf = c_walkforward(a, "2010-01-01", end)
        if wf.get("n_windows"):
            st.markdown("**🧪 Walk-forward 样本外自检（参数固定 → 看优势是否只来自某一段）**")
            wfc = st.columns(3)
            wfc[0].markdown(stat_card("OOS 跑赢持有比例", f"{wf['beat_rate']:.0%}",
                                      f"{wf['n_windows']} 段样本外", "#2BE6A8" if wf["beat_rate"] > 0.5 else "#FFD166"), unsafe_allow_html=True)
            wfc[1].markdown(stat_card("OOS 策略年化(中位)", f"{wf['strat_cagr_median']:+.0%}", "各样本外窗口", "#7C5CFC"), unsafe_allow_html=True)
            wfc[2].markdown(stat_card("OOS 持有年化(中位)", f"{wf['hold_cagr_median']:+.0%}", "同期对照", "#8A93A6"), unsafe_allow_html=True)
            st.caption(f"📖 {wf['note']}")

    with st.expander("🛡️ Regime 风险加权暴露（高波动/避险环境自动降仓 · 改善回撤）"):
        ro = c_regime_overlay(a, "2014-01-01", end)
        ex, ov = ro["exposure"], ro["overlay"]
        st.markdown(stat_card("今日建议暴露", f"{ex['exposure']:.0%}", "满仓的百分比(只降不加杠杆)",
                              "#2BE6A8" if ex["exposure"] >= 0.8 else ("#FFD166" if ex["exposure"] >= 0.5 else "#FF5C7A")),
                    unsafe_allow_html=True)
        if ex["factors"]:
            fdf = pd.DataFrame([{"状态因子": f["name"], "当前": f["state"], "暴露乘子": f"×{f['mult']:.2f}"} for f in ex["factors"]])
            st.dataframe(fdf, use_container_width=True, hide_index=True)
        st.caption(f"📖 {ex['note']}")
        st.markdown("**波动目标 overlay vs 闭眼持有（年化 / 夏普 / 回撤 · 只降不加杠杆）**")
        oc = st.columns(4)
        o_, hh_ = ov["overlay"], ov["hold"]
        oc[0].markdown(stat_card("overlay 年化", f"{o_['cagr']:+.0%}", f"持有 {hh_['cagr']:+.0%}", "#7C5CFC"), unsafe_allow_html=True)
        oc[1].markdown(stat_card("overlay 夏普", f"{o_['sharpe']:.2f}", f"持有 {hh_['sharpe']:.2f}",
                                 "#2BE6A8" if o_["sharpe"] >= hh_["sharpe"] else "#8A93A6", tip="Sharpe"), unsafe_allow_html=True)
        oc[2].markdown(stat_card("overlay 最大回撤", f"{o_['maxdd']:.0%}", f"持有 {hh_['maxdd']:.0%}",
                                 "#2BE6A8" if o_["maxdd"] >= hh_["maxdd"] else "#FF5C7A", tip="回撤"), unsafe_allow_html=True)
        oc[3].markdown(stat_card("平均暴露", f"{ov['avg_exposure']:.0%}", f"目标波动 {ov['target_vol']:.0%}", "#00D4FF"), unsafe_allow_html=True)
        st.plotly_chart(ch.equity_compare(ov["equity"], title="波动目标 overlay vs 持有"), use_container_width=True, config=ch.CHART_CONFIG)
        st.caption("📖 波动目标=按近期波动反比缩放仓位(平静加、动荡减，上限不加杠杆)。常以**更低回撤换更稳夏普**，"
                   "年化可能略低——这是风控 edge，不是择时预测。")

    with st.expander("🏁 横截面相对排名 edge（同组谁更强 · 动量+低波多空 · deflated Sharpe）"):
        peers_cs = tuple(t for t in _TICKER_GROUPS[grp] if t != "SPY")[:12]
        if len(peers_cs) >= 5:
            if st.button("运行横截面回测", key=f"runcs_{a}"):
                with st.spinner("横截面多空分位回测 + 多重检验折扣…"):
                    cse = c_cross_section(peers_cs, "2015-01-01", end)
                csc = st.columns(3)
                csc[0].markdown(stat_card("多空夏普", f"{cse['sharpe']:.2f}",
                                          f"CI[{cse['sharpe_ci_low']:.2f},{cse['sharpe_ci_high']:.2f}]",
                                          "#2BE6A8" if (cse['sharpe_ci_low'] > 0) else "#8A93A6", tip="Sharpe"), unsafe_allow_html=True)
                csc[1].markdown(stat_card("年化收益", f"{cse['ann_return']:+.0%}", "多顶分位/空底分位", "#7C5CFC"), unsafe_allow_html=True)
                csc[2].markdown(stat_card("deflated 概率", f"{cse['deflated_sharpe_prob']:.0%}",
                                          "稳健" if cse["robust"] else "未达0.95", "#2BE6A8" if cse["robust"] else "#FFD166", tip="deflated Sharpe"), unsafe_allow_html=True)
                # IC + Newey-West t（自相关稳健显著性）
                nwt = cse.get("ic_t_newey_west", float("nan"))
                ic_m = cse.get("ic_mean", float("nan"))
                ncs = st.columns(2)
                ncs[0].markdown(stat_card("因子 IC(均)", f"{ic_m:.3f}", f"{cse.get('ic_n_periods',0)} 期·已beta中性化", "#7C5CFC", tip="IC"), unsafe_allow_html=True)
                ncs[1].markdown(stat_card("IC 的 |t| (Newey-West)", f"{abs(nwt):.2f}" if nwt == nwt else "—",
                                          "显著(>2)" if (nwt == nwt and abs(nwt) > 2) else "不显著", "#2BE6A8" if (nwt == nwt and abs(nwt) > 2) else "#FF5C7A", tip="IR"), unsafe_allow_html=True)
                # Purged & Embargo CV：无标签泄漏的 OOS IC（比 walk-forward 更严）
                pcv_r = c_purged_cv(peers_cs, "2015-01-01", end)
                if pcv_r.get("n_folds", 0) >= 2:
                    pcc = st.columns(2)
                    pcc[0].markdown(stat_card("Purged-CV OOS IC", f"{pcv_r['mean_oos_ic']:.3f}",
                                              f"{pcv_r['n_folds']} 折·无标签泄漏", "#7C5CFC", tip="IC"), unsafe_allow_html=True)
                    pt = pcv_r.get("t_across_folds", float("nan"))
                    pcc[1].markdown(stat_card("跨折 |t|", f"{abs(pt):.2f}" if pt == pt else "—",
                                              "稳健(>2)" if (pt == pt and abs(pt) > 2) else "不稳健",
                                              "#2BE6A8" if (pt == pt and abs(pt) > 2) else "#FF5C7A"), unsafe_allow_html=True)
                    st.caption(f"📖 {pcv_r['note']}")
                # 记入全局多重检验账本（IC 的 NW p 值）
                try:
                    from analysis import mt_ledger as _mt
                    from scipy.stats import norm as _norm
                    if nwt == nwt:
                        _p = float(2 * (1 - _norm.cdf(abs(nwt))))
                        _mt.log_test("横截面因子", f"{grp}·动量低波", _p, stat=nwt)
                except Exception:  # noqa: BLE001
                    pass
                st.caption(f"📖 {cse['edge_note']}（标的：{', '.join(peers_cs)}）")
                # 信号衰减监控
                dec = c_signal_decay(peers_cs, "2015-01-01", end)
                if dec.get("recent_ic") == dec.get("recent_ic"):
                    dc = st.columns(3)
                    dc[0].markdown(stat_card("前半段 IC", f"{dec['early_ic']:.3f}", "动量因子", "#8A93A6", tip="IC"), unsafe_allow_html=True)
                    dc[1].markdown(stat_card("近半段 IC", f"{dec['recent_ic']:.3f}", "衰减?" + ("是⚠️" if dec["decayed"] else "否"),
                                             "#FF5C7A" if dec["decayed"] else "#2BE6A8", tip="IC"), unsafe_allow_html=True)
                    yic = dec.get("ic_yearly")
                    if yic is not None and len(yic):
                        dc[2].markdown(stat_card("最新年度 IC", f"{yic.iloc[-1]:.3f}", f"{yic.index[-1]}", "#7C5CFC", tip="IC"), unsafe_allow_html=True)
                    st.caption(f"📖 {dec['note']}")
        else:
            st.caption("该组标的不足 5 只，无法做横截面相对排名。")

    with st.expander("📦 筹码分布（Volume Profile / POC）"):
        ohlcv, vpf = c_volume_profile(a, "2015-01-01", end, 252)
        st.plotly_chart(ch.volume_profile_bars(vpf, title=f"{a} 筹码分布"), use_container_width=True, config=ch.CHART_CONFIG)

    with st.expander(f"🛰️ 同板块今日对比（{grp} · 建仓分 / 风险分排名）"):
        peers = tuple(t for t in _TICKER_GROUPS[grp] if t != "SPY")[:8]
        if len(peers) >= 2:
            _df, cs7, _ew, _rw = c_scan(peers, "2012-01-01", end, int(horizon))
            st.dataframe(cs7.style.background_gradient(subset=["建仓分"], cmap="Greens")
                         .background_gradient(subset=["风险分"], cmap="Reds")
                         .format({"建仓分": "{:.0f}", "风险分": "{:.0f}", "现价": "{:.1f}"}),
                         use_container_width=True, hide_index=True, column_config=_col_cfg(cs7.columns))
            st.caption("建仓分高 + 风险分低 = 历史上相对更值得关注的建仓时机倾斜。**非买卖指令**。")
        else:
            st.caption("该板块标的太少，无法横向对比。")

    nw = b.get("news")
    if nw is not None and len(nw):
        with st.expander("🗞️ 最近新闻（免费 · 仅线索 · 不入量化）"):
            for _, r in nw.head(6).iterrows():
                title = f"[{r['title']}]({r['url']})" if r.get("url") else r["title"]
                st.markdown(f"- `{r['date']}` {title} — *{r.get('provider','')}*")

    st.divider()
    from analysis import briefing as bf
    md = bf.render_markdown([b], bf.auto_weights([b]), horizon)
    st.download_button("📄 导出该股分析(Markdown)", md, file_name=f"{a}_analysis_{end}.md", mime="text/markdown")


# ---------------------------------------------------------------------------
# 路由（三视图）
# ---------------------------------------------------------------------------
_ADV_PAGES = {
    "🎯 个股进出场规则(回测器)": page_rule, "🔬 因子评估": page_factor, "🌊 大盘 regime(SPY/宏观)": page_regime,
    "📈 当前快照": page_snapshot, "🗞️ 事件时间线": page_events, "📅 财报 PEAD": page_earnings,
    "💰 建仓策略对比": page_strategies, "ℹ️ 关于/术语": page_overview,
}
try:
    if view == "📊 个股分析":
        page_panorama()
    elif view == "📋 多票对比":
        page_briefing()
    else:
        _ADV_PAGES.get(adv, page_overview)()
except Exception as e:  # noqa: BLE001
    st.error(f"出错了：{type(e).__name__}: {e}")
    st.exception(e)
