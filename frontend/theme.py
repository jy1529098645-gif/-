"""集中式主题：shadcn/ui (slate) 双主题（暗/亮）+ 一处切换。

设计取向：克制专业，去掉渐变/玻璃拟态/霓虹发光——扁平卡片 + 细边框 + 中性石板灰，
单一靛蓝主色，语义色（绿/红/蓝）在明暗下都保证对比度。参考 GitHub 上最流行的
shadcn/ui 设计 token（CSS 自定义属性 + 明暗双套值）。

对外接口：
  active()            -> "dark" | "light"（从 st.session_state 读，默认 dark）
  tokens()            -> 当前主题的 UI 配色 dict（给 app.py 内联 HTML/原生控件用真实 hex）
  chart_tokens()      -> 当前主题的图表配色 dict（给 charts.py 的 C 代理用）
  plotly_template()   -> "plotly_dark" | "plotly_white"
  inject(st)          -> 注入全套样式（含 Streamlit 原生表面覆盖）
  toggle(st)          -> 在侧边栏渲染明暗切换控件
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# 调色板：每个主题一套完整 token。语义键在两套里一一对应，便于按主题切换。
# ---------------------------------------------------------------------------
PALETTES: dict[str, dict] = {
    "dark": {
        # —— 结构面（背景/卡片/边框/文字）——
        "bg":            "#0f172a",   # slate-900：主背景
        "bg2":           "#0b1120",   # 侧边栏/次级背景（比卡片更深）
        "card":          "#1e293b",   # slate-800：卡片
        "card2":         "#243549",   # 卡片悬浮/次级
        "border":        "#334155",   # slate-700：细边框
        "text":          "#e2e8f0",   # slate-200：正文
        "heading":       "#f1f5f9",   # slate-100：标题
        "muted":         "#94a3b8",   # slate-400：弱化文字
        # —— 语义色 ——
        "primary":       "#6366f1",   # indigo-500：唯一主色
        "primary_weak":  "rgba(99,102,241,0.16)",
        "primary_border":"rgba(99,102,241,0.45)",
        "info":          "#38bdf8",   # sky-400
        "info_weak":     "rgba(56,189,248,0.14)",
        "good":          "#22c55e",   # green-500
        "good_weak":     "rgba(34,197,94,0.15)",
        "good_border":   "rgba(34,197,94,0.40)",
        "bad":           "#ef4444",   # red-500
        "bad_weak":      "rgba(239,68,68,0.15)",
        "bad_border":    "rgba(239,68,68,0.40)",
        "gold":          "#eab308",   # yellow-500
        "gold_weak":     "rgba(234,179,8,0.14)",
        "amber":         "#f59e0b",   # amber-500
        "amber_weak":    "rgba(245,158,11,0.12)",
        "amber_border":  "rgba(245,158,11,0.45)",
        "good2":         "#4ade80",   # green-400
        "shadow":        "0 1px 3px rgba(0,0,0,0.4)",
        # —— 图表专用 ——
        "grid":          "rgba(148,163,184,0.14)",
        "faint":         "rgba(148,163,184,0.22)",
        "spike":         "rgba(148,163,184,0.55)",
        "anno_bg":       "rgba(15,23,42,0.72)",
        "candle_up":     "#22c55e",
        "candle_down":   "#ef4444",
        "vol_up":        "rgba(34,197,94,0.42)",
        "vol_down":      "rgba(239,68,68,0.42)",
        "band":          "rgba(99,102,241,0.18)",
        "band2":         "rgba(56,189,248,0.12)",
        "clock_text":    "#bfd8ff",
        "clock_bg":      "rgba(15,23,42,0.85)",
        "plotly":        "plotly_dark",
    },
    "light": {
        "bg":            "#ffffff",
        "bg2":           "#f8fafc",   # slate-50：侧边栏
        "card":          "#ffffff",
        "card2":         "#f1f5f9",   # slate-100
        "border":        "#e2e8f0",   # slate-200
        "text":          "#0f172a",   # slate-900
        "heading":       "#0f172a",
        "muted":         "#64748b",   # slate-500
        "primary":       "#4f46e5",   # indigo-600
        "primary_weak":  "rgba(79,70,229,0.10)",
        "primary_border":"rgba(79,70,229,0.35)",
        "info":          "#2563eb",   # blue-600
        "info_weak":     "rgba(37,99,235,0.10)",
        "good":          "#16a34a",   # green-600
        "good_weak":     "rgba(22,163,74,0.12)",
        "good_border":   "rgba(22,163,74,0.35)",
        "bad":           "#dc2626",   # red-600
        "bad_weak":      "rgba(220,38,38,0.10)",
        "bad_border":    "rgba(220,38,38,0.35)",
        "gold":          "#ca8a04",   # yellow-600
        "gold_weak":     "rgba(202,138,4,0.12)",
        "amber":         "#d97706",   # amber-600
        "amber_weak":    "rgba(217,119,6,0.12)",
        "amber_border":  "rgba(217,119,6,0.40)",
        "good2":         "#16a34a",
        "shadow":        "0 1px 2px rgba(15,23,42,0.08)",
        "grid":          "rgba(15,23,42,0.08)",
        "faint":         "rgba(15,23,42,0.16)",
        "spike":         "rgba(15,23,42,0.42)",
        "anno_bg":       "rgba(255,255,255,0.78)",
        "candle_up":     "#16a34a",
        "candle_down":   "#dc2626",
        "vol_up":        "rgba(22,163,74,0.38)",
        "vol_down":      "rgba(220,38,38,0.38)",
        "band":          "rgba(79,70,229,0.14)",
        "band2":         "rgba(37,99,235,0.10)",
        "clock_text":    "#1e3a8a",
        "clock_bg":      "rgba(241,245,249,0.92)",
        "plotly":        "plotly_white",
    },
}

DEFAULT = "dark"


def active() -> str:
    """当前主题名。从 st.session_state 读，默认 dark；无 streamlit 上下文时回退 default。"""
    try:
        import streamlit as st
        t = st.session_state.get("theme", DEFAULT)
        return t if t in PALETTES else DEFAULT
    except Exception:  # noqa: BLE001
        return DEFAULT


def tokens() -> dict:
    """当前主题完整 token（图表与 UI 共用一张表）。"""
    return PALETTES[active()]


def chart_tokens() -> dict:
    """图表调色：把通用 token 映射成 charts.py 习惯的键名。"""
    t = tokens()
    return {
        "accent":   t["primary"],
        "accent2":  t["info"],
        "win":      t["good"],
        "loss":     t["bad"],
        "baseline": t["muted"],
        "band":     t["band"],
        "band2":    t["band2"],
        "grid":     t["grid"],
        "text":     t["text"],
        "faint":    t["faint"],
        "spike":    t["spike"],
        "anno_bg":  t["anno_bg"],
        "candle_up":   t["candle_up"],
        "candle_down": t["candle_down"],
        "vol_up":   t["vol_up"],
        "vol_down": t["vol_down"],
        "gold":     t["gold"],
        "amber":    t["amber"],
        "value":    t["primary_weak"],
    }


def plotly_template() -> str:
    return tokens()["plotly"]


# 旧霓虹配色（analysis/decision 等层仍硬编码返回）→ 当前主题 token，保证全局一致。
_LEGACY = {
    "#7c5cfc": "primary", "#00d4ff": "info", "#2be6a8": "good", "#34c779": "good",
    "#ff5c7a": "bad", "#ff9f45": "amber", "#ffd166": "gold", "#8a93a6": "muted",
    "#e6e9ef": "text", "#7cfc9e": "good2",
}


def remap(hexstr: str | None) -> str:
    """把分析层返回的旧硬编码 hex 翻译成当前主题对应色；未知值原样返回。"""
    if not hexstr:
        return tokens()["text"]
    key = _LEGACY.get(str(hexstr).strip().lower())
    return tokens()[key] if key else hexstr


# ---------------------------------------------------------------------------
# 样式：CSS 变量（按当前主题填值）+ 组件类 + Streamlit 原生表面覆盖。
# ---------------------------------------------------------------------------
def _vars_block(t: dict) -> str:
    return f"""
    :root {{
      --bg:{t['bg']}; --bg2:{t['bg2']}; --card:{t['card']}; --card2:{t['card2']};
      --border:{t['border']}; --text:{t['text']}; --heading:{t['heading']}; --muted:{t['muted']};
      --primary:{t['primary']}; --primary-weak:{t['primary_weak']}; --primary-border:{t['primary_border']};
      --info:{t['info']}; --info-weak:{t['info_weak']};
      --good:{t['good']}; --good-weak:{t['good_weak']}; --good-border:{t['good_border']};
      --bad:{t['bad']}; --bad-weak:{t['bad_weak']}; --bad-border:{t['bad_border']};
      --gold:{t['gold']}; --gold-weak:{t['gold_weak']};
      --amber:{t['amber']}; --amber-weak:{t['amber_weak']}; --amber-border:{t['amber_border']};
      --shadow:{t['shadow']};
    }}"""


def _light_overrides() -> str:
    """仅浅色主题追加：config.toml base=dark 让原生控件默认深色，浅色下需强制覆盖回浅色。
    暗色主题不注入此块（原生深色本就正确，避免 !important 干扰 hover/focus）。"""
    return """
    /* —— 浅色主题：把残留深色的原生控件强制刷成浅色 —— */
    /* 次级按钮 / 下载 / 表单提交（主按钮在主样式里另行保持靛蓝） */
    section[data-testid="stSidebar"] .stButton button,
    .stButton button:not([kind="primary"]),
    .stDownloadButton button, .stFormSubmitButton button,
    [data-testid="stFileUploaderDropzone"] button, [data-testid="stBaseButton-secondary"] {
        background: var(--card) !important; border: 1px solid var(--border) !important; color: var(--text) !important;
    }
    section[data-testid="stSidebar"] .stButton button:hover,
    .stButton button:not([kind="primary"]):hover,
    .stDownloadButton button:hover, .stFormSubmitButton button:hover {
        background: var(--card2) !important; border-color: var(--primary-border) !important;
    }
    /* 输入 / 数字 / 日期 / 文本域 / 选择框（含 baseweb 内层） */
    [data-baseweb="base-input"], [data-baseweb="input"], [data-baseweb="textarea"],
    .stTextInput input, .stNumberInput input, .stDateInput input, .stTextArea textarea,
    [data-baseweb="select"] > div:first-child {
        background: var(--card) !important; border-color: var(--border) !important; color: var(--text) !important;
    }
    .stNumberInput button, [data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {
        background: var(--card2) !important; color: var(--text) !important; border-color: var(--border) !important;
    }
    /* 下拉弹层 / 选项列表 */
    [data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"], ul[role="listbox"] {
        background: var(--card) !important; border: 1px solid var(--border) !important;
    }
    [data-baseweb="popover"] li, [data-baseweb="menu"] li, [role="option"] { color: var(--text) !important; }
    [data-baseweb="popover"] li:hover, [role="option"]:hover { background: var(--card2) !important; }
    /* 文件上传 dropzone */
    [data-testid="stFileUploaderDropzone"] { background: var(--card2) !important; border-color: var(--border) !important; }
    [data-testid="stFileUploaderDropzone"] * { color: var(--text) !important; }
    /* 折叠面板 */
    [data-testid="stExpander"] details, [data-testid="stExpander"] summary { background: var(--card) !important; }
    /* 单选 / 复选 / 开关 未选中底色 */
    [data-baseweb="radio"] div[aria-checked="false"] > div,
    [data-baseweb="checkbox"] span[aria-checked="false"] { background: var(--card) !important; border-color: var(--border) !important; }
    /* 滑块轨道 */
    [data-baseweb="slider"] [role="slider"] { background: var(--primary) !important; }
    /* 提示框 alert 文本 */
    [data-testid="stAlertContainer"], [data-testid="stNotification"] { color: var(--text); }
    """


def css() -> str:
    t = tokens()
    extra = _light_overrides() if active() == "light" else ""
    return f"""
    <style>
    {_vars_block(t)}
    {extra}

    /* 用系统字体栈，避免阻塞式拉取远程字体 */
    html, body, [class*="css"] {{
        font-family: -apple-system, 'Segoe UI', 'Microsoft YaHei', Roboto, sans-serif;
    }}

    /* ===== 全局表面：扁平、中性，无渐变/无玻璃 ===== */
    .stApp {{ background: var(--bg); color: var(--text); }}
    [data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stAppViewContainer"], .main, .block-container {{ color: var(--text); }}

    /* 标题 / 正文 / 弱化文字 */
    h1, h2, h3, h4, h5, h6 {{ color: var(--heading); }}
    p, li, span, label, .stMarkdown {{ color: var(--text); }}
    [data-testid="stCaptionContainer"], .stCaption, small {{ color: var(--muted) !important; }}
    a {{ color: var(--primary); }}

    /* ===== 侧边栏 ===== */
    section[data-testid="stSidebar"] {{
        background: var(--bg2); border-right: 1px solid var(--border);
    }}
    section[data-testid="stSidebar"] * {{ color: var(--text); }}

    /* ===== 标题块（替代原渐变 hero）===== */
    .hero-title {{
        font-size: 2.1rem; font-weight: 700; letter-spacing: -0.5px;
        margin-bottom: 2px; color: var(--heading);
    }}
    .hero-sub {{ color: var(--muted); font-size: 0.95rem; margin-top: 2px; line-height: 1.55; }}

    /* ===== 卡片（沿用 .glass 类名，改为扁平 shadcn 卡片）===== */
    .glass {{
        background: var(--card); border: 1px solid var(--border);
        border-radius: 12px; padding: 16px 18px; box-shadow: var(--shadow);
    }}
    .stat-label {{ color: var(--muted); font-size: 0.78rem; letter-spacing: .3px; }}
    .stat-value {{ font-size: 1.6rem; font-weight: 700; line-height: 1.2; color: var(--heading); }}

    /* ===== 徽标 pill ===== */
    .pill {{
        display: inline-block; padding: 3px 10px; border-radius: 6px; font-size: .74rem;
        font-weight: 600; margin-right: 6px; border: 1px solid transparent;
    }}
    .pill-good {{ background: var(--good-weak); color: var(--good); border-color: var(--good-border); }}
    .pill-warn {{ background: var(--bad-weak);  color: var(--bad);  border-color: var(--bad-border); }}
    .pill-info {{ background: var(--info-weak); color: var(--info); border-color: var(--primary-border); }}

    /* ===== 裁决块 ===== */
    .verdict {{
        border-left: 3px solid var(--primary); padding: 12px 16px; border-radius: 8px;
        background: var(--primary-weak); color: var(--text); font-size: 0.95rem; line-height: 1.65;
    }}

    /* ===== Streamlit 原生控件：跟随主题 ===== */
    /* 输入 / 下拉 / 文本域 */
    [data-baseweb="select"] > div, [data-baseweb="input"] > div,
    .stTextInput input, .stNumberInput input, .stTextArea textarea, .stDateInput input {{
        background-color: var(--card) !important; border-color: var(--border) !important;
        color: var(--text) !important;
    }}
    [data-baseweb="select"] svg, [data-baseweb="input"] svg {{ color: var(--muted) !important; }}
    /* 下拉弹层 */
    [data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"] {{
        background-color: var(--card) !important; border: 1px solid var(--border) !important;
    }}
    [data-baseweb="popover"] li {{ color: var(--text) !important; }}
    [data-baseweb="popover"] li:hover {{ background-color: var(--card2) !important; }}

    /* 按钮 */
    .stButton > button, .stDownloadButton > button, .stLinkButton > a {{
        border-radius: 8px; border: 1px solid var(--border);
        background: var(--card); color: var(--text); font-weight: 500;
        transition: background .12s ease, border-color .12s ease;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover, .stLinkButton > a:hover {{
        background: var(--card2); border-color: var(--primary-border);
    }}
    .stButton > button[kind="primary"], button[data-testid="baseButton-primary"] {{
        background: var(--primary) !important; border-color: var(--primary) !important;
        color: #ffffff !important;
    }}
    .stButton > button[kind="primary"]:hover {{ filter: brightness(1.08); }}

    /* 单选 / 视图切换 */
    [data-baseweb="radio"] div {{ color: var(--text) !important; }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid var(--border); }}
    .stTabs [data-baseweb="tab"] {{ color: var(--muted); }}
    .stTabs [aria-selected="true"] {{ color: var(--primary) !important; }}
    .stTabs [data-baseweb="tab-highlight"] {{ background-color: var(--primary) !important; }}

    /* Metric */
    [data-testid="stMetric"] {{
        background: var(--card); border: 1px solid var(--border);
        border-radius: 12px; padding: 12px 14px;
    }}
    [data-testid="stMetricValue"] {{ color: var(--heading); }}
    [data-testid="stMetricLabel"] {{ color: var(--muted); }}

    /* Expander */
    [data-testid="stExpander"] {{
        background: var(--card); border: 1px solid var(--border); border-radius: 12px;
    }}
    [data-testid="stExpander"] summary {{ color: var(--text); }}

    /* 分隔线 */
    hr, [data-testid="stDivider"] {{ border-color: var(--border) !important; }}

    /* 提示框 alert */
    [data-testid="stAlert"] {{ border-radius: 10px; }}

    /* 表格 / DataFrame 外框 */
    [data-testid="stTable"] {{ color: var(--text); }}
    [data-testid="stDataFrame"] {{ border: 1px solid var(--border); border-radius: 10px; }}

    #MainMenu, footer {{ visibility: hidden; }}

    /* ===================== 移动端适配 ===================== */
    @media (max-width: 768px) {{
        [data-testid="stMainBlockContainer"], .block-container {{
            padding-left: 0.7rem !important; padding-right: 0.7rem !important;
            padding-top: 3.2rem !important;
        }}
        [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap !important; gap: 0.55rem !important; }}
        [data-testid="stColumn"] {{
            min-width: calc(50% - 0.55rem) !important; flex: 1 1 calc(50% - 0.55rem) !important;
        }}
        .hero-title {{ font-size: 1.5rem !important; letter-spacing: -0.3px !important; }}
        .hero-sub   {{ font-size: 0.84rem !important; }}
        .stat-value {{ font-size: 1.25rem !important; }}
        .stat-label {{ font-size: 0.68rem !important; }}
        .glass      {{ padding: 12px 13px !important; border-radius: 11px !important; }}
        .verdict    {{ font-size: 0.88rem !important; padding: 10px 12px !important; }}
        .stTabs [data-baseweb="tab-list"] {{ overflow-x: auto !important; flex-wrap: nowrap !important; }}
        .stTabs [data-baseweb="tab"] {{ white-space: nowrap !important; }}
        [data-testid="stDataFrame"], [data-testid="stTable"] {{ overflow-x: auto !important; }}
        [data-testid="stMetricValue"] {{ font-size: 1.3rem !important; }}
    }}
    @media (max-width: 480px) {{
        [data-testid="stColumn"] {{ min-width: 100% !important; flex: 1 1 100% !important; }}
        .hero-title {{ font-size: 1.3rem !important; }}
        .hero-sub   {{ font-size: 0.8rem !important; }}
        [data-testid="stMainBlockContainer"], .block-container {{
            padding-left: 0.5rem !important; padding-right: 0.5rem !important;
        }}
    }}
    </style>
    """


def inject(st) -> None:
    st.markdown(css(), unsafe_allow_html=True)


def toggle(st) -> None:
    """侧边栏明暗切换。改动后重跑一次，让顶部样式按新主题重注入。"""
    st.session_state.setdefault("theme", DEFAULT)
    opts = {"🌙 暗色": "dark", "☀️ 浅色": "light"}
    cur = st.session_state["theme"]
    labels = list(opts)
    pick = st.radio("主题", labels, index=(0 if cur == "dark" else 1),
                    horizontal=True, label_visibility="collapsed", key="theme_pick")
    if opts[pick] != st.session_state["theme"]:
        st.session_state["theme"] = opts[pick]
        st.rerun()
