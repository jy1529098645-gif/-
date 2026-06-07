# 量化研究工具 (quant-lab)

面向长期个人投资者的量化研究工具。目标不是「预测价格」，而是**校准预期、管理风险、评估因子**。

## 三条铁律

1. **校准而非预测**：永远输出「分布 + 置信区间 + 样本数 N」，绝不输出单一目标价/单一概率。
2. **永远对比无条件基准**：任何条件结果必须并排显示「无脑随机买入并持有」基准。跑不赢基准的不是信号。
3. **反过拟合优先**：无幸存者偏差股票池 / block bootstrap CI / walk-forward / deflated Sharpe。

详见 [PROJECT_SPEC.md](PROJECT_SPEC.md)（若已放入根目录）。

## 安装

**一键安装**（推荐）：双击 `setup.bat`（建 venv + 装依赖 + 跑测试 + 预热数据）。

或手动：
```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt          # 或 requirements-lock.txt 复现确切版本
python scripts/warmup.py                  # 预热 SPY+七姐妹+宏观+财报缓存
```

> ⚠️ 务必用 `alphalens-reloaded` 等 *reloaded* 系列，勿装 quantopian 原版。

## 🚀 本地前端（交互仪表盘）

```bash
.venv/Scripts/streamlit run app.py          # 或双击 run_app.bat
```

浏览器打开 http://localhost:8501 。暗色现代仪表盘，11 个页面：
- **🏠 概览** — 三铁律 + 数据源状态 + 术语表
- **📋 多票作战简报** — 多只一屏总览(引擎桶/盈亏比/胜率/财报) + 每票**建仓档(技术位共振→目标/止损/RR)** + **当前状态桶 + 动量陷阱⚠️检测** + **全网多源新闻**(Google News 上千家媒体 + Yahoo，可选 GDELT 全球外媒)/结构化财报要点(营收/净利同比·仅供人读) + **🧠 新闻启发式推理**(规则式情绪+主题×引擎数字交叉，非 LLM、非信号) + **自动权重** + Markdown 导出。[analysis/briefing.py](analysis/briefing.py)、[analysis/news_reason.py](analysis/news_reason.py)
- **🛰️ 建仓作战室** — 单票校准:**条件价位带**(距前高回撤分带→价位区间 + 远期收益分布 + **盈亏比/期望值** + N/CI/超基准) + 未来日程(财报/期权到期) + 财报前后 drift(量化"市场提前消化") + **阶梯式建仓回测** vs lump/DCA。[regime/entry_cockpit.py](regime/entry_cockpit.py)
- **🔎 信号挖掘** — 各建仓/止盈状态的远期超额 + N + block bootstrap CI + **多重检验 FDR 校正**
- **🌊 建仓概率引擎** — 今日状态快照(U2) + 条件远期收益锥 + 状态桶表 + 当前指纹
- **🔬 因子评估** — 价格因子 IC / 分位 / 随机因子健全性
- **🎯 个股进出场规则** — 七姐妹池化 + N_eff + 基准 + 多信号 AND/OR + K线标记图 + **PEAD/市场状态条件门** + 保存/HTML导出 + **反过拟合体检(walk-forward+deflated Sharpe)** + quantstats 报告
- **📈 当前快照(U3)** — Volume Profile(POC/价值区) + 期权当前快照(IV/PutCall/OI，仅展示不可回测)
- **🗞️ 事件时间线(U4)** — SEC EDGAR filings + 财报 + 客观价格反应(仅复盘)
- **📅 财报 PEAD** — 事件研究 + 领先 IC + 蒙特卡洛假财报日对照
- **💰 建仓策略对比** — lump/DCA/补仓 + 置信区间

> 这是「决策与验证」仪表盘，不是看盘终端：只展示分布 + 置信区间 + 样本数 N，不标目标价/买卖点。
> **交互模型（重要）**：重计算页（规则、因子）采用「**配置 → 点🚀运行 → 看结果**」——
> 拖滑块/改参数**不会**自动重算（避免每次卡 8–15s）；改了参数会提示「点运行刷新」。这与专业回测器的习惯一致。
> 侧栏「📅 数据范围」用预设下拉（近10年/近15年/2010至今…），无需手打日期。
> **K 线图用 TradingView 官方开源引擎**（Lightweight Charts）：十字光标(轴上显示价/时间)、滚轮缩放、拖动平移、
> 缩放自动 y 适配、量价副图、▲买▼卖箭头标记、POC/价值区横线。操作与 TradingView 一致。
> 但**不提供画线工具、不标买卖点/目标价**——守住「决策验证而非盯盘」的产品边界（PRODUCT_VISION §6.1/§9）。
> （组件加载失败时自动回退到 plotly K 线，含区间按钮/十字光标。）
> **专业名词鼠标悬浮显示解释**（带虚线下划线的词，如 IC / N_eff / MAE / PEAD / deflated Sharpe）；
> 控件旁 `?` 图标有提示；概览页有完整术语表。词条见 [frontend/glossary.py](frontend/glossary.py)。
>
> **U1 体验闭环（已完成）**：规则页支持
> - 📈 **K 线进出场标记图**（plotly candlestick + 买卖点按盈亏着色，单票复核）
> - 🧩 **多信号 AND/OR 组合构建器**（1–3 个入场信号，组合数计入多重检验提示）
> - 💾 **规则保存（SQLite）+ 📄 一键 HTML 报告导出**（reports/{rule}/index.html，自包含图 + 裁决 + N/CI）
> - 🔗 **URL 深链**：`?p=rule`（overview/regime/factor/rule/earnings/strategies）可直达页面
> 需 OHLCV 数据：`loader.load_ohlcv(ticker)`（复权 open/high/low/close/volume + 缓存）。

## ☁️ 部署到云端（让别人也能用网址打开）

GitHub 本身**不能运行** Streamlit（GitHub Pages 只放静态网页）。用 **Streamlit Community Cloud（免费）** 从本仓库一键部署：

1. 代码已在 GitHub：`https://github.com/jy1529098645-gif/-`
2. 打开 **https://share.streamlit.io** → 用 GitHub 账号登录 → **Create app** → **Deploy a public app from GitHub**。
3. 填写：
   - Repository：`jy1529098645-gif/-`
   - Branch：`main`
   - Main file path：`app.py`
4. 点 **Advanced settings → Secrets**，粘贴（可留空，留空则宏观自动回退 ETF 代理）：
   ```toml
   FRED_API_KEY = "你的_FRED_API_KEY"
   ```
5. **Deploy**。几分钟装完依赖后得到一个公开网址（形如 `https://xxx.streamlit.app`），任何人任何设备打开即用。

> ⚠️ **密钥安全**：`config.yaml` 不再含任何 key；密钥只走环境变量 / Streamlit secrets（`.streamlit/secrets.toml` 已被 .gitignore 排除）。
> ⚠️ **免费层限制**：Community Cloud 约 1GB 内存，首次加载/大票池回测可能较慢或偶发重启。
> ⚠️ **数据持久化**：用户数据（校准信号 / 手填事件 / 检验账本 / 保存的规则）存 SQLite，云端**重启会清空**。两条出路：
>   1. 侧栏 **「💾 我的数据」→ 导出备份 JSON**，需要时上传恢复（零依赖，任何部署都能用）；
>   2. 自建带持久盘的主机：设环境变量 **`QUANTLAB_DB_PATH=/data/quantlab.db`** 指向挂载盘，自动持久、免手动备份。

本地启动等价命令：`streamlit run app.py`（会自动读 `.streamlit/secrets.toml`）。

## 🧠 深度作战简报 Skill（在 Claude 对话里触发，联网读正文）

工具内的新闻是**规则式、自给自足、免费**的常规功能（全网标题→情绪/主题→`📖读正文要点`抽全文关键句）。
**全面读各种新闻 + LLM 级深度推理**则在 **Claude 对话**里做——已封装为 skill `quant-deep-brief`
（[.claude/skills/quant-deep-brief/SKILL.md](.claude/skills/quant-deep-brief/SKILL.md)）：

- 触发：对 Claude 说「深度分析 NVDA」「读读 GOOGL 新闻再分析」「GOOGL NVDA MSFT 作战简报」等。
- 机制：先跑 `python -m analysis.brief_cli <T>` 取**工具的真实引擎数字**(价位档/超额/盈亏比/财报drift)接地气，
  再用 WebSearch/WebFetch **读正文全文**，合成分层简报——**数字来自工具📊 / 叙事来自原文📰+URL / 推断🧠** 分层标注。
- [analysis/brief_cli.py](analysis/brief_cli.py) 是 skill 的接地气入口（输出可计算简报 JSON）。

## 运行测试

```bash
.venv/Scripts/python -m pytest          # 122 passed
```

## 目录结构

```
量化工具/
├── config.yaml              # 全局配置：股票池、日期、成本、因子参数
├── config.py                # 配置加载器
├── data/loader.py           # 数据层（价格/宏观/股票池 + 缓存）
├── data/news.py             # 全网多源新闻聚合(Google News/Yahoo/GDELT) + 结构化财报要点(仅供人读、不入量化)
├── factors/                 # 因子库（先做价格类）
├── evaluation/factor_eval.py# 因子评估（封装 alphalens）
├── regime/conditional_returns.py # 建仓概率引擎（条件远期收益）
├── regime/entry_cockpit.py  # 建仓作战室（条件价位带/盈亏比/期望值/事件日程/阶梯回测）
├── analysis/briefing.py     # 多票作战简报（综合层：总览/建仓档/引擎桶/自动权重/Markdown）
├── stats/                   # 反过拟合基建（bootstrap/walkforward/deflated sharpe）
├── backtest/strategies.py   # vectorbt 策略
├── reports/                 # 输出目录
└── tests/                   # 各模块验收测试
```

> **校准式升级（作战室 / 多票简报）铁律**：价位带是「区间 + 经验分布」非目标价；盈亏比/期望值基于历史独立事件；
> 引擎报**当前状态桶**(在回撤就报回撤桶)并自动检测**动量陷阱**(回撤桶超额≤0→标⚠️降权)，
> 绝不用全局最优桶掩盖「动量股越跌买越没优势」；新闻/基本面/财报要点**仅供人读、不入量化、含前视风险**；
> 目标/止损是按技术位规则推导的**风险参考**(供算盈亏比)，**非预测点位**；权重为机械规则、非投资建议。

## 数据源说明（重要）

- **价格**：yfinance（Yahoo），复权收盘价，2000+，parquet 缓存，价格不前向填充。
- **宏观**：已配置 FRED API key，走 **FRED 官方 API 全历史**：
  - `credit_spread` = `BAA10Y`（Moody's Baa 公司债收益率 − 10Y 国债，日频 **1986+**）。
  - `yield_curve` = `T10Y2Y`（10Y − 2Y 期限利差，**1976+**）。
  - ⚠️ ICE BofA 的 `BAMLH0A0HYM2`（HY OAS）是授权数据，FRED API 仅返回最近约 3 年，故改用 `BAA10Y`。
  - 无 key 时自动回退 Yahoo 代理：`yield_curve`=^TNX−^IRX（10Y-3M，2000+）、`credit_spread`=−(HYG/IEF)（2007+）。
  - key 配置在 `config.yaml` 的 `macro.fred_api_key`，或用环境变量 `FRED_API_KEY` 覆盖。
    申请：[FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html)。
- `load_macro()` 的 `DataFrame.attrs['sources']` 记录每列实际来源。

> ⚠️ 当前 demo 股票池（`spy_demo`）为幸存者偏差版本，仅供 API 验证；无幸存者偏差池见 Phase 6。

## 分阶段进度

- [x] **Phase 0** — 脚手架：目录 + config + 空模块 + tests（pytest 可跑，12 项）
- [x] **Phase 1** — 数据层：价格/宏观/股票池 + 缓存 + NaN 策略（验收 4 项通过）
- [x] **Phase 2** — 价格因子 + alphalens 评估：momentum 出 tear sheet（PDF），随机因子 IC≈0（验收通过）
- [x] **Phase 3** — 反过拟合基建：block bootstrap / walk-forward / deflated Sharpe（合成数据验收 9 项通过）
- [x] **Phase 4** — 建仓概率引擎：回撤×估值×信用分桶的条件远期收益（N+CI+基准差值）+ 当前指纹面板（验收 6 项通过）
- [x] **Phase 5** — 回测整合：lump_sum/dca/average_down 对比（同预算、滚动窗口、block bootstrap CI）+ 因子分位回测（验收 6 项通过）
- [x] **Phase 6（免费部分）** — 多因子组合：[price_factors.blend](factors/price_factors.py)（z-score 等权混合）+ 因子页「多因子组合」区（组合 IC + 多空回测 + Sharpe CI）
  - zipline-reloaded：可装但 Pipeline/动态大票池对固定七姐妹场景无增益且需重型 bundle，不引入；qlib：Python 3.13 无发行版（技术不可行，非付费）。详见 [ROADMAP.md](ROADMAP.md)。

### 补充规格 A：个股进出场规则（七姐妹，纯价格/免费数据）

- [x] **Phase S1** — 入场信号 [factors/signals.py](factors/signals.py) + 出场规则 [backtest/exits.py](backtest/exits.py)，单票逐笔交易含成本 + 进出场标记图/收益分布/MAE 图（验收 7 项）
- [x] **Phase S2** — 池化评估 [evaluation/rule_eval.py](evaluation/rule_eval.py)：N_eff 折算 + block bootstrap CI + 随机基准对比；随机入场超额≈0（验收 4 项）
- [x] **Phase S3** — 规则选择 [evaluation/rule_select.py](evaluation/rule_select.py)：walk-forward(IS选/OOS报) + deflated Sharpe + 水下图/walk-forward 图（验收 5 项）
- [x] **Phase F1** — 财报日历因子（**免费 yfinance 数据**）：[factors/fundamentals.py](factors/fundamentals.py) PIT 因子（距财报天数/上次超预期/财报前后窗口）+ [evaluation/earnings_eval.py](evaluation/earnings_eval.py) 事件研究 + PEAD 领先 IC + 蒙特卡洛假财报日对照 + 财报日历结构图（验收 6 项）
- [x] **F3（免费子集）** — PEAD/财报日历作为 A 进出场**条件门**：`evaluate_rule(condition_fn=...)` + `earnings_condition()`（分条件池化评估）
- [ ] **Phase F2–F3（完整）** — SUE/预期上修/估值分位/质量趋势：**需 point-in-time 付费数据源（Sharadar/FMP/Tiingo ~$30-50/月），待用户决定付费**。免费 yfinance 财务快照有前视/重述偏差，不可用于历史验证。

可视化：[reports/style.py](reports/style.py) 统一样式（中文字体），[reports/plots.py](reports/plots.py) 决策/验证图，自动存到 `reports/{rule_name}/`。所有统计图强制标注 N（及 N_eff）与分位/CI，不标"目标价/最佳买卖点"。

### 🚦 验收闸门 + 可靠性升级（2026-06）

- **验收闸门** [evaluation/acceptance.py](evaluation/acceptance.py)：把三铁律合成一个 go/no-go——
  ① 跑赢随机基准(N_eff 折算后显著) ② OOS 年化夏普 ≥ 1.0(walk-forward 样本外日度收益) ③ 最大回撤 ≤ 35%。
  三条全过才 PASS；**PASS≠买入信号**，只代表过了反过拟合及格线。回测器页可选票池运行。
- **降偏差宽池** `diversified`(config.yaml)：在 7 个十年赢家外混入 ~38 只走平/下跌大盘股
  (INTC/PYPL/BA/T/GE/PFE/DIS…)。实测同一 dip 规则超额从七姐妹 **+5.0%** 缩到宽池 **+1.4%**——
  「优势」大半是选股偏差。⚠️ 仍**非真·无幸存者偏差**(已退市标的需付费 PIT 数据)，只是大幅降低偏差。
- **长周期 CI 诚实化**：条件远期收益新增 `n_independent`(重叠窗口的有效独立数)与 `low_power` 标记；
  h252 独立窗口常只剩 ~4 个 → 标⚠️并把「显著」降级，杜绝长周期假性显著。
- **操作预案** [analysis/playbook.py](analysis/playbook.py)：把引擎桶/价位档/MAE/动量陷阱/财报drift 翻成
  **if-then 条件操作指导**——在哪些价位分批建仓、涨了怎么减仓+移动止损、跌了该补还是该止损、
  时间/事件/风控。个股分析页「📋 操作预案」直接展示；brief_cli 输出含 `operation_playbook`。
  **诚实关键**：动量陷阱 / 未过验收闸门 → 预案自动转防守口径（「别越跌越补、轻仓、按止损」）；
  价位是「**若到达就行动**」的区间(非预测)，**非买卖指令**。

> 个股 A 轨铁律：池化统计（不做单票结论）、N_eff 折算高相关证据、入场×出场成对、walk-forward + deflated Sharpe。
> 关键发现：dip 回撤入场规则在七姐妹上 per-trade Sharpe 稳健为正，**但相对随机进场的超额经 N_eff 折算后不显著**——赚钱主要来自出场规则+大盘上行，而非入场择时。

## 运行示例

```python
from data import loader
from regime import conditional_returns as cr
from backtest import strategies as bt

px = loader.load_prices(['SPY'], '1995-01-01')
macro = loader.load_macro('1990-01-01')

# 建仓概率引擎：条件远期收益（带 N / CI / 基准差值）
tab = cr.conditional_forward_returns(px, macro, asset='SPY', horizons=(63, 252))
print(cr.format_bucket_verdict(tab[tab.grouping=='in_drawdown'].iloc[0]))
print(cr.current_fingerprint(px, macro))           # 当前指纹 vs 历次大跌

# 建仓策略对比（结果带置信区间）
res = bt.compare_entry_strategies(px, asset='SPY')
print(res['note']); print(res['vs_lump_sum'])
```

PEAD 关键结论（七姐妹 371 个财报事件，免费数据）：盈利超预期 → 公布后漂移，领先 IC h=1/5/21 = 0.13/0.15/0.10（置换 p<0.05 显著），h=63 衰减不显著；蒙特卡洛假财报日对照平均 IC≈0。

总测试：**97 passed**（`.venv/Scripts/python -m pytest`）。前端 [app.py](app.py) + [frontend/](frontend/)（charts/glossary/store/report）。路线图见 [ROADMAP.md](ROADMAP.md)。
