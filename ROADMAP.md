# 路线图 · 做成「真正可用」的免费量化工具

> ✅ 更新：U1–U6 **已全部完成** + 打磨轮 P1–P4（85 tests passing）。下方差距表的 🟡/❌ 大多已转 ✅。
> 详见 README「分阶段进度」。
>
> **打磨轮（P1–P4，免费）**：
> - P1 反过拟合体检改为围绕**当前规则**参数动态生成候选（`rule_select.candidate_grid_from`）。
> - P2 事件时间线加 filing 类型筛选 + 各类事件平均反应统计。
> - P3 完成「保存→载入复用」闭环（规则页可从 SQLite 一键载入到控件）。
> - P4 Phase 6（zipline/qlib）可行性评估 + 免费替代落地（见下）。
>
> **Phase 6 结论（main spec，可选）**：
> - **qlib**：Python 3.13 无可安装发行版（pyqlib no matching distribution）→ 技术不可行，需独立旧版 Python 环境，**非付费问题**，搁置。
> - **zipline-reloaded**：可装（3.1.1 有 cp313 wheel、不降级 pandas/numpy），但其核心价值 Pipeline + 动态大票池对「固定七姐妹」场景几乎用不上，且需重型 bundle ingestion 插桩；与现有 vectorbt 因子回测重叠。**故不引入重依赖**。
> - **免费替代已落地**：Phase 6「因子组合」用现有稳定栈实现为**多因子 z-score 等权组合**（`price_factors.blend` + 因子页「多因子组合」区，含组合 IC + 多空回测 + Sharpe CI）。
>
> **唯一未做**：F2/F3 完整版（SUE/预期上修/估值分位/质量趋势）需 point-in-time **付费**数据源。其余免费能力全部完成。


> 对照 [PRODUCT_VISION.md](PRODUCT_VISION.md) 的差距分析 + 分阶段方案。
> 三硬约束：① 只用免费数据 ② 用户友好（验证与决策，非看盘）③ 必须能回测。

## A. 现状对照（PRODUCT_VISION 要求 → 完成度）

| 能力（Vision 第 3/4/5 节） | 状态 | 说明 |
|---|---|---|
| 进出场规则引擎 + 回测（核心） | ✅ | signals/exits/rule_eval，vectorbt 含费用滑点 |
| 池化 + N_eff + block bootstrap + walk-forward + deflated Sharpe | ✅ | 反过拟合管线齐全 |
| 三种部署对比（一次性/补仓/定投） | ✅ | backtest/strategies |
| 建仓 regime（免费可观测状态） | 🟡 | 已有回撤×估值×信用×曲线；缺：实现波动率分位、趋势位置、七姐妹相互相关、财报日历作 regime |
| 财报日历因子（PEAD，免费无前视） | ✅ | F1 + 接入规则条件门 |
| 用户友好 Streamlit + 术语悬浮解释 | ✅ | app.py 6 页 + glossary 悬浮提示 |
| 规则构建器（下拉/滑块零代码） | 🟡 | 已有单信号；缺：多信号 AND/OR 组合 UI、规则保存/复用 |
| 一屏看懂一条规则 | 🟡 | 已集中展示；缺：K线进出场标记图进前端、一句话裁决置顶 |
| 进出场标记图（K线作底 + 买卖点） | 🟡 | 有 matplotlib 版；前端缺 plotly K线（需 OHLCV 数据） |
| Volume Profile（日线近似 + POC） | ❌ | 未做（需 OHLCV） |
| 期权当前状态快照面板（免费、仅展示） | ❌ | 未做（yfinance 期权链） |
| 事件时间线（SEC EDGAR） | ❌ | 未做（另见 event_research 补充规格） |
| quantstats 绩效 tear sheet | ❌ | 因子用 alphalens；规则净值未接 quantstats |
| 规则保存 / 结果存储（SQLite） | ❌ | 未做 |
| 一键导出报告（HTML/PDF） | ❌ | 有 PNG 落盘；缺汇总页 |
| 可解释（每个数字可展开"怎么算"） | 🟡 | 已有术语悬浮 + 措辞模板；可再加"算法/样本"抽屉 |

**已明确砍掉（需付费，Vision 第 8 节）**：历史期权回测、PIT 基本面（F2/F3）、日内精细 Volume Profile。

## B. 关键技术债：数据层只存复权收盘价

当前 `load_prices` 只返回复权收盘价。Vision 的 K 线标记图、Volume Profile、成交量都需要 **OHLCV**。
→ **U1 第一步必须扩展数据层到 OHLCV**，这是多个后续功能的前置。

## C. 分阶段方案（按"让工具真正可用"的优先级）

### Phase U1 — 体验闭环（最高优先，纯免费，让它每天能用）
1. **数据层扩 OHLCV**：`load_ohlcv(ticker)` 存 open/high/low/close/volume（复权），parquet 缓存；旧 `load_prices` 保持兼容。
2. **前端 K 线进出场标记图**：plotly candlestick 作底 + 入场▲/出场▼按盈亏着色（规则页核心视图）。
3. **多信号规则构建器**：UI 支持 2–3 个入场信号 AND/OR 组合（组合数计入多重检验计数）。
4. **规则保存/复用 + 一键 HTML 报告**：SQLite 存命名规则；导出 `reports/{rule}/index.html` 汇总页（图 + 裁决 + N/CI）。
   - 验收：保存一条规则→重开→一键复现报告；标记图能肉眼复核买卖点。

### Phase U2 — regime 深化（免费可观测状态，喂给规则做条件）
1. 新增免费状态：实现波动率分位、趋势位置（距 200 日线）、七姐妹滚动相互相关、财报日历窗口。
2. 这些状态都能在规则页作"条件门"（复用已建 condition_fn 机制）→ 分条件池化评估。
3. "今日状态面板"：一屏显示当前各 regime 处于历史什么分位 + 该状态历史远期收益分布。
   - 验收：随机状态对照超额≈0；当前状态面板可读。

### Phase U3 — 当前状态快照（免费、仅展示、不可回测，明确标注）
1. **Volume Profile**：日线 high-low 摊量 → 成交密集区 + POC，叠在 K 线标记图上（辅助，非信号）。
2. **期权快照面板**：yfinance 期权链算当前 IV / Put-Call / OI 集中行权价；明确"仅当前、不可回测"。
   - 验收：两者都带"辅助展示、未验证领先性"角标。

### Phase U4 — 事件时间线（免费，SEC EDGAR，认知抽屉）
1. SEC EDGAR 抓取 filings + 财报日历，事件对齐客观价格反应；默认只展示七姐妹。
2. 客观字段自动、主观判断只出待复核草稿；**产出供复盘，不作信号**。
   - 验收：事件与价格反应对齐；主观内容不流入回测。

### Phase U5 — 绩效与反过拟合全程上前端
1. 规则净值接 **quantstats** tear sheet（水下图/回撤/恢复天数）。
2. walk-forward + deflated Sharpe 结果上规则页（现仅在模块层）。
3. 每个回测结果强制显示 N / N_eff / CI / 基准（审一遍补齐）。

### Phase U6 — 打包成"真正可分发"的本地应用
1. 依赖固定版本 + 一键安装脚本（`setup.bat`：建 venv、装依赖、跑测试）。
2. 首次运行数据预热（七姐妹 + SPY 全历史缓存）。
3. 前端 helper 加测试（glossary/charts 冒烟）；CI 友好。
4. 可选：桌面启动器 / 任务栏图标。

## D. 立即可做的"快赢"
- U1.1 OHLCV 数据层（解锁 K 线图 + Volume Profile，1 步多用）。
- U1.4 规则保存 + HTML 导出（让它从"演示"变"工具"）。
- U5.1 quantstats 水下图（现成库，接上即得专业绩效页）。

## E. 不做（守住产品边界）
- 不做可缩放画线的行情终端；不预测价格 / 不给目标价 / 不给买卖结论。
- 主观判断（事件性质、Volume/期权看法）禁止进入回测与信号。
- 不用付费数据硬凑；缺失能力如实标注。
