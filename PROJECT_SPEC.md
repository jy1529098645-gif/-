# 量化研究工具 · 搭建规格书（给 Claude Code 执行）

> 把本文件放在项目根目录，命名为 `PROJECT_SPEC.md`（或 `CLAUDE.md`）。
> 按「分阶段执行计划」从 Phase 0 开始，**一个阶段做完、通过验收标准、再做下一个**。
> 不要一次性把所有模块都生成出来。

---

## 0. 项目目标与哲学（必须先读，违背则全盘失效）

本工具是一个**面向长期个人投资者的量化研究工具**，目标不是"预测价格"，而是：

1. **校准预期**：在某种市场设置下进场，历史上的收益分布长什么样、最坏会浮亏多少、多久回得来。
2. **管理风险**：把"建仓 / 加仓"从感觉变成有概率分布和置信区间的决策。
3. **评估因子**：用规范方法判断一个因子是真信号还是过拟合噪音。

**贯穿全局的三条铁律（任何模块都不得违反）：**

- **校准而非预测**：永远输出"分布 + 置信区间 + 样本数 N"，**绝不输出单一目标价或单一概率数字**。
- **永远对比无条件基准**：任何"条件收益/胜率"必须和"无脑随机买入并持有"的基准并排显示。跑不赢基准的就不是信号。
- **反过拟合优先于一切**：股票池必须含已退市标的（无幸存者偏差）；统计用 block bootstrap 给置信区间；策略评估全程 walk-forward；多候选筛选必须做 deflated Sharpe / 多重检验校正。

---

## 1. 技术栈与安装

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip

# 核心
pip install pandas numpy scipy
pip install yfinance pandas-datareader        # 数据（美股 + FRED 宏观）
pip install vectorbt                           # 向量化回测（快速扫参）
pip install alphalens-reloaded                 # 因子评估（注意是 -reloaded，原版已废弃）
pip install quantstats                         # 绩效 tear sheet
pip install matplotlib plotly                  # 画图
pip install pyyaml tqdm                         # 配置/进度

# 可选（后期）
# pip install zipline-reloaded                 # 多空因子组合 + Pipeline（动态股票池）
# pip install pandas-ta                        # 技术指标
# pip install pyportfolioopt riskfolio-lib     # 组合优化
```

> ⚠️ **务必用 `alphalens-reloaded` / `pyfolio-reloaded` / `zipline-reloaded`**，不要装 quantopian 的原版（2020 年后无维护，与新版 pandas/numpy 不兼容）。
> ⚠️ 若运行环境有网络白名单限制 Yahoo，yfinance 会拉不到数据；此时改用本地缓存的 CSV，或换数据源。

---

## 2. 目录结构

```
quant-lab/
├── PROJECT_SPEC.md
├── config.yaml                # 全局配置：股票池、日期、成本、因子参数
├── data/
│   ├── loader.py              # 下载 + 本地缓存（价格、宏观、股票池成分）
│   └── cache/                 # CSV/parquet 缓存
├── factors/
│   ├── base.py                # 因子基类/协议：输入价格面板，输出因子值面板
│   ├── price_factors.py       # 价格类因子（先做这些）
│   └── fundamental_factors.py # 基本面因子（后期，需 PIT 数据）
├── evaluation/
│   └── factor_eval.py         # 封装 alphalens：IC、分位收益、tear sheet
├── regime/
│   └── conditional_returns.py # 建仓概率引擎：回撤分桶 + 宏观状态 → 条件远期收益
├── stats/
│   ├── bootstrap.py           # block bootstrap 置信区间
│   ├── walkforward.py         # 滚动 IS/OOS 切分
│   └── deflated_sharpe.py     # deflated Sharpe + 多重检验校正
├── backtest/
│   └── strategies.py          # vectorbt 策略（含已有的补仓/DCA/lump-sum）
├── reports/
│   └── (输出目录)
└── tests/                     # 每个模块的验收测试
```

---

## 3. 模块规格

### 3.1 data/loader.py
- `load_prices(tickers, start, end) -> pd.DataFrame`：返回**复权收盘价**面板，index=日期，columns=ticker。带本地缓存（parquet/CSV），二次运行不重复下载。
- `load_universe(name) -> list[str]`：返回股票池成分。**必须支持含已退市成分的历史股票池**（无幸存者偏差）。初期可先用单指数（如 SPY）跑通，但在文档/代码注释中明确标注"当前为幸存者偏差版本，仅供 API 验证"。
- `load_macro() -> pd.DataFrame`：从 FRED 拉宏观状态序列：信用利差（如 BAA-AAA 或 BAMLH0A0HYM2）、收益率曲线（10Y-2Y, T10Y2Y）、可选 CAPE/盈利收益率。
- **验收**：能取到至少 2000 年至今的日线，缓存生效，缺失值处理明确（前向填充宏观、价格不填充）。

### 3.2 factors/ —— 因子库

**先只实现价格类因子**（只用价格即可干净计算，无前视偏差）：

| 因子 | 定义 | 备注 |
|---|---|---|
| `momentum_12_1` | 过去 12 个月收益，剔除最近 1 个月 | 经典动量 |
| `short_reversal` | 过去 1 个月收益的负值 | 短期反转 |
| `low_volatility` | 过去 60/120 日收益波动率的负值 | 低波动异象 |
| `trend` | 价格相对 200 日均线的位置 | 趋势/regime |

每个因子是一个**纯函数**：输入价格面板 → 输出同形状的因子值面板（截面可比）。统一在 `base.py` 定义协议。

**基本面因子（`fundamental_factors.py`）标记为 Phase 4，暂不实现**，并在文件顶部写明："需要 point-in-time（PIT）财报数据源，免费 yfinance 数据有前视/幸存者偏差，不可直接用于历史回测。"

### 3.3 evaluation/factor_eval.py —— 因子评估（封装 alphalens）
- `evaluate_factor(factor_values, prices, quantiles=5, periods=(1,5,21,63)) -> dict`：
  - 用 `alphalens.utils.get_clean_factor_and_forward_returns` 对齐因子值与远期收益。
  - 输出：**IC 均值/标准差/IR**、分位组合远期收益、多空价差、换手率、`create_full_tear_sheet`。
- **必须随每个 IC/收益数字标注样本期长度，并提示**："单因子 IC 0.03–0.05 即属可用；高得离谱要怀疑前视偏差。"
- **验收**：对 `momentum_12_1` 跑通，输出 tear sheet；对一个**随机因子**（np.random）跑出近 0 的 IC，作为健全性检查（sanity check）。

### 3.4 regime/conditional_returns.py —— 建仓概率引擎（核心）
这是回答"当前点位该不该建仓/加仓"的引擎。**粗分桶，不做精确情景匹配。**

- 状态变量（最多 3 个，每个粗切，保证每桶有足够多**独立事件**）：
  1. `in_drawdown`：当前是否处于距前高 >10% 的回撤中（二元）。
  2. `valuation_tercile`：估值在历史的高/中/低三分位（用盈利收益率或 CAPE）。
  3. `credit_trend`：信用利差走阔 or 收窄（二元，来自 FRED）。
- `conditional_forward_returns(prices, macro, horizons=(21,63,126,252,504)) -> DataFrame`：
  - 对每个状态桶，统计往后各 horizon 的远期收益**经验分布**：胜率、中位、10/25/75/90 分位、途中最大浮亏。
  - **每个桶必须输出**：(a) 独立事件数 N（不是天数，要按事件去重叠/聚类计数）；(b) block bootstrap 95% 置信区间；(c) **与无条件基准的差值**（条件中位 − 无条件中位）。
- 输出措辞模板（强制）：
  > "状态=[已回撤>10% & 估值低]：12 个月远期收益中位 +X%，10 分位 −Y%，相对无条件基准 +Z 个点，基于 N≈n 个独立事件（95% CI: [a, b]）。结论：弱倾斜，非信号。"
- **禁止**输出"建仓概率 73%"这种光秃秃的数字。
- 附带"**当前指纹 vs 历史指纹**"对比面板：把历次大跌当时的利率/曲线/信用利差/估值分位，与"今天"的观测值并排列出，**供人判断**，不自动给结论。

### 3.5 stats/ —— 反过拟合基建（横切，所有评估都要调用）
- `bootstrap.py`：`block_bootstrap_ci(returns, stat_fn, block_size, n=1000)` —— 用块自助法给统计量置信区间（重叠样本会虚增显著性，必须用块）。
- `walkforward.py`：`walk_forward_splits(index, train, test, step)` —— 生成滚动 IS/OOS 切分；任何参数选择只能在 IS 做、在 OOS 报。
- `deflated_sharpe.py`：`deflated_sharpe_ratio(sr, n_trials, ...)` —— 按尝试次数对夏普打折（López de Prado）；当筛选了多个因子/参数时**必须**用它，否则报告"未做多重检验校正，下列显著性不可信"。

### 3.6 backtest/strategies.py
- 迁移已有的 `lump_sum / average_down / dca` 三策略对比（用 vectorbt，含 fees+slippage）。
- 新增：基于因子分位的多空/纯多组合回测（后期可接 zipline-reloaded 的 Pipeline）。
- 所有回测结果走 `stats/` 出置信区间，不报点估计。

---

## 4. 分阶段执行计划（按序，逐阶段验收）

- **Phase 0 — 脚手架**：建目录、`config.yaml`、`requirements.txt`、空模块与 `tests/`。验收：`pytest` 能跑（哪怕全是占位）。
- **Phase 1 — 数据层**：`data/loader.py`。验收：拉到并缓存 SPY + 几只成分股 + FRED 宏观，2000 年至今。
- **Phase 2 — 价格因子 + 评估**：`factors/price_factors.py` + `evaluation/factor_eval.py`。验收：momentum 出 tear sheet；随机因子 IC≈0（sanity check 通过）。
- **Phase 3 — 反过拟合基建**：`stats/`。验收：block bootstrap、walk-forward、deflated Sharpe 各有单元测试，用合成数据验证（如已知夏普的随机序列）。
- **Phase 4 — 建仓概率引擎**：`regime/conditional_returns.py`，调用 Phase 3 的工具。验收：输出带 N 和 CI 的条件收益表 + 基准对比 + 当前指纹面板。
- **Phase 5 — 回测整合**：`backtest/strategies.py`，迁移补仓/DCA 对比，结果走置信区间。
- **Phase 6（可选/后期）**：基本面因子（需 PIT 数据源）、zipline-reloaded 因子组合、Qlib 接入。**在前 5 阶段稳定前不做。**

每个阶段产出：可运行代码 + 对应 `tests/` 验收测试 + 一段 README 说明如何运行。

---

## 5. 硬性禁止清单（DO NOT）

1. **不要**输出单一目标价或单一概率数字；一律分布 + 置信区间 + N。
2. **不要**用只含现存成分股的股票池做历史结论（幸存者偏差）；如暂用，必须显式标注。
3. **不要**用免费 yfinance 基本面数据做历史因子回测（前视偏差）。
4. **不要**只做一次 IS/OOS；用 walk-forward。
5. **不要**在未做 deflated Sharpe / 多重检验校正的情况下，报告"从 N 个因子里选出的最优"的显著性。
6. **不要**自动挖因子（遗传编程/LLM 进化）作为起步功能——它是多重检验灾难，列入远期、且必须配 deflated Sharpe 守门。
7. **不要**用重叠窗口的普通标准误算置信区间；用 block bootstrap。
8. **不要**声称能识别"当前更像哪次历史下跌"并据此给买卖结论——原因只能事后归类，实时不可解。

---

## 6. 术语表（给量化新手）

- **因子 (Factor)**：能用规则算出来的、怀疑和未来收益有关的股票特征（如动量、价值）。把模糊想法变成可计算的数字。
- **IC / 信息系数 (Information Coefficient)**：因子值与未来收益的截面相关系数。单因子 0.03–0.05 即可用；因子天生很弱，靠广度和一致性取胜。
- **IR / 信息比率**：信号的风险调整后强度，近似 IC × √广度（基本主动管理定律）。
- **分位组合 (Quantile Portfolios)**：按因子值把股票分成几档（如 5 档），比较最高档和最低档的收益。
- **远期收益 (Forward Return)**：从某时点往后 N 天的收益，用来检验"现在的因子值/状态"是否预示未来。
- **无条件基准 (Unconditional Baseline)**：不挑时点、随机买入持有的收益分布。任何"条件"结果都要和它比，差值才是信息。
- **幸存者偏差 (Survivorship Bias)**：只用"活到今天"的股票/市场做回测，系统性高估收益（退市、破产的被剔除了）。
- **前视偏差 (Look-ahead Bias)**：用了当时还拿不到的信息（如尚未公布的财报）。
- **Walk-forward**：滚动地"用过去训练、用紧接的未来检验"，反复多段，比单次切分更难自欺。
- **Block Bootstrap / 块自助法**：对有时间相关性的收益重采样时，按"块"抽取以保留相关结构，给出诚实的置信区间。
- **Deflated Sharpe Ratio**：按"试了多少次"对夏普打折，扣掉靠运气挑出来的部分（治多重检验/数据窥探）。
- **Regime / 市场状态**：市场所处的大环境（如高/低估值、信用走阔/收窄），不同状态下同一策略表现可能完全不同。

---

## 7. 推荐参考（库与读物）
- 库：`alphalens-reloaded`（因子评估）、`vectorbt`（向量回测）、`zipline-reloaded`（因子组合/Pipeline）、`quantstats`/`pyfolio-reloaded`（绩效）、`microsoft/qlib`（AI 因子平台，后期）、`wilsonfreitas/awesome-quant`（清单）。
- 读物：López de Prado《Advances in Financial Machine Learning》（反过拟合/deflated Sharpe）、Grinold & Kahn《Active Portfolio Management》（IC/IR/基本定律）、石川《因子投资：方法与实践》（中文因子）。
