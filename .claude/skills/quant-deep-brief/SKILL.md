---
name: quant-deep-brief
description: Deep, web-read, reasoned stock briefing grounded in the quant-lab tool's real engine numbers. Reads FULL news articles across the web (not just headlines) and synthesizes a layered analysis — one-line take, why-these-price-levels, what already happened, future catalysts, what the market has priced in, and mid/long-term positioning. Use WHENEVER the user wants a deep/comprehensive analysis of one or more US stocks (esp. Mag7: AAPL MSFT GOOGL AMZN NVDA META TSLA, or SPY), or says things like "深度分析/全面分析/读新闻分析/作战简报/帮我看看 <ticker>", "深度 brief", "research <ticker> and tell me where to buy", "why is <ticker> down and should I add". Triggers on a ticker plus any of: 深度/全面/读新闻/作战/简报/分析/deep/brief/research. This is the conversational counterpart to the in-tool briefing page: the tool gives calibrated numbers; this skill adds full-text news reading + reasoning that the standalone app cannot do.
---

# Quant Deep Brief — 深度作战简报（联网读正文 + 接地气推理）

把 quant-lab 工具的**校准数字** 与 **全网新闻正文** 融合，产出用户范例那种分层深度分析。
铁律：**数字来自工具、叙事来自原文、推断标注清楚**；不预测点位、不编造数字、不把新闻当量化信号。

## 工作目录
量化工具在 `H:/量化工具`（用其 venv：`H:/量化工具/.venv/Scripts/python`）。

## 步骤（每个标的都走一遍；多标的先各自做再综合）

### 1. 取工具的真实数字（接地气，必做第一步）
对每个 ticker 运行，拿到引擎桶/价位档/财报drift/新闻清单：
```bash
cd "H:/量化工具" && .venv/Scripts/python -m analysis.brief_cli <TICKER> --horizon 63 --broad
```
输出 JSON 含：`engine_state_bucket`/`engine_value_bucket`（中位/超额/胜率/N/CI/显著性/盈亏比）、
`momentum_trap`、`entry_tranches`（浅/中/重：价位/共振技术位/目标/止损/RR）、`next_earnings`、
`earnings_reaction`（财报日波动/财报前 drift/财报后分超预期漂移）、`financial_highlights`
（营收/净利同比、分析师目标）、`news_to_read`（标题 + URL 列表）、`news_reason_heuristic`。
**所有价位、超额、盈亏比、胜率、财报数字只能用这里的，不许自己编。**

### 2. 全网读正文（这是 skill 相对工具的增量）
- 用 `WebSearch` 检索每个 ticker 的最新动态：`"<公司名> stock news"`、`"<公司名> earnings"`、
  `"<公司名> guidance / lawsuit / AI / capex"` 等，覆盖多角度。
- 用 `WebFetch` **读 5–10 篇正文全文**：优先 `news_to_read` 里的 URL + 搜索结果里的权威源
  （公司 IR/8-K、Reuters/Bloomberg/WSJ/CNBC/SeekingAlpha 等）。提取：已发生的硬事件
  （营收/利润/订单/产品指标/融资/回购/诉讼/高管变动）、前瞻指引、分析师观点、争议点。
- 也可读工具抽取的关键句：`.venv/Scripts/python -c "import json;from data import news as n;..."` 或直接 WebFetch。

### 3. 合成分层简报（每票）
按以下结构写，**严格区分三类来源**（标注：📊工具 / 📰原文+URL / 🧠我的推断）：
- **一句话定性**：风险调整后这票是什么角色（强势+显著优势 / 深跌价值但接刀 / 稳健底仓 / 动量陷阱…）。
- **为什么是这些价位**（📊）：直接用 `entry_tranches` 的浅/中/重 价位 + 共振技术位 + 目标/止损/RR +
  引擎桶（中位/超额/胜率/CI/显著性）。若 `momentum_trap=true`，**必须点明「逢跌买无统计优势，等确认」**。
- **已发生事件**（📰，每条带来源链接）：从正文读到的营收/利润/产品/融资/诉讼等硬事实。
- **未来催化**：`next_earnings`（📊）+ 原文里的政策/产品/诉讼时间点（📰）。
- **市场提前消化**：结合 `earnings_reaction.pre_drift`（📊，量化的财报前 drift）+ 原文情绪（📰）+
  你的判断（🧠：这波回撤是情绪/技术性 还是 基本面破裂？引擎超额是否支持）。
- **中长期布局**：若多票，运行多次 CLI 后用引擎超额×显著性×反波动率给权重梯队与角色，并说明打法
  （强势票=支撑分批接情绪超调；被打趴票=用财报当扳机赚预期修复）。

### 4. 多票综合（用户给多只时）
出一张「一屏总览」表（现价/趋势·距前高/波动分位/引擎最优桶+超额+显著性✅/盈亏比/胜率/下次财报，
全来自 CLI）+ 权重梯队 + 财报时间轴节奏 + 风控铁律（同赛道高相关别同时满仓、高波动票仓位小、
财报隔夜 gap 风险、结构性变化是表外尾部风险引擎吃不进）。

## 铁律（违背则失败）
1. **价位/超额/盈亏比/胜率/财报数字一律取自 brief_cli**，不得自创或四舍五入到“好看”。
2. 价位是**工具按技术位+历史条件分布推导的校准参考**，措辞用「区间/支撑/分批位」，**不要写「目标价 X、上涨概率 Y%」当预测**。
3. **叙事必须给来源 URL**；读不到正文就说“未读到正文，仅据标题”，不要脑补。
4. `momentum_trap=true` 的票，不得渲染成“逢跌好买点”。
5. 明确分层标注 📊工具 / 📰原文 / 🧠推断；结尾声明“研究校准用途，非投资建议”。
6. 回复语言跟随用户（中文优先）。

## 触发示例
- “深度分析 NVDA” / “帮我读读 GOOGL 最近的新闻再分析” / “GOOGL NVDA MSFT 作战简报”
- “英伟达为什么跌，现在能不能加” / “deep brief on TSLA”
