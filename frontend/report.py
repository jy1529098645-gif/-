"""一键导出规则 HTML 报告（reports/{rule}/index.html）。

汇总：一句话裁决 + 池化统计（N/N_eff/CI/基准） + 图（收益分布/MAE/水下/相关）。
强制显示样本数与置信区间，不含目标价/买卖点。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from reports import plots


def _png_b64(path: str) -> str:
    import base64
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def export_rule_report(name: str, res: dict, specs, op: str, exit_spec: dict) -> str:
    """生成 PNG 图 + 自包含 index.html，返回 html 路径。"""
    safe = name.replace("/", "_").replace(" ", "_")
    rule_dir = Path(config.get_path("reports")) / safe
    rule_dir.mkdir(parents=True, exist_ok=True)

    trades = res["trades"]
    p = res["pooled"]

    # 图（matplotlib 落盘）
    imgs = []
    imgs.append(plots.return_hist(trades, safe, n_eff=p["n_eff"], baseline_median=p["baseline_median"], title=name))
    imgs.append(plots.mae_hist(trades, safe, n_eff=p["n_eff"], title=name))
    try:
        from data import loader
        tickers = loader.load_universe("mag7")
        rets = pd.DataFrame({t: loader.load_prices([t])[t].pct_change() for t in tickers})
        imgs.append(plots.correlation_heatmap(rets, safe, avg_corr=p["rho_bar"]))
    except Exception:
        pass

    rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in {
            "交易笔数 N": p["n_trades"],
            "有效独立样本 N_eff": f"{p['n_eff']:.1f}（ρ̄={p['rho_bar']:.2f}）",
            "胜率": f"{p['win_rate']:.0%}",
            "收益中位": f"{p['median_return']:+.1%}",
            "5 分位（最差区）": f"{p['p5_return']:+.1%}",
            "MAE 中位": f"{p['median_mae']:+.1%}",
            "最长连亏链": p["longest_losing_streak"],
            "随机基准中位": f"{p['baseline_median']:+.1%}",
            "超额（vs 基准）": f"{p['excess_median']:+.1%}　95% CI [{p['excess_ci_low']:+.1%}, {p['excess_ci_high']:+.1%}]",
            "是否显著": "是" if p["excess_significant"] else "否（不显著，非择时圣杯）",
        }.items()
    )
    img_html = "".join(f'<img src="{_png_b64(i)}" style="width:100%;max-width:820px;margin:12px 0;border-radius:10px"/>' for i in imgs)

    join_op = f" {op.upper()} "
    sig_desc = join_op.join(f"{s[0]}({s[1]},{s[2]})" for s in specs)

    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>规则报告 · {name}</title>
<style>
body{{background:#0B0E14;color:#E6E9EF;font-family:'Segoe UI',sans-serif;max-width:880px;margin:0 auto;padding:32px}}
h1{{background:linear-gradient(92deg,#7C5CFC,#00D4FF);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.verdict{{border-left:4px solid #7C5CFC;background:rgba(124,92,252,0.1);padding:14px 18px;border-radius:8px;line-height:1.7}}
table{{border-collapse:collapse;width:100%;margin:18px 0}}
td{{border-bottom:1px solid rgba(255,255,255,0.1);padding:8px 10px}}
td:first-child{{color:#8A93A6;width:42%}}
.foot{{color:#8A93A6;font-size:.85rem;margin-top:24px}}
</style></head><body>
<h1>规则报告 · {name}</h1>
<p>入场信号：{sig_desc}　|　出场：{exit_spec}</p>
<div class="verdict">{res.get('_verdict','')}</div>
<h3>池化统计（基于 N_eff，含 95% 置信区间）</h3>
<table>{rows}</table>
<h3>图表</h3>
{img_html}
<p class="foot">本报告为「决策与验证」用途：只给历史胜率与收益分布 + 置信区间，<b>不含目标价/买卖点</b>。
样本：七姐妹池化；CI 用 block bootstrap 并按 N_eff 折算。</p>
</body></html>"""

    out = rule_dir / "index.html"
    out.write_text(html, encoding="utf-8")
    return str(out)
