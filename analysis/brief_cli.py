"""CLI：输出单票「可计算简报」JSON——供 Claude 的 quant-deep-brief skill 接地气用。

skill 用本命令拿到**工具算出的真实数字**(引擎桶/价位档/目标止损RR/财报drift/新闻清单)，
再联网读新闻正文写深度叙事，确保数字不是编的、价位是工具规则推导的。

用法:
    .venv/Scripts/python -m analysis.brief_cli NVDA --horizon 63 [--broad] [--no-news]
"""
from __future__ import annotations

import argparse
import json
import warnings

warnings.filterwarnings("ignore")


def _bucket(x):
    if not x:
        return None
    return {k: x.get(k) for k in ("bucket", "median", "excess", "win_rate", "n_events",
                                  "reward_risk", "ci_low", "ci_high", "significant")}


def build(ticker: str, horizon: int = 63, broad: bool = False, with_news: bool = True) -> dict:
    from analysis import briefing as bf

    start = "1995-01-01" if ticker.upper() == "SPY" else "2008-01-01"
    sources = ("google", "yahoo", "gdelt") if broad else ("google", "yahoo")
    b = bf.stock_brief(ticker.upper(), start, None, horizon=horizon,
                       with_news=with_news, news_sources=sources)

    news = b.get("news")
    news_list = []
    if news is not None and not news.empty:
        for _, r in news.head(10).iterrows():
            news_list.append({"date": str(r.get("date")), "title": r.get("title"),
                              "provider": r.get("provider"), "url": r.get("url")})

    # 最佳入场区 + 锚点价（校准式，含置信分层）
    best_entry = None
    try:
        from data import loader as _ld
        from regime import entry_cockpit as _ec
        _px = _ld.load_prices([ticker.upper()], start, None)[ticker.upper()]
        # 跨持有期择优：自动挑置信最高的周期（而非固定 horizon），避免长周期低置信埋没好结果
        _bez = _ec.best_entry_across_horizons(_px, asset=ticker.upper(),
                                              single_name=(ticker.upper() != "SPY"))
        if _bez.get("has_zone"):
            best_entry = {
                "zone": _bez["zone_label"], "best_horizon_days": _bez.get("horizon"),
                "anchor_price": round(_bez["anchor_price"], 2),
                "price_band": [None if _bez["price_band"][0] is None else round(_bez["price_band"][0], 2),
                               round(_bez["price_band"][1], 2)],
                "anchor_distance_pct": (round(_bez["anchor_distance"], 4)
                                        if _bez.get("anchor_distance") == _bez.get("anchor_distance") else None),
                "median_fwd": round(_bez["median_fwd"], 4), "excess_median": round(_bez["excess_median"], 4),
                "reward_risk": (round(_bez["reward_risk"], 2) if _bez["reward_risk"] == _bez["reward_risk"] else None),
                "win_rate": round(_bez["win_rate"], 3), "n_events": _bez["n_events"],
                "ci": [round(_bez["ci"][0], 4), round(_bez["ci"][1], 4)],
                "tier": _bez["tier"], "open_ended": _bez.get("open_ended", False),
                "caveats": _bez["caveats"],
            }
        else:
            best_entry = {"has_zone": False, "verdict": _bez["verdict"], "tier": "防守"}
    except Exception:  # noqa: BLE001 —— 取价/历史不足时不阻断整份简报
        best_entry = None

    out = {
        "ticker": b["ticker"], "as_of": b["date"], "horizon_days": horizon,
        "price": b["price"], "trailing_high": b["trailing_high"],
        "trend": b["trend"], "trend_position": b["trend_position"],
        "drawdown_from_high": b["drawdown"], "vol_percentile": b["vol_percentile"],
        "current_state_bucket": b.get("current_state_bucket"),
        "engine_state_bucket": _bucket(b.get("engine_state")),
        "engine_value_bucket": _bucket(b.get("engine_value")),
        "momentum_trap": b.get("momentum_trap"),
        "best_entry_zone": best_entry,
        "entry_tranches": [
            {"tier": t["tier"], "price": round(t["price"], 2), "confluence": t["what"],
             "target": round(t["target"], 2), "stop": round(t["stop"], 2),
             "reward_risk": (round(t["rr"], 2) if t["rr"] == t["rr"] else None),
             "to_target_pct": round(t["to_target_pct"], 4), "to_stop_pct": round(t["to_stop_pct"], 4),
             "engine_win_rate": t["engine_win_rate"]}
            for t in b.get("tranches", [])
        ],
        "next_earnings": b.get("next_earnings"), "days_to_earnings": b.get("days_to_earnings"),
        "earnings_reaction": b.get("earnings_stats"),
        "financial_highlights": {k: v for k, v in (b.get("highlights") or {}).items() if not k.startswith("_")},
        "fundamentals_snapshot": {k: v for k, v in (b.get("fundamentals") or {}).items() if not k.startswith("_")},
        "news_tone": (b.get("news_analysis") or {}).get("tone_label"),
        "news_themes": [t for t, _ in (b.get("news_analysis") or {}).get("themes", [])],
        "news_reason_heuristic": b.get("news_reason"),
        "news_to_read": news_list,
        "operation_playbook": __import__("analysis.playbook", fromlist=["build_playbook"]).build_playbook(b),
        "_disclaimer": ("引擎数字/价位档=价格与历史条件分布(校准非预测)；价位档目标/止损=技术位规则推导的风险参考；"
                        "财报要点/新闻=仅供人读、不入量化、含前视。"),
    }
    return out


def main():
    ap = argparse.ArgumentParser(description="quant-lab 可计算简报 JSON（供 skill 接地气）")
    ap.add_argument("ticker")
    ap.add_argument("--horizon", type=int, default=63)
    ap.add_argument("--broad", action="store_true", help="新闻并入 GDELT 全球源")
    ap.add_argument("--no-news", action="store_true", help="跳过新闻(更快)")
    a = ap.parse_args()
    out = build(a.ticker, horizon=a.horizon, broad=a.broad, with_news=not a.no_news)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
