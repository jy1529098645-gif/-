"""验证'合理入场位'核心主张：在**多技术支撑共振**处(且趋势健康)分批，是否真有正向边际？

历史事件：趋势健康(价>MA200)日，数当前价 ±tol 内有几个**不同类**技术支撑共振
(MA50 / MA200 / 近60日低 / 近120日低)。按共振数分组，比其 H 日远期收益 vs 全体(无条件)。
若"共振≥2"组中位远期收益/胜率 > "共振0"组与无条件 → 合理入场位的'技术共振'口径站得住。
无前视：t 日只用 ≤t 信息；远期收益用 t→t+H。
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from data import loader

END = "2026-06-06"; START = "2008-01-01"; TOL = 0.025
POOL = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "AMD", "TSM",
        "INTC", "DIS", "QCOM", "NKE", "SPY"]
HORIZONS = [63, 126, 252]


def events(px):
    px = px.dropna()
    ma50 = px.rolling(50, min_periods=25).mean()
    ma200 = px.rolling(200, min_periods=100).mean()
    low60 = px.rolling(60, min_periods=30).min()
    low120 = px.rolling(120, min_periods=60).min()
    uptrend = px > ma200
    # 各支撑是否在 ±tol 内
    sup = pd.DataFrame({
        "MA50": (ma50 / px - 1.0).abs() <= TOL,
        "MA200": (ma200 / px - 1.0).abs() <= TOL,
        "low60": (low60 / px - 1.0).abs() <= TOL,
        "low120": (low120 / px - 1.0).abs() <= TOL,
    })
    conf = sup.sum(axis=1)
    return uptrend, conf


rows = {h: {0: [], 1: [], "2+": [], "all": []} for h in HORIZONS}
maes = {0: [], 1: [], "2+": []}
for tk in POOL:
    try:
        px = loader.load_ohlcv(tk, START, END)["close"].dropna()
        if len(px) < 504:
            continue
        up, conf = events(px)
        for h in HORIZONS:
            fwd = px.shift(-h) / px - 1.0
            valid = up & fwd.notna()
            rows[h]["all"].extend(fwd[valid].tolist())
            for c, key in [(0, 0), (1, 1)]:
                m = valid & (conf == c)
                rows[h][key].extend(fwd[m].tolist())
            m2 = valid & (conf >= 2)
            rows[h]["2+"].extend(fwd[m2].tolist())
        # MAE：进场后 126 日内最深浮亏（只算 horizon=126）
        H = 126
        arr = px.to_numpy()
        for i in range(len(arr) - H):
            if not (up.iloc[i]):
                continue
            c = conf.iloc[i]
            seg = arr[i:i + H]
            mae = seg.min() / arr[i] - 1.0
            key = 0 if c == 0 else (1 if c == 1 else "2+")
            maes[key].append(mae)
    except Exception as e:
        print(f"  {tk} 跳过 {type(e).__name__}")


def stat(x):
    a = np.array([v for v in x if v == v])
    if len(a) < 30:
        return None
    return dict(n=len(a), med=float(np.median(a)), win=float((a > 0).mean()),
                p25=float(np.percentile(a, 25)))


print("=" * 92)
print(f"合理入场位验证 · 技术共振 → 远期收益（趋势健康日 · tol±{TOL:.1%} · 2008→2026 · {len(POOL)}标的）")
print("=" * 92)
for h in HORIZONS:
    print(f"\n# 远期 {h} 日")
    base = stat(rows[h]["all"])
    print(f"{'共振数':<10}{'N':>8}{'中位远期':>10}{'胜率':>8}{'25分位':>9}{'vs无条件中位':>13}")
    print(f"{'无条件(全体)':<10}{base['n']:>8}{base['med']:>+10.1%}{base['win']:>8.0%}{base['p25']:>+9.1%}{'—':>13}")
    for key, lab in [(0, "共振 0"), (1, "共振 1"), ("2+", "共振 ≥2")]:
        s = stat(rows[h][key])
        if not s:
            print(f"{lab:<10}{'样本不足':>8}"); continue
        ex = s["med"] - base["med"]
        print(f"{lab:<10}{s['n']:>8}{s['med']:>+10.1%}{s['win']:>8.0%}{s['p25']:>+9.1%}{ex:>+13.1%}")

print("\n# 进场后 126 日最深浮亏(MAE) 中位 —— 共振越多，进场点是否越'抗砸'？")
for key, lab in [(0, "共振 0"), (1, "共振 1"), ("2+", "共振 ≥2")]:
    a = np.array(maes[key])
    if len(a) >= 30:
        print(f"  {lab:<8} N={len(a):>6} MAE中位 {np.median(a):+.1%}")
print("\n读法：若 共振≥2 的中位远期>无条件且 MAE 更浅 → '多支撑共振处分批'有正边际、是合理入场位。")
print("=" * 92)
