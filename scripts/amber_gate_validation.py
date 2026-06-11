"""回测决定：exit_warning 各"黄灯"条件对**新建仓**的真实影响——该"暂停"还是只需"小仓"？

口径：趋势健康日(价>MA200)进场，按每个黄灯条件是否触发分组，比 H 日远期收益 + 进场后 MAE。
若某条件触发后 远期明显更差/MAE 明显更深 → 该"暂停建仓"；若几乎无差 → 只需"小仓"即可。
黄灯条件(单票可算)：① 临近200线(0≤距<4%) ② 波动飙升(21日已实现波动>90分位) ③ 高位拉伸(乖离>90分位且距200>5%)。
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from data import loader

END = "2026-06-06"; START = "2008-01-01"
POOL = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "AMD", "TSM",
        "INTC", "DIS", "QCOM", "NKE", "JPM", "JNJ", "PG", "XOM", "SPY", "HD"]
HS = [63, 126, 252]


def signals(px):
    px = px.dropna()
    ma200 = px.rolling(200, min_periods=100).mean()
    dist = px / ma200 - 1.0
    up = px > ma200
    rv = px.pct_change().rolling(21, min_periods=10).std() * np.sqrt(252)
    volp = rv.rolling(252, min_periods=120).apply(lambda w: (w[-1] >= w).mean(), raw=True)
    tp = (px / ma200 - 1.0)
    overextp = tp.rolling(756, min_periods=200).apply(lambda w: (w[-1] >= w).mean(), raw=True)
    return dict(
        up=up,
        near200=up & (dist >= 0) & (dist < 0.04),
        volspike=up & (volp > 0.90),
        overext=up & (overextp > 0.90) & (dist > 0.05),
    )


acc = {k: {h: [] for h in HS} for k in ["up", "near200", "volspike", "overext"]}
mae = {k: [] for k in ["up", "near200", "volspike", "overext"]}
for tk in POOL:
    try:
        px = loader.load_ohlcv(tk, START, END)["close"].dropna()
        if len(px) < 600:
            continue
        sig = signals(px)
        arr = px.to_numpy()
        for h in HS:
            fwd = px.shift(-h) / px - 1.0
            for k in acc:
                m = sig[k] & fwd.notna()
                acc[k][h].extend(fwd[m].tolist())
        H = 126
        for k in mae:
            idx = np.where(sig[k].to_numpy())[0]
            for i in idx:
                if i + H >= len(arr):
                    continue
                mae[k].append(arr[i:i + H].min() / arr[i] - 1.0)
    except Exception as e:
        print(f"  {tk} 跳过 {type(e).__name__}")


def med(x):
    a = np.array([v for v in x if v == v])
    return (len(a), float(np.median(a)) if len(a) else float("nan"),
            float((a > 0).mean()) if len(a) else float("nan"))


print("=" * 96)
print("黄灯条件 → 新建仓 远期收益 / MAE（趋势健康日 · 2008→2026 · 20标的）")
print("=" * 96)
print(f"{'条件':<14}{'N':>8}{'63日中位':>10}{'126日中位':>11}{'252日中位':>11}{'126日胜率':>11}{'进场MAE中位':>12}")
labels = {"up": "趋势健康(基准)", "near200": "①临近200线", "volspike": "②波动飙升", "overext": "③高位拉伸"}
base = {h: med(acc["up"][h])[1] for h in HS}
for k in ["up", "near200", "volspike", "overext"]:
    n63, m63, _ = med(acc[k][63]); _, m126, w126 = med(acc[k][126]); _, m252, _ = med(acc[k][252])
    nm, mmae, _ = med(mae[k])
    tag = ""
    if k != "up":
        # 判定：126日远期比基准差 >3% 或 MAE 比基准深 >2% → 建议"暂停"，否则"小仓"
        d126 = m126 - base[126]; dmae = mmae - med(mae["up"])[1]
        tag = "  → 暂停" if (d126 < -0.03 or dmae < -0.02) else "  → 小仓即可"
    print(f"{labels[k]:<14}{n63:>8}{m63:>+10.1%}{m126:>+11.1%}{m252:>+11.1%}{w126:>11.0%}{mmae:>+12.1%}{tag}")
print("\n读法：基准=所有趋势健康日。某黄灯若 远期明显更差 或 进场MAE明显更深 → 暂停建仓；否则小仓即可。")
print("=" * 96)
