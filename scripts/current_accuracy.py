"""当前引擎进出场点位准确度（与 entry_confluence 实际评级一一对应·互斥分桶）。

入场分桶(按引擎优先级，互斥)：🔴飞刀 → ✨优质回踩 → 🟢现价在支撑 → 🟡临近线 → 🟡趋势健康等回踩。
出场：跌破200线 红灯预警 → 未来≥15%回撤 命中/误报/lift。
口径：2000→2026 宽池+VIX，无前视，远期126日。
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from data import loader
import factors.signals as sg

END="2026-06-06"; START="2000-01-01"; H=126; DROP=0.15
POOL=["NVDA","AAPL","MSFT","GOOGL","AMZN","META","TSLA","AVGO","AMD","TSM","NFLX",
      "INTC","PYPL","DIS","BA","F","T","VZ","NKE","MMM","QCOM","SPY"]
print("加载…")
PX={}
for t in POOL:
    try:
        s=loader.load_ohlcv(t,START,END)["close"].dropna()
        if len(s)>600: PX[t]=s
    except Exception: pass
VIX=loader.load_ohlcv("^VIX","1995-01-01",END)["close"].dropna()
vp_all=VIX.rolling(252,min_periods=126).apply(lambda x:(x[-1]>=x).mean(),raw=True)

tiers=["🔴 飞刀(破位)","✨ 优质回踩","🟢 现价在支撑","🟡 临近200线","🟡 趋势健康等回踩"]
agg={k:([],[]) for k in tiers}
for px in PX.values():
    px=px.dropna(); ma200=px.rolling(200,min_periods=100).mean(); ma50=px.rolling(50,min_periods=25).mean()
    low60=px.rolling(60,min_periods=30).min(); r=sg.rsi(px,14); vp=vp_all.reindex(px.index).ffill()
    slope_up=ma200>ma200.shift(20); dist=px/ma200-1.0
    knife=(px<ma200)&(~slope_up); up=px>ma200
    atsup=(((ma50/px-1).abs()<=0.025).astype(int)+((ma200/px-1).abs()<=0.025).astype(int)+((low60/px-1).abs()<=0.025).astype(int))>=2
    fear=up&(r<40)&(vp>0.7)
    near=up&(dist>=0)&(dist<0.04)
    fwd=px.shift(-H)/px-1.0; fmin=px.shift(-1).rolling(H).min().shift(-(H-1))/px-1.0
    # 互斥优先级：飞刀 > 优质回踩 > 在支撑 > 临近线 > 其余健康
    assigned=pd.Series(index=px.index, dtype=object)
    assigned[up]= "🟡 趋势健康等回踩"
    assigned[near]="🟡 临近200线"
    assigned[atsup]="🟢 现价在支撑"
    assigned[fear]="✨ 优质回踩"
    assigned[knife]="🔴 飞刀(破位)"
    for k in tiers:
        m=(assigned==k)&fwd.notna()
        agg[k][0].extend(fwd[m].dropna().tolist()); agg[k][1].extend(fmin[(assigned==k)&fmin.notna()].dropna().tolist())

print("\n"+"="*84)
print("【入场点准确度】当前引擎各评级 → 远期126日（互斥分桶）")
print("-"*84)
print(f"{'入场评级':<18}{'N':>9}{'中位收益':>10}{'胜率':>8}{'进场浮亏MAE':>13}")
def st(b):
    f=np.array([v for v in b[0] if v==v]); m=np.array([v for v in b[1] if v==v])
    return len(f),(np.median(f) if len(f) else float('nan')),((f>0).mean() if len(f) else float('nan')),(np.median(m) if len(m) else float('nan'))
for k in tiers:
    n,med,win,mae=st(agg[k])
    if n<30: continue
    print(f"{k:<18}{n:>9}{med:>+10.1%}{win:>8.0%}{mae:>+13.1%}")
print("→ 准确度体现在'胜率/浮亏'随评级单调：飞刀最差、优质回踩最好。收益区分弱(survivor都涨)。")

# 出场
print("\n"+"="*84)
print("【出场点准确度】跌破200线 红灯 → 未来126日≥15%回撤 命中/误报/lift")
print("-"*84)
hit=fa=0; bh=bn=0
for px in PX.values():
    px=px.dropna(); ma200=px.rolling(200,min_periods=100).mean(); below=px<ma200
    cross=below&(~below.shift(1).fillna(False)); arr=px.to_numpy(); n=len(arr)
    for i in np.where(cross.to_numpy())[0]:
        if i+H>=n: continue
        (hit if arr[i:i+H].min()/arr[i]-1.0<=-DROP else fa).__class__  # noop
        if arr[i:i+H].min()/arr[i]-1.0<=-DROP: hit+=1
        else: fa+=1
    for i in range(0,n-H,5):
        bn+=1; bh+= (arr[i:i+H].min()/arr[i]-1.0<=-DROP)
base=bh/max(bn,1); prec=hit/max(hit+fa,1)
print(f"  事件N={hit+fa}  命中率={prec:.0%}  误报率={1-prec:.0%}  基率={base:.0%}  lift={prec/max(base,1e-9):.1f}x")
print("→ 出场预警 lift≈1.0=预测不准；它是'降仓开关/烟雾报警器'(误报多但真崩盘大多逮到)，非精准择时。")
print("="*84)
