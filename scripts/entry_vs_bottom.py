"""工具入场点 vs 真实抄底位：买在信号处，比之后最低点高出多少%（诚实量化'离底有多远'）。

口径：2000→2026 宽池+VIX，无前视。
  · 各入场信号当天买入(=现价) → 比未来 63/126 日最低点 高出 entry/min-1（%，越小越接近底）。
  · 另算"分批回踩"：把买单分散到现价下方最近3个技术支撑(回踩到才成交)，看平均成本离底多远(更省)。
  · 同时报：买完后还会不会更低(命中率) / 买在底部±3%内的概率。
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from data import loader
import factors.signals as sg

END="2026-06-06"; START="2000-01-01"
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

def signals(px):
    px=px.dropna(); ma200=px.rolling(200,min_periods=100).mean(); ma50=px.rolling(50,min_periods=25).mean()
    low60=px.rolling(60,min_periods=30).min(); r=sg.rsi(px,14); vp=vp_all.reindex(px.index).ffill()
    slope_up=ma200>ma200.shift(20); up=px>ma200; knife=(px<ma200)&(~slope_up)
    atsup=(((ma50/px-1).abs()<=0.025).astype(int)+((ma200/px-1).abs()<=0.025).astype(int)+((low60/px-1).abs()<=0.025).astype(int))>=2
    fear=up&(r<40)&(vp>0.7)
    return {"✨优质回踩":fear, "🟢现价在支撑":atsup&up, "🟡趋势健康(泛)":up, "🔴飞刀(破位)":knife}

for W in (63, 126):
    print("\n"+"="*86)
    print(f"【买在信号当天】比未来 {W} 日最低点 高出多少% （中位/分布 · 越小越接近抄底）")
    print("-"*86)
    print(f"{'入场信号':<16}{'N':>9}{'离底中位':>10}{'离底p25':>9}{'离底p75':>9}{'买后还更低概率':>14}{'买在底±3%内':>12}")
    acc={}
    for px in PX.values():
        px=px.dropna(); sig=signals(px); arr=px.to_numpy(); n=len(arr)
        fmin=pd.Series(arr,index=px.index).shift(-1).rolling(W).min().shift(-(W-1))
        gap=(px/fmin-1.0)  # >0 = 买价高于未来最低
        for k,s in sig.items():
            m=s&gap.notna()
            acc.setdefault(k,[]).extend(gap[m].dropna().tolist())
    for k in ["✨优质回踩","🟢现价在支撑","🟡趋势健康(泛)","🔴飞刀(破位)"]:
        g=np.array([v for v in acc.get(k,[]) if v==v])
        if len(g)<30: continue
        print(f"{k:<16}{len(g):>9}{np.median(g):>+10.1%}{np.percentile(g,25):>+9.1%}{np.percentile(g,75):>+9.1%}"
              f"{(g>0.003).mean():>14.0%}{(g<=0.03).mean():>12.0%}")

# 分批回踩：把买单挂到现价下方最近3支撑，回踩到才成交，看平均成本离底多远
print("\n"+"="*86)
print("【分批回踩 vs 一次性买现价】平均成本 比未来126日最低点 高出多少%（趋势健康日触发）")
print("-"*86)
lump, ladder = [], []
for px in PX.values():
    px=px.dropna(); ma200=px.rolling(200,min_periods=100).mean(); ma50=px.rolling(50,min_periods=25).mean()
    low60=px.rolling(60,min_periods=30).min(); low120=px.rolling(120,min_periods=60).min()
    up=px>ma200; arr=px.to_numpy(); idx=px.index; n=len(arr)
    for i in np.where(up.to_numpy())[0]:
        if i+126>=n: continue
        cur=arr[i]; seg=arr[i:i+126]; bot=seg.min()
        # 现价下方最近3支撑(>0且<现价)
        sup=sorted([v for v in (ma50.iloc[i],ma200.iloc[i],low60.iloc[i],low120.iloc[i])
                    if v==v and 0<v<cur], reverse=True)[:3]
        fills=[cur]  # 第一批买现价
        for lv in sup:
            if seg.min()<=lv:   # 回踩到该支撑才成交
                fills.append(lv)
        avg=np.mean(fills)
        lump.append(cur/bot-1.0); ladder.append(avg/bot-1.0)
print(f"  一次性买现价   : 平均成本离底中位 {np.median(lump):+.1%}")
print(f"  分批回踩(现价+下方支撑): 平均成本离底中位 {np.median(ladder):+.1%}  → 比一次性省 {np.median(lump)-np.median(ladder):+.1%}")
print("="*86)
print("\n诚实读法：'离底中位'就是工具入场点平均比抄底位高出的%。它做不到精确抄底(那不存在)，")
print("但'在支撑/优质回踩'信号能把这个差距控制在个位数，分批回踩还能再省几个点。")
