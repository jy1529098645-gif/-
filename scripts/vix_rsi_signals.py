"""VIX(恐慌) + RSI(超买超卖) 信号检验，并与'趋势门+技术支撑'组合 → 找更准的进出场判断。

四段：
  1. VIX 单独：按 VIX 历史分位分桶 → 远期收益/胜率/浮亏（逆向：恐慌高时买更好吗？）。
  2. RSI 单独：按 RSI(14) 分桶 → 远期收益/胜率/浮亏（超卖买/超买卖 有区分力吗？）。
  3. 入场组合：趋势门 ×(RSI回踩 / 在支撑 / VIX高) 各组合的 胜率+浮亏(MAE)+远期收益 → 谁最准。
  4. 出场组合：跌破200线 / RSI超买拉伸 / VIX突刺 → 未来≥15%回撤的 命中率/误报/lift → 谁预警更准。
口径：2000→2026 宽池 + SPY，无前视(分位用滚动窗口、信号用≤t)。
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from data import loader
import factors.signals as sg

END = "2026-06-06"; START = "2000-01-01"
POOL = ["NVDA","AAPL","MSFT","GOOGL","AMZN","META","TSLA","AVGO","AMD","TSM","NFLX",
        "INTC","PYPL","DIS","BA","F","T","VZ","NKE","MMM","QCOM","SPY"]
print("加载宽池 + VIX…")
PX = {}
for t in POOL:
    try:
        s = loader.load_ohlcv(t, START, END)["close"].dropna()
        if len(s) > 600: PX[t] = s
    except Exception: pass
VIX = loader.load_ohlcv("^VIX", "1995-01-01", END)["close"].dropna()
print(f"  {len(PX)} 标的 + VIX\n")

def sec(t): print("="*96 + f"\n{t}\n" + "-"*96)
def roll_pctile(s, w=252):
    return s.rolling(w, min_periods=w//2).apply(lambda x: (x[-1] >= x).mean(), raw=True)
def stat(fwd, mae):
    f=np.array([v for v in fwd if v==v]); m=np.array([v for v in mae if v==v])
    if len(f)<50: return None
    return len(f), np.median(f), (f>0).mean(), (np.median(m) if len(m) else float('nan'))

# ── 1. VIX 单独 ──
sec("【1】VIX 单独：按 VIX 历史分位(滚动252) 分桶 → 标的远期126日 收益/胜率/浮亏（逆向买恐慌?）")
vixp = roll_pctile(VIX, 252)
buckets = {"极低<20%":(0,0.2),"低20-40":(0.2,0.4),"中40-60":(0.4,0.6),"高60-80":(0.6,0.8),"极高>80%":(0.8,1.01)}
res = {k:([],[]) for k in buckets}
for px in PX.values():
    px=px.dropna(); vp=vixp.reindex(px.index).ffill()
    fwd=px.shift(-126)/px-1.0; fmin=px.shift(-1).rolling(126).min().shift(-125)/px-1.0
    for k,(lo,hi) in buckets.items():
        m=(vp>=lo)&(vp<hi)&fwd.notna()
        res[k][0].extend(fwd[m].dropna().tolist()); res[k][1].extend(fmin[(vp>=lo)&(vp<hi)&fmin.notna()].dropna().tolist())
print(f"  {'VIX分位桶':<12}{'N':>8}{'中位收益':>10}{'胜率':>8}{'进场浮亏':>10}")
for k in buckets:
    s=stat(*res[k])
    if s: print(f"  {k:<12}{s[0]:>8}{s[1]:>+10.1%}{s[2]:>8.0%}{s[3]:>+10.1%}")
print("  → 若'极高VIX'桶远期收益/胜率明显更好=恐慌时买有逆向区分力。\n")

# ── 2. RSI 单独 ──
sec("【2】RSI(14) 单独：分桶 → 远期126日 收益/胜率/浮亏（超卖买/超买卖 有用吗?）")
rb = {"超卖<30":(0,30),"低30-45":(30,45),"中45-55":(45,55),"高55-70":(55,70),"超买>70":(70,101)}
rres = {k:([],[]) for k in rb}
for px in PX.values():
    px=px.dropna(); r=sg.rsi(px,14)
    fwd=px.shift(-126)/px-1.0; fmin=px.shift(-1).rolling(126).min().shift(-125)/px-1.0
    for k,(lo,hi) in rb.items():
        m=(r>=lo)&(r<hi)&fwd.notna()
        rres[k][0].extend(fwd[m].dropna().tolist()); rres[k][1].extend(fmin[(r>=lo)&(r<hi)&fmin.notna()].dropna().tolist())
print(f"  {'RSI桶':<12}{'N':>8}{'中位收益':>10}{'胜率':>8}{'进场浮亏':>10}")
for k in rb:
    s=stat(*rres[k])
    if s: print(f"  {k:<12}{s[0]:>8}{s[1]:>+10.1%}{s[2]:>8.0%}{s[3]:>+10.1%}")
print("  → 看 超卖桶 是否远期更好(逆向)、超买桶是否短期差(择时)。\n")

# ── 3. 入场组合：找最准的入场条件 ──
sec("【3】入场组合：趋势门 ×(RSI回踩/在支撑/VIX高) → 谁的 胜率高+浮亏浅(=入场最准)  · 远期126日")
def feats(px):
    px=px.dropna(); ma200=px.rolling(200,min_periods=100).mean(); ma50=px.rolling(50,min_periods=25).mean()
    low60=px.rolling(60,min_periods=30).min(); r=sg.rsi(px,14)
    up=px>ma200
    atsup=(((ma50/px-1).abs()<=0.025).astype(int)+((ma200/px-1).abs()<=0.025).astype(int)+((low60/px-1).abs()<=0.025).astype(int))>=2
    return px,up,r,atsup
vp_all=roll_pctile(VIX,252)
conds = {
 "A 趋势健康(基线)": lambda up,r,atsup,vp: up,
 "B 健康+RSI回踩<40": lambda up,r,atsup,vp: up&(r<40),
 "C 健康+在支撑": lambda up,r,atsup,vp: up&atsup,
 "D 健康+RSI<40+在支撑": lambda up,r,atsup,vp: up&(r<40)&atsup,
 "E 健康+VIX高>70%": lambda up,r,atsup,vp: up&(vp>0.7),
 "F 健康+RSI<40+VIX>70%": lambda up,r,atsup,vp: up&(r<40)&(vp>0.7),
 "G 在支撑+VIX>80%(逆向)": lambda up,r,atsup,vp: atsup&(vp>0.8),
}
agg = {k:([],[]) for k in conds}
for px in PX.values():
    px,up,r,atsup=feats(px); vp=vp_all.reindex(px.index).ffill()
    fwd=px.shift(-126)/px-1.0; fmin=px.shift(-1).rolling(126).min().shift(-125)/px-1.0
    for k,fn in conds.items():
        m=fn(up,r,atsup,vp)&fwd.notna()
        agg[k][0].extend(fwd[m].dropna().tolist()); agg[k][1].extend(fmin[fn(up,r,atsup,vp)&fmin.notna()].dropna().tolist())
print(f"  {'入场条件':<22}{'N':>8}{'中位收益':>10}{'胜率':>8}{'进场浮亏(MAE)':>14}")
rows=[]
for k in conds:
    s=stat(*agg[k])
    if s: rows.append((k,s)); print(f"  {k:<22}{s[0]:>8}{s[1]:>+10.1%}{s[2]:>8.0%}{s[3]:>+14.1%}")
best=max(rows,key=lambda x:(x[1][2]-abs(x[1][3])))  # 胜率-|浮亏| 最高=入场最稳
print(f"  → 入场最稳(胜率高+浮亏浅): **{best[0]}**(胜率{best[1][2]:.0%}·浮亏{best[1][3]:+.1%})。看比基线A改善多少。\n")

# ── 4. 出场组合：找最准的预警 ──
sec("【4】出场/预警组合：未来126日内≥15%回撤的 命中率/误报/lift/召回  · 各信号事件")
def warn_events(px):
    px=px.dropna(); ma200=px.rolling(200,min_periods=100).mean(); r=sg.rsi(px,14)
    ext=(px/ma200-1.0); extp=ext.rolling(756,min_periods=200).apply(lambda x:(x[-1]>=x).mean(),raw=True)
    below=px<ma200; vp=vp_all.reindex(px.index).ffill()
    ev = {
     "跌破200线": below&(~below.shift(1).fillna(False)),
     "RSI超买>75+高位拉伸": (r>75)&(extp>0.9),
     "VIX突破>80%分位": (vp>0.8)&(vp.shift(1)<=0.8),
     "破200线 OR RSI超买拉伸": (below&(~below.shift(1).fillna(False)))|((r>75)&(extp>0.9)),
    }
    return px, ev
DROP=0.15; H=126
acc = {k:[0,0] for k in ["跌破200线","RSI超买>75+高位拉伸","VIX突破>80%分位","破200线 OR RSI超买拉伸"]}
base_h=base_n=0
for px in PX.values():
    px,ev=warn_events(px); arr=px.dropna().to_numpy(); n=len(arr)
    for k,sig in ev.items():
        for i in np.where(sig.reindex(px.dropna().index).fillna(False).to_numpy())[0]:
            if i+H>=n: continue
            mdd=arr[i:i+H].min()/arr[i]-1.0
            acc[k][0 if mdd<=-DROP else 1]+=1
    for i in range(0,n-H,5):
        base_n+=1; base_h+= (arr[i:i+H].min()/arr[i]-1.0<=-DROP)
base=base_h/max(base_n,1)
print(f"  无条件基率(随便一天往后126日≥15%回撤) = {base:.0%}")
print(f"  {'预警信号':<22}{'事件N':>8}{'命中率':>8}{'误报率':>8}{'lift':>7}")
for k,(h,f) in acc.items():
    if h+f<30: continue
    prec=h/(h+f); print(f"  {k:<22}{h+f:>8}{prec:>8.0%}{1-prec:>8.0%}{prec/max(base,1e-9):>7.1f}x")
print("  → lift 越高、误报越低=预警越准。看 RSI/VIX 组合能否把 lift 拉过裸跌破200线。")
print("="*96)
