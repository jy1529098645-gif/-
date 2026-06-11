"""大量测试：跑遍多标的全历史，程序化检测前端裁决文字之间的逻辑冲突。

对每个(标的,每一天)算出前端实际会显示的三类裁决并比对：
  · 离场 exit_warning(red/amber/level，含市场宽度脆弱性)
  · 入场 entry_confluence(grade_tag/fear_pullback，已传入 warn_red/warn_amber/vix)
检测冲突：
  A. 离场红灯 but 入场评级≠🔴(可买)           —— 一边撤离一边可建仓
  B. fear_pullback(会弹'优质回踩·可加码'绿条) but 入场评级=🔴 —— 高亮与评级打架
  C. fear_pullback but 离场红灯/黄灯           —— 积极加码 vs 撤离
  D. 入场🟢现价在支撑 but 离场红灯            —— 同 A 子集
为速度：entry_confluence 传 best_entry={'has_zone':False} 跳过 bootstrap(冲突检测不依赖统计锚定价)。
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from data import loader
from analysis import decision as dec
from regime import entry_cockpit as ec

END="2026-06-06"; START="2008-01-01"
TEST=["NVDA","AAPL","MSFT","GOOGL","AMZN","META","TSLA","AVGO","AMD","TSM","NFLX",
      "INTC","PYPL","DIS","BA","F","T","VZ","NKE","MMM","QCOM","SPY"]
BREADTH=["AAPL","MSFT","GOOGL","AMZN","META","NVDA","JPM","BAC","V","UNH","JNJ","XOM",
         "WMT","HD","PG","KO","CAT","BA","DIS","INTC","CSCO","ORCL","CRM","PEP","MCD"]

print("加载宽度篮子 + VIX…")
bpx={}
for t in BREADTH:
    try: bpx[t]=loader.load_prices([t],START,END)[t].dropna()
    except Exception: pass
bdf=pd.DataFrame(bpx)
above=pd.DataFrame({t:(bdf[t]>bdf[t].rolling(200,min_periods=100).mean()) for t in bdf}).mean(axis=1)
breadth_pct=above.rolling(252,min_periods=120).apply(lambda x:(x[-1]>=x).mean(),raw=True)
fragile=breadth_pct<0.15
VIX=loader.load_ohlcv("^VIX","1995-01-01",END)["close"].dropna()
vixp=VIX.rolling(252,min_periods=126).apply(lambda x:(x[-1]>=x).mean(),raw=True)

DUMMY={"has_zone":False}
counts={k:0 for k in "ABCD"}; examples={k:[] for k in "ABCD"}; total=0
for tk in TEST:
    try:
        o=loader.load_ohlcv(tk,START,END).dropna()
    except Exception: continue
    px=o["close"]; ma200=px.rolling(200,min_periods=100).mean()
    rsi=__import__("factors.signals",fromlist=["rsi"]).rsi(px,14)
    frg=fragile.reindex(px.index).ffill().fillna(False); bpc=breadth_pct.reindex(px.index).ffill()
    vp=vixp.reindex(px.index).ffill()
    # 逐日(为速度每3日采样)
    idx=px.index[200:]
    for d in idx[::3]:
        i=px.index.get_loc(d)
        sub_px=px.iloc[:i+1]; sub_o=o.iloc[:i+1]
        if len(sub_px)<210: continue
        ew=dec.exit_warning(sub_px, bool(frg.loc[d]), float(bpc.loc[d]) if bpc.loc[d]==bpc.loc[d] else None)
        red,amber=bool(ew["red"]),bool(ew["amber"])
        ef=ec.entry_confluence(sub_o, asset=tk, best_entry=DUMMY, warn_red=red, warn_amber=amber,
                               warn_label=ew["level"], vix_pctile=float(vp.loc[d]) if vp.loc[d]==vp.loc[d] else None)
        g=ef["grade_tag"]; fp=ef["fear_pullback"]; total+=1
        if red and g!="🔴": counts["A"]+=1; examples["A"].append((tk,str(d.date()),ew["level"],g))
        if fp and g=="🔴": counts["B"]+=1; examples["B"].append((tk,str(d.date()),ew["level"],g))
        if fp and (red or amber): counts["C"]+=1; examples["C"].append((tk,str(d.date()),ew["level"],"fear+"+("red" if red else "amber")))
        if g=="🟢" and "支撑" in ef["grade"] and red: counts["D"]+=1; examples["D"].append((tk,str(d.date()),ew["level"],g))
    print(f"  扫完 {tk}")

print("\n"+"="*80)
print(f"大量测试结果：共检 {total} 个(标的×日)样本")
print("-"*80)
names={"A":"离场红灯 but 入场可买(≠🔴)","B":"fear_pullback弹优质回踩 but 评级=🔴",
       "C":"fear_pullback积极加码 but 离场红/黄灯","D":"入场🟢在支撑 but 离场红灯"}
for k in "ABCD":
    print(f"  [{k}] {names[k]:<34} 冲突 {counts[k]:>5} 次")
    for ex in examples[k][:4]:
        print(f"        例: {ex[0]} {ex[1]} 离场[{ex[2]}] 入场{ex[3]}")
print("="*80)
print("→ A/D=0 表示红灯门控严密；B/C>0 表示 fear_pullback 高亮需按评级/预警再门控(待修)。")
