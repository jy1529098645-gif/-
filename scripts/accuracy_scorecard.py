"""进/出场判断"准不准"——把信号当预测来检验：命中率 / 误报率 / 召回 / 提前量 / 分桶远期结果。

口径：2000→2026 宽池(含 dotcom+GFC)，无前视。诚实——工具自我定位是"校准非预测"，
所以这里量化的是"判断有没有真实区分力/预警力"，而非"能不能精准预测"。

A. 进场判断：entry_confluence 的可买/观望(飞刀)分桶 → 远期收益/胜率/浮亏，看判断有没有区分力。
B. 离场预警：跌破200线(红灯核心)作为"预警事件" → 命中率(未来真大跌?)/误报率/召回(抓住了几成真崩盘)/提前量。
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from data import loader

END = "2026-06-06"; START = "2000-01-01"
POOL = ["NVDA","AAPL","MSFT","GOOGL","AMZN","META","TSLA","AVGO","AMD","TSM","NFLX",
        "INTC","PYPL","DIS","BA","F","T","VZ","NKE","MMM","QCOM","SPY"]
print("加载 2000→2026 宽池…")
PX = {}
for t in POOL:
    try:
        s = loader.load_ohlcv(t, START, END)["close"].dropna()
        if len(s) > 600: PX[t] = s
    except Exception: pass
print(f"  {len(PX)} 标的\n")

def sec(t): print("="*92 + f"\n{t}\n" + "-"*92)
def md(x): a=np.array([v for v in x if v==v]); return (len(a), np.median(a) if len(a) else float('nan'),
                                                       (a>0).mean() if len(a) else float('nan'))

# ══ A. 进场判断的区分力 ══
sec("【A】进场判断准吗：entry_confluence 三态(🟢现价在支撑/🟡趋势健康等回踩/🔴飞刀) → 远期结果")
def entry_state(px):
    px=px.dropna(); ma200=px.rolling(200,min_periods=100).mean(); ma50=px.rolling(50,min_periods=25).mean()
    slope_up=ma200>ma200.shift(20); low60=px.rolling(60,min_periods=30).min()
    knife=(px<ma200)&(~slope_up)
    conf=(((ma50/px-1).abs()<=0.025).astype(int)+((ma200/px-1).abs()<=0.025).astype(int)+((low60/px-1).abs()<=0.025).astype(int))
    at_sup=(px>ma200)&(conf>=2)              # 🟢 现价在支撑共振区
    healthy=(px>ma200)&(conf<2)              # 🟡 趋势健康等回踩
    return knife, healthy, at_sup
for H in (63,126,252):
    buckets={"🔴飞刀":[], "🟡健康":[], "🟢在支撑":[]}
    maes={"🔴飞刀":[], "🟡健康":[], "🟢在支撑":[]}
    for px in PX.values():
        px=px.dropna(); knife,healthy,at_sup=entry_state(px)
        fwd=px.shift(-H)/px-1.0; fmin=px.shift(-1).rolling(H).min().shift(-(H-1))/px-1.0
        for k,cond in [("🔴飞刀",knife),("🟡健康",healthy),("🟢在支撑",at_sup)]:
            buckets[k].extend(fwd[cond&fwd.notna()].dropna().tolist())
            maes[k].extend(fmin[cond&fmin.notna()].dropna().tolist())
    print(f"  远期{H}日:")
    for k in ("🔴飞刀","🟡健康","🟢在支撑"):
        n,m,w=md(buckets[k]); _,mm,_=md(maes[k])
        print(f"    {k}  N={n:>7}  中位收益{m:+.1%}  胜率{w:.0%}  进场浮亏中位{mm:+.1%}")
print("  → 看 🔴飞刀 是否明显差于 🟡/🟢(=避飞刀有区分力)；🟢vs🟡收益是否接近(=技术共振非收益alpha,只改善浮亏)。\n")

# ══ B. 离场预警的命中率/误报率/召回 ══
sec("【B】离场预警准吗：'跌破200线'预警事件 → 未来是否真大跌（命中/误报/召回/提前量）")
DROP=0.15; HORIZON=126   # 预警后 126 日内是否出现 ≥15% 回撤算"命中"
hit=fa=0; leads=[]; base_events=base_hit=0; warned_crashes=total_crashes=0
for px in PX.values():
    px=px.dropna(); ma200=px.rolling(200,min_periods=100).mean()
    below=px<ma200; cross=below&(~below.shift(1).fillna(False))    # 首次跌破=预警事件
    arr=px.to_numpy(); n=len(arr)
    # 预警事件命中/误报
    for i in np.where(cross.to_numpy())[0]:
        if i+HORIZON>=n: continue
        seg=arr[i:i+HORIZON]; mdd=seg.min()/arr[i]-1.0
        if mdd<=-DROP:
            hit+=1; lead=int(np.argmin(seg)); leads.append(lead)
        else: fa+=1
    # 召回：所有"未来126日内≥15%回撤"的起点中，有多少在起点已处于/即将预警(前20日内跌破200线)
    for i in range(n-HORIZON):
        mdd=arr[i:i+HORIZON].min()/arr[i]-1.0
        if mdd<=-DROP:
            total_crashes+=1
            # 该大跌前夕(起点前20日内)是否出现过跌破200线预警
            lo=max(0,i-20)
            if below.iloc[lo:i+1].any(): warned_crashes+=1
    # 基率：随机起点 126 日内 ≥15% 回撤概率
    for i in range(0,n-HORIZON,5):
        base_events+=1
        if arr[i:i+HORIZON].min()/arr[i]-1.0<=-DROP: base_hit+=1
prec=hit/max(hit+fa,1); base=base_hit/max(base_events,1); recall=warned_crashes/max(total_crashes,1)
print(f"  预警事件总数 {hit+fa}（跌破200线）")
print(f"  命中率(预警后126日内真≥15%回撤) = {prec:.0%}   |   误报率 = {1-prec:.0%}")
print(f"  无条件基率(随便一天往后126日≥15%回撤) = {base:.0%}   →   预警把概率提升 {prec/max(base,1e-9):.1f} 倍(lift)")
print(f"  召回(所有≥15%大跌中，事前已有跌破200线预警的占比) = {recall:.0%}  ←抓住了几成真崩盘")
print(f"  提前量：命中时，最深回撤平均出现在预警后 {np.mean(leads):.0f} 个交易日（预警领先于最深处）")
print("\n  诚实读法：命中率≈基率的1.x倍=有真实预警力但**远非精准**；误报率高='降仓开关'非'崩盘预言'；")
print("  召回高=真崩盘大多被它逮到(代价是误报多)。这与工具自我定位'校准非预测'一致。")
print("="*92)
