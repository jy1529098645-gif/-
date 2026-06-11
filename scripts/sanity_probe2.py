"""大量测试②：统计有效性 + 跨源数字一致性 + 杠杆ETF口径 + 分布健康度。"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from collections import Counter
from data import loader
from regime import entry_cockpit as ec
from regime import observables as ob
from analysis import decision as dec
from analysis import position_guidance as pg

END = "2025-12-01"
NORMAL = ["SPY","QQQ","NVDA","AAPL","MSFT","GOOGL","AMZN","META","AMD","INTC","PYPL","DIS","BA","NFLX","MU","QCOM"]
LEV = ["TQQQ","SOXL","UPRO","TECL","TNA"]

# ---------- ① 杠杆ETF：entry_confluence 是否给"可分批长持"口径 + 算胜率（应被特殊处理）----------
print("="*80); print("① 杠杆3xETF 的入场裁决（entry_confluence 不含杠杆感知 → 与 decision_card 的'不宜久持'警告矛盾？）"); print("="*80)
for tk in LEV:
    try:
        o = loader.load_ohlcv(tk, "2010-01-01", END).dropna()
        ew = dec.exit_warning(o["close"], False, None)
        ef = ec.entry_confluence(o, asset=tk, best_entry={"has_zone": False},
                                 warn_red=ew["red"], warn_amber=ew["amber"], warn_label=ew["level"], vix_pctile=0.5)
        dc = dec.decision_card(tk, o["close"], None, False, "🟢", False, None)
        wn = ef.get("win_now")
        print(f"{tk:6} entry={ef['grade_tag']} {ef['grade'][:22]:22} | 现价买胜率={'%.0f%%'%(wn*100) if wn==wn and wn is not None else '—':>5} "
              f"| classify={dec.classify(tk)} | decision_card.action={dc.get('action','')[:30]}")
    except Exception as e:
        print(f"{tk}: {e}")
print("  ⚠️ 若 entry 给🟢/🟡'分批'且报具体胜率，但 classify=leveraged_etf/decision_card 说'不宜久持摊平' → 同一票两套口径打架。")

# ---------- ② 跨源 vol_percentile 一致性 ----------
print("\n"+"="*80); print("② 跨源『波动分位』一致性：observables vs position_guidance vs exit_warning"); print("="*80)
maxdiff = 0
for tk in NORMAL[:8]:
    o = loader.load_ohlcv(tk, "2006-01-01", END).dropna(); px = o["close"]
    v_ob = float(ob.realized_vol_percentile(px, 21, 252).iloc[-1])
    g = pg.position_guidance(tk, end=END); v_pg = g["regime"]["vol_percentile"]
    rv = px.pct_change().rolling(21, min_periods=10).std()*np.sqrt(252)
    v_ew = float((rv.iloc[-1] >= rv.tail(252)).mean())
    d = max(abs(v_ob-v_pg), abs(v_ob-v_ew), abs(v_pg-v_ew)); maxdiff = max(maxdiff, d)
    flag = "  ⚠️差异>5%" if d > 0.05 else ""
    print(f"{tk:6} observables={v_ob:.0%}  position_guidance={v_pg:.0%}  exit_warning={v_ew:.0%}{flag}")
print(f"  → 三源最大差异 {maxdiff:.0%}（应≈0；>5% 说明同名『波动分位』口径不一致）")

# ---------- ③ win_now（现价买胜率）分布健康度：是否多数NaN / 是否系统性偏高(不区分) ----------
print("\n"+"="*80); print("③ 『现价买胜率』分布：可用率 + 是否系统性偏高(失去区分度)"); print("="*80)
vals=[]; nan_cnt=0; tot=0
for tk in NORMAL+["PLTR","SMCI","ARM"]:
    try: o = loader.load_ohlcv(tk, "2006-01-01", END).dropna()
    except Exception: continue
    px=o["close"]
    cuts=[px.index[-1]] + ([px.index[int(len(px)*f)] for f in (0.5,0.7,0.85)] if len(px)>500 else [])
    for cut in cuts:
        sub=o.loc[:cut]
        if len(sub)<260: continue
        ef=ec.entry_confluence(sub, asset=tk, best_entry={"has_zone":False}, vix_pctile=0.5)
        wn=ef.get("win_now"); tot+=1
        if wn is None or wn!=wn: nan_cnt+=1
        else: vals.append(wn)
va=np.array(vals)
print(f"  样本 {tot}：可用胜率 {len(va)}（{len(va)/tot:.0%}）, NaN {nan_cnt}（{nan_cnt/tot:.0%}）")
if len(va):
    print(f"  胜率分布：min {va.min():.0%} / p25 {np.percentile(va,25):.0%} / 中位 {np.median(va):.0%} / p75 {np.percentile(va,75):.0%} / max {va.max():.0%}")
    print(f"  >60% 占比 {np.mean(va>0.6):.0%}、>70% 占比 {np.mean(va>0.7):.0%}（若几乎全>60% → 胜率永远好看、无区分度=误导）")

# ---------- ④ 暴露抖动(whipsaw)：单日暴露翻动>0.4 的频率 ----------
print("\n"+"="*80); print("④ 暴露序列抖动：单日变动>40% 的频率（频繁翻动=不可执行/交易成本高）"); print("="*80)
mc=pg.PROFILES["moderate"]
for tk in ["SPY","NVDA","TQQQ","INTC"]:
    px=loader.load_ohlcv(tk,"2010-01-01",END)["close"].dropna()
    es=pg._exposure_series(px, mc["tvol"], mc["max_lev"], floor=mc["floor"], slope_floor=mc["slope_floor"]).dropna()
    jumps=int((es.diff().abs()>0.4).sum()); ann=jumps/(len(es)/252)
    print(f"{tk:6} 暴露区间[{es.min():.2f},{es.max():.2f}] 单日>40%翻动 {jumps} 次（约 {ann:.1f} 次/年）")

# ---------- ⑤ grade 分布：是否几乎全🟡(裁决无信息量) ----------
print("\n"+"="*80); print("⑤ 入场评级分布（若几乎全🟡 → 裁决缺乏区分度）"); print("="*80)
gc=Counter()
for tk in NORMAL:
    o=loader.load_ohlcv(tk,"2006-01-01",END).dropna(); px=o["close"]
    for cut in [px.index[min(int(len(px)*f), len(px)-1)] for f in (0.4,0.55,0.7,0.85,0.999)]:
        sub=o.loc[:cut]
        if len(sub)<260: continue
        ew=dec.exit_warning(sub["close"],False,None)
        ef=ec.entry_confluence(sub,asset=tk,best_entry={"has_zone":False},warn_red=ew["red"],warn_amber=ew["amber"],vix_pctile=0.5)
        gc[ef["grade_tag"]]+=1
tot=sum(gc.values())
for g,n in gc.most_common(): print(f"  {g}: {n} ({n/tot:.0%})")
print("="*80)
