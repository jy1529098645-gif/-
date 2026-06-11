"""大量测试：跑遍全 universe × 多个 as-of 切点，调用 UI 实际引擎，检一整套"合理性不变量"。
目的=找工具输出里**不合理/会误导用户**的地方(越界值/自相矛盾/跨页数字打架/NaN当数字/样本不足却报具体数)。
无前视：每个 as-of 只用 o.loc[:cut] 的数据。
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from data import loader
from regime import entry_cockpit as ec
from analysis import decision as dec
from analysis import position_guidance as pg
import factors.signals as sg

END = "2025-12-01"
# 代表性 universe：赢家巨头 + 深跌票 + ETF + 3x杠杆 + 短历史新票
UNI = ["SPY", "QQQ", "IWM", "XLK", "SMH", "XLE", "XLF",
       "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA",
       "AMD", "INTC", "QCOM", "MU", "AVGO", "TSM",
       "PYPL", "DIS", "BA", "NFLX", "INTC",
       "PLTR", "SMCI", "ARM",          # 短历史
       "TQQQ", "SOXL", "UPRO"]          # 3x 杠杆
UNI = list(dict.fromkeys(UNI))

# VIX 分位（一次算好）
try:
    _vix = loader.load_ohlcv("^VIX", "1995-01-01", END)["close"].dropna()
    _vp = _vix.rolling(252, min_periods=126).apply(lambda x: (x[-1] >= x).mean(), raw=True)
except Exception:
    _vp = pd.Series(dtype=float)

V = []   # violations: (类别, 标的, asof, 描述)
def bad(cat, tk, asof, msg): V.append((cat, tk, str(asof), msg))

def vixp_at(d):
    try:
        s = _vp.reindex(_vp.index.union([d])).ffill().get(d)
        return float(s) if s == s else None
    except Exception:
        return None

n_eval = 0
for tk in UNI:
    try:
        o = loader.load_ohlcv(tk, "2006-01-01", END).dropna()
    except Exception as e:
        bad("加载", tk, "-", f"load 失败 {e}"); continue
    if len(o) < 60:
        continue
    px = o["close"]
    # 3 个 as-of 切点
    cuts = [px.index[-1]]
    if len(px) > 400: cuts.append(px.index[int(len(px)*0.6)])
    if len(px) > 700: cuts.append(px.index[int(len(px)*0.3)])
    for cut in cuts:
        sub = o.loc[:cut]
        if len(sub) < 60: continue
        cur = float(sub["close"].iloc[-1])
        n_eval += 1
        vp = vixp_at(cut)
        # ---- exit_warning ----
        try:
            ew = dec.exit_warning(sub["close"], False, None)
        except Exception as e:
            bad("exit崩溃", tk, cut, str(e)); ew = None
        if ew is not None:
            if ew.get("red") and ew.get("amber"):
                bad("exit", tk, cut, "red 与 amber 同时为真")
            dm = ew.get("dist_ma200")
            if dm is not None and dm == dm and abs(dm) > 5:
                bad("exit", tk, cut, f"dist_ma200 异常 {dm}")
        # ---- entry_confluence ----
        try:
            ef = ec.entry_confluence(sub, asset=tk, best_entry={"has_zone": False},
                                     warn_red=bool(ew["red"]) if ew else False,
                                     warn_amber=bool(ew["amber"]) if ew else False,
                                     warn_label=ew["level"] if ew else "", vix_pctile=vp)
        except Exception as e:
            bad("entry崩溃", tk, cut, str(e)); continue
        gt = ef.get("grade_tag")
        if gt not in ("🟢", "🟡", "🔴"):
            bad("entry", tk, cut, f"grade_tag 非法 {gt}")
        # 胜率边界 + 样本门
        wn, wnn = ef.get("win_now"), ef.get("win_now_n")
        if wn is not None and wn == wn and not (0 <= wn <= 1):
            bad("胜率越界", tk, cut, f"win_now={wn}")
        if wn == wn and wnn is not None and wnn < 30:
            bad("样本不足却报胜率", tk, cut, f"win_now={wn:.2f} 但 N={wnn}<30")
        rsi = ef.get("rsi")
        if rsi is not None and rsi == rsi and not (0 <= rsi <= 100):
            bad("RSI越界", tk, cut, f"rsi={rsi}")
        # supports_below 必须严格在现价下方、价>0、dist<0
        for s in (ef.get("supports_below") or []):
            if not (s["price"] > 0): bad("支撑", tk, cut, f"支撑价≤0 {s}")
            if s["price"] >= cur * 1.0: bad("支撑", tk, cut, f"回踩支撑 {s['label']} {s['price']:.2f} ≥现价 {cur:.2f}")
            if s["dist_pct"] >= 0: bad("支撑", tk, cut, f"回踩支撑 dist_pct≥0 {s}")
        # at_support_now ⟹ 确有支撑在容差内
        if ef.get("at_support_now"):
            near = ef.get("supports_near_now") or []
            if not near: bad("支撑", tk, cut, "at_support_now=True 但 supports_near_now 为空")
        # 门控一致性
        if ew is not None and ew.get("red") and gt != "🔴":
            bad("门控矛盾", tk, cut, f"离场红灯但入场={gt}")
        if ef.get("fear_pullback") and (gt != "🟢" or (ew and ew.get("red"))):
            bad("门控矛盾", tk, cut, f"fear_pullback 但 grade={gt}/red={ew.get('red') if ew else '?'}")

# ---- 重型：position_guidance 全量 + 跨页 dd 一致性（小子集，慢）----
SUBSET = ["SPY", "NVDA", "PYPL", "DIS", "INTC", "TQQQ", "PLTR"]
for tk in SUBSET:
    try:
        g = pg.position_guidance(tk, end=END)
    except Exception as e:
        bad("guidance崩溃", tk, END, str(e)); continue
    reg = g["regime"]; px_now = reg["price"]
    # 暴露边界
    for k, ex in g["exposure"].items():
        e_ = ex["exposure"]; ml = ex["max_lev"]
        if not (e_ == e_): bad("暴露NaN", tk, END, f"{k} 暴露 NaN")
        elif e_ < -1e-9 or e_ > ml + 1e-6:
            bad("暴露越界", tk, END, f"{k} 暴露 {e_:.3f} 超出 [0,{ml}]")
    # dd 跨页一致性：guidance 用的 rolling252 应与独立计算一致
    px_s = loader.load_ohlcv(tk, "2006-01-01", END)["close"].dropna()
    dd_indep = float(px_s.iloc[-1] / px_s.rolling(252, min_periods=120).max().iloc[-1] - 1)
    if abs(reg["drawdown_from_high"] - dd_indep) > 0.005:
        bad("跨页dd不一致", tk, END, f"guidance dd={reg['drawdown_from_high']:+.1%} vs rolling252 {dd_indep:+.1%}")
    if not (-1 < reg["drawdown_from_high"] <= 0.03):
        bad("dd越界", tk, END, f"dd={reg['drawdown_from_high']}")
    # zones 合理性
    for z in g["build"]["zones"]:
        if not (z["price_low"] < z["price_high"]): bad("zone", tk, END, f"low≥high {z}")
        if z["price_low"] <= 0: bad("zone", tk, END, f"价≤0 {z}")
        if z["reached"] != bool(px_now <= z["price_high"]):
            bad("zone", tk, END, f"reached 标志不一致 {z['band']} reached={z['reached']} px={px_now} hi={z['price_high']}")

print("="*78)
print(f"合理性大量测试：评估 {n_eval} 个(标的×as-of) + {len(SUBSET)} 个 guidance 全量")
print("="*78)
if not V:
    print("✅ 未发现违反不变量的情况")
else:
    from collections import Counter
    cc = Counter(v[0] for v in V)
    print(f"⚠️ 共 {len(V)} 处异常，按类别：")
    for cat, n in cc.most_common():
        print(f"  · {cat}: {n}")
    print("-"*78)
    for cat, _ in cc.most_common():
        ex = [v for v in V if v[0] == cat][:6]
        for v in ex:
            print(f"  [{v[0]}] {v[1]} @ {v[2]}: {v[3]}")
print("="*78)
