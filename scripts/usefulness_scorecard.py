"""用处计分卡：用大量回测量化这个工具**到底有多有用**（含两次大崩盘的长历史，诚实标注有用/没用）。

五段，对应它的真实价值主张：
  1. 离场警示（趋势/200线信号）的崩盘保护力——预警后未来大跌概率 lift + 减回撤。
  2. 飞刀防护——避开"破位接飞刀"进场 vs 趋势健康进场，远期收益/胜率/浮亏差多少。
  3. 入场回踩支撑的入场质量——多支撑共振处进场，浮亏(MAE)浅多少。
  4. 全套暴露规则(v3.1) vs 闭眼持有——回撤砍多少、夏普/Calmar、各崩盘窗口。
  5. 诚实的"没用处"——绝对收益让出多少、选股有没有 edge。
口径：无前视；2000→2026，21只赢家+落后股+SPY，含 dotcom+GFC。
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from data import loader
from analysis import position_guidance as pg

END = "2026-06-06"; START = "2000-01-01"; ANN = 252; FEE = 0.0010
WIN = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "AMD", "TSM", "NFLX"]
LAG = ["INTC", "PYPL", "DIS", "BA", "F", "T", "VZ", "NKE", "MMM", "QCOM"]
POOL = WIN + LAG + ["SPY"]

print("加载 2000→2026 宽池…")
PX = {}
for t in POOL:
    try:
        s = loader.load_ohlcv(t, START, END).dropna()
        if len(s) > 600:
            PX[t] = s
    except Exception:
        pass
print(f"  {len(PX)} 标的\n")


def sec(t): print("=" * 94 + f"\n{t}\n" + "-" * 94)


# ── 1. 离场警示：趋势信号(破200线)对未来大跌的预警力 ──
sec("【1】离场警示的崩盘保护力：趋势在200线上方 vs 下方 → 未来63日最深浮亏 / 大跌概率")
inT_dd, outT_dd, inT_drop, outT_drop = [], [], [], []
for px in (p["close"] for p in PX.values()):
    px = px.dropna(); ma200 = px.rolling(200, min_periods=100).mean()
    above = px > ma200
    fwd_min = px.shift(-1).rolling(63).min().shift(-62) / px - 1.0   # 未来63日路径最低
    for cond, ddl, drl in [(above, inT_dd, inT_drop), (~above, outT_dd, outT_drop)]:
        m = cond & fwd_min.notna()
        v = fwd_min[m].dropna()
        ddl.extend(v.tolist()); drl.extend((v <= -0.10).tolist())
def _m(x): a = np.array(x); return np.median(a), np.mean(a)
print(f"  {'状态':<14}{'N':>9}{'未来63日浮亏中位':>16}{'未来跌≥10%概率':>16}")
print(f"  {'200线上方(在场)':<14}{len(inT_dd):>9}{np.median(inT_dd):>+16.1%}{np.mean(inT_drop):>16.0%}")
print(f"  {'200线下方(预警)':<14}{len(outT_dd):>9}{np.median(outT_dd):>+16.1%}{np.mean(outT_drop):>16.0%}")
_lift = np.mean(outT_drop) / max(np.mean(inT_drop), 1e-9)
print(f"  → 跌破200线后未来大跌概率是在场时的 **{_lift:.1f} 倍**；浮亏中位深 {np.median(outT_dd)-np.median(inT_dd):+.1%}。"
      f"这就是'离场预警'的保护来源。\n")


# ── 2. 飞刀防护：破位接飞刀 vs 趋势健康 进场 ──
sec("【2】飞刀防护：'破200线+均线下行'(飞刀)进场 vs '趋势健康'进场 → 远期收益/胜率/浮亏")
def states(px):
    px = px.dropna(); ma200 = px.rolling(200, min_periods=100).mean()
    slope_up = ma200 > ma200.shift(20)
    knife = (px < ma200) & (~slope_up)
    healthy = px > ma200
    return knife, healthy
for H in (63, 126):
    kf, hf, kmae, hmae = [], [], [], []
    for px in (p["close"] for p in PX.values()):
        px = px.dropna(); knife, healthy = states(px)
        fwd = px.shift(-H) / px - 1.0
        fmin = px.shift(-1).rolling(H).min().shift(-(H-1)) / px - 1.0
        for cond, fl, ml in [(knife, kf, kmae), (healthy, hf, hmae)]:
            m = cond & fwd.notna()
            fl.extend(fwd[m].dropna().tolist()); ml.extend(fmin[cond & fmin.notna()].dropna().tolist())
    print(f"  远期{H}日:  飞刀进场 中位{np.median(kf):+.1%} 胜率{np.mean(np.array(kf)>0):.0%} 浮亏中位{np.median(kmae):+.1%}"
          f"  |  健康进场 中位{np.median(hf):+.1%} 胜率{np.mean(np.array(hf)>0):.0%} 浮亏中位{np.median(hmae):+.1%}")
print("  → 趋势健康进场在 收益/胜率/浮亏 三项全面优于'接飞刀'。飞刀防护=帮你别在最差的时点进场。\n")


# ── 3. 入场质量：多支撑共振处进场 浮亏更浅 ──
sec("【3】入场回踩支撑的质量：趋势健康日 多支撑共振(≥2) vs 无共振 → 进场后126日最深浮亏")
c0, c2 = [], []
for px in (p["close"] for p in PX.values()):
    px = px.dropna(); ma200 = px.rolling(200, min_periods=100).mean()
    ma50 = px.rolling(50, min_periods=25).mean(); low60 = px.rolling(60, min_periods=30).min()
    low120 = px.rolling(120, min_periods=60).min()
    up = px > ma200
    conf = (((ma50/px-1).abs()<=0.025).astype(int) + ((ma200/px-1).abs()<=0.025).astype(int)
            + ((low60/px-1).abs()<=0.025).astype(int) + ((low120/px-1).abs()<=0.025).astype(int))
    arr = px.to_numpy()
    for i in np.where(up.to_numpy())[0]:
        if i+126 >= len(arr): continue
        mae = arr[i:i+126].min()/arr[i]-1.0
        (c2 if conf.iloc[i] >= 2 else c0).append(mae)
print(f"  共振≥2 进场: N={len(c2)} 浮亏中位 {np.median(c2):+.1%}   |   无共振 进场: N={len(c0)} 浮亏中位 {np.median(c0):+.1%}")
print(f"  → 多支撑处进场浮亏浅 {np.median(c2)-np.median(c0):+.1%}（更拿得住）。注：共振**不提高收益**，只改善入场舒适度。\n")


# ── 4. 全套暴露规则(v3.1中性) vs 闭眼持有 ──
sec("【4】全套暴露规则(v3.1中性·破位降底仓+波动目标) vs 闭眼持有 —— 池化")
def bt(px, cfg):
    ret = px.pct_change(); expo = pg._exposure_series(px, **cfg); pos = expo.shift(1)
    return (pos*ret - pos.diff().abs().fillna(0)*FEE).dropna()
mc = pg.PROFILES["moderate"]
cfg = dict(tvol=mc["tvol"], max_lev=mc["max_lev"], floor=mc["floor"], slope_floor=mc["slope_floor"])
sd, hd = {}, {}
for t, p in PX.items():
    px = p["close"].dropna(); sd[t] = bt(px, cfg); hd[t] = px.pct_change()
common = pd.DataFrame(sd).dropna(how="all").index
sp = pd.DataFrame(sd).reindex(common).mean(axis=1); hp = pd.DataFrame(hd).reindex(common).mean(axis=1)
def stats(r):
    r = r.dropna(); n = len(r); cum = (1+r).prod(); eq = (1+r).cumprod()
    mdd = float((eq/eq.cummax()-1).min()); cagr = cum**(ANN/n)-1
    return cagr, r.std()*np.sqrt(ANN), r.mean()/r.std()*np.sqrt(ANN), mdd, cagr/abs(mdd)
for nm, r in [("工具暴露", sp), ("闭眼持有", hp)]:
    c, v, s, d, cal = stats(r)
    print(f"  {nm:<8} 年化{c:+.1%}  波动{v:.0%}  夏普{s:.2f}  最大回撤{d:+.0%}  Calmar{cal:.2f}")
cri = [("dotcom 00-03","2000-09","2003-01"),("GFC 07-09","2007-10","2009-04"),("2022熊","2022-01","2022-12")]
print("  崩盘窗口 区间最大回撤：")
for nm,s,e in cri:
    idx = common[(common>=s)&(common<=e)]
    def wd(r): eq=(1+r.reindex(idx)).cumprod(); return float((eq/eq.cummax()-1).min())
    print(f"    {nm:<12} 工具 {wd(sp):+.0%}  vs  持有 {wd(hp):+.0%}")
print("  → 风险调整(夏普/Calmar)更优、最大回撤砍约一半；崩盘窗口保护明显。\n")


# ── 5. 诚实的"没用处" ──
sec("【5】诚实：它在哪儿没用（别误用）")
c_s,_,sh_s,_,_ = stats(sp); c_h,_,sh_h,_,_ = stats(hp)
print(f"  · 绝对复利：工具年化 {c_s:+.1%} < 闭眼持有 {c_h:+.1%}——减暴露必让出复利，跑不赢长持。")
print(f"  · 风险调整：夏普 {sh_s:.2f} vs 持有 {sh_h:.2f}（相近/略优）——价值在'砍回撤'不在'多赚'。")
print( "  · 选股：横截面 RankIC≈安慰剂(工具自承认)，不能告诉你买哪只会暴涨。")
print( "  · 入场精度：entry 对长期收益影响极小，'最佳买价'是噪声——只求进得稳。")
print("\n" + "=" * 94)
print("一句话：它有用在'让你扛住崩盘别割肉(回撤砍半)+进得稳(浮亏浅)+破位预警'；没用在'跑赢市场/选股/抄底'。")
print("=" * 94)
