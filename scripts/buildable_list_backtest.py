"""把工具当前'可建仓'清单(AVGO/AMZN/QCOM/NVDA/GOOGL/SPY)按工具策略回测 + 具体 alpha/beta。

工具策略 = analysis.position_guidance._exposure_series(中性档:趋势门×波动目标×软地板) —— UI实际引擎。
对每只：策略暴露收益 vs 闭眼持有该股 vs SPY，算 CAGR/夏普/回撤 + 两套 alpha/beta(回归,带bootstrap CI)：
  · α vs 持有该股：工具的择时/减仓 相对'死拿这只票'有没有超额。
  · α/β vs SPY(CAPM)：相对大盘的超额与市场敏感度。
口径：2010→2026，无前视(昨暴露×今收益)−|Δ暴露|×10bps成本。
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from data import loader
from analysis import position_guidance as pg
from analysis import quant_edge as qe

END="2026-06-06"; START="2010-01-01"; ANN=252; FEE=0.0010
STOCKS=["AVGO","AMZN","QCOM","NVDA","GOOGL","SPY"]
mc=pg.PROFILES["moderate"]
CFG=dict(tvol=mc["tvol"], max_lev=mc["max_lev"], floor=mc["floor"], slope_floor=mc["slope_floor"])

def bt(px):
    ret=px.pct_change(); expo=pg._exposure_series(px, **CFG); pos=expo.shift(1)
    return (pos*ret - pos.diff().abs().fillna(0)*FEE).dropna()
def stats(r):
    r=r.dropna(); n=len(r); cum=(1+r).prod(); eq=(1+r).cumprod()
    mdd=float((eq/eq.cummax()-1).min()); cagr=cum**(ANN/n)-1
    return cagr, r.std()*np.sqrt(ANN), (r.mean()/r.std()*np.sqrt(ANN) if r.std()>0 else np.nan), mdd, cum

spy=loader.load_ohlcv("SPY",START,END)["close"].dropna(); spy_ret=spy.pct_change()
strat_all={}; hold_all={}
print("="*104)
print("工具策略(v3.1中性暴露) 回测 · 2010→2026 · 逐票")
print("="*104)
print(f"{'票':<6}{'策略年化':>9}{'持有年化':>9}{'策略夏普':>9}{'持有夏普':>9}{'策略回撤':>9}{'持有回撤':>9}"
      f"{'α vs持有':>10}{'α vsSPY':>10}{'β vsSPY':>8}")
for tk in STOCKS:
    px=loader.load_ohlcv(tk,START,END)["close"].dropna()
    sr=bt(px); hr=px.pct_change().dropna()
    strat_all[tk]=sr; hold_all[tk]=hr
    cs,vs,shs,dds,_=stats(sr); ch,vh,shh,ddh,_=stats(hr)
    idx=sr.index
    ab_h=qe.alpha_beta(sr.reindex(idx), hr.reindex(idx), n_boot=800)            # vs 持有该股
    ab_s=qe.alpha_beta(sr.reindex(idx), spy_ret.reindex(idx).fillna(0), n_boot=800)  # vs SPY
    sig_h="✔" if ab_h["alpha_significant"] else "✘"; sig_s="✔" if ab_s["alpha_significant"] else "✘"
    print(f"{tk:<6}{cs:>+9.0%}{ch:>+9.0%}{shs:>9.2f}{shh:>9.2f}{dds:>+9.0%}{ddh:>+9.0%}"
          f"{ab_h['alpha_ann']:>+8.1%}{sig_h}{ab_s['alpha_ann']:>+8.1%}{sig_s}{ab_s['beta']:>8.2f}")

# 池化等权这6只
print("-"*104)
common=pd.DataFrame(strat_all).dropna(how="all").index
sp_port=pd.DataFrame(strat_all).reindex(common).mean(axis=1)
hd_port=pd.DataFrame(hold_all).reindex(common).mean(axis=1)
cs,vs,shs,dds,muls=stats(sp_port); ch,vh,shh,ddh,mulh=stats(hd_port)
ab_h=qe.alpha_beta(sp_port.loc[common], hd_port.loc[common], n_boot=1500)
ab_s=qe.alpha_beta(sp_port.loc[common], spy_ret.reindex(common).fillna(0), n_boot=1500)
print(f"{'池化(等权6只)':<6}{cs:>+9.0%}{ch:>+9.0%}{shs:>9.2f}{shh:>9.2f}{dds:>+9.0%}{ddh:>+9.0%}"
      f"{ab_h['alpha_ann']:>+8.1%}{'✔' if ab_h['alpha_significant'] else '✘'}"
      f"{ab_s['alpha_ann']:>+8.1%}{'✔' if ab_s['alpha_significant'] else '✘'}{ab_s['beta']:>8.2f}")
print("\n池化详情：")
print(f"  策略组合: 年化{cs:+.1%} 终值{muls:.1f}x 夏普{shs:.2f} 回撤{dds:+.0%}")
print(f"  持有组合: 年化{ch:+.1%} 终值{mulh:.1f}x 夏普{shh:.2f} 回撤{ddh:+.0%}")
print(f"  α vs持有: {ab_h['alpha_ann']:+.1%} (β{ab_h['beta']:.2f}, CI[{ab_h['alpha_ann_ci'][0]:+.1%},{ab_h['alpha_ann_ci'][1]:+.1%}], "
      f"{'显著' if ab_h['alpha_significant'] else '不显著'})")
print(f"  α vs SPY : {ab_s['alpha_ann']:+.1%} (β{ab_s['beta']:.2f}, CI[{ab_s['alpha_ann_ci'][0]:+.1%},{ab_s['alpha_ann_ci'][1]:+.1%}], "
      f"{'显著' if ab_s['alpha_significant'] else '不显著'})")
print("="*104)
print("读法：α vs持有>0显著=工具择时相对死拿这票有超额(罕见)；α vsSPY=相对大盘超额(常因这些是高β成长股);")
print("      β vsSPY=策略对大盘敏感度(减仓后通常<1)。")
