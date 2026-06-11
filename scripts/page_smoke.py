"""无头页面冒烟测试：用 streamlit AppTest 跑每个 job/sub 页面，捕获异常与 st.error。
不点 run_gate 按钮(只测页面骨架+默认即时计算)，但 panorama/作战卡/组合配置/研究台多数页是加载即算。
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest

JOBS = ["ℹ️ 关于", "🎯 个股决策", "🔭 建仓扫描", "🛡️ 组合配置", "📋 多票简报", "🔬 研究台"]
STOCK_SUB = ["📊 全景图（图+裁决）", "🎖️ 作战卡（入场位 / 离场警示）", "📈 当前快照", "🗞️ 事件时间线", "📅 财报 PEAD"]
RESEARCH_SUB = None  # 用 index 遍历

def radio_by_label(at, label):
    for r in at.radio:
        if r.label == label:
            return r
    return None

def selectbox_by_label(at, label):
    for s in at.selectbox:
        if s.label == label:
            return s
    return None

def run_case(job, sub_label=None, sub_index=None, app_path=str(ROOT / "app.py")):
    at = AppTest.from_file(app_path, default_timeout=240)
    at.run()
    if at.exception:
        return ("启动", [str(e) for e in at.exception])
    r = radio_by_label(at, "任务")
    if r is None:
        return ("无任务radio", [])
    r.set_value(job); at.run()
    if (job in ("🎯 个股决策", "🔬 研究台")) and (sub_label or sub_index is not None):
        sb = selectbox_by_label(at, "看什么") or selectbox_by_label(at, "研究工具")
        if sb is not None:
            if sub_label: sb.set_value(sub_label)
            elif sub_index is not None: sb.set_value(sb.options[sub_index])
            at.run()
    errs = []
    if at.exception:
        errs += [f"EXC {type(e).__name__ if hasattr(e,'__class__') else ''}: {e}" for e in at.exception]
    for e in at.error:
        errs += [f"st.error: {e.value}"]
    return ("ok" if not errs else "FAIL", errs)

cases = [("ℹ️ 关于", None, None)]
for s in STOCK_SUB: cases.append(("🎯 个股决策", s, None))
cases.append(("🔭 建仓扫描", None, None))
cases.append(("🛡️ 组合配置", None, None))
cases.append(("📋 多票简报", None, None))
for i in range(6): cases.append(("🔬 研究台", None, i))

print("="*84)
print("页面冒烟测试（AppTest 无头跑每页，捕获异常/st.error）")
print("="*84)
nfail = 0
for job, sl, si in cases:
    name = f"{job}" + (f" › {sl}" if sl else (f" › 研究#{si}" if si is not None else ""))
    try:
        status, errs = run_case(job, sl, si)
    except Exception as e:
        status, errs = "HARNESS", [f"{type(e).__name__}: {e}"]
    mark = "✅" if status == "ok" else "❌"
    if status != "ok": nfail += 1
    print(f"{mark} {name:42} [{status}]")
    for e in errs[:3]:
        print(f"      ↳ {e[:160]}")
print("="*84)
print(f"完成：{len(cases)} 页，{nfail} 个有异常/错误")
print("="*84)
