"""UI 文字质量扫描：渲染每个页面，抓所有文本元素，找 NaN/None/inf/未格式化 泄漏到界面的 bug。
(page_smoke 只抓崩溃；这个抓"显示出来但是脏数据/格式错"的问题，如 'nan%'、'None'、'$nan'、'{...}'.)
"""
import sys, re, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest

JOBS = ["ℹ️ 关于", "🎯 个股决策", "🔭 建仓扫描", "🛡️ 组合配置", "📋 多票简报", "🔬 研究台"]
STOCK_SUB = ["📊 全景图（图+裁决）", "🎖️ 作战卡（入场位 / 离场警示）", "📈 当前快照", "🗞️ 事件时间线", "📅 财报 PEAD"]

# 脏数据/格式 bug 模式（命中=可能 UI 显示了未处理的值）
BAD = re.compile(
    r"(?i)(?<![a-z])nan(?![a-z])"      # 裸 nan
    r"|(?<![a-z])none(?![a-z])"        # 裸 None
    r"|[+\-]?\binf\b"                  # inf
    r"|nan%|±nan|\+nan|\-nan"          # nan 百分比
    r"|\$nan|nan x|nanx"
    r"|\{['\"][a-z_]+['\"]\s*:"        # 未格式化的 dict repr
    r"|\[\s*\{['\"]"                   # 未格式化的 list[dict]
    r"|0nan|nan倍|nan个"
)
# 允许的误报豁免（中文/正常词里含 none/nan 子串极少；这里防 'Nasdaq' 之类）
ALLOW = re.compile(r"(?i)nasdaq|tennant|nano|finance|annot"
                   r"|transform:\s*none|:\s*none[;\"']|none\s*\}|outline:|border:|appearance"  # CSS value none
                   r"|nan(?=oseconds)")

def radio_by_label(at, label):
    for r in at.radio:
        if r.label == label: return r
def selectbox_by_label(at, label):
    for s in at.selectbox:
        if s.label in (label, "看什么", "研究工具"): return s

def collect_texts(at):
    out = []
    for attr in ("markdown", "caption", "info", "warning", "error", "success", "title", "header", "subheader", "text"):
        try:
            for el in getattr(at, attr):
                v = getattr(el, "value", None) or getattr(el, "body", None)
                if isinstance(v, str): out.append((attr, v))
        except Exception:
            pass
    for el_attr in ("metric",):
        try:
            for el in getattr(at, el_attr):
                for f in ("label", "value", "delta"):
                    v = getattr(el, f, None)
                    if isinstance(v, str): out.append((el_attr, v))
        except Exception:
            pass
    return out

def run_case(job, sub_label=None, sub_index=None):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)
    at.run()
    r = radio_by_label(at, "任务")
    if r is None: return []
    r.set_value(job); at.run()
    if job in ("🎯 个股决策", "🔬 研究台"):
        sb = selectbox_by_label(at, "看什么")
        if sb is not None:
            if sub_label: sb.set_value(sub_label)
            elif sub_index is not None: sb.set_value(sb.options[sub_index])
            at.run()
    hits = []
    for attr, txt in collect_texts(at):
        for line in txt.splitlines():
            for m in BAD.finditer(line):
                seg = line[max(0, m.start()-30):m.start()+30]
                if ALLOW.search(seg): continue
                hits.append((attr, m.group(0), seg.strip()))
    return hits

cases = [("ℹ️ 关于", None, None)]
for s in STOCK_SUB: cases.append(("🎯 个股决策", s, None))
cases += [("🔭 建仓扫描", None, None), ("🛡️ 组合配置", None, None), ("📋 多票简报", None, None)]
for i in range(6): cases.append(("🔬 研究台", None, i))

print("="*84); print("UI 文字质量扫描（NaN/None/inf/未格式化 泄漏）"); print("="*84)
total = 0
for job, sl, si in cases:
    name = f"{job}" + (f" › {sl}" if sl else (f" › 研究#{si}" if si is not None else ""))
    try:
        hits = run_case(job, sl, si)
    except Exception as e:
        print(f"⚠️ {name}: 渲染异常 {type(e).__name__}: {e}"); continue
    if hits:
        total += len(hits)
        print(f"❌ {name}: {len(hits)} 处")
        seen = set()
        for attr, tok, seg in hits:
            if seg in seen: continue
            seen.add(seg)
            print(f"      [{attr}] …{seg}…")
    else:
        print(f"✅ {name}")
print("="*84); print(f"完成：共 {total} 处可疑 UI 文字"); print("="*84)
