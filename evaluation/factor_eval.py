"""因子评估，封装 alphalens-reloaded。

铁律：每个 IC/收益数字都随样本期长度一并给出，并附判读提示
——「单因子 IC 0.03–0.05 即属可用；高得离谱要怀疑前视偏差」。
分位收益必须能和无条件基准对照（最高/最低分位、多空价差）。
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

import config

IC_HINT = (
    "判读：单因子 IC 0.03–0.05 即属可用（因子天生很弱，靠广度与一致性取胜）；"
    "IC 高得离谱（如 >0.2）通常意味着前视偏差/数据泄漏，需怀疑。"
)


def _to_alphalens_factor(factor_values: pd.DataFrame) -> pd.Series:
    """宽表（date × ticker）→ alphalens 需要的 MultiIndex Series (date, asset)。"""
    s = factor_values.stack()
    s.index = s.index.set_names(["date", "asset"])
    return s.dropna()


def evaluate_factor(
    factor_values: pd.DataFrame,
    prices: pd.DataFrame,
    quantiles: int = 5,
    periods: tuple[int, ...] = (1, 5, 21, 63),
    tearsheet: bool = False,
    report_name: str | None = None,
) -> dict:
    """对齐因子值与远期收益，输出 IC 统计、分位收益、多空价差、换手率等。

    返回 dict 含：sample_start/end、n_obs（因子样本点数）、ic（每周期 mean/std/IR）、
    quantile_returns、long_short_spread、note。
    tearsheet=True 时把 alphalens full tear sheet 存到 reports/{report_name}_tearsheet.pdf。
    """
    import alphalens as al

    factor = _to_alphalens_factor(factor_values)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # alphalens 内部大量 pandas 兼容告警
        factor_data = al.utils.get_clean_factor_and_forward_returns(
            factor,
            prices,
            quantiles=quantiles,
            periods=periods,
        )

        # --- IC ---
        ic = al.performance.factor_information_coefficient(factor_data)
        ic_summary = pd.DataFrame(
            {
                "IC_mean": ic.mean(),
                "IC_std": ic.std(),
                "IR": ic.mean() / ic.std(),
                "n_days": ic.notna().sum(),
            }
        )

        # --- 分位平均收益（与基准对照的原料）---
        mean_q, _ = al.performance.mean_return_by_quantile(factor_data)
        # 多空价差：最高分位 − 最低分位
        top, bot = mean_q.index.max(), mean_q.index.min()
        long_short = mean_q.loc[top] - mean_q.loc[bot]

        # --- 换手率（首个周期）---
        try:
            q_top_turnover = al.performance.quantile_turnover(
                factor_data["factor_quantile"], top
            )
            turnover_mean = float(q_top_turnover.mean())
        except Exception:  # noqa: BLE001
            turnover_mean = float("nan")

    dates = factor_data.index.get_level_values("date")
    start, end = dates.min(), dates.max()
    n_years = (end - start).days / 365.25

    result = {
        "sample_start": start,
        "sample_end": end,
        "sample_years": round(n_years, 2),
        "n_obs": int(len(factor_data)),
        "n_assets": int(factor_data.index.get_level_values("asset").nunique()),
        "quantiles": quantiles,
        "periods": list(periods),
        "ic": ic_summary,
        "quantile_returns": mean_q,
        "long_short_spread": long_short,
        "top_quantile_turnover": turnover_mean,
        "note": (
            f"样本期 {start.date()} ~ {end.date()}（约 {n_years:.1f} 年，"
            f"{factor_data.index.get_level_values('asset').nunique()} 只标的）。{IC_HINT}"
        ),
    }

    if tearsheet:
        result["tearsheet_path"] = _save_tearsheet(factor_data, report_name or "factor")

    return result


def _save_tearsheet(factor_data: pd.DataFrame, report_name: str) -> str:
    """生成 alphalens full tear sheet 并把所有图存成单个 PDF。

    alphalens 每段绘图后会 gf.close() 关掉图（plt.show 本为 notebook 内联显示）。
    因此临时打补丁 plt.show：每次调用时把当前图存进 PDF（此刻图尚未被 close）。
    """
    import matplotlib

    matplotlib.use("Agg")  # 非交互后端，便于保存
    import alphalens as al
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    reports = config.get_path("reports")
    out = Path(reports) / f"{report_name}_tearsheet.pdf"

    plt.close("all")
    orig_show = plt.show
    with PdfPages(out) as pdf:
        def _capture(*_a, **_k):
            fig = plt.gcf()
            if fig.get_axes():  # 跳过空图
                pdf.savefig(fig, bbox_inches="tight")

        plt.show = _capture
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                al.tears.create_full_tear_sheet(factor_data)
        finally:
            plt.show = orig_show
    plt.close("all")
    return str(out)


def print_report(result: dict) -> None:
    """把评估结果打印成人类可读报告。"""
    print(f"=== 因子评估 ===\n{result['note']}\n")
    print(f"样本点数 N = {result['n_obs']}，标的数 = {result['n_assets']}，"
          f"分位数 = {result['quantiles']}\n")
    print("IC 统计（按远期周期）：")
    print(result["ic"].round(4).to_string())
    print("\n多空价差（最高分位 − 最低分位，按周期）：")
    print(result["long_short_spread"].round(5).to_string())
    print(f"\n最高分位换手率（均值）：{result['top_quantile_turnover']:.3f}")
