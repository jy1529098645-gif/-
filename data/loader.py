"""数据层：价格 / 宏观 / 股票池加载，带本地缓存（parquet）。

缺失值策略（验收要求）：
- 价格：**不前向填充**（停牌/退市必须留 NaN，避免造假）。
- 宏观：**前向填充**（低频序列对齐到交易日）。

缓存：每个 ticker / 每个 FRED 序列各存一个 parquet，二次运行命中缓存不重复下载。
缓存按已下载日期范围做并集扩展：请求区间被缓存覆盖时直接切片返回。
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

import config

_CFG = config.load_config()
_CACHE = config.get_path("cache", _CFG)


def _ensure_ascii_ca_bundle() -> None:
    """curl_cffi（yfinance 底层）无法读取非 ASCII 路径下的 CA 证书。

    本项目路径含中文，certifi.where() 落在非 ASCII 路径，curl 报 error 77。
    解决：把 cacert.pem 复制到纯 ASCII 临时目录，并设置相关环境变量。
    """
    if os.environ.get("CURL_CA_BUNDLE") and Path(os.environ["CURL_CA_BUNDLE"]).exists():
        return
    try:
        import certifi

        src = certifi.where()
    except Exception:
        return
    if src.isascii():  # 路径已是 ASCII，无需处理
        os.environ.setdefault("CURL_CA_BUNDLE", src)
        os.environ.setdefault("SSL_CERT_FILE", src)
        return
    dst = Path(tempfile.gettempdir()) / "quantlab_cacert.pem"
    if not dst.exists() or dst.stat().st_size == 0:
        shutil.copyfile(src, dst)
    if not str(dst).isascii():
        return  # 临时目录本身也含非 ASCII，无能为力，留给上层报错
    os.environ["CURL_CA_BUNDLE"] = str(dst)
    os.environ["SSL_CERT_FILE"] = str(dst)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", str(dst))


_ensure_ascii_ca_bundle()


# ---------------------------------------------------------------------------
# 股票池
# ---------------------------------------------------------------------------
def load_universe(name: str | None = None) -> list[str]:
    """返回股票池成分。

    ⚠️ 当前 demo 池（spy_demo）为**幸存者偏差版本**，仅供 API 验证。
    无幸存者偏差（含已退市标的）的历史股票池见 Phase 6。
    """
    name = name or _CFG["universe"]["default"]
    if name == "sp500":  # 大票池：降低选股偏差（成分表 data/sp500_constituents.txt，离线 bundled）
        return load_sp500()
    entry = _CFG["universe"].get(name)
    if entry is None:
        raise KeyError(f"未知股票池 '{name}'，可选：{[k for k in _CFG['universe'] if k != 'default'] + ['sp500']}")
    return list(entry["tickers"])


def load_sp500() -> list[str]:
    """S&P 500 现成成分（离线 bundled，约 500 只）。用于横截面研究，大幅降低选股偏差。

    ⚠️ 仍是**现成**成分（含幸存者偏差：已剔除退市股）；真·无偏差需付费 PIT 数据。"""
    f = Path(config.ROOT) / "data" / "sp500_constituents.txt"
    if not f.exists():
        return []
    return [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]


# ---------------------------------------------------------------------------
# 价格
# ---------------------------------------------------------------------------
def _price_cache_path(ticker: str) -> Path:
    safe = ticker.replace("/", "_").replace("^", "_")
    return _CACHE / f"price_{safe}.parquet"


def _download_one(ticker: str, start: str, end: str | None) -> pd.Series:
    """下载单只标的的复权收盘价（auto_adjust=True）。返回以日期为 index 的 Series。"""
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df is None or df.empty:
        raise ValueError(f"{ticker}: 未取到数据（区间 {start}~{end}）")

    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # 多层列时取该 ticker 列
        close = close.iloc[:, 0]
    s = close.dropna()
    s.name = ticker
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def _load_one_cached(ticker: str, start: str, end: str | None) -> pd.Series:
    """带缓存地取单只标的复权收盘价。缓存覆盖请求区间则切片，否则下载并合并缓存。"""
    path = _price_cache_path(ticker)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    start_ts = pd.Timestamp(start)

    cached: pd.Series | None = None
    if path.exists():
        cached = pd.read_parquet(path).iloc[:, 0]
        cached.index = pd.to_datetime(cached.index)
        # 缓存已覆盖请求区间（容忍 5 个自然日的尾部新鲜度差）
        if cached.index.min() <= start_ts and cached.index.max() >= end_ts - pd.Timedelta(days=5):
            return cached.loc[start_ts:end_ts]

    # 下载（缓存缺失或不够新）。下载请求区间并与旧缓存并集后写回。
    fresh = _download_one(ticker, start, end)
    merged = fresh if cached is None else pd.concat([cached, fresh])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.to_frame().to_parquet(path)
    return merged.loc[start_ts:end_ts]


def load_prices(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """返回复权收盘价面板，index=日期(DatetimeIndex)，columns=ticker。带本地缓存。

    价格不前向填充：停牌/退市/上市前留 NaN。
    """
    start = start or _CFG["dates"]["start"]
    end = end or _CFG["dates"]["end"]  # 可能为 None → 取到今天

    series = {}
    for t in tickers:
        series[t] = _load_one_cached(t, start, end)

    panel = pd.DataFrame(series)
    panel.index.name = "date"
    return panel  # 不 ffill：保留 NaN


# ---------------------------------------------------------------------------
# OHLCV（K 线 / 成交量 / Volume Profile 用，复权）
# ---------------------------------------------------------------------------
def _ohlcv_cache_path(ticker: str) -> Path:
    safe = ticker.replace("/", "_").replace("^", "_")
    return _CACHE / f"ohlcv_{safe}.parquet"


def load_ohlcv(ticker: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """返回单只标的的复权 OHLCV：列 open/high/low/close/volume，index=日期。带缓存。

    用于 K 线进出场标记图、成交量、Volume Profile。auto_adjust=True（复权）。
    """
    import yfinance as yf

    start = start or _CFG["dates"]["start"]
    end = end or _CFG["dates"]["end"]
    path = _ohlcv_cache_path(ticker)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    start_ts = pd.Timestamp(start)

    cached: pd.DataFrame | None = None
    if path.exists():
        cached = pd.read_parquet(path)
        cached.index = pd.to_datetime(cached.index)
        if cached.index.min() <= start_ts and cached.index.max() >= end_ts - pd.Timedelta(days=5):
            return cached.loc[start_ts:end_ts]

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"{ticker}: 未取到 OHLCV（区间 {start}~{end}）")
    if isinstance(df.columns, pd.MultiIndex):  # 多层列时压平
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"

    merged = df if cached is None else pd.concat([cached, df])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.to_parquet(path)
    return merged.loc[start_ts:end_ts]


# ---------------------------------------------------------------------------
# 财报日期（补充规格 B / Phase F1，免费 yfinance）
# ---------------------------------------------------------------------------
def load_earnings_dates(ticker: str, limit: int = 80, refresh: bool = False) -> pd.DataFrame:
    """返回财报日历：index=财报公布日(tz-naive)，列 EPS Estimate / Reported EPS / Surprise(%)。

    PIT 性质：财报**日期**是提前公布的（无前视）；Surprise(%) 只有在公布日之后才可知。
    带 parquet 缓存。未来已排期但未公布的行 Reported EPS 为 NaN。
    """
    path = _CACHE / f"earnings_{ticker.replace('/', '_').replace('^', '_')}.parquet"
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    import yfinance as yf

    raw = yf.Ticker(ticker).get_earnings_dates(limit=limit)
    if raw is None or raw.empty:
        raise ValueError(f"{ticker}: 未取到财报日期")
    df = raw.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()  # 转 tz-naive 日期
    df.index.name = "earnings_date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_parquet(path)
    return df


# ---------------------------------------------------------------------------
# 宏观
#   优先级：FRED 官方 API（需 api_key，全历史，最贴合规格书）
#           → 否则 Yahoo 代理回退（^TNX-^IRX 收益率曲线；HYG/IEF 信用利差代理）。
#
#   ⚠️ 本网络环境对 FRED 公开 fredgraph.csv 只返回最近 ~3 年（代理改写日期参数），
#      且 /downloaddata 全历史端点被屏蔽。但 api.stlouisfed.org 可达，
#      因此只要在 config.macro.fred_api_key 或环境变量 FRED_API_KEY 填入免费 key，
#      即自动切换到 FRED 官方 API 取完整历史。
#      申请 key：https://fred.stlouisfed.org/docs/api/api_key.html
# ---------------------------------------------------------------------------
def _fred_api_key() -> str | None:
    """从环境变量或 config 取 FRED API key（环境变量优先）。"""
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key.strip()
    key = _CFG.get("macro", {}).get("fred_api_key")
    return str(key).strip() if key else None


def _macro_cache_path(tag: str) -> Path:
    safe = tag.replace("/", "_").replace("^", "_")
    return _CACHE / f"macro_{safe}.parquet"


def _fetch_fred_api(series_id: str, start: str, end: str | None, key: str) -> pd.Series:
    """FRED 官方 API 取单条全历史序列（curl_cffi 直连 api.stlouisfed.org）。"""
    import time

    from curl_cffi import requests as creq

    retries = int(_CFG["macro"].get("fred_max_retries", 4))
    timeout = int(_CFG["macro"].get("fred_timeout", 25))
    end_str = end or pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={key}&file_type=json"
        f"&observation_start={start}&observation_end={end_str}"
    )

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = creq.get(url, impersonate="chrome", timeout=timeout)
            r.raise_for_status()
            obs = r.json().get("observations", [])
            if not obs:
                raise ValueError("空响应")
            s = pd.Series(
                {pd.Timestamp(o["date"]): o["value"] for o in obs}, name=series_id
            )
            s = pd.to_numeric(s, errors="coerce").dropna()  # "." → NaN → 丢弃
            if s.empty:
                raise ValueError("全为缺失")
            return s.sort_index()
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"FRED API 序列 {series_id} 拉取失败（重试 {retries} 次）：{last_err}")


def _load_fred_one(series_id: str, start: str, end: str | None) -> pd.Series:
    """带缓存地经 FRED 官方 API 取单条序列。无 api_key 则抛错（交上层回退）。"""
    key = _fred_api_key()
    if not key:
        raise RuntimeError("未配置 FRED_API_KEY（环境变量或 config.macro.fred_api_key）")

    path = _macro_cache_path(f"fred_{series_id}")
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    start_ts = pd.Timestamp(start)

    cached: pd.Series | None = None
    if path.exists():
        cached = pd.read_parquet(path).iloc[:, 0]
        cached.index = pd.to_datetime(cached.index)
        if cached.index.min() <= start_ts and cached.index.max() >= end_ts - pd.Timedelta(days=10):
            return cached.loc[start_ts:end_ts]

    fresh = _fetch_fred_api(series_id, start, end, key)
    fresh.index = pd.to_datetime(fresh.index)
    merged = fresh if cached is None else pd.concat([cached, fresh])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.to_frame().to_parquet(path)
    return merged.loc[start_ts:end_ts]


def _yahoo_proxy_series(proxy: dict, start: str, end: str | None) -> pd.Series:
    """按 proxy.op 从 Yahoo 价格构造宏观代理序列。

    op:
      diff       → tickers[0] - tickers[1]            （如 ^TNX-^IRX = 10Y-3M 曲线）
      neg_ratio  → -(tickers[0] / tickers[1])          （如 -(HYG/IEF) 作信用利差代理，越高=越紧）
      ratio      → tickers[0] / tickers[1]
    """
    tickers = list(proxy["tickers"])
    op = proxy.get("op", "diff")
    px = load_prices(tickers, start, end)
    a, b = px[tickers[0]], px[tickers[1]]
    if op == "diff":
        s = a - b
    elif op == "ratio":
        s = a / b
    elif op == "neg_ratio":
        s = -(a / b)
    else:
        raise ValueError(f"未知 proxy.op '{op}'")
    return s.dropna()


def _load_macro_series(friendly: str, spec: dict, start: str, end: str | None) -> tuple[pd.Series, str]:
    """取单条宏观序列。优先 FRED 官方 API；不可达/无 key 时按 yahoo_proxy 回退。"""
    fred_id = spec.get("fred")
    if fred_id and _fred_api_key():
        try:
            s = _load_fred_one(fred_id, start, end)
            s.name = friendly
            return s, f"fred_api:{fred_id}"
        except Exception as e:  # noqa: BLE001
            if not spec.get("yahoo_proxy"):
                raise
            import warnings

            warnings.warn(
                f"宏观 '{friendly}' 的 FRED API 源 {fred_id} 失败（{type(e).__name__}），回退 Yahoo 代理。",
                stacklevel=2,
            )

    proxy = spec.get("yahoo_proxy")
    if proxy:
        s = _yahoo_proxy_series(proxy, start, end)
        s.name = friendly
        tag = f"{proxy.get('op', 'diff')}:{'/'.join(proxy['tickers'])}"
        return s, f"yahoo_proxy:{tag}"

    raise KeyError(f"宏观序列 '{friendly}' 既无可用 FRED（缺 key）也无 yahoo_proxy 配置")


def load_macro(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """加载宏观状态序列：信用利差、收益率曲线（可选 CAPE/盈利收益率）。

    有 FRED API key → 走官方 API 全历史；否则按 config 的 yahoo_proxy 回退。
    宏观序列**前向填充**对齐到日频（低频/节假日缺口）。
    DataFrame.attrs['sources'] 记录每列实际来源。
    """
    start = start or _CFG["dates"]["start"]
    end = end or _CFG["dates"]["end"]

    series_specs = _CFG["macro"]["series"]  # {friendly: {fred, yahoo_proxy?}}
    cols, sources = {}, {}
    for friendly, spec in series_specs.items():
        s, src = _load_macro_series(friendly, spec, start, end)
        cols[friendly] = s
        sources[friendly] = src

    macro = pd.DataFrame(cols).sort_index()
    macro = macro.ffill()  # 宏观前向填充
    macro.index.name = "date"
    macro.attrs["sources"] = sources
    return macro
