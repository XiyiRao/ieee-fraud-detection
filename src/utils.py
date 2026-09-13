# -*- coding: utf-8 -*-
"""
通用工具函数。

包含：
- setup_plot_style()   统一 matplotlib 风格（含中文字体）
- save_fig()           保存图片到 reports/figures/
- log()                print 的同时追加写入 reports/run_log.txt
- reduce_mem_usage()   DataFrame 内存压缩
- save_df() / load_df() 中间文件读写（优先 parquet，无 pyarrow 自动降级 pkl）
- check_input()        检查前置脚本产出的输入文件是否存在
- build_uid_features() uid 构造与聚合特征（03/05 共用，保证口径一致）
- build_velocity_features() velocity 时序频度特征（严格无未来信息口径）
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# 让本模块无论从哪个目录被 import，都能找到 src.config
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
def setup_plot_style():
    """统一 matplotlib 绘图风格，处理中文字体与负号显示问题。

    Windows 上常用中文字体是 Microsoft YaHei（微软雅黑）和 SimHei（黑体），
    按顺序 fallback；axes.unicode_minus=False 避免负号被渲染成方块。
    """
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 100


def save_fig(fig, name: str) -> Path:
    """把 matplotlib figure 保存到 reports/figures/ 目录。

    参数
    ----
    fig : matplotlib.figure.Figure
    name : str
        文件名（含扩展名），例如 "02_amt_distribution.png"

    返回
    ----
    Path : 保存的完整路径
    """
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = config.FIGURES_DIR / name
    fig.savefig(out, bbox_inches="tight")
    log(f"[图表已保存] {out}")
    return out


# ---------------------------------------------------------------------------
# 模型文件读写
# ---------------------------------------------------------------------------
def save_booster(booster, path: Path) -> Path:
    """保存 LightGBM booster 到 path。

    LightGBM 底层用 ANSI fopen 读写模型文件，项目路径含中文等非 ASCII
    字符时会报 "not available for writes"；先写到系统临时目录（纯 ASCII）
    再移动过来，绕过该缺陷。
    """
    tmp = Path(tempfile.gettempdir()) / f"lgbm_save_{os.getpid()}.txt"
    booster.save_model(str(tmp))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp), str(path))
    return path


def load_booster(path: Path):
    """从 path 加载 LightGBM booster（与 save_booster 相同的绕法）。"""
    from lightgbm import Booster

    tmp = Path(tempfile.gettempdir()) / f"lgbm_load_{os.getpid()}.txt"
    shutil.copyfile(str(path), str(tmp))
    booster = Booster(model_file=str(tmp))
    tmp.unlink(missing_ok=True)
    return booster


# ---------------------------------------------------------------------------
# 日志：屏幕打印 + 追加写入 reports/run_log.txt
# ---------------------------------------------------------------------------
def log(msg=""):
    """打印关键输出，同时追加写入 reports/run_log.txt，方便事后回溯各阶段数字。"""
    text = str(msg)
    print(text)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RUN_LOG, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def log_section(title: str):
    """打印一个分节标题，让日志更易读。"""
    log("")
    log("=" * 60)
    log(title)
    log("=" * 60)


# ---------------------------------------------------------------------------
# 内存压缩
# ---------------------------------------------------------------------------
def reduce_mem_usage(df: pd.DataFrame, name: str = "df") -> pd.DataFrame:
    """通过向下转型数值列来压缩 DataFrame 内存占用。

    IEEE-CIS 数据有 394 列，float64 全量加载很占内存；
    把能放下的列降级为 float32/int32/int16，通常能省 50%+ 内存，
    对 LightGBM 训练精度影响可忽略。

    注意：类别列（object）不做处理，留给建模脚本转 category。
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isna(c_min) or pd.isna(c_max):
                continue
            if pd.api.types.is_integer_dtype(col_type):
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                # float 统一尝试降为 float32（float16 精度损失较大，不用）
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    log(f"[内存压缩] {name}: {start_mem:.1f} MB -> {end_mem:.1f} MB"
        f"（节省 {100 * (start_mem - end_mem) / max(start_mem, 1e-9):.1f}%）")
    return df


# ---------------------------------------------------------------------------
# 中间文件读写：优先 parquet，没有 pyarrow 则降级为 pkl
# ---------------------------------------------------------------------------
def _resolve_path(base_path: Path) -> Path:
    """根据 pyarrow 是否可用，决定中间文件的实际扩展名。"""
    try:
        import pyarrow  # noqa: F401
        return base_path.with_suffix(".parquet")
    except ImportError:
        return base_path.with_suffix(".pkl")


def save_df(df: pd.DataFrame, name: str) -> Path:
    """把 DataFrame 存到 data/processed/<name>.parquet（或 .pkl）。"""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    base = config.PROCESSED_DIR / name
    path = _resolve_path(base)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_pickle(path)
    log(f"[中间文件已保存] {path}  形状={df.shape}")
    return path


def load_df(name: str) -> pd.DataFrame:
    """从 data/processed/ 读取中间文件（自动识别 parquet/pkl）。"""
    base = config.PROCESSED_DIR / name
    pq, pk = base.with_suffix(".parquet"), base.with_suffix(".pkl")
    if pq.exists():
        return pd.read_parquet(pq)
    if pk.exists():
        return pd.read_pickle(pk)
    raise FileNotFoundError(f"找不到中间文件：{pq} 或 {pk}")


def check_input(name: str, hint: str):
    """检查前置脚本产出的中间文件是否存在，不存在则友好退出。

    参数
    ----
    name : str
        中间文件名（不含扩展名），如 "train_merged"
    hint : str
        提示信息，告诉用户应该先运行哪个脚本
    """
    base = config.PROCESSED_DIR / name
    if not (base.with_suffix(".parquet").exists() or base.with_suffix(".pkl").exists()):
        print("!" * 60)
        print(f"缺少输入文件：{base}.parquet / .pkl")
        print(hint)
        print("!" * 60)
        sys.exit(1)


def check_raw_csv(path: Path, hint: str):
    """检查原始 CSV 是否存在，不存在则提示去下载数据。"""
    if not path.exists():
        print("!" * 60)
        print(f"缺少原始数据文件：{path}")
        print(hint)
        print("请按 data/README.md 的说明从 Kaggle 下载数据后放入 data/ 目录。")
        print("!" * 60)
        sys.exit(1)


# ---------------------------------------------------------------------------
# UID 构造与聚合特征（03 与 05 共用，保证切分点敏感性实验的口径一致）
# ---------------------------------------------------------------------------
def build_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """构造 velocity（时序频度）特征：只统计"当前交易时刻之前"的历史。

    在完整时间线上统一计算（每笔交易只用它之前的数据，无未来信息），
    因此切回 tr/te 后，te 早期的交易天然能看到 tr 时段的历史——
    这正是"把 tr 和 te 拼回完整时间线再算"的意义。

    特征（按 uid 分组，窗口均为左开右开区间 (t - window, t)，不含当前这笔）：
    - uid_cnt_1h          : 该 uid 在当前交易之前 1 小时（3600 秒）内的历史笔数
    - uid_cnt_24h         : 同理，24 小时（86400 秒）窗口
    - uid_secs_since_last : 距该 uid 上一笔交易的秒数；uid 的首笔交易填 -1
                            （不能用 0——0 会被误读为"上一笔就在同一秒"）

    实现：按 TransactionDT 排序后，对每个 uid 的时间序列做
    np.searchsorted 求窗口左边界下标，O(n log n)，全量 59 万条约几秒。

    与 uid_cnt / uid_amt_mean 等"全训练期聚合"特征不同，velocity 是
    严格无泄漏口径，可直接用于线上实时特征工程。
    """
    df = df.copy()
    original_order = df.index.to_numpy()
    df = df.sort_values(config.TIME_COL, kind="mergesort")  # 稳定排序，保序

    t = df[config.TIME_COL].to_numpy(dtype=np.float64)
    n = len(df)
    cnt_1h = np.zeros(n, dtype=np.int32)
    cnt_24h = np.zeros(n, dtype=np.int32)
    prev = np.full(n, np.nan)

    for pos in df.groupby("uid").indices.values():
        times = t[pos]                       # df 已按时间排序，组内单调不减
        prev[pos[1:]] = times[:-1]
        # 窗口左边界：第一个时间 > t - window 的下标；
        # 当前笔之前的笔数 = 当前下标 - 左边界下标（天然不含当前这笔）
        left_1h = np.searchsorted(times, times - 3600.0, side="right")
        cnt_1h[pos] = np.arange(len(times)) - left_1h
        left_24h = np.searchsorted(times, times - 86400.0, side="right")
        cnt_24h[pos] = np.arange(len(times)) - left_24h

    secs = t - prev
    secs[np.isnan(prev)] = -1.0              # uid 首笔交易，无上一笔

    df["uid_cnt_1h"] = cnt_1h
    df["uid_cnt_24h"] = cnt_24h
    df["uid_secs_since_last"] = secs.astype(np.float32)
    return df.loc[original_order]            # 恢复调用前的行顺序


def build_uid_features(df: pd.DataFrame) -> pd.DataFrame:
    """构造 uid 及其聚合特征。

    uid = card1_card2_addr1_P_emaildomain
    业务含义：把"同一张卡 + 同一账单地址 + 同一邮箱域名"视作同一个用户实体，
    用来捕捉团伙/设备的聚集行为。

    聚合特征：
    - uid_cnt        : 该 uid 在数据中出现的交易次数（频次）
    - uid_amt_mean   : 该 uid 的平均交易金额
    - uid_amt_std    : 该 uid 交易金额的标准差
    - amt_to_uid_mean: 当前金额 / uid 平均金额（偏离自己习惯多少倍）
    - uid_cnt_1h / uid_cnt_24h / uid_secs_since_last
                     : velocity 时序频度特征，见 build_velocity_features
                       （严格"只统计当前时刻之前"的口径，无未来信息）

    【口径声明】uid_cnt / uid_amt_* 严格的做法是"截至当前交易时刻"的历史
    统计（避免未来信息泄漏）；这里用全训练期统计的简化版，属于近似口径，
    报告中必须声明。OOT 切分已经把未来样本挡在测试集里，泄漏程度有限，
    但不是零。velocity 特征（build_velocity_features）是严格无泄漏口径。
    """
    df = df.copy()
    for c in ["card1", "card2", "addr1", "P_emaildomain"]:
        if c not in df.columns:
            df[c] = "unknown"
    df["uid"] = (
        df["card1"].astype(str) + "_"
        + df["card2"].astype(str) + "_"
        + df["addr1"].astype(str) + "_"
        + df["P_emaildomain"].astype(str)
    )

    grp = df.groupby("uid")["TransactionAmt"]
    df["uid_cnt"] = grp.transform("count")
    df["uid_amt_mean"] = grp.transform("mean")
    df["uid_amt_std"] = grp.transform("std")
    # 加 1e-6 防止除以 0
    df["amt_to_uid_mean"] = df["TransactionAmt"] / (df["uid_amt_mean"] + 1e-6)
    # velocity 特征：严格"只统计当前时刻之前"的历史，与上面的全期聚合互补
    df = build_velocity_features(df)
    return df
