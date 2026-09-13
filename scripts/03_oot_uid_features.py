# -*- coding: utf-8 -*-
"""
阶段 2：OOT 时间切分 + UID 实体聚合特征
========================================
所属阶段：第 1 周 · 阶段 2（特征工程：UID 聚合 + OOT 切分）

前置依赖：需先运行 scripts/01_load_and_inspect.py
          （读取 data/processed/train_merged.parquet / .pkl）

运行方式：
    python scripts/03_oot_uid_features.py

预期产出：
    - data/processed/oot_train.parquet（或 .pkl）  早期 80%，训练集 tr
    - data/processed/oot_test.parquet（或 .pkl）   晚期 20%，测试集 te
    - reports/figures/03_uid_cnt_fraud_rate.png    假设 A 验证图
    - reports/figures/03_velocity_fraud_rate.png   velocity 特征验证图
    - reports/run_log.txt 追加：切分点、tr/te 欺诈率、新 uid 欺诈率、分箱结果

核心设计（面试高频考点）：
    1. 为什么按时间 OOT 切分而不是随机切分？
       风控模型上线后面对的永远是"未来"的交易。随机切分会让同一 uid 的
       行为同时出现在训练集和测试集（信息泄漏），指标虚高；OOT 切分
       模拟真实上线场景，才能回答"模型对未来欺诈的泛化能力如何"。

    2. uid = card1_card2_addr1_P_emaildomain
       把"卡 + 账单地址 + 邮箱域名"拼成用户实体标识，用来捕捉
       团伙/设备的聚集行为——单个交易看不出异常，聚合到实体维度就看出来了。

    3. 【口径声明】严格做法是"截至当前交易时刻"的历史统计（无任何未来信息）；
       本脚本用全训练期统计的简化版，属于近似口径，报告中必须声明这一点。
       OOT 切分已把未来样本挡在 te 里，泄漏有限但不为零——
       面试时主动说出来是加分项（说明你意识到了 leakage 问题）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config
from src import utils


def validate_velocity_features(te):
    """按 uid_secs_since_last 分箱，验证 velocity 特征与欺诈率的关系。

    业务直觉：距上一笔交易越近（刷单/卡测试式的"连击"），欺诈率应越高；
    首笔交易（uid_secs_since_last = -1，uid 首笔或 te 中的新 uid）欺诈率
    应接近新 uid 水平。在 te（OOT 集）上验证，口径与模型评估一致。

    分箱样本量 < 1000 的箱与相邻箱合并后再输出（小样本箱的欺诈率
    不可信，直接展示会误导业务方）。
    """
    MIN_SAMPLES = 1000
    secs = te["uid_secs_since_last"]
    # pd.cut 左开右闭：-1（首笔）单独一箱，其余按秒数切
    cat = pd.cut(secs, bins=[-np.inf, -0.5, 60, 3600, 86400, np.inf],
                 labels=["首笔交易(-1)", "<60秒", "60~3600秒",
                         "3600~86400秒", ">86400秒"])

    stats = (te.groupby(cat, observed=False)[config.TARGET_COL]
               .agg(交易数="count", 欺诈率="mean"))
    utils.log("\nvelocity 验证 —— uid_secs_since_last 分箱欺诈率（te 上，原始分箱）：")
    for idx, row in stats.iterrows():
        utils.log(f"  {idx:<14s} 交易数 {int(row['交易数']):>7d}  "
                  f"欺诈率 {row['欺诈率']:.2%}")

    # 小样本箱（<1000 笔）与相邻箱合并：逐个折叠，保序合并到下一箱
    rows = [(str(idx), int(r["交易数"]), float(r["欺诈率"]) * int(r["交易数"]))
            for idx, r in stats.iterrows()]
    merged = []
    for label, cnt, fraud in rows:
        if merged and merged[-1][1] < MIN_SAMPLES:
            pl, pc, pf = merged.pop()
            merged.append((f"{pl}+{label}", pc + cnt, pf + fraud))
        else:
            merged.append((label, cnt, fraud))
    if len(merged) > 1 and merged[-1][1] < MIN_SAMPLES:   # 末箱仍小则并回前一箱
        label, cnt, fraud = merged.pop()
        pl, pc, pf = merged.pop()
        merged.append((f"{pl}+{label}", pc + cnt, pf + fraud))

    utils.log(f"\n合并小样本箱（<{MIN_SAMPLES} 笔）后：")
    plot_labels, plot_rates, plot_counts = [], [], []
    for label, cnt, fraud in merged:
        rate = fraud / cnt
        utils.log(f"  {label:<24s} 交易数 {cnt:>7d}  欺诈率 {rate:.2%}")
        plot_labels.append(label)
        plot_rates.append(rate)
        plot_counts.append(cnt)

    overall = te[config.TARGET_COL].mean()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(plot_labels, plot_rates, color="steelblue")
    for bar, cnt in zip(bars, plot_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"n={cnt:,}", ha="center", va="bottom", fontsize=9)
    ax.axhline(overall, color="red", linestyle="--",
               label=f"te 整体欺诈率 {overall:.2%}")
    ax.set_xlabel("距 uid 上一笔交易的秒数（uid_secs_since_last）")
    ax.set_ylabel("欺诈率")
    ax.set_title("velocity 验证：uid 交易间隔 vs 欺诈率（te）")
    ax.legend()
    plt.xticks(rotation=15)
    utils.save_fig(fig, "03_velocity_fraud_rate.png")
    plt.close(fig)


def main():
    utils.setup_plot_style()
    utils.log_section("阶段 2：OOT 切分 + UID 聚合特征")

    utils.check_input(config.MERGED_TRAIN_NAME,
                      "请先运行：python scripts/01_load_and_inspect.py")
    df = utils.load_df(config.MERGED_TRAIN_NAME)
    utils.log(f"读取合并数据: {df.shape}")

    # ------------------------------------------------------------------
    # 1. 按时间排序，80% 分位点 OOT 切分
    # ------------------------------------------------------------------
    df = df.sort_values(config.TIME_COL).reset_index(drop=True)
    cut = int(len(df) * config.OOT_SPLIT_RATIO)
    cut_time = df.loc[cut, config.TIME_COL]
    utils.log(f"OOT 切分点: 第 {cut} 行，{config.TIME_COL} = {cut_time:.0f}"
              f"（前 {config.OOT_SPLIT_RATIO:.0%} 为 tr，其余为 te）")

    # ------------------------------------------------------------------
    # 2. 构造 uid 聚合特征 + velocity 时序频度特征（完整时间线上统一计算）
    # ------------------------------------------------------------------
    # velocity 特征（uid_cnt_1h / uid_cnt_24h / uid_secs_since_last）在
    # 完整时间线上统一计算——本脚本的 df 就是按 TransactionDT 排序的
    # 完整时间线，等价于"tr 和 te 拼回一起算再切回去"：每笔交易只统计
    # 该时刻之前的历史（无未来信息），所以切回 tr/te 后，te 早期的交易
    # 天然能看到 tr 时段的历史。这与 uid_cnt 等"全训练期聚合"的近似
    # 口径不同，velocity 是严格无泄漏口径（见 utils.build_velocity_features）。
    df = utils.build_uid_features(df)
    utils.log(f"uid 总数: {df['uid'].nunique()}，平均每 uid "
              f"{len(df) / df['uid'].nunique():.2f} 笔交易")
    utils.log("velocity 特征已生成: uid_cnt_1h / uid_cnt_24h / "
              "uid_secs_since_last（严格只统计当前时刻之前的历史）")

    # ------------------------------------------------------------------
    # 3. 切分 tr / te
    # ------------------------------------------------------------------
    tr = df.iloc[:cut].copy()
    te = df.iloc[cut:].copy()
    utils.log(f"tr 形状: {tr.shape}，欺诈率 {tr[config.TARGET_COL].mean():.2%}")
    utils.log(f"te 形状: {te.shape}，欺诈率 {te[config.TARGET_COL].mean():.2%}")
    utils.log("（tr/te 欺诈率通常不同——欺诈模式随时间漂移，这正是 OOT 要面对的）")

    # ------------------------------------------------------------------
    # 4. te 的新 uid 分析（冷启动问题）
    #    新 uid = 在 tr 里从未出现过的 uid。线上总会遇到新用户，
    #    他们的聚合特征是"一笔交易的统计"，模型对冷启动用户的表现
    #    必须单独看——这也是报告第 4 章"冷启动发现"的素材。
    # ------------------------------------------------------------------
    known_uids = set(tr["uid"].unique())
    te["is_new_uid"] = (~te["uid"].isin(known_uids)).astype(int)
    tr["is_new_uid"] = 0  # tr 内部不标记（简化口径）
    new_rate = te.loc[te["is_new_uid"] == 1, config.TARGET_COL].mean()
    old_rate = te.loc[te["is_new_uid"] == 0, config.TARGET_COL].mean()
    utils.log(f"\n冷启动分析（te 上）:")
    utils.log(f"  新 uid 交易占比: {te['is_new_uid'].mean():.1%}")
    utils.log(f"  新 uid 欺诈率: {new_rate:.2%}  vs  老 uid 欺诈率: {old_rate:.2%}")

    # ------------------------------------------------------------------
    # 5. 验证假设 A：按 uid_cnt 分箱看欺诈率
    #    业务直觉：只出现 1 次的 uid 是"一次性盗刷"，高频 uid 是"卡测试/团伙"，
    #    两端的欺诈率都可能显著高于中间段——画出曲线即见分晓。
    # ------------------------------------------------------------------
    df["_uid_cnt_bin"] = pd.cut(
        df["uid_cnt"], bins=[0, 1, 3, 10, np.inf],
        labels=["1 次", "2~3 次", "4~10 次", ">10 次"])
    bin_rate = (df.dropna(subset=["_uid_cnt_bin"])
                  .groupby("_uid_cnt_bin", observed=True)[config.TARGET_COL]
                  .agg(["mean", "count"]))
    utils.log("\n假设 A 验证 —— uid_cnt 分箱欺诈率：")
    for idx, row in bin_rate.iterrows():
        utils.log(f"  {idx:<8s} 交易数 {int(row['count']):>7d}  "
                  f"欺诈率 {row['mean']:.2%}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(bin_rate.index.astype(str), bin_rate["mean"], color="steelblue")
    ax.axhline(df[config.TARGET_COL].mean(), color="red", linestyle="--",
               label=f"整体欺诈率 {df[config.TARGET_COL].mean():.2%}")
    ax.set_xlabel("uid 交易次数分箱")
    ax.set_ylabel("欺诈率")
    ax.set_title("假设 A 验证：uid 频次 vs 欺诈率")
    ax.legend()
    utils.save_fig(fig, "03_uid_cnt_fraud_rate.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 6. 验证 velocity 特征：按 uid_secs_since_last 分箱看欺诈率
    #    业务直觉：uid 的"连击"交易（距上一笔很近）更像卡测试/盗刷脚本，
    #    欺诈率应显著高于隔了很久的正常复购——画出分箱曲线即见分晓。
    # ------------------------------------------------------------------
    validate_velocity_features(te)

    # ------------------------------------------------------------------
    # 7. 保存 tr / te
    # ------------------------------------------------------------------
    utils.save_df(tr, config.OOT_TRAIN_NAME)
    utils.save_df(te, config.OOT_TEST_NAME)

    utils.log("\n阶段 2 完成。下一步：python scripts/04_train_lgbm.py")


if __name__ == "__main__":
    main()
