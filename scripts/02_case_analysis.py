# -*- coding: utf-8 -*-
"""
阶段 1：欺诈 case 分析（看 200 条欺诈样本长什么样）
====================================================
所属阶段：第 1 周 · 阶段 1（欺诈模式假设）

前置依赖：需先运行 scripts/01_load_and_inspect.py
          （读取 data/processed/train_merged.parquet / .pkl）

运行方式：
    python scripts/02_case_analysis.py

预期产出：
    - reports/figures/02_amt_distribution.png     ① 金额分布对比（log 轴）
    - reports/figures/02_hourly_fraud_rate.png    ② 按小时的欺诈率曲线
    - reports/figures/02_emaildomain_top15.png    ③ 邮箱域名 Top15 欺诈率
    - reports/figures/02_dist1_binned.png         ④ dist1 分箱欺诈率
    - reports/figures/02_card1_top20.png          ⑤ card1 Top20 高频卡欺诈率
    - reports/figures/02_identity_missing.png     ⑥ identity 有无 vs 欺诈率
    - reports/run_log.txt 追加各图对应的数字结论
    - 屏幕打印"欺诈模式假设清单"模板，供你根据图填结论

为什么要做这一步：建模之前先用肉眼理解欺诈长什么样。
好的风控分析师不是先跑模型，而是先提出可验证的假设（A/B/C），
再用图逐一验证——这也是面试里"你怎么理解业务"的标准答法。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config
from src import utils


def fraud_rate_by(df, col):
    """按某列分组计算欺诈率，返回排序后的 Series。"""
    return df.groupby(col)[config.TARGET_COL].mean().sort_values(ascending=False)


def main():
    utils.setup_plot_style()
    utils.log_section("阶段 1：欺诈 case 分析")

    utils.check_input(config.MERGED_TRAIN_NAME,
                      "请先运行：python scripts/01_load_and_inspect.py")
    df = utils.load_df(config.MERGED_TRAIN_NAME)
    utils.log(f"读取合并数据: {df.shape}")

    # 抽 200 条欺诈样本"肉眼过一遍"（打印到屏幕，建立直观感受）
    fraud_sample = df[df[config.TARGET_COL] == 1].sample(
        n=min(200, int(df[config.TARGET_COL].sum())),
        random_state=config.RANDOM_SEED)
    show_cols = [c for c in [config.ID_COL, "TransactionAmt", config.TIME_COL,
                             "card1", "card2", "addr1", "P_emaildomain",
                             "dist1", "DeviceType", "DeviceInfo"]
                 if c in df.columns]
    utils.log("\n=== 200 条欺诈样本速览（前 20 行打印，全部样本在内存中可自行查看）===")
    utils.log(fraud_sample[show_cols].head(20).to_string())

    fraud = df[df[config.TARGET_COL] == 1]
    normal = df[df[config.TARGET_COL] == 0]

    # ------------------------------------------------------------------
    # 图① 金额分布对比（log 轴 + 标注整数金额聚集）
    #   观察点：欺诈金额是否偏向"整数/整百"（脚本批量跑出来的金额往往取整），
    #   以及小额盗刷 vs 大额冒用的分布差异。
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.logspace(-1, 4, 60)
    ax.hist(normal["TransactionAmt"].dropna(), bins=bins, alpha=0.6,
            density=True, label="正常交易")
    ax.hist(fraud["TransactionAmt"].dropna(), bins=bins, alpha=0.6,
            density=True, label="欺诈交易")
    ax.set_xscale("log")
    ax.set_xlabel("TransactionAmt（美元，log 轴）")
    ax.set_ylabel("密度")
    ax.set_title("① 欺诈 vs 正常：交易金额分布")
    ax.legend()
    # 标注整数金额聚集：统计欺诈样本中金额为整数的占比
    int_ratio_f = (fraud["TransactionAmt"].dropna() % 1 == 0).mean()
    int_ratio_n = (normal["TransactionAmt"].dropna() % 1 == 0).mean()
    ax.annotate(f"整数金额占比：欺诈 {int_ratio_f:.1%} / 正常 {int_ratio_n:.1%}",
                xy=(0.02, 0.95), xycoords="axes fraction", va="top", fontsize=9)
    utils.save_fig(fig, "02_amt_distribution.png")
    plt.close(fig)
    utils.log(f"整数金额占比：欺诈 {int_ratio_f:.2%}，正常 {int_ratio_n:.2%}")

    # ------------------------------------------------------------------
    # 图② 按小时的欺诈率曲线
    #   TransactionDT 是相对起点的秒数，//3600%24 得到一天中的小时。
    #   观察点：欺诈是否在凌晨（真人睡觉、机器干活）显著更高。
    # ------------------------------------------------------------------
    df["_hour"] = (df[config.TIME_COL] // 3600 % 24).astype(int)
    hourly = fraud_rate_by(df, "_hour")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hourly.index, hourly.values, marker="o")
    overall = df[config.TARGET_COL].mean()
    ax.axhline(overall, color="red", linestyle="--",
               label=f"整体欺诈率 {overall:.2%}")
    ax.set_xlabel("一天中的小时（由 TransactionDT 推算）")
    ax.set_ylabel("欺诈率")
    ax.set_title("② 按小时的欺诈率")
    ax.legend()
    utils.save_fig(fig, "02_hourly_fraud_rate.png")
    plt.close(fig)
    utils.log(f"欺诈率最高的小时: {int(hourly.idxmax())} 时 "
              f"({hourly.max():.2%})，最低: {int(hourly.idxmin())} 时 "
              f"({hourly.min():.2%})")

    # ------------------------------------------------------------------
    # 图③ P_emaildomain Top15 域名欺诈率
    #   观察点：免费/匿名邮箱（如 protonmail）是否欺诈率显著高于企业邮箱。
    # ------------------------------------------------------------------
    top15_domains = df["P_emaildomain"].value_counts().head(15).index
    sub = df[df["P_emaildomain"].isin(top15_domains)]
    dom_rate = fraud_rate_by(sub, "P_emaildomain")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(dom_rate.index[::-1], dom_rate.values[::-1])
    ax.set_xlabel("欺诈率")
    ax.set_title("③ P_emaildomain Top15 域名的欺诈率")
    utils.save_fig(fig, "02_emaildomain_top15.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 图④ dist1 分箱欺诈率
    #   dist1 是账单地址与收货/常用地址间的距离（或类似距离度量）。
    #   观察点：距离越大是否欺诈率越高（异地盗刷模式）。
    # ------------------------------------------------------------------
    if "dist1" in df.columns:
        df["_dist1_bin"] = pd.cut(
            df["dist1"], bins=[-1, 1, 10, 50, 100, 500, np.inf],
            labels=["0~1", "1~10", "10~50", "50~100", "100~500", ">500"])
        dist_rate = fraud_rate_by(df.dropna(subset=["_dist1_bin"]), "_dist1_bin")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(dist_rate.index.astype(str), dist_rate.values)
        ax.set_xlabel("dist1 分箱")
        ax.set_ylabel("欺诈率")
        ax.set_title("④ dist1 分箱的欺诈率")
        utils.save_fig(fig, "02_dist1_binned.png")
        plt.close(fig)
    else:
        utils.log("（本数据无 dist1 列，跳过图④）")

    # ------------------------------------------------------------------
    # 图⑤ card1 Top20 高频卡欺诈率
    #   观察点：高频卡是否欺诈率异常高（一张卡短时间刷很多笔是典型的卡测试行为）。
    # ------------------------------------------------------------------
    top20_cards = df["card1"].value_counts().head(20).index
    sub = df[df["card1"].isin(top20_cards)]
    card_rate = fraud_rate_by(sub, "card1")
    card_cnt = sub["card1"].value_counts()
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(card_rate))
    ax.bar(x, card_rate.values)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{c}\n({card_cnt[c]}笔)" for c in card_rate.index],
                       rotation=45, fontsize=8)
    ax.axhline(overall, color="red", linestyle="--",
               label=f"整体欺诈率 {overall:.2%}")
    ax.set_ylabel("欺诈率")
    ax.set_title("⑤ card1 Top20 高频卡的欺诈率（括号内为交易笔数）")
    ax.legend()
    utils.save_fig(fig, "02_card1_top20.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 图⑥ identity 缺失 vs 有的欺诈率对比
    #   这是本项目最重要的观察之一：没有设备指纹的交易欺诈率是否显著更高？
    #   若是，则"identity 缺失"本身就是一个强特征（无需任何复杂加工）。
    # ------------------------------------------------------------------
    has_identity = df["id_01"].notna() if "id_01" in df.columns else None
    if has_identity is not None:
        rate_has = df.loc[has_identity, config.TARGET_COL].mean()
        rate_no = df.loc[~has_identity, config.TARGET_COL].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["有 identity（有设备指纹）", "无 identity（缺失）"],
               [rate_has, rate_no], color=["steelblue", "salmon"])
        ax.set_ylabel("欺诈率")
        ax.set_title("⑥ identity 有无 vs 欺诈率")
        for i, v in enumerate([rate_has, rate_no]):
            ax.text(i, v, f"{v:.2%}", ha="center", va="bottom")
        utils.save_fig(fig, "02_identity_missing.png")
        plt.close(fig)
        utils.log(f"有 identity 的欺诈率: {rate_has:.2%}，"
                  f"无 identity 的欺诈率: {rate_no:.2%}")

    # ------------------------------------------------------------------
    # 欺诈模式假设清单（打印模板，请你看着上面的图填结论）
    # ------------------------------------------------------------------
    utils.log_section("欺诈模式假设清单（请根据 reports/figures/02_*.png 填结论）")
    utils.log("""
假设 A（设备/卡聚集）：
  内容：同一 uid（卡+地址+邮箱）短时间高频交易的欺诈率显著更高。
  证据图：02_card1_top20.png（阶段 2 的 03 脚本会用 uid 正式验证）
  结论：【待填写】

假设 B（金额分布异常）：
  内容：欺诈金额分布与正常不同，且整数金额占比更高（脚本批量作案特征）。
  证据图：02_amt_distribution.png
  结论：【待填写】

假设 C（无设备指纹欺诈率更高）：
  内容：没有 identity 信息的交易欺诈率显著更高（欺诈者会规避设备检测）。
  证据图：02_identity_missing.png
  结论：【待填写】

补充观察（小时模式 / 邮箱域名 / 距离）：
  证据图：02_hourly_fraud_rate.png / 02_emaildomain_top15.png / 02_dist1_binned.png
  结论：【待填写】
""")

    utils.log("阶段 1 完成。下一步：python scripts/03_oot_uid_features.py")


if __name__ == "__main__":
    main()
