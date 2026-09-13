# -*- coding: utf-8 -*-
"""
阶段 5：三级处置矩阵（拦截 / 二次验证 / 放行）+ 成本权衡
==========================================================
所属阶段：第 2~3 周 · 阶段 5（策略落地）

前置依赖：需先运行 scripts/04_train_lgbm.py
          （读取 data/processed/oot_test 和 models/lgbm_model.txt）

运行方式：
    python scripts/06_three_tier_strategy.py

预期产出：
    - reports/figures/06_net_benefit_heatmap.png   净收益-阈值对热力图
    - reports/run_log.txt 追加：最优 (T1, T2)、三级处置表、回测句式

核心思想：
    模型输出的是"分数"，业务要的是"动作"。把所有交易一刀切（拦/放）浪费
    模型的区分能力——中间分段的灰名单用低成本的二次验证（短信/人脸）
    来兜底，既拦欺诈又保体验。这就是"三级处置矩阵"：
        分数 ≥ T1        → 直接拦截（高风险）
        T2 ≤ 分数 < T1   → 二次验证（中风险，验证通过则放行）
        分数 < T2        → 直接放行（低风险）
    最优 (T1, T2) 通过扫描候选阈值对、比较净收益选出。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src import utils

# ---------------------------------------------------------------------------
# 成本假设（元/笔）——面试时要能讲清楚每个数字的业务含义和敏感性
# ---------------------------------------------------------------------------
COST_FALSE_KILL = 50.0    # 误杀 1 笔正常交易的损失（用户流失/投诉/客服成本）
COST_VERIFY = 5.0         # 1 笔二次验证的成本（短信/人脸通道费）
# 拦截 1 笔欺诈避免的损失 = 该笔交易金额（逐笔不同，用真实 TransactionAmt 算）
# 二次验证假设能挽回其中 90% 的正常用户、放走 10% 的欺诈（可调）
VERIFY_SAVE_NORMAL = 0.9
VERIFY_MISS_FRAUD = 0.1


def score_te():
    """加载模型对 te 打分，返回 (te, pred)。"""
    model_path = config.MODELS_DIR / "lgbm_model.txt"
    if not model_path.exists():
        print("缺少模型文件，请先运行：python scripts/04_train_lgbm.py")
        sys.exit(1)
    te = utils.load_df(config.OOT_TEST_NAME)
    booster = utils.load_booster(model_path)
    drop_cols = [c for c in [config.ID_COL, config.TARGET_COL, "uid",
                             "is_new_uid", "TransactionDT"] if c in te.columns]
    X = te.drop(columns=drop_cols)
    for c in X.select_dtypes(include=["object"]).columns:
        X[c] = X[c].astype("category")
    return te, booster.predict(X)


def evaluate_tier(te, pred, t1, t2):
    """给定阈值对 (T1, T2)，在 te 上回测三级处置的各项指标与净收益。

    净收益 = 拦截层避免的欺诈损失 + 验证层拦住的欺诈损失
             - 误杀正常用户的损失 - 二次验证的通道成本
    """
    y = te[config.TARGET_COL].values
    amt = te["TransactionAmt"].fillna(0).values

    block = pred >= t1                        # 拦截层
    verify = (pred >= t2) & (pred < t1)       # 二次验证层
    # 其余为放行层

    n_total = len(te)
    n_fraud_total = y.sum()

    # --- 拦截层 ---
    blk_fraud_amt = amt[block & (y == 1)].sum()      # 避免的欺诈损失
    blk_fraud_cnt = int((block & (y == 1)).sum())
    blk_normal_cnt = int((block & (y == 0)).sum())   # 误杀数
    cost_block = blk_normal_cnt * COST_FALSE_KILL

    # --- 二次验证层 ---
    vf_fraud_amt = amt[verify & (y == 1)].sum() * (1 - VERIFY_MISS_FRAUD)
    vf_fraud_cnt = int((verify & (y == 1)).sum())
    vf_normal_cnt = int((verify & (y == 0)).sum())
    cost_verify = verify.sum() * COST_VERIFY
    # 验证层里被放走的 10% 欺诈不扣损失（损失已隐含在未挽回里），
    # 被误验的正常用户只付通道费，不算误杀（体验损失远小于直接拦截）。

    net_benefit = blk_fraud_amt + vf_fraud_amt - cost_block - cost_verify

    return {
        "T1": t1, "T2": t2,
        "net_benefit": net_benefit,
        "拦截层交易占比": block.mean(),
        "拦截层欺诈覆盖率": blk_fraud_cnt / max(n_fraud_total, 1),
        "拦截层误杀数": blk_normal_cnt,
        "验证层交易占比": verify.mean(),
        "验证层欺诈数": vf_fraud_cnt,
        "验证层正常数": vf_normal_cnt,
        "放行层交易占比": 1 - block.mean() - verify.mean(),
        "总欺诈拦截率": (blk_fraud_cnt + vf_fraud_cnt * (1 - VERIFY_MISS_FRAUD))
                        / max(n_fraud_total, 1),
        "总交易干预率": (block.sum() + verify.sum()) / n_total,
    }


def main():
    utils.setup_plot_style()
    utils.log_section("阶段 5：三级处置矩阵与成本权衡")

    utils.check_input(config.OOT_TEST_NAME,
                      "请先运行：python scripts/03_oot_uid_features.py")
    te, pred = score_te()
    utils.log(f"te {te.shape}，欺诈 {int(te[config.TARGET_COL].sum())} 笔")

    # ------------------------------------------------------------------
    # 1. 扫描候选阈值对 (T1, T2)
    #    T1 从 99.5%~99.99% 分位扫（拦截层必须极克制），
    #    T2 从 97%~99.5% 分位扫，且要求 T2 < T1。
    # ------------------------------------------------------------------
    q1s = np.array([99.5, 99.7, 99.8, 99.9, 99.95, 99.99])
    q2s = np.array([97.0, 98.0, 99.0, 99.3, 99.5, 99.7])
    t1_list = np.quantile(pred, q1s / 100)
    t2_list = np.quantile(pred, q2s / 100)

    records = []
    for t1 in t1_list:
        for t2 in t2_list:
            if t2 >= t1:
                continue
            records.append(evaluate_tier(te, pred, t1, t2))
    res_df = pd.DataFrame(records)

    best = res_df.loc[res_df["net_benefit"].idxmax()]
    utils.log(f"\n最优阈值对：T1 = {best['T1']:.4f}，T2 = {best['T2']:.4f}，"
              f"净收益 = {best['net_benefit']:,.0f} 元")

    # ------------------------------------------------------------------
    # 2. 三级处置表
    # ------------------------------------------------------------------
    utils.log("\n=== 三级处置表（最优 T1/T2 下） ===")
    utils.log(f"  高风险 ≥T1 拦截     : 覆盖交易 {best['拦截层交易占比']:.3%}，"
              f"欺诈覆盖 {best['拦截层欺诈覆盖率']:.1%}，"
              f"误杀 {int(best['拦截层误杀数'])} 笔")
    utils.log(f"  中风险 T2~T1 二次验证: 覆盖交易 {best['验证层交易占比']:.3%}，"
              f"其中欺诈 {int(best['验证层欺诈数'])} 笔 / "
              f"正常 {int(best['验证层正常数'])} 笔")
    utils.log(f"  低风险 <T2 放行     : 覆盖交易 {best['放行层交易占比']:.3%}")

    # 回测句式（可直接抄进报告/简历，X 已替换成真实数字）
    utils.log("\n=== 回测结论句式 ===")
    utils.log(f"拦截层覆盖 {best['拦截层欺诈覆盖率']:.1%} 欺诈（拦截率），"
              f"误杀率 {best['拦截层误杀数'] / max((te[config.TARGET_COL] == 0).sum(), 1):.3%}；"
              f"二次验证层覆盖 {best['验证层交易占比']:.2%} 交易，"
              f"预计挽回误杀 {VERIFY_SAVE_NORMAL:.0%}（假设值）；"
              f"整体欺诈拦截率 {best['总欺诈拦截率']:.1%}，"
              f"总交易干预率仅 {best['总交易干预率']:.2%}，"
              f"回测净收益 {best['net_benefit']:,.0f} 元。")

    # ------------------------------------------------------------------
    # 3. 净收益-阈值热力图
    # ------------------------------------------------------------------
    pivot = res_df.pivot_table(index="T1", columns="T2",
                               values="net_benefit")
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.3f}" for v in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.3f}" for v in pivot.index])
    ax.set_xlabel("T2（二次验证阈值）")
    ax.set_ylabel("T1（拦截阈值）")
    ax.set_title("三级处置净收益热力图（越绿净收益越高）")
    fig.colorbar(im, ax=ax, label="净收益（元）")
    utils.save_fig(fig, "06_net_benefit_heatmap.png")
    plt.close(fig)

    # 全部扫描结果也存一份，方便报告里挑数
    out_csv = config.REPORTS_DIR / "tier_scan_results.csv"
    res_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    utils.log(f"[扫描结果已保存] {out_csv}（共 {len(res_df)} 组阈值对）")

    utils.log("\n阶段 5 完成。下一步：python scripts/07_rules_extraction.py")


if __name__ == "__main__":
    main()
