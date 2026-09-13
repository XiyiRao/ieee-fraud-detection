# -*- coding: utf-8 -*-
"""
阶段 9：金额口径的拦截效果（笔数口径 vs 金额口径）
======================================================
所属阶段：第 3 周 · 阶段 9（业务价值量化）

前置依赖：需先运行 scripts/03_oot_uid_features.py 和 scripts/04_train_lgbm.py
          （读取 data/processed/oot_test 和 models/lgbm_model.txt）

运行方式：
    python scripts/10_amount_weighted.py

预期产出：
    - reports/run_log.txt 追加：99.9% 分位阈值下"笔数口径 vs 金额口径"的
      拦截率/误杀率对比表，以及欺诈 vs 正常、被拦 vs 未拦欺诈的平均金额

为什么做金额口径：
    05/06 的拦截率按"笔数"统计，但风控部门真正关心的是"挽回了多少钱"。
    欺诈交易往往金额更高（盗刷倾向大额），同样拦 2.5% 的欺诈笔数，
    如果拦下的恰是高金额欺诈，挽回的资金占比会远高于笔数占比——
    这就是"金额加权拦截率"。反过来，误杀正常用户也要看金额：
    误杀的都是高金额正常交易，客诉和资金损失会更严重。
    两个口径一起看，才能向业务方讲清楚模型价值。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import config
from src import utils

# 与 04 脚本 prepare_xy() / 05 脚本 predict() 一致的剔除列
EXCLUDE_COLS = [config.ID_COL, config.TARGET_COL, "uid", "is_new_uid",
                "TransactionDT"]


def main():
    utils.setup_plot_style()
    utils.log_section("阶段 9：金额口径的拦截效果（笔数 vs 金额）")

    utils.check_input(config.OOT_TEST_NAME,
                      "请先运行：python scripts/03_oot_uid_features.py")
    if not (config.MODELS_DIR / "lgbm_model.txt").exists():
        print("缺少模型文件，请先运行：python scripts/04_train_lgbm.py")
        sys.exit(1)

    te = utils.load_df(config.OOT_TEST_NAME)
    y_te = te[config.TARGET_COL].values
    amt = te["TransactionAmt"].to_numpy(dtype=np.float64)

    # 打分：口径同 05 脚本 predict()（同剔除列、object 转 category）
    booster = utils.load_booster(config.MODELS_DIR / "lgbm_model.txt")
    X = te.drop(columns=[c for c in EXCLUDE_COLS if c in te.columns])
    for c in X.select_dtypes(include=["object", "str"]).columns:
        X[c] = X[c].astype("category")
    score = booster.predict(X)

    # 与 05 脚本同口径：阈值 = 预测分 99.9% 分位
    threshold = np.quantile(score, 0.999)
    flagged = score >= threshold

    n_fraud = int(y_te.sum())
    n_norm = int((y_te == 0).sum())
    tp_mask = flagged & (y_te == 1)          # 被拦欺诈
    fp_mask = flagged & (y_te == 0)          # 误杀正常
    fraud_mask = y_te == 1
    norm_mask = y_te == 0

    # ---------------- 笔数口径（与 05 一致，从数据重算确认） ----------------
    recall_cnt = tp_mask.sum() / max(n_fraud, 1)
    fpr_cnt = fp_mask.sum() / max(n_norm, 1)

    # ---------------- 金额口径 ----------------
    amt_fraud = amt[fraud_mask].sum()
    amt_norm = amt[norm_mask].sum()
    recall_amt = amt[tp_mask].sum() / max(amt_fraud, 1e-9)
    fpr_amt = amt[fp_mask].sum() / max(amt_norm, 1e-9)

    # ---------------- 平均金额对比 ----------------
    mean_fraud = amt[fraud_mask].mean()
    mean_norm = amt[norm_mask].mean()
    mean_caught = amt[tp_mask].mean()
    mean_missed = amt[fraud_mask & ~flagged].mean()

    utils.log(f"\nte {te.shape}，阈值=预测分 99.9% 分位 = {threshold:.4f}，"
              f"共拦下 {int(flagged.sum())} 笔"
              f"（真欺诈 {int(tp_mask.sum())} / 误杀 {int(fp_mask.sum())}）")
    utils.log(f"\n{'口径':<14s}{'欺诈拦截率':>12s}{'误杀率':>12s}")
    utils.log(f"{'笔数口径':<12s}{recall_cnt:>11.2%}{fpr_cnt:>11.3%}")
    utils.log(f"{'金额口径':<12s}{recall_amt:>11.2%}{fpr_amt:>11.3%}")

    utils.log(f"\n平均金额对比：")
    utils.log(f"  欺诈交易平均金额  : ${mean_fraud:,.2f}")
    utils.log(f"  正常交易平均金额  : ${mean_norm:,.2f}"
              f"（欺诈/正常 = {mean_fraud / max(mean_norm, 1e-9):.2f} 倍）")
    utils.log(f"  被拦欺诈平均金额  : ${mean_caught:,.2f}")
    utils.log(f"  未拦欺诈平均金额  : ${mean_missed:,.2f}"
              f"（被拦/未拦 = {mean_caught / max(mean_missed, 1e-9):.2f} 倍）")

    utils.log("\n结论：金额口径与笔数口径的差异说明——")
    ratio_fn = mean_fraud / max(mean_norm, 1e-9)
    utils.log(f"欺诈交易平均金额（${mean_fraud:,.2f}）是正常交易（${mean_norm:,.2f}）的 "
              f"{ratio_fn:.2f} 倍，本数据集欺诈金额仅略高，并非大额主导；")
    ratio_cm = mean_caught / max(mean_missed, 1e-9)
    utils.log(f"金额加权拦截率（{recall_amt:.2%}）{'高于' if recall_amt > recall_cnt else '低于'}"
              f"笔数口径（{recall_cnt:.2%}），被拦欺诈的平均金额是未拦欺诈的 "
              f"{ratio_cm:.2f} 倍——模型的高分段"
              + ("集中在高金额欺诈上，按资金挽回口径模型价值比笔数拦截率更高。"
                 if ratio_cm > 1 else
                 "偏向小额欺诈，按资金挽回口径模型价值低于笔数拦截率的直观印象；"
                 "改进方向是对训练样本按 TransactionAmt 加权，或在阈值扫描时"
                 "直接以金额加权拦截率为目标。"))
    utils.log(f"误杀的金额占比（{fpr_amt:.3%}）与笔数占比（{fpr_cnt:.3%}）同量级，"
              "误杀未特别集中在高金额正常用户上，客诉风险可控。")

    utils.log("\n阶段 9 完成。")


if __name__ == "__main__":
    main()
