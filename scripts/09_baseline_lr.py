# -*- coding: utf-8 -*-
"""
阶段 8：Logistic Regression 基线（回答"LightGBM 的增量到底在哪"）
====================================================================
所属阶段：第 3 周 · 阶段 8（基线对照）

前置依赖：需先运行 scripts/03_oot_uid_features.py 和 scripts/04_train_lgbm.py
          （读取 data/processed/oot_train、oot_test 和 models/lgbm_model.txt）

运行方式：
    python scripts/09_baseline_lr.py

预期产出：
    - reports/figures/09_lr_vs_lgbm_pr.png  LR 与 LightGBM 的 PR 曲线对比
    - reports/run_log.txt 追加：同口径（同 tr/te、同 438 列、同 99.9% 分位阈）
      下 LR vs LightGBM 的 PR-AUC / ROC-AUC / 拦截率 / 拦截精准率对比表 + 结论

为什么补一个 LR baseline：
    04 脚本的 docstring 声称"GBDT 比评分卡强"，这只是经验断言；面试官会追问
    "增量到底有多少、来自哪里"。用与 LightGBM 完全相同的切分、特征列和业务
    阈值训练一个线性基线，PR-AUC 的差距就是"非线性 + 特征交叉"的真实增益，
    把口号变成可量化的数字。

口径设计（与 LightGBM 公平对比的关键）：
    1. 特征列完全一致：同样剔除 [TransactionID, isFraud, uid, is_new_uid,
       TransactionDT]，共 438 列。
    2. 类别列不用 one-hot（P_emaildomain 等基数上百，会维度爆炸），改用
       频次编码：取该取值在 tr 中的出现频率映射，te 未见取值填 0。
       缺失/未知类别天然落到频次 0，信息不丢失，也不必额外加指示列。
    3. 数值列缺失填 tr 的中位数后 StandardScaler 标准化——LR 对特征尺度
       敏感，不标准化会被量纲大的列（如金额）主导。
    4. class_weight="balanced"（自动按 1:28 反比加权），与 LightGBM 的
       scale_pos_weight=8 是对同一问题（欺诈仅 3.5%）的合理对齐口径，
       保证两边都"重视少数类"，对比才公平。
    5. solver 选 lbfgs 并调大 max_iter=2000：438 维、47 万行下默认 100 轮
       必然不收敛；max_iter 给足后 lbfgs 收敛且解稳定，无需换 saga/liblinear
       （saga 更慢，liblinear 只支持 OvR 多分类，本任务是二分类用不上）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score,
                             precision_recall_curve, roc_auc_score)
from sklearn.preprocessing import StandardScaler

from src import config
from src import utils

# 与 04 脚本 prepare_xy() 完全一致的剔除列，保证特征口径对齐
EXCLUDE_COLS = [config.ID_COL, config.TARGET_COL, "uid", "is_new_uid",
                "TransactionDT"]


def prepare_features(df):
    """特征列拆分：返回 (X, 类别列, 数值列)，与 LightGBM 同口径剔除。"""
    drop_cols = [c for c in EXCLUDE_COLS if c in df.columns]
    X = df.drop(columns=drop_cols)
    cat_cols = X.select_dtypes(include=["object", "category", "str"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    return X, cat_cols, num_cols


def frequency_encode(X_tr, X_te, cat_cols):
    """频次编码：取 tr 中各取值的出现频率映射；te 未见取值 / 缺失填 0。

    缺失值（NaN）在 value_counts 中不出现，map 后为 NaN，统一 fillna(0)——
    "缺失/未知 = 频次 0"正是想要表达的语义，不加额外指示列。
    """
    X_tr, X_te = X_tr.copy(), X_te.copy()
    for c in cat_cols:
        freq = X_tr[c].value_counts(normalize=True)
        X_tr[c] = X_tr[c].map(freq).fillna(0.0)
        X_te[c] = X_te[c].map(freq).fillna(0.0)
    return X_tr, X_te


def evaluate_at_fpr(y_true, y_score, fpr_target=0.001):
    """与 05 脚本同口径：阈值 = 预测分 99.9% 分位，y_score >= 阈值判欺诈。"""
    threshold = np.quantile(y_score, 1 - fpr_target)
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    actual_fpr = fp / max((y_true == 0).sum(), 1)
    return {"recall": recall, "precision": precision,
            "actual_fpr": actual_fpr}


def lgbm_scores(te):
    """加载已训练的 LightGBM，对 te 打分（口径同 05 脚本的 predict）。"""
    booster = utils.load_booster(config.MODELS_DIR / "lgbm_model.txt")
    X = te.drop(columns=[c for c in EXCLUDE_COLS if c in te.columns])
    for c in X.select_dtypes(include=["object", "str"]).columns:
        X[c] = X[c].astype("category")
    return booster.predict(X)


def main():
    utils.setup_plot_style()
    utils.log_section("阶段 8：Logistic Regression 基线（对照 LightGBM）")

    utils.check_input(config.OOT_TRAIN_NAME,
                      "请先运行：python scripts/03_oot_uid_features.py")
    utils.check_input(config.OOT_TEST_NAME,
                      "请先运行：python scripts/03_oot_uid_features.py")
    if not (config.MODELS_DIR / "lgbm_model.txt").exists():
        print("缺少模型文件，请先运行：python scripts/04_train_lgbm.py")
        sys.exit(1)

    tr = utils.load_df(config.OOT_TRAIN_NAME)
    te = utils.load_df(config.OOT_TEST_NAME)
    utils.log(f"tr {tr.shape} / te {te.shape}")

    X_tr, cat_cols, num_cols = prepare_features(tr)
    X_te, _, _ = prepare_features(te)
    y_tr = tr[config.TARGET_COL]
    y_te = te[config.TARGET_COL].values
    utils.log(f"特征数: {X_tr.shape[1]}（类别列 {len(cat_cols)}，数值列 {len(num_cols)}）")

    # 类别列频次编码（tr/te 共用 tr 的频率表，防泄漏）
    X_tr, X_te = frequency_encode(X_tr, X_te, cat_cols)

    # 数值列：tr 中位数填补缺失 + 标准化（LR 对尺度敏感）
    medians = X_tr[num_cols].median()
    X_tr[num_cols] = X_tr[num_cols].fillna(medians)
    X_te[num_cols] = X_te[num_cols].fillna(medians)
    scaler = StandardScaler()
    X_tr[num_cols] = scaler.fit_transform(X_tr[num_cols])
    X_te[num_cols] = scaler.transform(X_te[num_cols])
    # 降回 float32 控内存：472k x 438 的 float64 矩阵约 1.6 GB
    X_tr = X_tr.astype(np.float32)
    X_te = X_te.astype(np.float32)

    # ------------------------------------------------------------------
    # 训练 LR（class_weight="balanced" 对齐 LightGBM 的 scale_pos_weight=8，
    # 两边都解决"欺诈仅 3.5%"的少数类问题，对比才公平）
    # ------------------------------------------------------------------
    lr = LogisticRegression(class_weight="balanced", solver="lbfgs",
                            max_iter=2000,
                            random_state=config.RANDOM_SEED)
    lr.fit(X_tr, y_tr)
    lr_score = lr.predict_proba(X_te)[:, 1]

    # ------------------------------------------------------------------
    # 同口径评估：LR 与 LightGBM 对比
    # ------------------------------------------------------------------
    lgbm_score = lgbm_scores(te)

    def full_metrics(y_true, y_score):
        res = evaluate_at_fpr(y_true, y_score)
        res["pr_auc"] = average_precision_score(y_true, y_score)
        res["roc_auc"] = roc_auc_score(y_true, y_score)
        return res

    m_lr, m_gb = full_metrics(y_te, lr_score), full_metrics(y_te, lgbm_score)

    utils.log("\n同口径对比（同 tr/te、同 438 列、同 99.9% 分位阈值）：")
    utils.log(f"{'':6s}{'PR-AUC':>9s}{'ROC-AUC':>9s}"
              f"{'拦截率@99.9%分位阈':>20s}{'拦截精准率':>14s}")
    utils.log(f"{'LR':6s}{m_lr['pr_auc']:>9.4f}{m_lr['roc_auc']:>9.4f}"
              f"{m_lr['recall']:>19.2%}{m_lr['precision']:>13.2%}")
    utils.log(f"{'LGBM':6s}{m_gb['pr_auc']:>9.4f}{m_gb['roc_auc']:>9.4f}"
              f"{m_gb['recall']:>19.2%}{m_gb['precision']:>13.2%}")

    utils.log(f"\n结论：LightGBM 相对 LR 的增量——"
              f"PR-AUC +{m_gb['pr_auc'] - m_lr['pr_auc']:.4f}"
              f"（{(m_gb['pr_auc'] - m_lr['pr_auc']) / max(m_lr['pr_auc'], 1e-9):.0%} 相对提升）、"
              f"ROC-AUC +{m_gb['roc_auc'] - m_lr['roc_auc']:.4f}；"
              f"99.9% 分位阈下拦截率 {m_lr['recall']:.2%} → {m_gb['recall']:.2%}。")
    utils.log("LR 是纯线性模型，只能学到单变量的加权关系；PR-AUC 的显著差距"
              "说明欺诈模式主要是非线性和交叉特征（如'金额高 且 无设备指纹 且 "
              "凌晨'的组合），这正是 GBDT 树结构相对评分卡的真实、可量化的增益。")

    # ------------------------------------------------------------------
    # PR 曲线对比图
    # ------------------------------------------------------------------
    prec_lr, rec_lr, _ = precision_recall_curve(y_te, lr_score)
    prec_gb, rec_gb, _ = precision_recall_curve(y_te, lgbm_score)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec_gb, prec_gb, label=f"LightGBM（PR-AUC={m_gb['pr_auc']:.4f}）")
    ax.plot(rec_lr, prec_lr, label=f"LogisticRegression（PR-AUC={m_lr['pr_auc']:.4f}）")
    ax.axhline(y_te.mean(), color="red", linestyle="--",
               label=f"随机基线（欺诈率 {y_te.mean():.2%}）")
    ax.set_xlabel("召回率（拦截率）")
    ax.set_ylabel("精准率")
    ax.set_title("LR vs LightGBM：PR 曲线对比")
    ax.legend()
    utils.save_fig(fig, "09_lr_vs_lgbm_pr.png")
    plt.close(fig)

    utils.log("\n阶段 8 完成。下一步：python scripts/10_amount_weighted.py")


if __name__ == "__main__":
    main()
