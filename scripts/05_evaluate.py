# -*- coding: utf-8 -*-
"""
阶段 4：模型评估（PR-AUC 为主，业务化口径 + bootstrap 置信区间）
================================================================
所属阶段：第 2 周 · 阶段 4（评估）

前置依赖：需先运行 scripts/04_train_lgbm.py
          （读取 data/processed/oot_train、oot_test 和 models/lgbm_model.txt）

运行方式：
    python scripts/05_evaluate.py

预期产出：
    - reports/figures/05_pr_curve.png      PR 曲线
    - reports/run_log.txt 追加：PR-AUC、ROC-AUC、FPR≈0.1% 下的拦截率、
      拦截率 95% 置信区间、切分点 0.75/0.8/0.85 敏感性对比表

评估口径设计（面试高频考点）：
    1. 为什么主指标是 PR-AUC 而不是 ROC-AUC？
       欺诈率只有约 3.5%，极度不平衡下 ROC-AUC 会虚高——负样本太多，
       大量"容易判对的正常样本"把曲线撑起来了。PR 曲线只关心"判为欺诈的
       里面有多少是真的"（精准率）和"真欺诈抓住了多少"（召回率），
       对不平衡数据诚实得多。
    2. 业务化口径：阈值取预测分的 99.9% 分位，即只拦最高分的 0.1% 交易
       （FPR≈0.1%——反欺诈线上硬约束：误杀率不能高，否则正常用户被拦
       会直接投诉流失）。报告指标 = 该阈值下的 recall（欺诈拦截率）。
    3. bootstrap 置信区间：te 只有约 12 万笔、欺诈几千笔，单个数字有运气
       成分，必须给区间并诚实打印区间宽度——"我们 95% 确信拦截率在
       [a, b] 之间"比"拦截率 55%"专业得多。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)

from src import config
from src import utils


def predict(model_path, df):
    """加载模型并对一个 DataFrame 打欺诈分。"""
    booster = utils.load_booster(model_path)
    drop_cols = [c for c in [config.ID_COL, config.TARGET_COL, "uid",
                             "is_new_uid", "TransactionDT"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    for c in X.select_dtypes(include=["object"]).columns:
        X[c] = X[c].astype("category")
    return booster.predict(X)


def evaluate_at_fpr(y_true, y_score, fpr_target=0.001):
    """在 FPR≈fpr_target 的业务阈值下计算拦截率（recall）。

    实现方式：阈值取预测分的 (1 - fpr_target) 分位数，
    即只把得分最高的 0.1% 交易判为欺诈。
    """
    threshold = np.quantile(y_score, 1 - fpr_target)
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    recall = tp / max(tp + fn, 1)          # 欺诈拦截率
    precision = tp / max(tp + fp, 1)       # 拦截精准率
    actual_fpr = fp / max((y_true == 0).sum(), 1)
    return {"threshold": float(threshold), "recall": recall,
            "precision": precision, "actual_fpr": actual_fpr,
            "tp": tp, "fp": fp, "fn": fn}


def bootstrap_recall_ci(y_true, y_score, n_boot=1000, fpr_target=0.001,
                        seed=config.RANDOM_SEED):
    """对 FPR≈0.1% 阈值下的拦截率做 bootstrap，返回 95% 置信区间。

    做法：每次有放回地抽与 te 等大的样本，重算阈值和拦截率；
    取 2.5% / 97.5% 分位作为区间。区间宽 = 指标不稳，要诚实报告。
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    recalls = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if yt.sum() == 0:      # 极端重采样下可能没抽到欺诈样本
            continue
        recalls.append(evaluate_at_fpr(yt, ys, fpr_target)["recall"])
    lo, hi = np.percentile(recalls, [2.5, 97.5])
    return float(lo), float(hi), np.array(recalls)


def sensitivity_to_split(ratios=(0.75, 0.8, 0.85)):
    """换切分点重训，看指标波动（代码框架，函数化便于复用）。

    流程：重新读 merged 数据 → 按给定比例重建 uid 特征与切分 → 调
    04 脚本的 train() → 评估 PR-AUC 和 FPR0.1% 拦截率。
    如果三个切分点指标波动大，说明模型对时间漂移敏感，报告里要写。
    """
    # 延迟 import：避免循环依赖，也让本函数只在需要时才依赖 04
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_lgbm", Path(__file__).resolve().parent / "04_train_lgbm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    df = utils.load_df(config.MERGED_TRAIN_NAME)
    df = df.sort_values(config.TIME_COL).reset_index(drop=True)
    df = utils.build_uid_features(df)

    results = []
    for r in ratios:
        cut = int(len(df) * r)
        tr, te = df.iloc[:cut].copy(), df.iloc[cut:].copy()
        model, _, X_te = mod.train(tr, te, save_path=None, verbose=False)
        pred = model.predict_proba(X_te)[:, 1]
        y_te = te[config.TARGET_COL].values
        pr_auc = average_precision_score(y_te, pred)
        roc_auc = roc_auc_score(y_te, pred)
        recall = evaluate_at_fpr(y_te, pred)["recall"]
        results.append({"split_ratio": r, "pr_auc": pr_auc,
                        "roc_auc": roc_auc, "recall@fpr0.1%": recall})
        utils.log(f"切分点 {r:.2f}: PR-AUC={pr_auc:.4f}, "
                  f"ROC-AUC={roc_auc:.4f}, 拦截率@FPR0.1%={recall:.2%}")
    return results


def main():
    utils.setup_plot_style()
    utils.log_section("阶段 4：模型评估")

    utils.check_input(config.OOT_TEST_NAME,
                      "请先运行：python scripts/03_oot_uid_features.py")
    model_path = config.MODELS_DIR / "lgbm_model.txt"
    if not model_path.exists():
        print("缺少模型文件，请先运行：python scripts/04_train_lgbm.py")
        sys.exit(1)

    te = utils.load_df(config.OOT_TEST_NAME)
    y_te = te[config.TARGET_COL].values
    pred = predict(model_path, te)

    # ------------------------------------------------------------------
    # 1. 主指标 PR-AUC + 参考 ROC-AUC
    # ------------------------------------------------------------------
    pr_auc = average_precision_score(y_te, pred)
    roc_auc = roc_auc_score(y_te, pred)
    utils.log(f"PR-AUC（主指标） : {pr_auc:.4f}")
    utils.log(f"ROC-AUC（参考）  : {roc_auc:.4f}")
    utils.log("（典型现象：ROC-AUC 看起来很漂亮，PR-AUC 低得多——"
              "不平衡数据下 PR-AUC 才是诚实指标）")

    # ------------------------------------------------------------------
    # 2. PR 曲线
    # ------------------------------------------------------------------
    prec, rec, _ = precision_recall_curve(y_te, pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec)
    ax.axhline(y_te.mean(), color="red", linestyle="--",
               label=f"随机基线（欺诈率 {y_te.mean():.2%}）")
    ax.set_xlabel("召回率（拦截率）")
    ax.set_ylabel("精准率")
    ax.set_title(f"PR 曲线（PR-AUC = {pr_auc:.4f}）")
    ax.legend()
    utils.save_fig(fig, "05_pr_curve.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3. 业务化口径：FPR≈0.1% 阈值下的拦截率
    # ------------------------------------------------------------------
    res = evaluate_at_fpr(y_te, pred, fpr_target=0.001)
    utils.log(f"\n业务化口径（阈值=预测分 99.9% 分位 = {res['threshold']:.4f}）：")
    utils.log(f"  实际 FPR（误杀率）: {res['actual_fpr']:.3%}")
    utils.log(f"  欺诈拦截率 recall : {res['recall']:.2%}  "
              f"（{res['tp']} 笔真欺诈被拦 / 共 {res['tp'] + res['fn']} 笔）")
    utils.log(f"  拦截精准率        : {res['precision']:.2%}  "
              f"（拦下的 {res['tp'] + res['fp']} 笔里 {res['tp']} 笔是真的）")

    # ------------------------------------------------------------------
    # 4. bootstrap 1000 次算拦截率 95% 置信区间
    # ------------------------------------------------------------------
    lo, hi, boots = bootstrap_recall_ci(y_te, pred, n_boot=1000)
    utils.log(f"\n拦截率 95% 置信区间（bootstrap 1000 次）: "
              f"[{lo:.2%}, {hi:.2%}]，区间宽度 {hi - lo:.2%}")
    utils.log("（诚实声明：区间越宽说明该业务口径下指标越不稳定，"
              "面试时主动报区间比只报点估计专业得多）")

    # ------------------------------------------------------------------
    # 5. 切分点敏感性：0.75 / 0.8 / 0.85 重训对比
    #    （会重新训练 3 次，耗时较长；想跳过可注释掉下面两行）
    # ------------------------------------------------------------------
    utils.log("\n切分点敏感性实验（重训 3 次，请耐心等待）...")
    sensitivity_to_split(ratios=(0.75, 0.8, 0.85))

    utils.log("\n阶段 4 完成。下一步：python scripts/06_three_tier_strategy.py")


if __name__ == "__main__":
    main()
