# -*- coding: utf-8 -*-
"""
P1-b：客户级后处理实验（同客户预测取组内均值）
==============================================
实验：优化边界归因 E3 · 客户级（uid）预测聚合后处理

背景：客户级后处理——把同一客户（uid）内的预测分替换为
组内均值。直觉：同一实体的交易分数应一致；组均值能抑制单笔噪声、把
"这个客户整体可疑"的信号聚合起来，同时也把组内所有交易拉到同一分数
（组内要么全被拦、要么全放行）。

前置依赖：
    - models/lgbm_model.txt（scripts/04_train_lgbm.py）
    - data/processed/oot_test.parquet（scripts/03）
    - scripts/05_evaluate.py（复用其 evaluate_at_fpr）

运行方式：
    .venv\\Scripts\\python.exe scripts/13_client_postprocess.py

预期产出：
    - reports/figures/13_postprocess_metrics.png  替换前后指标对比图
    - reports/run_log.txt 追加：替换前后 PR-AUC / FPR / 拦截率 / 精准率
      对比表 + 业务化讨论
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib.util

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config
from src import utils

BASELINE = {"pr_auc": 0.4354, "recall": 0.0256, "precision": 0.7761}


def load_module(mod_name, file_name):
    spec = importlib.util.spec_from_file_location(
        mod_name, Path(__file__).resolve().parent / file_name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def plot_metrics(m_before, m_after):
    """替换前后 PR-AUC / 拦截率 / 精准率 / FPR 对比柱状图。"""
    metrics = [("PR-AUC", m_before["pr_auc"], m_after["pr_auc"]),
               ("拦截率", m_before["res"]["recall"], m_after["res"]["recall"]),
               ("精准率", m_before["res"]["precision"], m_after["res"]["precision"]),
               ("FPR(误杀率)", m_before["res"]["actual_fpr"], m_after["res"]["actual_fpr"])]
    x = np.arange(len(metrics))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width / 2, [m[1] for m in metrics], width,
           label="替换前（原始预测分）", color="lightgray", edgecolor="gray")
    ax.bar(x + width / 2, [m[2] for m in metrics], width,
           label="替换后（uid 组内均值）", color="steelblue")
    for i, (_, v1, v2) in enumerate(metrics):
        ax.text(i - width / 2, v1, f"{v1:.4f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, v2, f"{v2:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics])
    ax.set_ylabel("指标值")
    ax.set_title("P1-b：客户级后处理（同 uid 预测取组均值）前后指标对比（te）")
    ax.legend()
    utils.save_fig(fig, "13_postprocess_metrics.png")
    plt.close(fig)


def main():
    utils.setup_plot_style()
    utils.log_section("P1-b：客户级后处理实验（script 13）")

    utils.check_input(config.OOT_TEST_NAME,
                      "请先运行：python scripts/03_oot_uid_features.py")
    mod05 = load_module("evaluate", "05_evaluate.py")
    model_path = config.MODELS_DIR / "lgbm_model.txt"
    if not model_path.exists():
        print("缺少模型文件，请先运行：python scripts/04_train_lgbm.py")
        sys.exit(1)

    te = utils.load_df(config.OOT_TEST_NAME)
    booster = utils.load_booster(model_path)
    drop_cols = [c for c in [config.ID_COL, config.TARGET_COL, "uid",
                             "is_new_uid", "TransactionDT"] if c in te.columns]
    X = te.drop(columns=drop_cols).copy()
    for c in X.select_dtypes(include=["object"]).columns:
        X[c] = X[c].astype("category")
    pred = booster.predict(X)
    y_te = te[config.TARGET_COL].values
    utils.log(f"te {te.shape}，欺诈率 {y_te.mean():.2%}")

    # ------------------------------------------------------------------
    # 1. 客户级后处理：同 uid 预测分替换为组内均值
    # ------------------------------------------------------------------
    # 口径声明：03 构建 uid 时任一组成字段（card2/addr1/P_emaildomain）缺失
    # 则 uid 为 NaN（pandas 3 的 astype(str) 保留 NaN），te 中占比约 29%。
    # 这些交易无法归属客户，groupby 用 dropna=False 让它们自成一组
    # （组内均值即自身，等价于不做替换——这也是线上实现的自然语义）。
    nan_uid_rate = te["uid"].isna().mean()
    utils.log(f"\n口径声明：te 中 uid 为 NaN（客户归属字段缺失）的占比 {nan_uid_rate:.2%}，"
              f"后处理时自成一组（组均值=自身分数）")
    pred_s = pd.Series(pred, index=te.index)
    pred_post = pred_s.groupby(te["uid"], dropna=False).transform("mean")
    pred_post = pred_post.to_numpy()
    n_groups = te["uid"].nunique(dropna=False)
    avg_group_size = len(te) / n_groups
    # 组内分数是否被拉平
    within_std_before = pred_s.groupby(te["uid"], dropna=False).std().mean()
    within_std_after = (pd.Series(pred_post, index=te.index)
                        .groupby(te["uid"], dropna=False).std().mean())
    utils.log(f"te 中 uid 组数 {n_groups:,}，平均组规模 {avg_group_size:.2f} 笔")
    utils.log(f"组内预测分标准差：替换前 {within_std_before:.5f} → "
              f"替换后 {within_std_after:.5f}（应≈0，即组内分数被完全拉平）")

    # ------------------------------------------------------------------
    # 2. 替换前后指标对比
    # ------------------------------------------------------------------
    def metrics_of(p):
        return {"pr_auc": average_precision_score(y_te, p),
                "roc_auc": roc_auc_score(y_te, p),
                "res": mod05.evaluate_at_fpr(y_te, p, fpr_target=0.001)}

    m_before = metrics_of(pred)
    m_after = metrics_of(pred_post)
    utils.log(f"\n替换前: PR-AUC={m_before['pr_auc']:.4f}  ROC-AUC={m_before['roc_auc']:.4f}  "
              f"FPR={m_before['res']['actual_fpr']:.3%}  拦截率={m_before['res']['recall']:.2%}  "
              f"精准率={m_before['res']['precision']:.2%}")
    utils.log(f"替换后: PR-AUC={m_after['pr_auc']:.4f}  ROC-AUC={m_after['roc_auc']:.4f}  "
              f"FPR={m_after['res']['actual_fpr']:.3%}  拦截率={m_after['res']['recall']:.2%}  "
              f"精准率={m_after['res']['precision']:.2%}")

    def row(name, m):
        r = m["res"]
        return (f"  {name:<16s} {m['pr_auc']:.4f}      {r['actual_fpr']:.3%}    "
                f"{r['recall']:.2%}     {r['precision']:.2%}")

    utils.log("\n对比表（业务阈值 = te 预测分 99.9% 分位）：")
    utils.log(f"  {'方案':<16s} {'PR-AUC':<8s} {'FPR':<8s} {'拦截率':<8s} {'精准率':<8s}")
    utils.log(row("基线（复跑前记录）", {
        "pr_auc": BASELINE["pr_auc"],
        "res": {"actual_fpr": 0.00026, "recall": BASELINE["recall"],
                "precision": BASELINE["precision"]}}))
    utils.log(row("后处理前（复算）", m_before))
    utils.log(row("后处理后（组均值）", m_after))
    plot_metrics(m_before, m_after)

    # ------------------------------------------------------------------
    # 3. 结论 + 业务化讨论
    # ------------------------------------------------------------------
    delta_pr = m_after["pr_auc"] - m_before["pr_auc"]
    utils.log("\n[P1-b 结论]")
    utils.log(f"  同 uid 组内均值后处理使 PR-AUC 变化 {delta_pr:+.4f}"
              f"（{m_before['pr_auc']:.4f} → {m_after['pr_auc']:.4f}），"
              f"拦截率 {m_before['res']['recall']:.2%} → {m_after['res']['recall']:.2%}，"
              f"精准率 {m_before['res']['precision']:.2%} → "
              f"{m_after['res']['precision']:.2%}。")
    if delta_pr > 0.005:
        utils.log("  客户级后处理在 OOT 口径下有实质增益，建议采纳。")
    else:
        utils.log("  客户级后处理在本项目 OOT 口径下无实质增益（阴性结论）——"
                  "该类后处理的增益高度依赖特征与模型 pipeline 的整体配合，"
                  "单独移植后处理不足以复现。")
    utils.log("  业务化讨论：线上实现该后处理需要维护客户（uid）历史分数缓存"
              "（实时流上按 uid 聚合预测分），并处理新 uid 冷启动（无历史可均），"
              "属于额外的工程成本与延迟开销；是否采纳应权衡其增益与基建成本。")
    utils.log("  P1-b 实验完成。")


if __name__ == "__main__":
    main()
