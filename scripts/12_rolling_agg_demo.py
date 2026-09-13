# -*- coding: utf-8 -*-
"""
P1-a：滚动口径聚合示范（把"已声明的全期统计轻微泄漏"量化并示范修复）
=========================================================================
实验：优化边界归因 E2 · 滚动口径聚合示范（量化全期统计泄漏）

背景：03 脚本的 uid_cnt / uid_amt_mean 等聚合特征用的是"全训练期统计"，
属于近似口径（报告已声明存在轻微未来信息泄漏）。本脚本挑 2 个核心聚合
特征（uid 维度的 TransactionAmt 均值、交易次数），用"截至当前交易时刻"
的 expanding window 滚动统计版（shift 一位，不计入当前行）替代全期版，
量化两者的差异与重训后的指标差异。

前置依赖：
    - data/processed/oot_train.parquet / oot_test.parquet（scripts/03）
    - scripts/04_train_lgbm.py（复用其 train()，同参数同口径）
    - scripts/05_evaluate.py（复用其 evaluate_at_fpr）

运行方式：
    .venv\\Scripts\\python.exe scripts/12_rolling_agg_demo.py

预期产出：
    - reports/figures/12_rolling_vs_full_scatter.png  滚动版 vs 全期版散点图
    - reports/run_log.txt 追加：相关性、重训对比表、泄漏量化结论

口径说明：
    - 滚动统计在 tr+te 拼回的完整时间线上计算（与 velocity 同口径），
      每笔交易只用它之前的历史：cnt_exp = 该 uid 之前的交易笔数，
      amt_mean_exp = 该 uid 之前交易的金额均值；uid 首笔无历史，
      两特征均留 NaN（LightGBM 原生处理缺失，并保留"是否首笔"的信息）。
    - 只替换 uid_cnt 与 uid_amt_mean 两列（其余特征保持 03 口径），
      特征数与基线一致，保证对比干净。
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
ROLLING_COLS = ["uid_cnt", "uid_amt_mean"]


def load_module(mod_name, file_name):
    spec = importlib.util.spec_from_file_location(
        mod_name, Path(__file__).resolve().parent / file_name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute_expanding(df_all: pd.DataFrame) -> pd.DataFrame:
    """在 tr+te 拼接时间线上计算 expanding（shift 1）版聚合特征。

    返回带 TransactionID / uid_cnt_exp / uid_amt_mean_exp 的 DataFrame。
    实现为向量化 cumcount/cumsum，避免逐组 Python 回调，59 万笔秒级完成。
    """
    df = df_all.sort_values(config.TIME_COL, kind="mergesort").reset_index(drop=True)
    grp = df.groupby("uid")
    cnt_prev = grp.cumcount()                       # 当前笔之前该 uid 的笔数
    amt_cum_prev = grp["TransactionAmt"].cumsum() - df["TransactionAmt"]

    out = pd.DataFrame({
        config.ID_COL: df[config.ID_COL],
        "uid_cnt_exp": cnt_prev.astype(np.float32),
        "uid_amt_mean_exp": np.where(cnt_prev > 0,
                                     amt_cum_prev / cnt_prev, np.nan),
    })
    out["uid_amt_mean_exp"] = out["uid_amt_mean_exp"].astype(np.float32)
    return out


def plot_scatter(tr_full, tr_roll):
    """全期版 vs 滚动版散点图（子采样 3 万条），存 12_rolling_vs_full_scatter.png。

    滚动版在数据集中原位替换同名列（uid_cnt / uid_amt_mean），
    因此 tr_roll 取同名列即为滚动值。
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    idx = rng.choice(len(tr_full), size=min(30000, len(tr_full)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    specs = [("uid_cnt", "uid 交易次数"), ("uid_amt_mean", "uid 平均交易金额")]
    for ax, (c, title) in zip(axes, specs):
        xs = tr_full[c].to_numpy()[idx]
        ys = tr_roll[c].to_numpy()[idx]
        ax.scatter(xs, ys, s=4, alpha=0.15, color="steelblue")
        lim = [0, np.nanmax(xs) * 1.05]
        ax.plot(lim, lim, color="red", linestyle="--", linewidth=1,
                label="y = x（两版一致）")
        ax.set_xlabel(f"全期统计版 {c}")
        ax.set_ylabel(f"滚动版（expanding, shift1）{c}")
        ax.set_title(title)
        ax.legend()
    fig.suptitle("P1-a：全期统计 vs 滚动口径（tr 子采样 3 万条；首笔交易滚动版为 NaN 不显示）")
    utils.save_fig(fig, "12_rolling_vs_full_scatter.png")
    plt.close(fig)


def train_and_eval(mod04, mod05, tr, te, tag):
    model, X_tr, X_te = mod04.train(tr, te, save_path=None, verbose=False)
    pred = model.predict_proba(X_te)[:, 1]
    y_te = te[config.TARGET_COL].values
    pr_auc = average_precision_score(y_te, pred)
    roc_auc = roc_auc_score(y_te, pred)
    res = mod05.evaluate_at_fpr(y_te, pred, fpr_target=0.001)
    utils.log(f"[{tag}] PR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}  "
              f"FPR={res['actual_fpr']:.3%}  拦截率={res['recall']:.2%}  "
              f"精准率={res['precision']:.2%}")
    return {"pr_auc": pr_auc, "roc_auc": roc_auc, "res": res}


def main():
    utils.setup_plot_style()
    utils.log_section("P1-a：滚动口径聚合示范（script 12）")

    utils.check_input(config.OOT_TRAIN_NAME,
                      "请先运行：python scripts/03_oot_uid_features.py")
    utils.check_input(config.OOT_TEST_NAME,
                      "请先运行：python scripts/03_oot_uid_features.py")
    mod04 = load_module("train_lgbm", "04_train_lgbm.py")
    mod05 = load_module("evaluate", "05_evaluate.py")

    tr = utils.load_df(config.OOT_TRAIN_NAME)
    te = utils.load_df(config.OOT_TEST_NAME)
    utils.log(f"tr {tr.shape} / te {te.shape}")

    # ------------------------------------------------------------------
    # 1. 在 tr+te 拼接时间线上计算 expanding 版聚合特征，回接替换
    # ------------------------------------------------------------------
    tr["_is_te"] = 0
    te["_is_te"] = 1
    df_all = pd.concat([tr, te], ignore_index=True)
    rolling = compute_expanding(df_all).set_index(config.ID_COL)

    tr_roll = tr.drop(columns=["_is_te"]).copy()
    te_roll = te.drop(columns=["_is_te"]).copy()
    for c_full, c_exp in [("uid_cnt", "uid_cnt_exp"),
                          ("uid_amt_mean", "uid_amt_mean_exp")]:
        tr_roll[c_full] = tr_roll[config.ID_COL].map(rolling[c_exp]).to_numpy()
        te_roll[c_full] = te_roll[config.ID_COL].map(rolling[c_exp]).to_numpy()
    assert tr_roll[ROLLING_COLS].isna().sum().sum() > 0, \
        "滚动版首笔应存在 NaN，断言失败说明口径可能有误"

    nan_rate = tr_roll["uid_cnt"].isna().mean()
    utils.log(f"\n滚动版特征已生成（首笔交易无历史 → NaN，tr 中占比 {nan_rate:.2%}，"
              f"交予 LightGBM 原生处理缺失）")

    # ------------------------------------------------------------------
    # 2. 特征相关性：滚动版 vs 全期版
    # ------------------------------------------------------------------
    tr_full = tr.drop(columns=["_is_te"])
    te_full = te.drop(columns=["_is_te"]).copy()
    utils.log("\n相关性（tr 上，Pearson / Spearman）：")
    pearson_vals = {}
    for c in ROLLING_COLS:
        pear = tr_full[c].corr(tr_roll[c])
        spear = tr_full[c].corr(tr_roll[c], method="spearman")
        pearson_vals[c] = pear
        utils.log(f"  {c:<14s} Pearson={pear:.4f}   Spearman={spear:.4f}")
    plot_scatter(tr_full, tr_roll)

    # ------------------------------------------------------------------
    # 3. 重训对比：全期版（同口径复现基线）vs 滚动版（修复泄漏）
    # ------------------------------------------------------------------
    utils.log("\n重训 2 次（全期版复现 + 滚动版修复，同 04 口径，请耐心等待）...")
    m_full = train_and_eval(mod04, mod05, tr_full, te_full, "全期统计版")
    m_roll = train_and_eval(mod04, mod05, tr_roll, te_roll, "滚动口径版")

    def row(name, m):
        r = m["res"]
        return (f"  {name:<20s} {m['pr_auc']:.4f}      {r['actual_fpr']:.3%}    "
                f"{r['recall']:.2%}     {r['precision']:.2%}")

    utils.log("")
    utils.log("对比表（业务阈值 = te 预测分 99.9% 分位）：")
    utils.log(f"  {'模型':<20s} {'PR-AUC':<8s} {'FPR':<8s} {'拦截率':<8s} {'精准率':<8s}")
    utils.log(row("基线（复跑前记录）", {
        "pr_auc": BASELINE["pr_auc"],
        "res": {"actual_fpr": 0.00026, "recall": BASELINE["recall"],
                "precision": BASELINE["precision"]}}))
    utils.log(row("全期统计版（重训）", m_full))
    utils.log(row("滚动口径版（修复）", m_roll))

    # ------------------------------------------------------------------
    # 4. 结论：量化"全期统计轻微泄漏"的实际影响
    # ------------------------------------------------------------------
    delta = m_roll["pr_auc"] - m_full["pr_auc"]
    utils.log("\n[P1-a 结论]")
    pear_min, pear_max = min(pearson_vals.values()), max(pearson_vals.values())
    utils.log(f"  滚动版与全期版 Pearson 相关性 {pear_min:.2f}~{pear_max:.2f}——"
              f"整体同源（同一实体的前后统计必然趋同），差异主要来自实体的"
              f"前若干笔交易（全期版此时已经“看到”了未来）。")
    utils.log(f"  修复泄漏后 PR-AUC 变化 {delta:+.4f}"
              f"（{m_full['pr_auc']:.4f} → {m_roll['pr_auc']:.4f}）——"
              f"全期统计的轻微泄漏对 OOT 评估指标的虚增程度约为 "
              f"{abs(delta):.4f}，{'可忽略，原报告的声明成立' if abs(delta) < 0.01 else '不可忽略，报告需更新口径'}。"
              f"滚动版本身可作为上线实时特征工程的参考实现。")
    utils.log("  P1-a 实验完成。")


if __name__ == "__main__":
    main()
