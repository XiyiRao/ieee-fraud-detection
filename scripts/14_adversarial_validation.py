# -*- coding: utf-8 -*-
"""
P1-c：对抗验证筛漂移特征（训练前筛除 train/test 分布漂移特征）
==============================================================
实验：优化边界归因 E4 · 对抗验证筛漂移特征

背景：对抗验证（Adversarial Validation）——把 tr 标 0、te 标 1，训练一个
二分类器区分样本来自哪个集合。若分类器 AUC 显著高于 0.5，说明 tr/te
存在分布漂移；AUC 越接近 1 漂移越严重。重要性最高的特征即"漂移特征"：
模型可能学到的是"区分 tr/te 的捷径"而非欺诈模式，在 OOT 上损害泛化。
这与 08 脚本的 PSI（上线后监控）形成闭环：PSI 管上线后，对抗验证管训练前。

前置依赖：
    - data/processed/oot_train.parquet / oot_test.parquet（scripts/03）
    - scripts/04_train_lgbm.py（复用其 prepare_xy / train，同口径）
    - scripts/05_evaluate.py（复用其 evaluate_at_fpr）

运行方式：
    .venv\\Scripts\\python.exe scripts/14_adversarial_validation.py

预期产出：
    - reports/figures/14_adv_importance.png  对抗验证特征重要性 Top20
    - reports/run_log.txt 追加：整体 AUC、Top20 重要性、剔除漂移特征
      前后的模型对比表与结论
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib.util

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src import config
from src import utils

BASELINE = {"pr_auc": 0.4354, "recall": 0.0256, "precision": 0.7761}
N_DRIFT_DROP = 10   # 剔除重要性最高的前 10 个漂移特征


def load_module(mod_name, file_name):
    spec = importlib.util.spec_from_file_location(
        mod_name, Path(__file__).resolve().parent / file_name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    utils.setup_plot_style()
    utils.log_section("P1-c：对抗验证筛漂移特征（script 14）")

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
    # 1. 对抗验证：tr 标 0、te 标 1，训练区分器
    #    特征处理与 04 完全一致（prepare_xy：剔主键/uid/时间，类别转 category）
    # ------------------------------------------------------------------
    X_tr, _, cat_cols = mod04.prepare_xy(tr)
    X_te, _, _ = mod04.prepare_xy(te)
    for c in cat_cols:                       # te 的类别集合与 tr 对齐
        X_te[c] = X_te[c].astype("category")

    X_adv = pd.concat([X_tr, X_te], ignore_index=True)
    # 修复：不只转 cat_cols，而是把 concat 后所有字符串类型的列统一转 category
    #（04 的 prepare_xy 可能未覆盖全部 str 列；pandas concat 也会把
    #  类别不一致的 category 列上cast为 str，旧版 LightGBM 不认 str dtype）
    cat_cols = []
    for c in X_adv.columns:
        if X_adv[c].dtype == object or isinstance(X_adv[c].dtype, pd.StringDtype) \
                or str(X_adv[c].dtype) == "category":
            X_adv[c] = X_adv[c].astype("category")
            cat_cols.append(c)
    y_adv = np.r_[np.zeros(len(X_tr)), np.ones(len(X_te))]

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_adv, y_adv, test_size=0.3, random_state=config.RANDOM_SEED,
        shuffle=True)
    adv_model = LGBMClassifier(
        n_estimators=2000, learning_rate=0.05, num_leaves=64,
        subsample=0.8, colsample_bytree=0.8,
        random_state=config.RANDOM_SEED, n_jobs=-1, verbose=-1)
    adv_model.fit(
        X_fit, y_fit, eval_set=[(X_val, y_val)], eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[early_stopping(100, verbose=False)])
    adv_auc = roc_auc_score(y_val, adv_model.predict_proba(X_val)[:, 1])
    utils.log(f"\n对抗验证器 AUC（tr=0 / te=1，留出 30%）: {adv_auc:.4f}")
    if adv_auc > 0.6:
        utils.log("  → 明显高于 0.5：tr/te 存在可学习的分布漂移，"
                  "有必要筛除漂移特征。")
    else:
        utils.log("  → 接近 0.5：tr/te 几乎不可区分，漂移有限"
                  "（本实验照常跑完，验证筛除是否无害）。")

    # ------------------------------------------------------------------
    # 2. 特征重要性 Top20 与累计贡献
    # ------------------------------------------------------------------
    imp = pd.Series(adv_model.booster_.feature_importance(importance_type="gain"),
                    index=X_adv.columns).sort_values(ascending=False)
    total_gain = imp.sum()
    top20_share = imp.head(20).sum() / total_gain
    topn_share = imp.head(N_DRIFT_DROP).sum() / total_gain
    utils.log(f"\n对抗验证特征重要性 Top20（gain 占全部重要性的 {top20_share:.1%}）：")
    for i, (name, val) in enumerate(imp.head(20).items(), 1):
        cum = imp.head(i).sum() / total_gain
        mark = " ← 剔除" if i <= N_DRIFT_DROP else ""
        utils.log(f"  {i:>2d}. {name:<25s} gain={val:>12,.0f}  累计占比={cum:.1%}{mark}")

    fig, ax = plt.subplots(figsize=(10, 7))
    imp.head(20)[::-1].plot.barh(ax=ax, color="steelblue")
    ax.set_title(f"对抗验证特征重要性 Top20（AUC={adv_auc:.4f}；"
                 f"前 {N_DRIFT_DROP} 名累计 {topn_share:.1%}）")
    ax.set_xlabel("gain")
    utils.save_fig(fig, "14_adv_importance.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3. 剔除漂移特征后按 04 口径重训，对比业务指标
    # ------------------------------------------------------------------
    drift_cols = imp.head(N_DRIFT_DROP).index.tolist()
    utils.log(f"\n剔除漂移特征 Top{N_DRIFT_DROP}（累计重要性 {topn_share:.1%}）：")
    utils.log("  " + ", ".join(drift_cols))

    tr_drop = tr.drop(columns=drift_cols)
    te_drop = te.drop(columns=drift_cols)
    model, X_tr, X_te = mod04.train(tr_drop, te_drop, save_path=None,
                                    verbose=False)
    pred = model.predict_proba(X_te)[:, 1]
    y_te = te[config.TARGET_COL].values
    from sklearn.metrics import average_precision_score
    pr_auc = average_precision_score(y_te, pred)
    roc_auc = roc_auc_score(y_te, pred)
    res = mod05.evaluate_at_fpr(y_te, pred, fpr_target=0.001)
    utils.log(f"\n剔除后重训: PR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}  "
              f"FPR={res['actual_fpr']:.3%}  拦截率={res['recall']:.2%}  "
              f"精准率={res['precision']:.2%}")

    def row(name, m):
        r = m["res"]
        return (f"  {name:<22s} {m['pr_auc']:.4f}      {r['actual_fpr']:.3%}    "
                f"{r['recall']:.2%}     {r['precision']:.2%}")

    utils.log("\n对比表（业务阈值 = te 预测分 99.9% 分位）：")
    utils.log(f"  {'模型':<22s} {'PR-AUC':<8s} {'FPR':<8s} {'拦截率':<8s} {'精准率':<8s}")
    utils.log(row("基线（复跑前记录）", {
        "pr_auc": BASELINE["pr_auc"],
        "res": {"actual_fpr": 0.00026, "recall": BASELINE["recall"],
                "precision": BASELINE["precision"]}}))
    utils.log(row(f"剔除 Top{N_DRIFT_DROP} 漂移特征", {
        "pr_auc": pr_auc, "res": res}))

    # ------------------------------------------------------------------
    # 4. 结论
    # ------------------------------------------------------------------
    delta = pr_auc - BASELINE["pr_auc"]
    utils.log("\n[P1-c 结论]")
    utils.log(f"  对抗验证器 AUC={adv_auc:.4f}，漂移最严重的 Top{N_DRIFT_DROP} "
              f"特征累计贡献 {topn_share:.1%}；剔除后 PR-AUC 变化 {delta:+.4f}"
              f"（基线 0.4354 → {pr_auc:.4f}），拦截率 "
              f"{BASELINE['recall']:.2%} → {res['recall']:.2%}。")
    if delta > 0.005:
        utils.log("  剔除漂移特征带来实质提升，建议纳入训练前特征筛选流程。")
    else:
        utils.log("  剔除漂移特征无实质提升（阴性/无害结论）——说明基线模型的"
                  "OOT 口径本身已把大部分分布漂移挡在评估之外，漂移特征更多是"
                  "'可区分 tr/te'而非'误导欺诈学习'；对抗验证可作为训练前的"
                  "体检项保留（成本低），但不指望它提分。")
    utils.log("  与 08 脚本 PSI 监控形成闭环：对抗验证管训练前（离线筛查），"
              "PSI 管上线后（在线监控）。")
    utils.log("  P1-c 实验完成。")


if __name__ == "__main__":
    main()
