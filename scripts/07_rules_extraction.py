# -*- coding: utf-8 -*-
"""
阶段 6：SHAP 解释 + 规则提炼（把黑盒模型翻译成业务规则）
==========================================================
所属阶段：第 3 周 · 阶段 6（可解释性与规则化）

前置依赖：需先运行 scripts/04_train_lgbm.py（模型）
          和 scripts/06_three_tier_strategy.py（T1 阈值口径，本脚本会重新
          扫一遍取最优 T1，保证可独立运行）

运行方式：
    python scripts/07_rules_extraction.py

预期产出：
    - reports/figures/07_shap_summary.png      SHAP 特征重要性蜂群图
    - reports/rules_summary.csv                3~5 条提炼规则的汇总表
                                               （规则文本/精准率/覆盖率）
    - reports/run_log.txt 追加：SHAP Top20、决策树规则文本、逐规则验证结果

为什么要提炼规则：GBDT 是黑盒，业务方（风控运营）要的是"看得懂、能直接
配置进拦截系统"的规则。做法：在模型判高分的欺诈样本上拟合一棵浅决策树
（深度 3，最多 8 条规则），让它"模仿"黑盒模型的判断逻辑，再逐条在 te 上
独立验证精准率和覆盖率——规则不是从模型里"拆"出来的，而是"提炼+验证"
出来的，这个方法论本身就是面试亮点。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from src import config
from src import utils


def load_and_score():
    """加载模型与 te，返回 (te, pred, X_te)。"""
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
    return te, pred, X, booster


def shap_analysis(booster, X):
    """采样 5000 条计算 SHAP 值，输出 Top20 特征并画 summary 蜂群图。

    为什么采样：SHAP 计算复杂度随样本数线性增长，全量 12 万条太慢；
    5000 条已足够估计特征重要性的排序。
    """
    import shap

    utils.log("\n计算 SHAP 值（采样 5000 条，可能需要几分钟）...")
    sample_idx = np.random.default_rng(config.RANDOM_SEED).choice(
        len(X), size=min(5000, len(X)), replace=False)
    X_sample = X.iloc[sample_idx]
    # 直接用 LightGBM 原生 pred_contrib 接口算 SHAP 值，保留 category 列
    # 让 LightGBM 自己处理类别切分（shap.TreeExplainer 内部也是调这个接口，
    # 但它会把类别列转成 int 编码，与训练时的 categorical_feature 声明冲突）
    shap_out = booster.predict(X_sample, pred_contrib=True)
    # 最后一列是基准分（expected value），去掉后才是各特征的 SHAP 分量
    shap_values = shap_out[:, :-1]
    # 蜂群图坐标轴需要数值：另做一份 int 编码的展示用数据
    X_num = X_sample.copy()
    for c in X_num.columns:
        if str(X_num[c].dtype) == "category":
            X_num[c] = X_num[c].cat.codes

    mean_abs = np.abs(shap_values).mean(axis=0)
    top20 = pd.Series(mean_abs, index=X_num.columns).sort_values(ascending=False)
    utils.log("\nSHAP 特征重要性 Top20：")
    for i, (name, val) in enumerate(top20.head(20).items(), 1):
        utils.log(f"  {i:>2d}. {name:<25s} {val:.4f}")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_num, max_display=20, show=False)
    fig = plt.gcf()
    utils.save_fig(fig, "07_shap_summary.png")
    plt.close(fig)
    return top20


def extract_rules(te, pred, X):
    """在模型高分（≥T1）欺诈样本上训练深度 3 决策树，提炼可读规则。"""
    # 重新扫一遍阈值取最优 T1（与 06 口径一致；保证本脚本可独立运行）
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tier", Path(__file__).resolve().parent / "06_three_tier_strategy.py")
    tier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tier)
    q1s = np.quantile(pred, [0.995, 0.997, 0.998, 0.999, 0.9995, 0.9999])
    q2s = np.quantile(pred, [0.97, 0.98, 0.99, 0.993, 0.995, 0.997])
    best_nb, best_t1 = -np.inf, None
    for t1 in q1s:
        for t2 in q2s:
            if t2 < t1:
                nb = tier.evaluate_tier(te, pred, t1, t2)["net_benefit"]
                if nb > best_nb:
                    best_nb, best_t1 = nb, t1
    utils.log(f"\n沿用阶段 5 口径的最优 T1 = {best_t1:.4f}")

    # 决策树训练集：模型判高分的样本（分数 ≥ T1），标签用真实 isFraud
    # 直觉：让树在这些"模型认为最可疑"的样本里学"什么特征组合是真欺诈"
    mask = pred >= best_t1
    X_high = X[mask].copy()
    y_high = te.loc[mask, config.TARGET_COL]
    utils.log(f"模型高分样本 {mask.sum()} 笔，其中真欺诈 {y_high.sum()} 笔")

    # 决策树不能直接吃 category/缺失，统一转数值编码
    X_tree = X_high.copy()
    for c in X_tree.columns:
        if str(X_tree[c].dtype) == "category":
            X_tree[c] = X_tree[c].cat.codes
    X_tree = X_tree.fillna(-999)   # 缺失用哨兵值，树可以学"是否缺失"这个模式

    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30,
                                  class_weight="balanced",
                                  random_state=config.RANDOM_SEED)
    tree.fit(X_tree, y_high)
    rules_text = export_text(tree, feature_names=list(X_tree.columns))
    utils.log("\n=== 提炼出的决策树规则（深度 3） ===")
    utils.log(rules_text)

    return tree, X_tree.columns.tolist(), best_t1


def validate_rules(tree, feature_names, te, X):
    """把决策树的每条叶子路径还原成"条件组合"，逐条在全量 te 上独立验证。

    对每条预测为欺诈的叶子：
      精准率 = 命中样本中真欺诈的比例
      覆盖率 = 命中的真欺诈 / te 全部真欺诈
    """
    X_all = X.copy()
    for c in X_all.columns:
        if str(X_all[c].dtype) == "category":
            X_all[c] = X_all[c].cat.codes
    X_all = X_all.fillna(-999)
    y = te[config.TARGET_COL].values
    total_fraud = y.sum()

    leaf_id = tree.apply(X_all[feature_names])           # 每条样本落到哪个叶子
    leaf_value = tree.tree_.value                          # 叶子上的类别分布
    fraud_leaves = [i for i in range(tree.tree_.node_count)
                    if tree.tree_.children_left[i] == -1   # 是叶子
                    and leaf_value[i][0].argmax() == 1]    # 预测为欺诈

    rows = []
    for leaf in fraud_leaves:
        hit = leaf_id == leaf
        if hit.sum() == 0:
            continue
        prec = y[hit].mean()
        cover = y[hit].sum() / max(total_fraud, 1)
        # 还原该叶子的路径条件为可读文本
        path = describe_path(tree, feature_names, leaf)
        rows.append({"规则文本": path,
                     "命中笔数": int(hit.sum()),
                     "精准率": round(float(prec), 4),
                     "覆盖率": round(float(cover), 4)})
    rules_df = (pd.DataFrame(rows)
                .sort_values("精准率", ascending=False)
                .head(5)
                .reset_index(drop=True))
    return rules_df


def describe_path(tree, feature_names, leaf):
    """把从根到叶子的决策路径翻译成可读的"条件 AND 条件"文本。"""
    t = tree.tree_
    conds = []

    def walk(node, trail):
        if node == leaf:
            conds.extend(trail)
            return True
        left, right = t.children_left[node], t.children_right[node]
        # 注意：分裂特征的下标存在 t.feature[node]，不是 node 本身
        feat, thr = feature_names[t.feature[node]], t.threshold[node]
        if left != -1 and walk(left, trail + [f"{feat} <= {thr:.3f}"]):
            return True
        if right != -1 and walk(right, trail + [f"{feat} > {thr:.3f}"]):
            return True
        return False

    walk(0, [])
    return " AND ".join(conds) if conds else "（空路径）"


def main():
    utils.setup_plot_style()
    utils.log_section("阶段 6：SHAP 解释 + 规则提炼")

    utils.check_input(config.OOT_TEST_NAME,
                      "请先运行：python scripts/03_oot_uid_features.py")
    te, pred, X, booster = load_and_score()

    top20 = shap_analysis(booster, X)
    tree, feature_names, _ = extract_rules(te, pred, X)

    utils.log("\n逐条规则在全量 te 上独立验证...")
    rules_df = validate_rules(tree, feature_names, te, X)
    utils.log("\n=== 规则汇总表（Top5，按精准率排序） ===")
    utils.log(rules_df.to_string(index=False))

    out_csv = config.REPORTS_DIR / "rules_summary.csv"
    rules_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    utils.log(f"\n[规则汇总已保存] {out_csv}")
    utils.log("\n阶段 6 完成。最后一步：按 reports/报告模板.md 撰写结项报告。")


if __name__ == "__main__":
    main()
