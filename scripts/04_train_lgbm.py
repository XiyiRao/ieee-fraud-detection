# -*- coding: utf-8 -*-
"""
阶段 3：LightGBM 建模（tr 尾部 15% 时间验证集早停）
==================================
所属阶段：第 2 周 · 阶段 3（基线模型）

前置依赖：需先运行 scripts/03_oot_uid_features.py
          （读取 data/processed/oot_train / oot_test）

运行方式：
    python scripts/04_train_lgbm.py

预期产出：
    - models/lgbm_model.txt                    训练好的模型（供 05/06/07 加载）
    - reports/figures/04_feature_importance.png 特征重要性 Top30
    - reports/run_log.txt 追加：特征数、最佳迭代轮数、te 上的 AUC 快览

【面试必答】为什么反欺诈用 GBDT 而不是传统评分卡（Logistic 评分卡）？
    1. 非线性捕捉能力：欺诈模式是强非线性的（如"金额高 且 无设备指纹 且
       凌晨"的交叉组合），GBDT 的树结构天然学交叉，评分卡需要人工做大量
       交叉变量和分箱才能逼近。
    2. 对脏数据鲁棒：IEEE-CIS 有 394 列、大量缺失和异常值，GBDT 不需要
       缺失值填充、对单调变换不敏感；评分卡要求严格的分箱、WOE 编码和
       单变量筛选，前期人力成本极高。
    3. 迭代效率：欺诈模式快速漂移，GBDT 可以每天自动重训；
       评分卡依赖专家经验，迭代周期以周/月计。评分卡的优势只剩"可解释性
       监管合规"，可以用 SHAP + 规则提炼（本项目的 07 脚本）补足。

为什么 scale_pos_weight=8：欺诈样本只占约 3.5%，类别极度不平衡，
给正类加权让模型重视少数类（代价是预测分整体偏高，需要后校准阈值，
06 脚本的阈值扫描就是干这个的）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from src import config
from src import utils

# 排除列：主键、标签、uid 字符串（uid 信息已聚合进数值特征，原字符串列高基数无用）
# is_new_uid 是 OOT 切分口径衍生的辅助标记，05/06/07 打分口径均剔除它，
# 训练口径需保持一致，否则会出现特征数不一致
EXCLUDE_COLS = [config.ID_COL, config.TARGET_COL, "uid", "is_new_uid",
                "TransactionDT"]


def prepare_xy(df):
    """特征/标签拆分 + 类别列转 category（LightGBM 原生支持 category，无需 one-hot）。

    为什么不用 one-hot：P_emaildomain 等列基数上百，one-hot 会维度爆炸；
    LightGBM 的 category 支持用直方图直接切分，效果和效率都更好。
    """
    drop_cols = [c for c in EXCLUDE_COLS if c in df.columns]
    # 阶段 2 辅助列不进模型
    drop_cols += [c for c in ["_hour", "_dist1_bin", "_uid_cnt_bin"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[config.TARGET_COL]
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    return X, y, cat_cols


def train(tr, te, save_path=None, verbose=True):
    """训练 LightGBM，在 tr 尾部 15%（按 TransactionDT 排序切出）上早停。

    参数化的原因：05 脚本做切分点敏感性实验（0.75/0.85）时会复用本函数。
    注意：不再用 te（OOT 集）早停，te 完全无污染，只用于训练后的 AUC 快览
    （见 main）；验证集从 tr 尾部按时间切出，模拟时间外推，更严格。
    """
    # 按时间排序后切尾部 15% 做验证集，模拟"用过去预测未来"
    tr = tr.sort_values(config.TIME_COL).reset_index(drop=True)
    cut = int(len(tr) * 0.85)
    tr_fit, tr_val = tr.iloc[:cut].copy(), tr.iloc[cut:].copy()

    X_tr, y_tr, cat_cols = prepare_xy(tr_fit)
    X_val, y_val, _ = prepare_xy(tr_val)
    X_te, y_te, _ = prepare_xy(te)
    # 保证验证集/te 的 category 类别与训练集一致（新类别会被置为缺失，可接受）
    for c in cat_cols:
        X_val[c] = X_val[c].astype("category")
        X_te[c] = X_te[c].astype("category")

    model = LGBMClassifier(
        n_estimators=1500,          # 上限给足，交给早停决定实际轮数
        learning_rate=0.01,         # 小学习率 + 多轮数，泛化更稳
        num_leaves=64,
        scale_pos_weight=8,         # 类别不平衡加权（见 docstring）
        subsample=0.8,              # 行采样防过拟合
        colsample_bytree=0.8,       # 列采样防过拟合（394 列里噪声很多）
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[early_stopping(200, verbose=False),
                   log_evaluation(0 if not verbose else 100)],
    )
    if save_path is not None:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        utils.save_booster(model.booster_, save_path)
        utils.log(f"[模型已保存] {save_path}  最佳迭代轮数={model.best_iteration_}")
    return model, X_tr, X_te


def main():
    utils.setup_plot_style()
    utils.log_section("阶段 3：LightGBM 建模")

    utils.check_input(config.OOT_TRAIN_NAME,
                      "请先运行：python scripts/03_oot_uid_features.py")
    utils.check_input(config.OOT_TEST_NAME,
                      "请先运行：python scripts/03_oot_uid_features.py")
    tr = utils.load_df(config.OOT_TRAIN_NAME)
    te = utils.load_df(config.OOT_TEST_NAME)
    utils.log(f"tr {tr.shape} / te {te.shape}")

    model_path = config.MODELS_DIR / "lgbm_model.txt"
    model, X_tr, X_te = train(tr, te, save_path=model_path)
    utils.log(f"特征数: {X_tr.shape[1]}")

    # 快览 te 上的 AUC（正式评估见 05 脚本，这里只确认模型没训崩）
    from sklearn.metrics import roc_auc_score
    pred = model.predict_proba(X_te)[:, 1]
    utils.log(f"te 上 ROC-AUC 快览: {roc_auc_score(te[config.TARGET_COL], pred):.4f}")

    # 特征重要性 Top30（gain 口径：对损失下降的实际贡献，比 split 次数更有意义）
    imp = pd.Series(model.booster_.feature_importance(importance_type="gain"),
                    index=model.booster_.feature_name()).sort_values(ascending=False)
    utils.log("\n特征重要性 Top15（gain）:")
    for name, val in imp.head(15).items():
        utils.log(f"  {name:<25s} {val:,.0f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    imp.head(30)[::-1].plot.barh(ax=ax)
    ax.set_title("LightGBM 特征重要性 Top30（gain）")
    ax.set_xlabel("gain")
    utils.save_fig(fig, "04_feature_importance.png")
    plt.close(fig)

    utils.log("\n阶段 3 完成。下一步：python scripts/05_evaluate.py")


if __name__ == "__main__":
    main()
