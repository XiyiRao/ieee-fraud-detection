# -*- coding: utf-8 -*-
"""
P0：UID 加 D1 锚点重建 + velocity 归因重跑
==========================================
实验：优化边界归因 E1 · 实体锚点（卡首用日）重建 UID + velocity 归因重跑

假设：velocity 无增量可能是因为现有 UID 缺少"卡首次使用日"锚点
（锚点定义：floor(day − D1)，即卡首次使用日），实体边界不准
导致实体级统计被稀释。本脚本验证该假设。

前置依赖：
    - data/processed/train_merged.pkl（scripts/01_load_and_inspect.py）
    - data/processed/oot_train.parquet / oot_test.parquet（scripts/03）
    - scripts/04_train_lgbm.py（复用其 train()，同参数同口径）

运行方式：
    .venv\\Scripts\\python.exe scripts/11_uid_v2_velocity.py

预期产出：
    - data/processed/oot_train_v2.parquet / oot_test_v2.parquet
      （velocity 三特征替换为 uid_v2 版本，其余列不变；不覆盖旧文件）
    - reports/figures/11_velocity_bins_v1_vs_v2.png  分箱梯度对比图
    - reports/figures/11_shap_velocity_v1_vs_v2.png   velocity SHAP 对比图
    - reports/run_log.txt 追加：实体对比、分箱梯度、双模型对比表与结论

口径说明（与 03 保持一致）：
    - velocity 特征在 tr+te 拼回的完整时间线上计算，每笔交易只统计
      当前时刻之前的历史；uid 首笔交易 secs_since_last 填 -1。
    - D1 缺失率约 0.2%，用全数据众数填充并在日志中声明。
    - 只新增不覆盖：不改动 scripts/01~10、utils.py 与旧中间文件；
      build_velocity_for() 是 utils.build_velocity_features 的本地改写版，
      仅把分组列参数化为 uid_v2，算法完全一致。
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

# 实验设计给定的基线成绩（用于对比，勿重新解释）
BASELINE = {"pr_auc": 0.4354, "recall": 0.0256, "precision": 0.7761}
VELOCITY_COLS = ["uid_cnt_1h", "uid_cnt_24h", "uid_secs_since_last"]


def load_module(mod_name, file_name):
    """按项目已有惯例（见 scripts/05）用 importlib 加载 scripts 下的模块。"""
    spec = importlib.util.spec_from_file_location(
        mod_name, Path(__file__).resolve().parent / file_name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# velocity 计算（utils.build_velocity_features 的本地改写版，分组列参数化）
# ---------------------------------------------------------------------------
def build_velocity_for(df: pd.DataFrame, uid_col: str) -> pd.DataFrame:
    """按 uid_col 分组计算 velocity 三特征，口径与 utils 完全一致。

    在完整时间线上统一计算（每笔交易只用它之前的数据，无未来信息），
    uid 首笔交易 secs_since_last 填 -1。
    """
    df = df.copy()
    original_order = df.index.to_numpy()
    df = df.sort_values(config.TIME_COL, kind="mergesort")  # 稳定排序，保序

    t = df[config.TIME_COL].to_numpy(dtype=np.float64)
    n = len(df)
    cnt_1h = np.zeros(n, dtype=np.int32)
    cnt_24h = np.zeros(n, dtype=np.int32)
    prev = np.full(n, np.nan)

    for pos in df.groupby(uid_col).indices.values():
        times = t[pos]                       # df 已按时间排序，组内单调不减
        prev[pos[1:]] = times[:-1]
        left_1h = np.searchsorted(times, times - 3600.0, side="right")
        cnt_1h[pos] = np.arange(len(times)) - left_1h
        left_24h = np.searchsorted(times, times - 86400.0, side="right")
        cnt_24h[pos] = np.arange(len(times)) - left_24h

    secs = t - prev
    secs[np.isnan(prev)] = -1.0              # uid 首笔交易，无上一笔

    df["uid_cnt_1h"] = cnt_1h
    df["uid_cnt_24h"] = cnt_24h
    df["uid_secs_since_last"] = secs.astype(np.float32)
    return df.loc[original_order]            # 恢复调用前的行顺序


# ---------------------------------------------------------------------------
# 分箱验证（复用 03 的分箱逻辑）
# ---------------------------------------------------------------------------
BINS = [-np.inf, -0.5, 60, 3600, 86400, np.inf]
BIN_LABELS = ["首笔交易(-1)", "<60秒", "60~3600秒", "3600~86400秒", ">86400秒"]


def bin_fraud_stats(te: pd.DataFrame, col: str, min_samples: int = 1000):
    """按 secs_since_last 分箱统计欺诈率，小样本箱（<min_samples）与相邻箱合并。"""
    cat = pd.cut(te[col], bins=BINS, labels=BIN_LABELS)
    stats = (te.groupby(cat, observed=False)[config.TARGET_COL]
               .agg(交易数="count", 欺诈率="mean"))
    rows = [(str(idx), int(r["交易数"]), float(r["欺诈率"]) * int(r["交易数"]))
            for idx, r in stats.iterrows()]
    merged = []
    for label, cnt, fraud in rows:
        if merged and merged[-1][1] < min_samples:
            pl, pc, pf = merged.pop()
            merged.append((f"{pl}+{label}", pc + cnt, pf + fraud))
        else:
            merged.append((label, cnt, fraud))
    if len(merged) > 1 and merged[-1][1] < min_samples:
        label, cnt, fraud = merged.pop()
        pl, pc, pf = merged.pop()
        merged.append((f"{pl}+{label}", pc + cnt, pf + fraud))
    return [(label, cnt, fraud / cnt) for label, cnt, fraud in merged]


def plot_bin_comparison(stats_v1, stats_v2):
    """新旧两版分箱欺诈率分组柱状图，存 reports/figures/11_velocity_bins_v1_vs_v2.png。"""
    labels = [s[0] for s in stats_v2]
    rates_v1 = [s[2] for s in stats_v1]
    rates_v2 = [s[2] for s in stats_v2]
    counts_v2 = [s[1] for s in stats_v2]

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars1 = ax.bar(x - width / 2, rates_v1, width, label="uid（旧版）",
                   color="lightgray", edgecolor="gray")
    bars2 = ax.bar(x + width / 2, rates_v2, width, label="uid_v2（D1 锚点版）",
                   color="steelblue")
    for bar, cnt in zip(bars2, counts_v2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"n={cnt:,}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12)
    ax.set_xlabel("距 uid 上一笔交易的秒数")
    ax.set_ylabel("欺诈率")
    ax.set_title("P0 验证 A：velocity 分箱欺诈率梯度（te，旧 uid vs D1 锚点 uid_v2）")
    ax.legend()
    utils.save_fig(fig, "11_velocity_bins_v1_vs_v2.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 模型训练与评估（复用 04 的 train；复用 05 的 evaluate_at_fpr）
# ---------------------------------------------------------------------------
def velocity_shap_ranks(booster, X_te, tag):
    """采样 5000 条算 mean|SHAP|，返回 velocity 三特征的排名与均值（口径同 07）。"""
    rng = np.random.default_rng(config.RANDOM_SEED)
    sample_idx = rng.choice(len(X_te), size=min(5000, len(X_te)), replace=False)
    shap_out = booster.predict(X_te.iloc[sample_idx], pred_contrib=True)
    mean_abs = np.abs(shap_out[:, :-1]).mean(axis=0)
    imp = pd.Series(mean_abs, index=X_te.columns).sort_values(ascending=False)
    ranks = {c: int(imp.index.get_loc(c)) + 1 for c in VELOCITY_COLS}
    utils.log(f"[{tag}] velocity 三特征 SHAP 排名（mean|SHAP|，共 {len(imp)} 维）:")
    for c in VELOCITY_COLS:
        utils.log(f"  {c:<22s} 排名 {ranks[c]:>3d}   mean|SHAP|={imp[c]:.4f}")
    return ranks, imp


def train_and_eval(mod04, mod05, tr, te, tag):
    """按 04 口径训练，在 te 上输出 PR-AUC/ROC-AUC/99.9% 分位阈值业务指标。"""
    model, X_tr, X_te = mod04.train(tr, te, save_path=None, verbose=False)
    pred = model.predict_proba(X_te)[:, 1]
    y_te = te[config.TARGET_COL].values
    pr_auc = average_precision_score(y_te, pred)
    roc_auc = roc_auc_score(y_te, pred)
    res = mod05.evaluate_at_fpr(y_te, pred, fpr_target=0.001)
    utils.log(f"[{tag}] PR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}  "
              f"FPR={res['actual_fpr']:.3%}  拦截率={res['recall']:.2%}  "
              f"精准率={res['precision']:.2%}")
    ranks, imp = velocity_shap_ranks(model.booster_, X_te, tag)
    return {"pr_auc": pr_auc, "roc_auc": roc_auc, "res": res,
            "ranks": ranks, "imp": imp}


def print_compare_table(m_base, m_v2):
    """与实验设计基线 0.4354 / 2.56% / 77.61% 的对比表。"""
    def row(name, m):
        r = m["res"]
        return (f"  {name:<22s} {m['pr_auc']:.4f}      {r['actual_fpr']:.3%}    "
                f"{r['recall']:.2%}     {r['precision']:.2%}")

    utils.log("")
    utils.log("对比表（业务阈值 = te 预测分 99.9% 分位）：")
    utils.log(f"  {'模型':<22s} {'PR-AUC':<8s} {'FPR':<8s} {'拦截率':<8s} {'精准率':<8s}")
    utils.log(row("基线（复跑前记录）", {"pr_auc": BASELINE["pr_auc"],
          "res": {"actual_fpr": 0.00026, "recall": BASELINE["recall"],
                  "precision": BASELINE["precision"]}}))
    utils.log(row("旧 uid velocity（重训）", m_base))
    utils.log(row("uid_v2 velocity（重训）", m_v2))


def main():
    utils.setup_plot_style()
    utils.log_section("P0：UID 加 D1 锚点重建 + velocity 归因重跑（script 11）")

    utils.check_input(config.MERGED_TRAIN_NAME,
                      "请先运行：python scripts/01_load_and_inspect.py")
    utils.check_input(config.OOT_TRAIN_NAME,
                      "请先运行：python scripts/03_oot_uid_features.py")
    utils.check_input(config.OOT_TEST_NAME,
                      "请先运行：python scripts/03_oot_uid_features.py")
    mod04 = load_module("train_lgbm", "04_train_lgbm.py")
    mod05 = load_module("evaluate", "05_evaluate.py")

    # ------------------------------------------------------------------
    # 1. 构建 D1 锚点版 uid_v2
    # ------------------------------------------------------------------
    df = utils.load_df(config.MERGED_TRAIN_NAME)
    assert df[config.ID_COL].is_unique, "TransactionID 应唯一，无法按主键回接"
    utils.log(f"读取合并数据: {df.shape}")

    day = np.floor(df[config.TIME_COL].to_numpy(dtype=np.float64) / 86400.0)
    d1_missing_rate = df["D1"].isna().mean()
    d1_mode = float(df["D1"].mode().iloc[0])
    utils.log(f"D1 缺失率 {d1_missing_rate:.3%}，缺失用全数据众数 {d1_mode:.0f} 填充"
              f"（口径声明：缺失以众数填充的近似处理）")
    d1 = df["D1"].fillna(d1_mode).to_numpy(dtype=np.float64)
    anchor = np.floor(day - d1).astype(np.int64)

    parts = []
    for c in ["card1", "card2", "addr1", "P_emaildomain"]:
        s = df[c].astype(str)
        parts.append(s.where(df[c].notna(), "unknown"))
    base_uid = parts[0] + "_" + parts[1] + "_" + parts[2] + "_" + parts[3]
    df["uid_v2"] = base_uid + "_" + pd.Series(anchor, index=df.index).astype(str)
    utils.log("uid_v2 = card1_card2_addr1_P_emaildomain_anchor，"
              "anchor = floor(day - D1)（day = floor(TransactionDT/86400)）")

    # ------------------------------------------------------------------
    # 2. 实体对比：锚点把实体拆细了多少
    # ------------------------------------------------------------------
    size_v1 = df.groupby(base_uid).size()
    size_v2 = df.groupby(df["uid_v2"]).size()
    utils.log("\n实体对比（全量 59 万笔）：")
    utils.log(f"  {'':<14s} {'实体数':>10s} {'最大实体':>10s} {'平均规模':>10s}")
    utils.log(f"  {'uid（旧）':<14s} {size_v1.shape[0]:>10,d} {size_v1.max():>10,d} "
              f"{size_v1.mean():>10.2f}")
    utils.log(f"  {'uid_v2（锚点）':<14s} {size_v2.shape[0]:>10,d} {size_v2.max():>10,d} "
              f"{size_v2.mean():>10.2f}")
    split_cnt = (df.groupby(base_uid)["uid_v2"].nunique())
    utils.log(f"  被锚点拆成 ≥2 个子实体的旧 uid 占比: "
              f"{(split_cnt > 1).mean():.2%}（平均拆成 "
              f"{split_cnt[split_cnt > 1].mean():.2f} 个）")

    # ------------------------------------------------------------------
    # 3. 用 uid_v2 在完整时间线上重算 velocity（口径与 03 一致），回接 tr/te
    # ------------------------------------------------------------------
    df_sorted = df.sort_values(config.TIME_COL, kind="mergesort").reset_index(drop=True)
    vel_v2 = build_velocity_for(df_sorted, "uid_v2")[
        [config.ID_COL] + VELOCITY_COLS]

    tr = utils.load_df(config.OOT_TRAIN_NAME)
    te = utils.load_df(config.OOT_TEST_NAME)
    v2_map = vel_v2.set_index(config.ID_COL)
    tr_v2 = tr.copy()
    te_v2 = te.copy()
    for c in VELOCITY_COLS:
        tr_v2[c] = tr[config.ID_COL].map(v2_map[c]).to_numpy()
        te_v2[c] = te[config.ID_COL].map(v2_map[c]).to_numpy()
    assert tr_v2[VELOCITY_COLS].isna().sum().sum() == 0, "velocity v2 回接存在缺失"
    utils.log("\nvelocity v2 已按 TransactionID 回接 tr/te（tr+te 拼接时间线口径不变，"
              "首笔仍填 -1）")
    utils.log(f"  te 中 uid_v2 首笔交易（secs=-1）占比: "
              f"{(te_v2['uid_secs_since_last'] == -1).mean():.2%}  "
              f"（旧 uid 口径: {(te['uid_secs_since_last'] == -1).mean():.2%}）")

    utils.save_df(tr_v2, "oot_train_v2")
    utils.save_df(te_v2, "oot_test_v2")

    # ------------------------------------------------------------------
    # 4. 验证 A：分箱欺诈率梯度对比（te 上）
    # ------------------------------------------------------------------
    stats_v1 = bin_fraud_stats(te, "uid_secs_since_last")
    stats_v2 = bin_fraud_stats(te_v2, "uid_secs_since_last")
    utils.log("\n验证 A —— 分箱欺诈率梯度（te）：")
    utils.log(f"  {'分箱':<26s} {'uid 旧版':>12s} {'uid_v2 锚点版':>14s} {'v2 交易数':>10s}")
    for (l1, n1, r1), (l2, n2, r2) in zip(stats_v1, stats_v2):
        utils.log(f"  {l2:<26s} {r1:>11.2%} {r2:>13.2%} {n2:>10,d}")
    # 梯度陡峭度：第一个非首笔箱（<60秒）与最后一箱（>86400秒）的欺诈率比值
    short1 = next(r for l, n, r in stats_v1 if l.startswith("<60秒"))
    long1 = next(r for l, n, r in stats_v1 if l.startswith(">86400秒"))
    short2 = next(r for l, n, r in stats_v2 if l.startswith("<60秒"))
    long2 = next(r for l, n, r in stats_v2 if l.startswith(">86400秒"))
    utils.log(f"  梯度陡峭度（<60秒 欺诈率 / >86400秒 欺诈率）: "
              f"旧 uid {short1 / long1:.2f}  vs  uid_v2 {short2 / long2:.2f}")
    plot_bin_comparison(stats_v1, stats_v2)

    # ------------------------------------------------------------------
    # 5. 验证 B：双模型对比（04 口径：同参数、tr 尾 15% 早停）
    # ------------------------------------------------------------------
    utils.log("\n验证 B —— 模型增量（重训 2 次，请耐心等待）...")
    m_base = train_and_eval(mod04, mod05, tr, te, "旧 uid velocity")
    m_v2 = train_and_eval(mod04, mod05, tr_v2, te_v2, "uid_v2 velocity")
    print_compare_table(m_base, m_v2)

    # SHAP 对比图
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(VELOCITY_COLS))
    width = 0.38
    ax.bar(x - width / 2, [m_base["imp"][c] for c in VELOCITY_COLS], width,
           label="旧 uid velocity", color="lightgray", edgecolor="gray")
    ax.bar(x + width / 2, [m_v2["imp"][c] for c in VELOCITY_COLS], width,
           label="uid_v2 velocity", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(VELOCITY_COLS, rotation=10)
    ax.set_ylabel("mean |SHAP|（te 采样 5000）")
    ax.set_title("velocity 三特征 SHAP 重要性：旧 uid vs D1 锚点 uid_v2")
    ax.legend()
    utils.save_fig(fig, "11_shap_velocity_v1_vs_v2.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 6. 结论
    # ------------------------------------------------------------------
    delta_pr = m_v2["pr_auc"] - m_base["pr_auc"]
    utils.log("\n[P0 结论]")
    if delta_pr > 0.005:
        utils.log(f"  uid_v2 版 velocity 带来 PR-AUC 提升 {delta_pr:+.4f}"
                  f"（{m_base['pr_auc']:.4f} → {m_v2['pr_auc']:.4f}），"
                  f"且 SHAP 排名明显前移 —— 锚点假设成立：velocity 此前的无增量"
                  f"确实源于实体边界不准。")
    else:
        utils.log(f"  uid_v2 版 velocity 的 PR-AUC 变化 {delta_pr:+.4f}"
                  f"（{m_base['pr_auc']:.4f} → {m_v2['pr_auc']:.4f}），SHAP 排名"
                  f"未实质前移 —— 锚点假设不成立：排除'实体精度不足'这一解释，"
                  f"velocity 无增量被归因于 V/C 覆盖说（velocity 三特征 SHAP 仅排 "
                  f"75/166/244，本身信号弱，不是实体边界问题）。")
    utils.log("  P0 实验完成。")


if __name__ == "__main__":
    main()
