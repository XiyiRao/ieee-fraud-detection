# -*- coding: utf-8 -*-
"""
阶段 7：特征漂移监控（PSI）
============================
所属阶段：第 3 周 · 阶段 7（上线后监控方案）

前置依赖：需先运行 scripts/04_train_lgbm.py（模型，用于取 SHAP Top20 特征）
          和 scripts/03_oot_uid_features.py（tr/te 中间文件）

运行方式：
    python scripts/08_psi_monitoring.py

预期产出：
    - reports/figures/08_psi_heatmap.png  Top10 漂移特征的 PSI 热力图
    - reports/run_log.txt 追加：各特征 PSI 变化表、超阈值标注、漂移结论

为什么做 PSI（面试高频考点）：
    1. 风控是攻防对抗：欺诈团伙会适应模型，特征分布随时间漂移，
       上线即开始失效。PSI（Population Stability Index）是最常用的
       分布漂移量化指标，逐周/逐月计算可提前发现"模型输入已不再是
       训练时见过的世界"。
    2. PSI = Σ(实际占比 - 预期占比) × ln(实际占比 / 预期占比)，
       分箱后对每箱的占比差异加权（差异越大权重越大）。
       经验阈值：PSI < 0.1 稳定；0.1~0.2 观察（需留意）；
       > 0.2 告警（应触发重训或下线该特征）。
    3. 本项目用 tr 的三个等长时间片（T1/T2/T3）+ te（T4）模拟
       "模型服役期间的四个观察窗"，对 SHAP Top20 特征算相邻时间片的
       PSI——既演示了监控方法，也顺带验证了 OOT 假设：如果 Top 特征
       在 T1→T4 漂移显著，恰好说明时间外推的难度是真实存在的。

口径说明：
    - 分箱边界取参照期（相邻两片中较早的一片）的十分位数，10 箱；
      占比为 0 的箱裁剪到 eps=1e-4 后归一化，避免 ln(0) 除零。
    - 类别特征（card6 / R_emaildomain 等）转 category 编码后参与分箱，
      PSI 衡量的是编码分布的漂移，对高基数类别是近似口径。
    - 缺失值不参与分箱（占比按非缺失样本归一），缺失率本身的漂移
      建议另设监控，不在本脚本范围。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src import utils

# PSI 经验阈值（见 docstring）
PSI_WATCH = 0.1    # 观察线
PSI_ALARM = 0.2    # 告警线
EPS = 1e-4


def load_xy():
    """加载 tr/te 并做与 04/05/07 一致的特征列处理，返回 (tr, te, X_tr, X_te)。"""
    tr = utils.load_df(config.OOT_TRAIN_NAME)
    te = utils.load_df(config.OOT_TEST_NAME)
    drop_cols = [c for c in [config.ID_COL, config.TARGET_COL, "uid",
                             "is_new_uid", "TransactionDT"] if c in tr.columns]
    X_tr = tr.drop(columns=drop_cols).copy()
    X_te = te.drop(columns=drop_cols).copy()
    # 类别编码必须用 tr/te 共享的类别表：两边各自 astype("category") 会按
    # 各自的出现顺序分配编码，同一取值的编码不一致，PSI 会把"编码差异"
    # 误判为"分布漂移"（card6 曾因此虚报 PSI=9.6）
    obj_cols = X_tr.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        cats = pd.Categorical(
            pd.concat([X_tr[c], X_te[c]], ignore_index=True)).categories
        X_tr[c] = pd.Categorical(X_tr[c], categories=cats)
        X_te[c] = pd.Categorical(X_te[c], categories=cats)
    return tr, te, X_tr, X_te


def shap_top20(X_te):
    """复用 07 的口径取 SHAP Top20 特征名（同种子采样 5000 条，pred_contrib）。"""
    model_path = config.MODELS_DIR / "lgbm_model.txt"
    if not model_path.exists():
        print("缺少模型文件，请先运行：python scripts/04_train_lgbm.py")
        sys.exit(1)
    booster = utils.load_booster(model_path)
    sample_idx = np.random.default_rng(config.RANDOM_SEED).choice(
        len(X_te), size=min(5000, len(X_te)), replace=False)
    X_sample = X_te.iloc[sample_idx]
    utils.log("计算 SHAP 值取 Top20 特征（采样 5000 条，与 07 口径一致）...")
    shap_out = booster.predict(X_sample, pred_contrib=True)
    mean_abs = np.abs(shap_out[:, :-1]).mean(axis=0)
    top20 = pd.Series(mean_abs, index=X_sample.columns).sort_values(ascending=False)
    return top20.head(20).index.tolist()


def slice_by_time(tr, X_tr, X_te, n_slices=3):
    """把 tr 按 TransactionDT 等时长切成 n_slices 片，te 作为最后一片。

    等时长而非等样本量：漂移监控关心的是"同样长的一段时间"，样本量
    本身的涨落（大促/节假日）是业务现象，不应被切分抹平。
    """
    t = tr[config.TIME_COL].to_numpy()
    t0, t3 = t.min(), t.max()
    dur = (t3 - t0) / n_slices
    slices = []
    for i in range(n_slices):
        lo, hi = t0 + i * dur, t0 + (i + 1) * dur
        mask = (t >= lo) & (t <= hi) if i == n_slices - 1 else (t >= lo) & (t < hi)
        slices.append(X_tr[mask])
    slices.append(X_te)
    labels = [f"T{i+1}" for i in range(n_slices)] + [f"T{n_slices+1}(te)"]
    utils.log(f"时间片切分（等时长 {dur/3600:.1f} 小时/片）：")
    for lab, s in zip(labels, slices):
        utils.log(f"  {lab:<8s} 样本量 {len(s):>7d}")
    return slices, labels


def numeric_values(s: pd.Series) -> np.ndarray:
    """把一列（含 category）转成可用于分箱的 float 数组，去掉缺失。"""
    if str(s.dtype) == "category":
        s = s.cat.codes          # 缺失为 -1，作为普通编码值参与分箱
    v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    return v[~np.isnan(v)]


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """PSI = Σ(实际占比-预期占比)*ln(实际占比/预期占比)。

    分箱边界取预期期（参照期）的十分位；零占比箱裁剪到 eps 后归一化，
    防止 ln(0) 和除零。
    """
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:                       # 常量列：无分布可言
        return 0.0
    e_cnt, _ = np.histogram(expected, bins=edges)
    a_cnt, _ = np.histogram(actual, bins=edges)
    e_pct = np.clip(e_cnt / e_cnt.sum(), EPS, None)
    a_pct = np.clip(a_cnt / a_cnt.sum(), EPS, None)
    e_pct /= e_pct.sum()
    a_pct /= a_pct.sum()
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def main():
    utils.setup_plot_style()
    utils.log_section("阶段 7：特征漂移监控（PSI）")

    utils.check_input(config.OOT_TRAIN_NAME,
                      "请先运行：python scripts/03_oot_uid_features.py")
    utils.check_input(config.OOT_TEST_NAME,
                      "请先运行：python scripts/03_oot_uid_features.py")
    tr, te, X_tr, X_te = load_xy()
    utils.log(f"tr {tr.shape} / te {te.shape}")

    top20 = shap_top20(X_te)
    utils.log("\n监控范围（SHAP Top20 特征）：")
    utils.log("  " + ", ".join(top20))

    slices, labels = slice_by_time(tr, X_tr, X_te)
    pairs = [(0, 1), (1, 2), (2, 3)]         # 相邻时间片：(T1,T2) (T2,T3) (T3,T4)
    pair_labels = [f"{labels[i]}→{labels[j]}" for i, j in pairs]

    # ------------------------------------------------------------------
    # 逐特征计算相邻时间片 PSI
    # ------------------------------------------------------------------
    rows = []
    for feat in top20:
        vals = [numeric_values(s[feat]) for s in slices]
        row = {"特征": feat}
        for (i, j), pl in zip(pairs, pair_labels):
            row[pl] = round(psi(vals[i], vals[j]), 4)
        row["max"] = max(row[pl] for pl in pair_labels)
        rows.append(row)
    table = (pd.DataFrame(rows)
             .sort_values("max", ascending=False)
             .reset_index(drop=True))

    # ------------------------------------------------------------------
    # 输出 PSI 变化表，标注超阈值特征
    # ------------------------------------------------------------------
    utils.log("\n=== SHAP Top20 特征 PSI 变化表（相邻时间片）===")
    utils.log(table.to_string(index=False))
    alarm = table[table["max"] > PSI_ALARM]
    watch = table[(table["max"] > PSI_WATCH) & (table["max"] <= PSI_ALARM)]
    utils.log(f"\n超阈值标注（max PSI 口径）：")
    utils.log(f"  [告警 PSI>{PSI_ALARM}] "
              + (", ".join(f"{r['特征']}({r['max']:.3f})" for _, r in alarm.iterrows())
                 if len(alarm) else "无"))
    utils.log(f"  [观察 {PSI_WATCH}<PSI≤{PSI_ALARM}] "
              + (", ".join(f"{r['特征']}({r['max']:.3f})" for _, r in watch.iterrows())
                 if len(watch) else "无"))

    # ------------------------------------------------------------------
    # Top10 漂移特征 PSI 热力图
    # ------------------------------------------------------------------
    top10 = table.head(10)
    mat = top10[pair_labels].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=max(PSI_ALARM * 1.5, np.nanmax(mat)))
    ax.set_xticks(range(len(pair_labels)), pair_labels)
    ax.set_yticks(range(len(top10)), top10["特征"])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                    fontsize=9, color="black")
    # 告警阈值参考线：在 colorbar 上标出 0.2
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("PSI")
    ax.set_title("SHAP Top10 特征相邻时间片 PSI 热力图\n"
                 "(T1~T3 为 tr 等时长切片，T4=te；PSI>0.1 观察，>0.2 告警)")
    utils.save_fig(fig, "08_psi_heatmap.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 结论
    # ------------------------------------------------------------------
    worst = table.iloc[0]
    utils.log("\n=== 漂移监控结论 ===")
    utils.log(f"漂移最严重的特征：{worst['特征']}（max PSI={worst['max']:.3f}，"
              f"出现在 {pair_labels[int(np.argmax([worst[pl] for pl in pair_labels]))]}）")
    if len(alarm):
        utils.log(f"达到告警线（>{PSI_ALARM}）的特征共 {len(alarm)} 个："
                  + ", ".join(alarm["特征"]))
    if len(watch):
        utils.log(f"处于观察区（{PSI_WATCH}~{PSI_ALARM}）的特征共 {len(watch)} 个："
                  + ", ".join(watch["特征"]))
    stable = table[table["max"] <= PSI_WATCH]
    utils.log(f"分布稳定（max PSI ≤ {PSI_WATCH}）的特征 {len(stable)}/20 个："
              + ", ".join(stable["特征"]))
    utils.log("上线预案：对告警特征触发重训并回测新旧模型并排；对观察特征"
              "缩短监控周期至周级；PSI 监控应与拦截率/误杀率业务指标联动，"
              "分布未漂移但业务指标恶化同样要复盘。")

    utils.log("\n阶段 7 完成。PSI 监控方案已演示，可平移到线上定时任务。")


if __name__ == "__main__":
    main()
