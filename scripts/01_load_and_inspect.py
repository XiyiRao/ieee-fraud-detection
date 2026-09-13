# -*- coding: utf-8 -*-
"""
阶段 0：数据加载与初探
======================
所属阶段：第 1 周 · 阶段 0（数据理解）

前置依赖：无（第一个脚本），但需要先把 4 个 CSV 下载到 data/ 目录
          （见 data/README.md）。

运行方式：
    python scripts/01_load_and_inspect.py

预期产出：
    - data/processed/train_merged.parquet（无 pyarrow 时为 train_merged.pkl）
    - reports/run_log.txt 追加：形状、欺诈率、缺失率 Top20、内存占用、字段分组速览

本脚本做的事：
    1. 加载 train_transaction + train_identity
    2. 按 TransactionID 左连接（为什么是左连接见下）
    3. 输出形状 / 欺诈率 / 缺失率 Top20 / 内存占用
    4. 字段速览分组打印（金额/时间/卡/邮箱/距离/C列/M列/V列/设备）
    5. 存一份合并后的中间文件，供后续所有脚本使用
"""
import sys
from pathlib import Path

# 把项目根目录加入 sys.path，以便 import src 下的模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src import utils


def main():
    utils.log_section("阶段 0：数据加载与初探")

    # ------------------------------------------------------------------
    # 1. 检查并加载原始数据
    # ------------------------------------------------------------------
    utils.check_raw_csv(config.TRAIN_TRANSACTION_CSV, "缺少 train_transaction.csv")
    utils.check_raw_csv(config.TRAIN_IDENTITY_CSV, "缺少 train_identity.csv")

    utils.log("正在加载 train_transaction.csv ...")
    df_trans = pd_read(config.TRAIN_TRANSACTION_CSV)
    utils.log("正在加载 train_identity.csv ...")
    df_id = pd_read(config.TRAIN_IDENTITY_CSV)

    utils.log(f"train_transaction 形状: {df_trans.shape}")
    utils.log(f"train_identity    形状: {df_id.shape}")

    # ------------------------------------------------------------------
    # 2. 左连接（LEFT JOIN）
    #    为什么不用内连接：identity 表只覆盖约 1/4 的交易，内连接会把
    #    没有设备指纹的 3/4 交易全部丢掉，而"没有设备指纹"恰恰是重要的
    #    风险信号（阶段 1 会专门验证这一点），绝不能丢。
    # ------------------------------------------------------------------
    df = df_trans.merge(df_id, on=config.ID_COL, how="left")
    utils.log(f"左连接后形状: {df.shape}")
    utils.log(f"identity 覆盖率: {df['id_01'].notna().mean():.1%}"
              if "id_01" in df.columns else
              f"identity 覆盖率（按 Identity 表行数估算）: "
              f"{len(df_id) / len(df_trans):.1%}")

    # ------------------------------------------------------------------
    # 3. 标签与基础统计
    # ------------------------------------------------------------------
    fraud_rate = df[config.TARGET_COL].mean()
    utils.log(f"欺诈率（isFraud 均值）: {fraud_rate:.2%}")
    utils.log(f"欺诈样本数: {int(df[config.TARGET_COL].sum())} / {len(df)}")

    # 内存占用（压缩前）
    mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    utils.log(f"合并后内存占用（压缩前）: {mem_mb:.1f} MB")

    # ------------------------------------------------------------------
    # 4. 缺失率 Top20
    #    缺失不是敌人而是信号：V 列大量缺失意味着没有通过设备指纹检测，
    #    后续会把"缺失模式"本身当作特征来用。
    # ------------------------------------------------------------------
    na_rate = df.isna().mean().sort_values(ascending=False)
    utils.log("\n缺失率 Top20 字段：")
    for col, rate in na_rate.head(20).items():
        utils.log(f"  {col:<20s} {rate:.1%}")

    # ------------------------------------------------------------------
    # 5. 字段速览分组：394 列直接看不过来，按业务含义分组建立直觉
    # ------------------------------------------------------------------
    groups = {
        "金额相关": ["TransactionAmt"],
        "时间相关": [config.TIME_COL],
        "卡信息": [c for c in df.columns if c.startswith("card")],
        "邮箱域名": [c for c in df.columns if "emaildomain" in c],
        "距离特征": [c for c in df.columns if c.startswith("dist")],
        "C 列（计数类）": [c for c in df.columns
                       if c.startswith("C") and c[1:].isdigit()],
        "M 列（匹配类）": [c for c in df.columns
                       if c.startswith("M") and c[1:].isdigit()],
        "V 列（Vesta 设备指纹）": [c for c in df.columns
                               if c.startswith("V") and c[1:].isdigit()],
        "D 列（时间差）": [c for c in df.columns
                       if c.startswith("D") and c[1:].isdigit()],
        "identity 设备信息": [c for c in df.columns
                          if c.startswith("id_") or c.startswith("Device")],
    }
    utils.log("\n字段速览分组：")
    for gname, cols in groups.items():
        utils.log(f"  {gname:<22s} 共 {len(cols)} 列，示例: {cols[:5]}")

    # ------------------------------------------------------------------
    # 6. 内存压缩并保存中间文件
    # ------------------------------------------------------------------
    df = utils.reduce_mem_usage(df, name="train_merged")
    utils.save_df(df, config.MERGED_TRAIN_NAME)

    utils.log("\n阶段 0 完成。下一步：python scripts/02_case_analysis.py")


def pd_read(path):
    """读取 CSV 的小封装（保持 main 里逻辑清爽）。"""
    import pandas as pd
    return pd.read_csv(path)


if __name__ == "__main__":
    main()
