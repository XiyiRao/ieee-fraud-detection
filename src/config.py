# -*- coding: utf-8 -*-
"""
全局配置：路径常量、随机种子、OOT 切分比例。

所有 scripts/ 下的脚本都从这里读取路径，保证目录约定只有一处定义。
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径常量（基于本文件位置推导，不依赖运行时的当前目录）
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"      # 脚本间传递的中间文件都放这里
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"       # 所有图表统一输出到这里
MODELS_DIR = PROJECT_ROOT / "models"        # 训练好的模型
RUN_LOG = REPORTS_DIR / "run_log.txt"       # 关键数字的运行日志

# 原始数据文件名（下载后放到 data/ 目录）
TRAIN_TRANSACTION_CSV = DATA_DIR / "train_transaction.csv"
TRAIN_IDENTITY_CSV = DATA_DIR / "train_identity.csv"
TEST_TRANSACTION_CSV = DATA_DIR / "test_transaction.csv"
TEST_IDENTITY_CSV = DATA_DIR / "test_identity.csv"

# ---------------------------------------------------------------------------
# 建模常量
# ---------------------------------------------------------------------------
RANDOM_SEED = 42            # 全局随机种子，保证抽样/bootstrap 可复现
TARGET_COL = "isFraud"      # 标签列
ID_COL = "TransactionID"    # 主键列
TIME_COL = "TransactionDT"  # 时间列（相对某个起点的秒数偏移量）

# OOT（Out-Of-Time）切分比例：按时间排序后，前 80% 做训练、后 20% 做测试。
# 为什么用 OOT 而不是随机切分：风控模型上线后面对的永远是"未来"的交易，
# 随机切分会让未来信息泄漏进训练集（同一 uid 的行为模式跨期出现），
# 指标会虚高；OOT 切分模拟真实上线场景，评估结果才可信。
OOT_SPLIT_RATIO = 0.8

# 中间文件名（不含扩展名，扩展名由 utils.save_df 根据 pyarrow 是否可用决定）
MERGED_TRAIN_NAME = "train_merged"   # 01 脚本产出
OOT_TRAIN_NAME = "oot_train"         # 03 脚本产出（tr，早期 80%）
OOT_TEST_NAME = "oot_test"           # 03 脚本产出（te，晚期 20%）
