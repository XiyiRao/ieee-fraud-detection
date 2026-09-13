# 数据下载说明

本项目使用 Kaggle 的 **IEEE-CIS Fraud Detection** 数据集。

## 下载步骤

1. 打开 Kaggle，搜索 **"IEEE-CIS Fraud Detection"**（或直接访问该竞赛页面）。
2. 需要登录 Kaggle 账号并接受竞赛规则后才能下载。
3. 下载数据压缩包，解压后将以下 **4 个 CSV 文件** 放到本目录（`data/`）下：

| 文件名 | 说明 | 大致规模 |
|---|---|---|
| `train_transaction.csv` | 训练集交易表，含 `isFraud` 标签 | 约 59 万行 × 394 列 |
| `train_identity.csv` | 训练集身份/设备表 | 约 14 万行 × 41 列 |
| `test_transaction.csv` | 测试集交易表（无标签） | 约 51 万行 |
| `test_identity.csv` | 测试集身份/设备表 | 约 14 万行 |

> 本项目脚本主要使用两个 train 文件（有标签才能做有监督分析与评估）；
> test 文件用于最终提交预测，可作为进阶练习。

## 合并方式

两张表按 `TransactionID` **左连接**（以 transaction 表为主表）：
只有约 1/4 的交易有 identity 信息，内连接会丢掉大量样本，
而"没有设备指纹"本身就是重要的风险信号，不能丢。

## 目录约定

- 原始 CSV → 放在 `data/`（本目录）
- 脚本产出的中间文件 → 自动写入 `data/processed/`（由脚本自动创建）
