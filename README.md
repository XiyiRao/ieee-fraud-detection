# IEEE-CIS 电商交易反欺诈 · 风控策略闭环项目

基于 Kaggle **IEEE-CIS Fraud Detection** 公开数据集（59 万笔脱敏交易 × 394 列，欺诈率 3.5%），
完成一次"业务约束定义 → 欺诈假设验证 → 特征工程 → 建模 → 业务化评估 → 三级处置策略
→ 规则提炼 → 漂移监控 → 优化边界归因"的完整风控项目闭环（离线 OOT 回测口径）。

## 核心结果

| 维度 | 结果 |
|---|---|
| 主指标 | PR-AUC **0.4354**（prevalence 基线的 12.7 倍），ROC-AUC 0.8738 |
| 硬约束拦截 | 99.9% 分位阈值下实际 FPR **0.026%**（严于 0.1% 误杀红线），欺诈拦截率 **2.56%**（95% CI [2.04%, 3.36%]，bootstrap 1000 次），拦截精准率 **77.6%** |
| 三级处置策略 | 整体欺诈触达 **39.1%**（拦截层 12.8% + 验证层 26.3%，后者基于 90% 验证有效率假设），总干预率仅 3.0%，回测净收益估算 **15.8 万元**（成本系数显式可校准） |
| 稳健性 | 3 组 OOT 切分点重训（拦截率波动 0.16pp）；LR 对照实验（PR-AUC +137% 的非线性增量）；Top20 特征 PSI 时间稳定性验证 19/20 稳定 |
| 优化边界 | 四轮归因实验（实体锚点重建 / 滚动口径修泄漏 / 客户级后处理 / 对抗验证筛漂移）均无实质增量，确认特征侧局部最优；剩余空间定位在金额加权方向（金额口径拦截率 1.28% vs 笔数 2.56%） |

## 方法论要点

- **可证伪假设驱动**：代入攻击者视角提出 3 条欺诈模式假设（设备聚集 / 金额异常 / 无指纹高危），
  分层欺诈率验证后 2 条被数据证伪——把错误直觉拦在特征工程之前。
- **无泄漏工程纪律**：OOT 时间切分（对照实验量化随机切分虚高 19.8%/59%）；
  训练集尾部切时间验证集早停（不用测试集早停）；velocity 时序特征按 tr/te 拼接
  时间线计算（每笔只统计当前时刻之前的历史）；全期聚合的轻微泄漏已量化（虚增仅 0.0001）。
- **业务化评估**：主指标 PR-AUC 而非 ROC-AUC（28:1 不平衡下后者虚高）；报 bootstrap
  置信区间而非单点；笔数口径与金额口径分开计价。
- **模型 → 策略的翻译**：双阈值三级处置矩阵（拦截/二次验证/放行），净收益公式完全透明；
  SHAP + 浅层决策树提炼机读补充规则（最优规则精准率 23.8%，基线 6.9 倍）。
- **漂移治理闭环**：对抗验证（训练前筛漂移特征，tr/te 可区分度 AUC 0.92）
  + PSI 时间切分监控（上线后盯漂移），同一思想的两端。

## 项目结构

```
ieee-fraud-project/
├── README.md                  # 本文件
├── requirements.txt           # 依赖清单
├── data/
│   ├── README.md              # 数据下载说明（Kaggle 4 个 CSV，不进版本库）
│   └── processed/             # 脚本自动创建的中间文件（不进版本库）
├── src/
│   ├── config.py              # 路径常量 / 随机种子 / OOT 比例
│   └── utils.py               # 内存压缩 / 存图 / 日志 / uid 特征
├── scripts/
│   ├── 01_load_and_inspect.py        # 数据加载、左连接、字段速览
│   ├── 02_case_analysis.py           # 欺诈 case 分析 + 假设 A/B/C 验证
│   ├── 03_oot_uid_features.py        # OOT 切分 + UID 聚合 + velocity 特征
│   ├── 04_train_lgbm.py              # LightGBM（tr 尾部时间验证集早停）
│   ├── 05_evaluate.py                # PR-AUC / FPR 约束拦截率 / bootstrap CI
│   ├── 06_three_tier_strategy.py     # 三级处置矩阵 + 净收益扫描
│   ├── 07_rules_extraction.py        # SHAP + 决策树规则提炼
│   ├── 08_psi_monitoring.py          # Top20 特征时间切分 PSI 监控
│   ├── 09_baseline_lr.py             # Logistic Regression 对照实验
│   ├── 10_amount_weighted.py         # 金额口径拦截效果分析
│   ├── 11_uid_v2_velocity.py         # 归因 E1：实体锚点重建 + velocity 重验
│   ├── 12_rolling_agg_demo.py        # 归因 E2：滚动口径聚合（量化泄漏）
│   ├── 13_client_postprocess.py      # 归因 E3：客户级预测聚合后处理
│   └── 14_adversarial_validation.py  # 归因 E4：对抗验证筛漂移特征
├── models/                    # 训练好的模型（不进版本库，04 脚本可复现）
└── reports/
    ├── 结项报告.md            # 主报告（11 章，含面试防御清单）
    ├── run_log.txt            # 全部脚本关键数字的运行日志（证据链）
    ├── figures/               # 全部图表
    ├── rules_summary.csv      # 规则精准率/覆盖率
    └── tier_scan_results.csv  # 33 组阈值对扫描结果
```

## 快速开始

```bash
python -m venv .venv && .venv\Scripts\activate    # Windows
pip install -r requirements.txt
# 按 data/README.md 从 Kaggle 下载 4 个 CSV 放入 data/
python scripts/01_load_and_inspect.py
python scripts/02_case_analysis.py
python scripts/03_oot_uid_features.py
python scripts/04_train_lgbm.py
python scripts/05_evaluate.py
python scripts/06_three_tier_strategy.py
python scripts/07_rules_extraction.py
python scripts/08_psi_monitoring.py
# 以下为补强/归因实验，可选
python scripts/09_baseline_lr.py
python scripts/10_amount_weighted.py
python scripts/11_uid_v2_velocity.py   # 耗时较长（从合并数据重建实体 + 重训 2 次）
python scripts/12_rolling_agg_demo.py
python scripts/13_client_postprocess.py
python scripts/14_adversarial_validation.py
```

每个脚本开头会检查输入文件是否存在，缺了会提示先运行哪个前置脚本；
关键数字自动追加到 `reports/run_log.txt`。

## 局限声明

- **标签滞后**：isFraud 为事后标签（拒付/投诉产生），存在数周空窗，线上表现会低于离线评估。
- **聚合特征口径**：UID 全期统计有轻微未来泄漏（已量化为 PR-AUC 虚增 0.0001，可忽略；
  velocity 与滚动口径实验为严格无泄漏的参照实现）。
- **净收益为估算**：误杀 50 元/笔、验证 5 元/笔、验证有效率 90% 均为显式假设，需业务校准。
- **特征脱敏**：V/C/D 列为第三方脱敏指纹，规则的业务可读性受限，落地需与特征平台核对语义。
