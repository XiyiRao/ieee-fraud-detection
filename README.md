# IEEE-CIS Fraud Detection · End-to-End Risk Control Strategy Project

An end-to-end fraud risk-control project on the Kaggle **IEEE-CIS Fraud Detection** dataset
(590K anonymized transactions × 394 features, 3.5% fraud rate), covering the full loop:
**business constraint definition → fraud hypothesis testing → feature engineering → modeling →
business-oriented evaluation → three-tier decision strategy → rule extraction → drift monitoring
→ optimization-boundary attribution** (offline out-of-time backtest).

## Key Results

| Dimension | Result |
|---|---|
| Primary metric | PR-AUC **0.4354** (12.7× the prevalence baseline), ROC-AUC 0.8738 |
| Interception under hard constraint | At the 99.9th-percentile score threshold: actual FPR **0.026%** (well under the 0.1% false-decline budget), fraud recall **2.56%** (95% CI [2.04%, 3.36%], 1000× bootstrap), precision **77.6%** |
| Three-tier decision strategy | Overall fraud coverage **39.1%** (block tier 12.8% + step-up verification tier 26.3%, the latter assuming 90% verification pass-through), only 3.0% of transactions touched; estimated net benefit **¥158K** (all cost coefficients explicit and calibratable) |
| Robustness | Retrained on 3 OOT split points (recall variance 0.16pp); Logistic Regression control (LightGBM delivers +137% PR-AUC — the non-linear gain is real); PSI stability check: 19/20 top features stable |
| Optimization boundary | Four attribution experiments (entity-anchor rebuild / rolling-window leakage fix / client-level post-processing / adversarial-validation drift filtering) — none yields a material gain, confirming a local optimum on the feature side; the remaining headroom is amount-weighted training (amount-weighted recall 1.28% vs count-based 2.56%) |

## Methodology Highlights

- **Falsifiable-hypothesis-driven analysis**: three attacker-mindset fraud hypotheses
  (device aggregation / amount anomaly / missing-fingerprint risk) — two were **falsified by data**,
  stopping wrong intuitions from becoming features or rules.
- **Leakage discipline**: OOT time split (a controlled experiment quantifies random-split inflation
  at +19.8% PR-AUC / +59% recall); early stopping on a time-ordered holdout carved from the training
  tail (never the test set); velocity features computed on a stitched tr/te timeline so every
  transaction only sees its own past; the residual leakage of full-period aggregates is
  **quantified at 0.0001 PR-AUC** — negligible.
- **Business-first evaluation**: PR-AUC over ROC-AUC (inflated under 28:1 imbalance);
  bootstrap confidence intervals instead of point estimates; count-based and amount-weighted
  views reported separately.
- **Model → strategy translation**: dual-threshold three-tier matrix (block / step-up verification /
  pass) with a fully transparent net-benefit function; SHAP + shallow decision tree distills
  machine-readable fallback rules (best rule precision 23.8%, 6.9× base rate).
- **Drift governance loop**: adversarial validation (pre-training drift screening — tr/te are
  distinguishable at AUC 0.92) + PSI time-slice monitoring (post-deployment watch): two ends
  of the same idea.

## Project Structure

```
ieee-fraud-project/
├── README.md                  # this file
├── requirements.txt
├── data/
│   ├── README.md              # dataset download instructions (4 CSVs from Kaggle, not in repo)
│   └── processed/             # intermediate files (not in repo)
├── src/
│   ├── config.py              # path constants / random seed / OOT ratio
│   └── utils.py               # memory reduction / plotting / logging / uid & velocity features
├── scripts/
│   ├── 01_load_and_inspect.py        # load, left-join, field overview
│   ├── 02_case_analysis.py           # fraud case analysis + hypothesis A/B/C tests
│   ├── 03_oot_uid_features.py        # OOT split + UID aggregates + velocity features
│   ├── 04_train_lgbm.py              # LightGBM (time-holdout early stopping)
│   ├── 05_evaluate.py                # PR-AUC / recall under FPR constraint / bootstrap CI
│   ├── 06_three_tier_strategy.py     # three-tier matrix + net-benefit threshold scan
│   ├── 07_rules_extraction.py        # SHAP + decision-tree rule distillation
│   ├── 08_psi_monitoring.py          # PSI time-slice monitoring on top-20 features
│   ├── 09_baseline_lr.py             # Logistic Regression control experiment
│   ├── 10_amount_weighted.py         # amount-weighted interception analysis
│   ├── 11_uid_v2_velocity.py         # attribution E1: entity-anchor rebuild + velocity re-test
│   ├── 12_rolling_agg_demo.py        # attribution E2: rolling-window aggregates (leakage quantified)
│   ├── 13_client_postprocess.py      # attribution E3: client-level prediction averaging
│   └── 14_adversarial_validation.py  # attribution E4: adversarial-validation drift filtering
├── models/                    # trained model (not in repo; reproducible via script 04)
└── reports/
    ├── 结项报告.md            # main report in Chinese (11 chapters incl. interview Q&A defense)
    ├── run_log.txt            # run log of all key numbers (evidence chain)
    ├── figures/               # all figures
    ├── rules_summary.csv      # rule precision / coverage
    └── tier_scan_results.csv  # 33 threshold-pair scan results
```

> The main report is written in Chinese (the project's working language); all numbers are
> reproducible from `reports/run_log.txt`.

## Quick Start

```bash
python -m venv .venv && .venv\Scripts\activate    # Windows
pip install -r requirements.txt
# download the 4 CSVs from Kaggle into data/ (see data/README.md)
python scripts/01_load_and_inspect.py
python scripts/02_case_analysis.py
python scripts/03_oot_uid_features.py
python scripts/04_train_lgbm.py
python scripts/05_evaluate.py
python scripts/06_three_tier_strategy.py
python scripts/07_rules_extraction.py
python scripts/08_psi_monitoring.py
# optional: control & attribution experiments
python scripts/09_baseline_lr.py
python scripts/10_amount_weighted.py
python scripts/11_uid_v2_velocity.py   # slow (rebuilds entities from merged data + 2 retrains)
python scripts/12_rolling_agg_demo.py
python scripts/13_client_postprocess.py
python scripts/14_adversarial_validation.py
```

Each script checks its inputs up front and tells you which prerequisite to run first;
key numbers are appended to `reports/run_log.txt`.

## Limitations

- **Label lag**: `isFraud` comes from post-hoc chargebacks/disputes (weeks of delay);
  online performance will be lower than offline evaluation.
- **Aggregation window**: full-period UID statistics carry minor future leakage
  (quantified at 0.0001 PR-AUC, negligible); velocity and the rolling-window experiment
  are strictly leakage-free reference implementations.
- **Net benefit is an estimate**: false-decline cost ¥50, verification cost ¥5,
  90% verification pass-through are explicit assumptions awaiting business calibration.
- **Anonymized features**: V/C/D columns are third-party anonymized fingerprints;
  rule semantics must be validated with the feature platform before production use.
