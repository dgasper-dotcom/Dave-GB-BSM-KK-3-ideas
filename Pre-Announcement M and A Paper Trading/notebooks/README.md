# Notebooks

The first implementation keeps notebooks as reproducible scripts and report generators so that leakage checks can run in CI.

Recommended notebook/script mapping:

- `01_aemd_case_study.ipynb`: use `python -m src.aemd_case_study`.
- `02_dataset_validation.ipynb`: use `src.data_quality` and `src.leakage_tests`.
- `03_baselines.ipynb`: use `src.models.train_and_evaluate` with logistic and random forest.
- `04_model_training.ipynb`: add optional XGBoost/LightGBM/CatBoost/MLP.
- `05_walk_forward.ipynb`: iterate `ChronologicalSplit`.
- `06_event_study.ipynb`: use `src.event_study.event_study_returns`.
- `07_feature_importance.ipynb`: run ablations and SHAP where available.

Convert these into `.ipynb` once real historical data is wired in; the source modules are intentionally notebook-friendly.
