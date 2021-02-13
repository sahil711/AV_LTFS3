LGBM_PARAMS = {
    "boosting_type": "gbdt",
    "colsample_bytree": 0.8,
    "learning_rate": 0.1,
    "min_child_weight": 20.0,
    "n_estimators": 5000,
    "metric": "None",
    "n_jobs": -1,
    "objective": "multiclass",
    "subsample": 0.8,
    "subsample_freq": 5,
}

META_LGB_PARAMS = {
    "boosting_type": "gbdt",
    "colsample_bytree": 0.9,
    "learning_rate": 0.1,
    "min_child_weight": 80.0,
    "n_estimators": 5000,
    "n_jobs": -1,
    "num_leaves": 24,
    "metric": "None",
    "objective": "multiclass",
    "subsample": 0.7000000000000001,
    "subsample_freq": 5,
}


XGB_PARAMS = {
    "boosting_type": "gbdt",
    "colsample_bytree": 0.8,
    "learning_rate": 0.1,
    "min_child_weight": 20.0,
    "n_estimators": 5000,
    "metric": "None",
    "n_jobs": -1,
    "max_depth": 6,
    "objective": "multi:softmax",
    "subsample": 0.8,
    "disable_default_eval_metric": 1,
}
