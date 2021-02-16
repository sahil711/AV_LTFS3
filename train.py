import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from param_config import LGBM_PARAMS, META_LGB_PARAMS, XGB_PARAMS
from sklearn.metrics import f1_score
import argparse

sys.path.append("ml_lib/")
from custom_classifier_mutliclass import Estimator

target_map = {
    "No Top-up Service": 0,
    "12-18 Months": 1,
    "18-24 Months": 2,
    "24-30 Months": 3,
    "30-36 Months": 4,
    "36-48 Months": 5,
    " > 48 Months": 6,
}


def get_data(DATA_DIR):
    train = pd.read_csv(DATA_DIR + "train_Data.csv")
    test = pd.read_csv(DATA_DIR + "test_Data.csv")
    train["Top-up Month"] = train["Top-up Month"].map(target_map)
    #     reverse_map = {v:k for k,v in target_map.items()}
    train["DisbursalDate"] = pd.to_datetime(train["DisbursalDate"])
    test["DisbursalDate"] = pd.to_datetime(test["DisbursalDate"])
    df = pd.read_feather("features_500.fr")
    df2 = pd.read_pickle("nikhil_feats_3.pkl")
    df = df.merge(
        df2[["ID"] + df2.columns[~df2.columns.isin(df.columns)].tolist()], on="ID"
    )
    del df2
    cat_cols = []
    target = "Top-up Month"
    drop_cols = ["ID", "DisbursalDate", "MaturityDAte", "AuthDate", "AssetID"]
    num_cols = df.columns[~df.columns.isin([target] + drop_cols + cat_cols)].tolist()
    use_cols = cat_cols + num_cols
    train = df[df.ID.isin(train.ID)]
    test = df[df.ID.isin(test.ID)]
    train = train.sort_values("ID", ascending=True)
    test = test.sort_values("ID", ascending=True)
    train[target] = train[target].astype("int")

    print("columns used in model {}".format(len(use_cols)))

    return train, test, use_cols, target


def train_xgb(train, test, use_cols, target):
    oof_list = []
    test_list = []
    print("******************** TRAINING XGBOOST **********************************")
    for i in [2, 100, 200]:
        print("FOLD {}".format(i))
        folds = StratifiedKFold(5, shuffle=True, random_state=i)
        folds = [(x, y) for x, y in folds.split(train, train[target])]
        est = Estimator(
            model=XGBClassifier(**XGB_PARAMS),
            validation_scheme=folds,
            early_stopping_rounds=100,
            n_jobs=12,
        )
        temp = est.fit_transform(train[use_cols].values, train[target].values)
        meta_est = Estimator(
            model=LGBMClassifier(**META_LGB_PARAMS),
            validation_scheme=folds,
            early_stopping_rounds=100,
            n_jobs=-1,
        )
        oof_preds = meta_est.fit_transform(temp, train[target].values)
        test_preds = [
            est.predict_proba(test[use_cols].values) for est in est.fitted_models
        ]
        meta_test_df = np.mean(test_preds, axis=0)
        test_preds = [est.predict_proba(meta_test_df) for est in meta_est.fitted_models]
        test_preds = np.mean(test_preds, axis=0)
        oof_list.append(oof_preds)
        test_list.append(test_preds)
        #     est.save_model('saved_models/XGB_682_feats_seed{}.pkl'.format(i))
        #     meta_est.save_model('saved_models/meta_models/meta_XGB_682_feats_seed{}.pkl'.format(i))
        print(
            meta_est.cv_scores,
            meta_est.overall_cv_score,
            print(est.cv_scores, est.overall_cv_score),
        )
    final_test_preds = np.mean(test_list, axis=0)
    final_oof_preds = np.mean(oof_list, axis=0)
    print(f1_score(train[target], final_oof_preds.argmax(1), average="macro"))
    b = pd.DataFrame(final_test_preds)
    b["preds"] = b.values.argmax(1)
    b["ID"] = test["ID"].values
    a = pd.DataFrame(final_oof_preds)
    a["ID"] = train["ID"]
    a["Top-up Month"] = train["Top-up Month"]
    a["preds"] = a.iloc[:, :-2].values.argmax(1)
    a.to_pickle("xgb_oof_preds_682_feats_3_seeds_avg.pkl")
    b.to_pickle("xgb_test_preds_682_feats_3_seeds_avg.pkl")


def train_lgb(train, test, use_cols, target):
    print("******************** TRAINING LIGHTGBM **********************************")
    num_leaves = [128, 32, 192]
    for leaves in num_leaves:
        LGBM_PARAMS["num_leaves"] = leaves
        oof_list = []
        test_list = []
        for i in [100, 200]:
            print("FOLD {}".format(i))
            folds = StratifiedKFold(5, shuffle=True, random_state=i)
            folds = [(x, y) for x, y in folds.split(train, train[target])]

            est = Estimator(
                model=LGBMClassifier(**LGBM_PARAMS),
                validation_scheme=folds,
                early_stopping_rounds=100,
                n_jobs=-1,
            )
            temp = est.fit_transform(train[use_cols].values, train[target].values)
            #         est.save_model('saved_models/LGBM_682_depth{}_feats_seed{}.pkl'.format(leaves,i))
            meta_est = Estimator(
                model=LGBMClassifier(**META_LGB_PARAMS),
                validation_scheme=folds,
                early_stopping_rounds=100,
                n_jobs=-1,
            )
            oof_preds = meta_est.fit_transform(temp, train[target].values)
            print(
                meta_est.cv_scores,
                meta_est.overall_cv_score,
                print(est.cv_scores, est.overall_cv_score),
            )
            #         meta_est.save_model('saved_models/meta_models/meta_LGBM_682_depth{}_feats_seed{}.pkl'.format(leaves,i))
            test_preds = [
                est.predict_proba(test[use_cols].values) for est in est.fitted_models
            ]
            meta_test_df = np.mean(test_preds, axis=0)
            test_preds = [
                est.predict_proba(meta_test_df) for est in meta_est.fitted_models
            ]
            test_preds = np.mean(test_preds, axis=0)
            oof_list.append(oof_preds)
            test_list.append(test_preds)

        final_test_preds = np.mean(test_list, axis=0)
        final_oof_preds = np.mean(oof_list, axis=0)
        score = f1_score(train[target], final_oof_preds.argmax(1), average="macro")
        print("FINAL SCORE {}", format(score))
        b = pd.DataFrame(final_test_preds)
        b["preds"] = b.values.argmax(1)
        b["ID"] = test["ID"].values
        a = pd.DataFrame(final_oof_preds)
        a["ID"] = train["ID"]
        a["Top-up Month"] = train["Top-up Month"]
        a["preds"] = a.iloc[:, :-2].values.argmax(1)
        a.to_pickle("oof_preds_lgbm_depth_{}.pkl".format(leaves))
        b.to_pickle("test_preds_lgbm_depth_{}.pkl".format(leaves))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", help="pass the path where the data is stored", type=str
    )
    args = parser.parse_args()
    path = args.data_path
    tr, tst, cols, tar = get_data(path)
    train_lgb(tr, tst, cols, tar)
    train_xgb(tr, tst, cols, tar)
