import pandas as pd
import pickle
import argparse
import sys

sys.path.append("ml_lib/")
from encoding import FreqeuncyEncoding


def save_obj(obj, name):
    with open(name, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, "rb") as f:
        return pickle.load(f)


def feature_join(DATA_DIR):
    train = pd.read_csv(DATA_DIR + "train_Data.csv")
    test = pd.read_csv(DATA_DIR + "test_Data.csv")
    target_map = {
        "No Top-up Service": 0,
        "12-18 Months": 1,
        "18-24 Months": 2,
        "24-30 Months": 3,
        "30-36 Months": 4,
        "36-48 Months": 5,
        " > 48 Months": 6,
    }
    train["Top-up Month"] = train["Top-up Month"].map(target_map)
    df = pd.concat((train, test), axis=0)

    bur_df = pd.read_pickle("bureau_future_feats_start_2_maturity_plus_2.pkl")
    bur_df2 = pd.read_pickle("ltfs_bureau_future_feats_start_2_maturity_plus_2.pkl")
    bur_df3 = pd.read_pickle("bur_future_feats.pkl")
    bur_df4 = pd.read_pickle("bureau_lead_lag_numeric_feats.pkl")

    df = (
        df.merge(bur_df, on="ID", how="left")
        .merge(bur_df2, on="ID", how="left")
        .merge(bur_df3, on="ID", how="left")
        .merge(bur_df4, on="ID", how="left")
    )
    df["DisbursalDate"] = pd.to_datetime(df["DisbursalDate"])
    df["MaturityDAte"] = pd.to_datetime(df["MaturityDAte"])
    df["pin1"] = df["ZiPCODE"].fillna("000000").astype("str").str[:2]
    df["pin2"] = df["ZiPCODE"].fillna("000000").astype("str").str[2:4]
    df["pin3"] = df["ZiPCODE"].fillna("000000").astype("str").str[4:6:]
    df["cal_tenor"] = (df["MaturityDAte"] - df["DisbursalDate"]).dt.days
    df["emi_sal_ratio"] = df["EMI"] / df["MonthlyIncome"]
    df["ltv2"] = df["DisbursalAmount"] / df["AssetCost"]
    df["disb_year"] = df["DisbursalDate"].dt.year
    df["disb_mon"] = df["DisbursalDate"].dt.month
    df["disb_day"] = df["DisbursalDate"].dt.day
    df["disb_dow"] = df["DisbursalDate"].dt.dayofweek

    df["mat_year"] = df["MaturityDAte"].dt.year
    df["mat_mon"] = df["MaturityDAte"].dt.month
    df["mat_day"] = df["MaturityDAte"].dt.day
    df["mat_dow"] = df["MaturityDAte"].dt.dayofweek
    cat_cols = [
        "Frequency",
        "InstlmentMode",
        "LoanStatus",
        "PaymentMode",
        "Area",
        "ManufacturerID",
        "SupplierID",
        "pin1",
        "pin2",
        "pin3",
        "SEX",
        "City",
        "State",
        "BranchID",
    ]
    target = "Top-up Month"
    drop_cols = [
        "ID",
        "DisbursalDate",
        "MaturityDAte",
        "AuthDate",
        "AssetID",
        "ZiPCODE",
    ]
    num_cols = df.columns[~df.columns.isin([target] + drop_cols + cat_cols)].tolist()
    use_cols = cat_cols + num_cols
    fe = FreqeuncyEncoding(categorical_columns=cat_cols, normalize=True, return_df=True)
    df = fe.fit_transform(df)
    cols = df[use_cols].columns[
        (df[use_cols].max() < 32000) & (df[use_cols].min() > -32000)
    ]
    for i, col in enumerate(cols):
        df[col] = df[col].astype("float32")
        if i % 100 == 0:
            print(i)
    print(df.shape)
    df = df[["ID"] + use_cols + [target]]

    df3 = pd.read_pickle("nikhil_feats_1.pkl")
    df2 = pd.read_pickle("nikhil_feats_2.pkl")
    df2 = df2[df2.columns[~df2.columns.isin(df.columns)].tolist() + ["ID"]]
    df3 = df3[df3.columns[~df3.columns.isin(df.columns)].tolist() + ["ID"]]
    df2.shape, df3.shape
    df = df.merge(df2, on="ID").merge(df3, on="ID")
    del df2, df3
    cat_cols = []
    target = "Top-up Month"
    drop_cols = ["ID", "DisbursalDate", "MaturityDAte", "AuthDate", "AssetID"]
    num_cols = df.columns[~df.columns.isin([target] + drop_cols + cat_cols)].tolist()
    use_cols = cat_cols + num_cols
    print(len(use_cols))
    cols = df.columns[(df.max() < 32000) & (df.min() > -32000)]
    for i, col in enumerate(cols):
        df[col] = df[col].astype("float32")
        if i % 100 == 0:
            print(i)
    final_cols = load_obj("column_list_500.pkl")
    df[final_cols].to_feather("features_500.fr")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", help="pass the path where the data is stored", type=str
    )
    args = parser.parse_args()
    path = args.data_path
    feature_join(path)
