import pandas as pd
import argparse
import datetime as dt
from bureau_fc import get_bureau_feats, get_bureau_feats_2
from multiprocessing import Pool
import warnings
import copy

warnings.filterwarnings("ignore")


def get_post_feats(grp):
    id_, temp = grp
    temp.MaturityDAte = temp.MaturityDAte + dt.timedelta(
        days=730
    )  # loans post 2 years of completetion of the current one
    cond1 = (temp["DISBURSED-DT"] >= temp.DisbursalDate) & (
        temp["DISBURSED-DT"] <= temp.MaturityDAte
    )
    account_df = temp[cond1]
    return get_bureau_feats(account_df)


def create_features(DATA_DIR):
    train = pd.read_csv(DATA_DIR + "train_Data.csv")
    test = pd.read_csv(DATA_DIR + "test_Data.csv")
    bur_df = pd.read_pickle(DATA_DIR + "bureau_data.pkl")
    train["DisbursalDate"] = pd.to_datetime(train["DisbursalDate"])
    test["DisbursalDate"] = pd.to_datetime(test["DisbursalDate"])
    train["MaturityDAte"] = pd.to_datetime(train["MaturityDAte"])
    test["MaturityDAte"] = pd.to_datetime(test["MaturityDAte"])

    bur_df["DATE-REPORTED"] = pd.to_datetime(bur_df["DATE-REPORTED"])
    bur_df["dpd_strin_var"] = (
        bur_df["dpd_string"]
        .fillna("")
        .apply(
            lambda x: [x[y - 3 : y] for y in range(3, len(x) + 3, 3)]
            if x.find("E") == -1
            else ["000"]
        )
    )
    df = pd.concat((train, test), axis=0)
    bur_df = bur_df.merge(
        df[["ID", "DisbursalDate", "MaturityDAte", "DisbursalAmount"]], on="ID"
    )
    bur_df.sort_values(["ID", "DISBURSED-DT"], inplace=True)

    temp_df = copy.deepcopy(bur_df)
    temp_df.MaturityDAte = temp_df.MaturityDAte + dt.timedelta(
        days=730
    )  # loans post 2 years of completetion of the current one
    cond1 = (temp_df["DISBURSED-DT"] >= temp_df.DisbursalDate) & (
        temp_df["DISBURSED-DT"] <= temp_df.MaturityDAte
    )
    temp_df = temp_df[cond1]
    grps = temp_df.groupby("ID")

    try:
        pool = Pool(8)
        data_outputs = pool.map(get_post_feats, grps)
    finally:
        pool.close()
        pool.join()

    bur_feats = pd.DataFrame(data_outputs)
    bur_feats.to_pickle("bureau_future_feats_start_2_maturity_plus_2.pkl")
    print(bur_feats.shape)

    temp_df = copy.deepcopy(bur_df)
    temp_df.MaturityDAte = temp_df.MaturityDAte + dt.timedelta(
        days=730
    )  # loans post 2 years of completetion of the current one
    cond1 = (temp_df["DISBURSED-DT"] >= temp_df.DisbursalDate) & (
        temp_df["DISBURSED-DT"] <= temp_df.MaturityDAte
    )
    cond2 = bur_df["SELF-INDICATOR"]
    temp_df = temp_df[cond1 & cond2]
    grps = temp_df.groupby("ID")

    try:
        pool = Pool(8)
        data_outputs = pool.map(get_post_feats, grps)
    finally:
        pool.close()
        pool.join()

    ltfs_bur_feats = pd.DataFrame(data_outputs)
    ltfs_bur_feats.columns = [
        "ltfs_" + x if x != "ID" else x for x in ltfs_bur_feats.columns
    ]
    ltfs_bur_feats.to_pickle("ltfs_bureau_future_feats_start_2_maturity_plus_2.pkl")

    print(ltfs_bur_feats.shape)

    intervals = [0, 1, 2, 3, 4, 10]
    master_feats = df[["ID"]]

    for i in range(len(intervals) - 1):
        temp_df = copy.deepcopy(bur_df)
        cond1 = (
            temp_df["DISBURSED-DT"]
            > (temp_df.DisbursalDate + dt.timedelta(days=365 * intervals[i]))
        ) & (
            temp_df["DISBURSED-DT"]
            <= (temp_df.DisbursalDate + dt.timedelta(days=365 * intervals[i + 1]))
        )
        temp_df = temp_df[cond1]
        print(temp_df.ID.nunique(), temp_df.shape)
        grps = temp_df.groupby("ID")

        try:
            pool = Pool(8)
            data_outputs = pool.map(get_bureau_feats_2, grps)
        finally:  # To make sure processes are closed in the end, even if errors happen
            pool.close()
            pool.join()

        bur_feats = pd.DataFrame(data_outputs)
        print(bur_feats.shape)
        bur_feats.columns = [
            "{}_yr_frm_disb".format(i) + x if x != "ID" else x
            for x in bur_feats.columns
        ]
        master_feats = master_feats.merge(bur_feats, on="ID", how="left")

    for i in range(len(intervals) - 1):
        temp_df = copy.deepcopy(bur_df)
        cond1 = (
            (
                temp_df["DISBURSED-DT"]
                > (temp_df.DisbursalDate + dt.timedelta(days=365 * intervals[i]))
            )
            & (
                temp_df["DISBURSED-DT"]
                <= (temp_df.DisbursalDate + dt.timedelta(days=365 * intervals[i + 1]))
            )
            & (temp_df["ACCT-TYPE"] == "Tractor Loan")
        )
        temp_df = temp_df[cond1]
        print(temp_df.ID.nunique(), temp_df.shape)
        grps = temp_df.groupby("ID")

        try:
            pool = Pool(8)
            data_outputs = pool.map(get_bureau_feats_2, grps)
        finally:  # To make sure processes are closed in the end, even if errors happen
            pool.close()
            pool.join()

        bur_feats = pd.DataFrame(data_outputs)
        print(bur_feats.shape)
        bur_feats.columns = [
            "trac_{}_yr_frm_disb".format(i) + x if x != "ID" else x
            for x in bur_feats.columns
        ]
        master_feats = master_feats.merge(bur_feats, on="ID", how="left")

    for i in range(len(intervals) - 1):
        temp_df = copy.deepcopy(bur_df)
        cond1 = (
            (
                temp_df["DISBURSED-DT"]
                > (temp_df.DisbursalDate + dt.timedelta(days=365 * intervals[i]))
            )
            & (
                temp_df["DISBURSED-DT"]
                <= (temp_df.DisbursalDate + dt.timedelta(days=365 * intervals[i + 1]))
            )
            & (temp_df["ACCT-TYPE"] == "Gold Loan")
        )
        temp_df = temp_df[cond1]
        print(temp_df.ID.nunique(), temp_df.shape)
        grps = temp_df.groupby("ID")

        try:
            pool = Pool(8)
            data_outputs = pool.map(get_bureau_feats_2, grps)
        finally:  # To make sure processes are closed in the end, even if errors happen
            pool.close()
            pool.join()
        bur_feats = pd.DataFrame(data_outputs)
        print(bur_feats.shape)
        bur_feats.columns = [
            "gold_{}_yr_frm_disb".format(i) + x if x != "ID" else x
            for x in bur_feats.columns
        ]
        master_feats = master_feats.merge(bur_feats, on="ID", how="left")

    print(master_feats.shape)
    master_feats.to_pickle("bur_future_feats.pkl")

    bur_df["tenor"] = ((bur_df["DATE-REPORTED"] - bur_df["DISBURSED-DT"]).dt.days).clip(
        upper=7300
    )

    num_cols = ["correctedDISBURSED-AMT/HIGH CREDIT", "correctedCURRENT-BAL", "tenor"]
    for col in num_cols:
        bur_df["prev_{}_value".format(col)] = bur_df.groupby("ID")[col].shift(1)
        bur_df["next_{}_value".format(col)] = bur_df.groupby("ID")[col].shift(-1)

    feats = df[["ID", "DisbursalDate", "DisbursalAmount"]].merge(
        bur_df[
            [
                "ID",
                "DISBURSED-DT",
                "correctedDISBURSED-AMT/HIGH CREDIT",
                "prev_correctedDISBURSED-AMT/HIGH CREDIT_value",
                "next_correctedDISBURSED-AMT/HIGH CREDIT_value",
                "prev_correctedCURRENT-BAL_value",
                "next_correctedCURRENT-BAL_value",
                "prev_tenor_value",
                "next_tenor_value",
            ]
        ]
        .rename(
            columns={
                "DISBURSED-DT": "DisbursalDate",
                "correctedDISBURSED-AMT/HIGH CREDIT": "DisbursalAmount",
            }
        )
        .drop_duplicates(["ID", "DisbursalDate", "DisbursalAmount"]),
        on=["ID", "DisbursalDate", "DisbursalAmount"],
        how="left",
    )
    feats.drop(["DisbursalDate", "DisbursalAmount"], axis=1, inplace=True)
    feats.fillna(-1).to_pickle("bureau_lead_lag_numeric_feats.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", help="pass the path where the data is stored", type=str
    )
    args = parser.parse_args()
    path = args.data_path
    create_features(path)
