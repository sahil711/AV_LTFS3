import pandas as pd
import numpy as np
import re
from bureau_fc import save_as_csv
import argparse


def process_data(DATA_DIR):
    for file in [
        "test_Data.xlsx",
        "test_bureau.xlsx",
        "train_bureau.xlsx",
        "train_Data.xlsx",
    ]:
        save_as_csv(DATA_DIR + file)

    train = pd.read_csv(DATA_DIR + "train_Data.csv")
    train_bur = pd.read_csv(DATA_DIR + "train_bureau.csv")
    test_bur = pd.read_csv(DATA_DIR + "test_bureau.csv")
    bur_df = pd.concat((train_bur, test_bur), axis=0)
    train["DisbursalDate"] = pd.to_datetime(train["DisbursalDate"])
    bur_df.sort_values(["ID", "DATE-REPORTED"], ascending=[True, False], inplace=True)
    bur_df = bur_df.drop_duplicates(
        ["ID", "DISBURSED-DT", "CONTRIBUTOR-TYPE", "DISBURSED-AMT/HIGH CREDIT"],
        keep="first",
    )
    bur_df["min_reported_date"] = (
        bur_df["REPORTED DATE - HIST"]
        .str[:-1]
        .fillna("")
        .str.split(",")
        .apply(min)
        .replace("", np.NaN)
    )
    bur_df["max_reported_date"] = (
        bur_df["REPORTED DATE - HIST"]
        .str[:-1]
        .fillna("")
        .str.split(",")
        .apply(max)
        .replace("", np.NaN)
    )
    bur_df["DISBURSED-DT"] = pd.to_datetime(bur_df["DISBURSED-DT"])
    bur_df["CLOSE-DT"] = pd.to_datetime(bur_df["CLOSE-DT"], errors="coerce")
    bur_df["LAST-PAYMENT-DATE"] = pd.to_datetime(
        bur_df["LAST-PAYMENT-DATE"], errors="coerce"
    )
    bur_df["max_reported_date"] = pd.to_datetime(
        bur_df["max_reported_date"], format="%Y%m%d", errors="coerce"
    )
    bur_df["min_reported_date"] = pd.to_datetime(
        bur_df["min_reported_date"], format="%Y%m%d", errors="coerce"
    )

    bur_df["DPD - HIST"].isnull().sum(), bur_df["REPORTED DATE - HIST"].isnull().sum()
    bur_df["dpd_str_len"] = 3 * bur_df["REPORTED DATE - HIST"].str[:-1].str.split(
        ","
    ).fillna("").apply(len)

    out = []
    for i, x in bur_df.iterrows():
        if not (
            (str(x["DPD - HIST"]).find("E") > -1) | (type(x["DPD - HIST"]) == float)
        ):
            out.append(
                "0" * (x["dpd_str_len"] - len(x["DPD - HIST"])) + x["DPD - HIST"]
            )
        else:
            out.append(x["DPD - HIST"])

    bur_df["dpd_string"] = out

    for col in ["OVERDUE-AMT", "DISBURSED-AMT/HIGH CREDIT", "CURRENT-BAL"]:
        amt = []
        for x in bur_df[col]:
            if type(x) != float:
                x = x.split(",")
                if len(x) > 1:
                    x[1] = "0" + x[1] if len(x[1]) == 1 else x[1]
                x = ",".join(x)
            amt.append(x)
        bur_df["corrected" + col] = amt
    amount_cols = [
        "CREDIT-LIMIT/SANC AMT",
        "correctedDISBURSED-AMT/HIGH CREDIT",
        "INSTALLMENT-AMT",
        "correctedCURRENT-BAL",
        "correctedOVERDUE-AMT",
    ]
    for col in amount_cols:
        bur_df[col] = (
            bur_df[col]
            .apply(lambda x: "".join(re.findall("[0-9]+", x)) if str(x) != "nan" else x)
            .astype("float")
        )
    bur_df.to_pickle(DATA_DIR + "bureau_data.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", help="pass the path where the data is stored", type=str
    )
    args = parser.parse_args()
    path = args.data_path
    process_data(path)
