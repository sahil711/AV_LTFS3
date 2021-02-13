import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
from pandarallel import pandarallel
from tqdm import tqdm
import argparse
import warnings

pandarallel.initialize()
warnings.filterwarnings("ignore")


def clean(x):
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return np.nan


ignore_fts = [
    "DisbursalDate",
    "MaturityDAte",
    "AuthDate",
    "last_reported_date",
    "first_reported_date",
    "first_DISBURSED-AMT/HIGH CREDIT - DisbursalAmount",
    "last_DISBURSED-AMT/HIGH CREDIT - DisbursalAmount",
    "n_loans_before_curr_loan",
    "ACCT-TYPE_and_next_ACCT-TYPE_same",
    "substandard_asset_class_count",
    "mean_days_number_of_loans_applied",
    "total_sum_disbursed_after_loan_applied/day_diff_last",
    "total_sum_disbursed_after_loan_applied/DisbursalAmount",
    "sum CUR BAL - HIST",
    "mean_disbursed_amt_forward_diff",
    "Branch_ID_nunq_ACCT_TYPE",
    "ACCT_TYPE_Branch_ID_nunq",
    "next_ACCOUNT-STATUS",
    "ACCT-TYPE",
    "last 3 mean CUR BAL - HIST",
    "DisbursalAmount/AssetCost",
    "EMI/DisbursalAmount",
    "last_DISBURSED-DT",
    "first_DISBURSED-DT",
]

drop_fts = [
    "first_DISBURSED-AMT/HIGH CREDIT - last_DISBURSED-AMT/HIGH CREDIT",
    "ACCT-TYPE_Telco Landline_total_for_user",
    "ACCT-TYPE_Pradhan Mantri Awas Yojana - CLSS_total_for_user",
    "ACCT-TYPE_Business Non-Funded Credit Facility-Priority Sector-Others_total_for_user",
    "ACCT-TYPE_Commercial Equipment Loan_total_for_user",
    "ACCT-TYPE_Staff Loan_total_for_user",
    "ACCT-TYPE_Corporate Credit Card_total_for_user",
    "ACCT-TYPE_Prime Minister Jaan Dhan Yojana - Overdraft_total_for_user",
    "ACCT-TYPE_Fleet Card_total_for_user",
    "ACCT-TYPE_JLG Group_total_for_user",
    "ACCT-TYPE_SHG Individual_total_for_user",
    "ACCT-TYPE_Leasing_total_for_user",
    "ACCT-TYPE_Loan on Credit Card_total_for_user",
    "ACCT-TYPE_SHG Group_total_for_user",
    "ACCT-TYPE_Microfinance Personal Loan_total_for_user",
    "ACCT-TYPE_Microfinance Housing Loan_total_for_user",
]
use_acct_types = [
    "Gold Loan",
    "Personal",
    "Tractor Loan",
    "Overdraft",
    "Business Loan Priority Sector  Agriculture",
]
target_mapper = pd.Series(
    {
        "No Top-up Service": 0,
        "12-18 Months": 1,
        "18-24 Months": 2,
        "24-30 Months": 3,
        "30-36 Months": 4,
        "36-48 Months": 5,
        " > 48 Months": 6,
    }
)


def feature_creation3(DATA_DIR):
    train_bureau = pd.read_csv(DATA_DIR + "train_bureau.csv")
    test_bureau = pd.read_csv(DATA_DIR + "test_bureau.csv")
    train = pd.read_csv(DATA_DIR + "train_Data.csv")
    test = pd.read_csv(DATA_DIR + "test_Data.csv")
    ID_COL, TARGET_COL = "ID", "Top-up Month"
    train[TARGET_COL] = train[TARGET_COL].map(target_mapper)
    df = train.append(test).reset_index(drop=True)
    date_cols = ["DisbursalDate", "MaturityDAte", "AuthDate"]
    df[date_cols] = df[date_cols].apply(lambda x: pd.to_datetime(x))
    del df["AssetID"]
    df_bureau = train_bureau.append(test_bureau).reset_index(drop=True)
    df_bureau = df_bureau.drop_duplicates(["ID", "DISBURSED-DT"], keep="first")
    date_cols = ["DATE-REPORTED", "DISBURSED-DT", "CLOSE-DT", "LAST-PAYMENT-DATE"]
    df_bureau[date_cols] = df_bureau[date_cols].apply(
        lambda x: pd.to_datetime(x, errors="coerce")
    )
    df_bureau = df_bureau.sort_values(by=["ID", "DISBURSED-DT"]).reset_index(drop=True)
    df_bureau["app_dd"] = df_bureau["ID"].map(df.set_index("ID")["DisbursalDate"])
    df_bureau["DISBURSED-AMT/HIGH CREDIT"] = df_bureau[
        "DISBURSED-AMT/HIGH CREDIT"
    ].apply(lambda x: clean(x))
    df_bureau["DISBURSED-DT_days_since_start"] = (
        df_bureau["DISBURSED-DT"] - df_bureau["DISBURSED-DT"].min()
    ).dt.days
    # ACCT-TYPE features
    use_acct_types = [
        "Gold Loan",
        "Tractor Loan",
        "Overdraft",
        "Business Loan Priority Sector  Agriculture",
        "Commercial Vehicle Loan",
    ]

    for a in tqdm(use_acct_types):
        fltr = df_bureau["ACCT-TYPE"] == a
        df[f"ACCT-TYPE_{a}_DISBURSED_DT_max"] = df["ID"].map(
            df_bureau[fltr].groupby("ID")["DISBURSED-DT_days_since_start"].max()
        )

        df["tmp"] = df["ID"].map(df_bureau[fltr].groupby("ID")["DISBURSED-DT"].max())
        df[f"ACCT-TYPE_{a}_DISBURSED_DT_max - DisbursalDate"] = (
            df["tmp"] - df["DisbursalDate"]
        ).dt.days

        df[f"ACCT-TYPE_{a}_DISBURSED-AMT/HIGH CREDIT_last"] = df["ID"].map(
            df_bureau[fltr].groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].last()
        )
        df[f"ACCT-TYPE_{a}_DISBURSED-AMT/HIGH CREDIT_max"] = df["ID"].map(
            df_bureau[fltr].groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].max()
        )
        df[f"ACCT-TYPE_{a}_DISBURSED-AMT/HIGH CREDIT_last - DisbursalAmount"] = (
            df[f"ACCT-TYPE_{a}_DISBURSED-AMT/HIGH CREDIT_last"] - df["DisbursalAmount"]
        )

    del df["tmp"]
    for a in tqdm(use_acct_types):
        fltr = df_bureau["ACCT-TYPE"] == a
        df[f"ACCT-TYPE_{a}_DISBURSED_DT_max"] = df["ID"].map(
            df_bureau[fltr].groupby("ID")["DISBURSED-DT_days_since_start"].max()
        )
        fltr2 = fltr & (df_bureau["DISBURSED-DT"] > df_bureau["app_dd"])

        df[f"ACCT-TYPE_{a}_DISBURSED_DT_next_after_loan"] = df["ID"].map(
            df_bureau[fltr2].groupby("ID")["DISBURSED-DT_days_since_start"].first()
        )
        df["tmp"] = df["ID"].map(
            df_bureau[df_bureau["DISBURSED-DT"] == df_bureau["app_dd"]].set_index("ID")[
                "DISBURSED-DT_days_since_start"
            ]
        )

        f = f"ACCT-TYPE_{a}_DISBURSED_DT_next_after_loan - DisbursalDate"
        df[f] = df[f"ACCT-TYPE_{a}_DISBURSED_DT_next_after_loan"] - df["tmp"]

        df["tmp"] = df["ID"].map(df_bureau[fltr].groupby("ID")["DISBURSED-DT"].max())
        df[f"ACCT-TYPE_{a}_DISBURSED_DT_max - DisbursalDate"] = (
            df["tmp"] - df["DisbursalDate"]
        ).dt.days

        df["tmp"] = df["ID"].map(df_bureau[fltr].groupby("ID")["DISBURSED-DT"].max())
        df[f"ACCT-TYPE_{a}_DISBURSED_DT_max - DisbursalDate"] = (
            df["tmp"] - df["DisbursalDate"]
        ).dt.days

        df[f"ACCT-TYPE_{a}_DISBURSED-AMT/HIGH CREDIT_last"] = df["ID"].map(
            df_bureau[fltr].groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].last()
        )
        df[f"ACCT-TYPE_{a}_DISBURSED-AMT/HIGH CREDIT_max"] = df["ID"].map(
            df_bureau[fltr].groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].max()
        )
        df[f"ACCT-TYPE_{a}_DISBURSED-AMT/HIGH CREDIT_last - DisbursalAmount"] = (
            df[f"ACCT-TYPE_{a}_DISBURSED-AMT/HIGH CREDIT_last"] - df["DisbursalAmount"]
        )

    del df["tmp"]

    tmp = df_bureau[df_bureau["DISBURSED-DT"] > df_bureau["app_dd"]]

    f = f"TOTAL_DISBURSED-AMT/HIGH CREDIT_after_DisbursalDate"
    df[f] = df["ID"].map(tmp.groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].sum())

    f = f"MAX_DISBURSED-AMT/HIGH CREDIT_after_DisbursalDate"
    df[f] = df["ID"].map(tmp.groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].max())

    f = f"LAST_DISBURSED-AMT/HIGH CREDIT"
    df[f] = df["ID"].map(df_bureau.groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].last())
    df_bureau["first_DISBURSED-DT"] = df_bureau.groupby("ID")["DISBURSED-DT"].transform(
        "first"
    )
    df_bureau["DISBURSED-DT_days_since_first_loan"] = (
        df_bureau["DISBURSED-DT"] - df_bureau["first_DISBURSED-DT"]
    ).dt.days
    df_bureau["DISBURSED-DT_days_since_DisbursalDate"] = (
        df_bureau["DISBURSED-DT"] - df_bureau["app_dd"]
    ).dt.days
    df_bureau["DISBURSED-DT_days_since_DisbursalDate2"] = (
        df_bureau.groupby("ID")["DISBURSED-DT"].shift(-1) - df_bureau["app_dd"]
    ).dt.days
    for a in tqdm(df_bureau["SELF-INDICATOR"].unique()):
        fltr = df_bureau["SELF-INDICATOR"] == a
        df[f"SELF-INDICATOR-TYPE_{a}_DISBURSED_DT_max"] = df["ID"].map(
            df_bureau[fltr].groupby("ID")["DISBURSED-DT_days_since_start"].max()
        )

        df["tmp"] = df["ID"].map(df_bureau[fltr].groupby("ID")["DISBURSED-DT"].max())
        df[f"ACCT-TYPE_{a}_SELF-INDICATOR_{a}_DISBURSED_DT_max - DisbursalDate"] = (
            df["tmp"] - df["DisbursalDate"]
        ).dt.days

        df[f"SELF-INDICATOR_{a}_DISBURSED-AMT/HIGH CREDIT_last"] = df["ID"].map(
            df_bureau[fltr].groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].last()
        )

    del df["tmp"]

    for s in tqdm(df_bureau["SELF-INDICATOR"].unique()):
        for c in tqdm(
            [
                "Personal Loan",
                "Gold Loan",
                "Kisan Credit Card",
                "Consumer Loan",
            ]
        ):
            fltr = (df_bureau["SELF-INDICATOR"] == s) & (df_bureau[f"ACCT-TYPE"] == c)
            df[
                f"SELF-INDICATOR-TYPE_{a}_ACCT-TYPE_{c}_DISBURSED_DT_max_days_since_first_loan"
            ] = df["ID"].map(
                df_bureau[fltr]
                .groupby("ID")["DISBURSED-DT_days_since_first_loan"]
                .max()
            )
            df[
                f"SELF-INDICATOR-TYPE_{a}_ACCT-TYPE_{c}_DISBURSED_DT_max_days_since_DisbursalDate"
            ] = df["ID"].map(
                df_bureau[fltr]
                .groupby("ID")["DISBURSED-DT_days_since_DisbursalDate"]
                .max()
            )

            fltr_add = (df_bureau["DISBURSED-DT"] > df_bureau["app_dd"]) & fltr

            df[
                f"SELF-INDICATOR-TYPE_{a}_ACCT-TYPE_{c}_DISBURSED_DT_just_after_DisbursalDate"
            ] = df["ID"].map(
                df_bureau[fltr_add]
                .groupby("ID")["DISBURSED-DT_days_since_DisbursalDate"]
                .nth(0)
            )

            df[
                f"SELF-INDICATOR-TYPE_{a}_ACCT-TYPE_{c}_DISBURSED_DT_2nd_last_days_since_DisbursalDate"
            ] = df["ID"].map(
                df_bureau[fltr]
                .groupby("ID")["DISBURSED-DT_days_since_DisbursalDate"]
                .nth(-2)
            )

            df[
                f"SELF-INDICATOR-TYPE_{a}_ACCT-TYPE_{c}_DISBURSED_DT_3rd_last_days_since_DisbursalDate"
            ] = df["ID"].map(
                df_bureau[fltr]
                .groupby("ID")["DISBURSED-DT_days_since_DisbursalDate"]
                .nth(-3)
            )

            df[
                f"SELF-INDICATOR-TYPE_{a}_ACCT-TYPE_{c}_DISBURSED_DT_4th_last_days_since_DisbursalDate"
            ] = df["ID"].map(
                df_bureau[fltr]
                .groupby("ID")["DISBURSED-DT_days_since_DisbursalDate"]
                .nth(-4)
            )

            df[
                f"SELF-INDICATOR-TYPE_{a}_ACCT-TYPE_{c}_DISBURSED_DT_5th_last_days_since_DisbursalDate"
            ] = df["ID"].map(
                df_bureau[fltr]
                .groupby("ID")["DISBURSED-DT_days_since_DisbursalDate"]
                .nth(-5)
            )

            df[
                f"SELF-INDICATOR-TYPE_{a}_ACCT-TYPE_{c}_last_DISBURSED-AMT/HIGH CREDIT"
            ] = df["ID"].map(
                df_bureau[fltr].groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].last()
            )

    for s in tqdm(df_bureau["SELF-INDICATOR"].unique()):

        fltr = (df_bureau["SELF-INDICATOR"] == s) & (
            df_bureau[f"ACCT-TYPE"].isin(["Personal Loan", "Gold Loan"])
        )
        df[
            f"SELF-INDICATOR-TYPE_{a}_Non Tractor Loan_DISBURSED_DT_max_days_since_first_loan"
        ] = df["ID"].map(
            df_bureau[fltr].groupby("ID")["DISBURSED-DT_days_since_first_loan"].max()
        )

        df[
            f"SELF-INDICATOR-TYPE_{a}_Non Tractor Loan_DISBURSED_DT_max_days_since_DisbursalDate"
        ] = df["ID"].map(
            df_bureau[fltr].groupby("ID")["DISBURSED-DT_days_since_DisbursalDate"].max()
        )

        df[
            f"SELF-INDICATOR-TYPE_{a}_Non Tractor Loan_DISBURSED_DT_just_after_DisbursalDate"
        ] = df["ID"].map(
            df_bureau[fltr_add]
            .groupby("ID")["DISBURSED-DT_days_since_DisbursalDate"]
            .nth(0)
        )

        df[
            f"SELF-INDICATOR-TYPE_{a}_Non Tractor Loan_DISBURSED_DT_2nd_last_days_since_DisbursalDate"
        ] = df["ID"].map(
            df_bureau[fltr]
            .groupby("ID")["DISBURSED-DT_days_since_DisbursalDate"]
            .nth(-2)
        )

        df[
            f"SELF-INDICATOR-TYPE_{a}_Non Tractor Loan_DISBURSED_DT_3rd_last_days_since_DisbursalDate"
        ] = df["ID"].map(
            df_bureau[fltr]
            .groupby("ID")["DISBURSED-DT_days_since_DisbursalDate"]
            .nth(-3)
        )

        df[
            f"SELF-INDICATOR-TYPE_{a}_Non Tractor Loan_DISBURSED_DT_4th_last_days_since_DisbursalDate"
        ] = df["ID"].map(
            df_bureau[fltr]
            .groupby("ID")["DISBURSED-DT_days_since_DisbursalDate"]
            .nth(-4)
        )

        df[
            f"SELF-INDICATOR-TYPE_{a}_Non Tractor Loan_DISBURSED_DT_5th_last_days_since_DisbursalDate"
        ] = df["ID"].map(
            df_bureau[fltr]
            .groupby("ID")["DISBURSED-DT_days_since_DisbursalDate"]
            .nth(-5)
        )

        df[
            f"SELF-INDICATOR-TYPE_{a}_Non Tractor Loan_last_DISBURSED-AMT/HIGH CREDIT"
        ] = df["ID"].map(
            df_bureau[fltr].groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].last()
        )

    df_bureau["Top-up Month"] = df_bureau["ID"].map(df.set_index("ID")["Top-up Month"])

    for s in tqdm(df_bureau["SELF-INDICATOR"].unique()):

        fltr = (df_bureau["SELF-INDICATOR"] == s) & (
            df_bureau[f"ACCT-TYPE"].isin(
                ["Business Loan Priority Sector  Agriculture", "Tractor Loan"]
            )
        )

        df[
            f"SELF-INDICATOR-TYPE_{a}_Non Tractor Loan2_DISBURSED_DT_max_days_since_DisbursalDate"
        ] = df["ID"].map(
            df_bureau[fltr].groupby("ID")["DISBURSED-DT_days_since_DisbursalDate"].max()
        )

        df[
            f"SELF-INDICATOR-TYPE_{a}_Non Tractor Loan2_DISBURSED_DT_just_after_DisbursalDate"
        ] = df["ID"].map(
            df_bureau[fltr_add]
            .groupby("ID")["DISBURSED-DT_days_since_DisbursalDate"]
            .nth(0)
        )

    col = "ACCOUNT-STATUS"
    for a in ["Tractor Loan"]:
        for o in tqdm(df_bureau[col].unique()):
            for s in df_bureau["SELF-INDICATOR"].unique():

                fltr = (
                    (df_bureau["SELF-INDICATOR"] == s)
                    & (df_bureau[f"ACCT-TYPE"] == a)
                    & (df_bureau[col] == o)
                )

                df[
                    f"ACCT_TYPE_{a}_SELF-INDICATOR_{s}_{col}_{o}_Non Tractor Loan2_DISBURSED_DT_max_days_since_DisbursalDate"
                ] = df["ID"].map(
                    df_bureau[fltr]
                    .groupby("ID")["DISBURSED-DT_days_since_DisbursalDate"]
                    .max()
                )

                df[
                    f"ACCT_TYPE_{a}_SELF-INDICATOR_{s}_{col}_{o}_Non Tractor Loan2_DISBURSED_DT_just_after_DisbursalDate"
                ] = df["ID"].map(
                    df_bureau[fltr_add]
                    .groupby("ID")["DISBURSED-DT_days_since_DisbursalDate"]
                    .nth(0)
                )

                df[
                    f"ACCT_TYPE_{a}_SELF-INDICATOR_{s}_{col}_{o}_Non Tractor Loan2_DISBURSED_DT_second_after_DisbursalDate"
                ] = df["ID"].map(
                    df_bureau[fltr_add]
                    .groupby("ID")["DISBURSED-DT_days_since_DisbursalDate"]
                    .nth(1)
                )
    col = "OWNERSHIP-IND"
    for a in ["Tractor Loan"]:
        for o in tqdm(df_bureau[col].unique()):
            for s in df_bureau["SELF-INDICATOR"].unique():

                fltr = (
                    (df_bureau["SELF-INDICATOR"] == s)
                    & (df_bureau[f"ACCT-TYPE"] == a)
                    & (df_bureau[col] == o)
                )

                df[
                    f"ACCT_TYPE_{a}_SELF-INDICATOR_{s}_{col}_{o}_Non Tractor Loan2_DISBURSED_DT_max_days_since_DisbursalDate"
                ] = df["ID"].map(
                    df_bureau[fltr]
                    .groupby("ID")["DISBURSED-DT_days_since_DisbursalDate"]
                    .max()
                )

                df[
                    f"ACCT_TYPE_{a}_SELF-INDICATOR_{s}_{col}_{o}_Non Tractor Loan2_DISBURSED_DT_just_after_DisbursalDate"
                ] = df["ID"].map(
                    df_bureau[fltr_add]
                    .groupby("ID")["DISBURSED-DT_days_since_DisbursalDate"]
                    .nth(0)
                )

    for i in range(1, 3):
        fltr = df_bureau["DISBURSED-DT"] >= df_bureau["app_dd"]

        df["tmp"] = df["ID"].map(df_bureau[fltr].groupby("ID")["DISBURSED-DT"].nth(i))
        df[f"DisbursalDT - DISBURSED-DT_next_{i}"] = (
            df["DisbursalDate"] - df["tmp"]
        ).dt.days

        df["tmp"] = df["ID"].map(df_bureau[fltr].groupby("ID")["DISBURSED-DT"].nth(-i))
        df[f"DisbursalDT - DISBURSED-DT_last_{i}"] = (
            df["DisbursalDate"] - df["tmp"]
        ).dt.days

        df["tmp"] = df["ID"].map(
            df_bureau[fltr].groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].nth(-i)
        )
        df[f"DisbursalAmount - DISBURSED-AMT/HIGH CREDIT_last_{i}"] = (
            df["DisbursalAmount"] - df["tmp"]
        )

    del df["tmp"]

    tmp = df_bureau[df_bureau["DISBURSED-DT"] > df_bureau["app_dd"]]
    for c in tqdm(
        [
            "Personal Loan",
            "Gold Loan",
            "Kisan Credit Card",
            "Consumer Loan",
            "Business Loan",
        ]
    ):
        df[f"ACCT-TYPE_{c}_after_DisbursalDate_DISBURSED-AMT/HIGH CREDIT_mean"] = df[
            "ID"
        ].map(
            tmp[tmp["ACCT-TYPE"] == c].groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].mean()
        )
        df[f"ACCT-TYPE_{c}_after_DisbursalDate_DISBURSED-AMT/HIGH CREDIT_max"] = df[
            "ID"
        ].map(
            tmp[tmp["ACCT-TYPE"] == c].groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].max()
        )
        df[f"ACCT-TYPE_{c}_after_DisbursalDate_DISBURSED-AMT/HIGH CREDIT_first"] = df[
            "ID"
        ].map(
            tmp[tmp["ACCT-TYPE"] == c]
            .groupby("ID")["DISBURSED-AMT/HIGH CREDIT"]
            .first()
        )
        df[f"ACCT-TYPE_{c}_after_DisbursalDate_DISBURSED-AMT/HIGH CREDIT_second"] = df[
            "ID"
        ].map(
            tmp[tmp["ACCT-TYPE"] == c].groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].nth(1)
        )
        df[
            f"ACCT-TYPE_{c}_after_DisbursalDate_DISBURSED-AMT/HIGH CREDIT_max - DisbursalAmount"
        ] = (
            df[f"ACCT-TYPE_{c}_after_DisbursalDate_DISBURSED-AMT/HIGH CREDIT_max"]
            - df["DisbursalAmount"]
        )
        df[
            f"ACCT-TYPE_{c}_after_DisbursalDate_DISBURSED-AMT/HIGH CREDIT_first - DisbursalAmount"
        ] = (
            df[f"ACCT-TYPE_{c}_after_DisbursalDate_DISBURSED-AMT/HIGH CREDIT_first"]
            - df["DisbursalAmount"]
        )
        df[
            f"ACCT-TYPE_{c}_after_DisbursalDate_DISBURSED-AMT/HIGH CREDIT_second - DisbursalAmount"
        ] = (
            df[f"ACCT-TYPE_{c}_after_DisbursalDate_DISBURSED-AMT/HIGH CREDIT_second"]
            - df["DisbursalAmount"]
        )
        df[f"ACCT-TYPE_{c}_after_DisbursalDate_total_loans"] = df["ID"].map(
            tmp[tmp["ACCT-TYPE"] == c].groupby("ID").size()
        )

    df["ID_freq"] = df["ID"].map(df_bureau["ID"].value_counts())
    df["ID_freq"] = df["ID_freq"] == 1

    fltr = df_bureau["app_dd"] == df_bureau["DISBURSED-DT"]
    df_bureau["total_loan_days"] = (
        df_bureau["CLOSE-DT"] - df_bureau["DISBURSED-DT"]
    ).dt.days
    df["total_loan_days"] = df["ID"].map(
        df_bureau[fltr].set_index("ID")["total_loan_days"]
    )
    df["total_loan_days_max"] = df["ID"].map(
        df_bureau.groupby("ID")["total_loan_days"].max()
    )
    df["total_loan_days_min"] = df["ID"].map(
        df_bureau.groupby("ID")["total_loan_days"].min()
    )
    df["total_loan_days_first"] = df["ID"].map(
        df_bureau.groupby("ID")["total_loan_days"].first()
    )
    df["total_loan_days_last"] = df["ID"].map(
        df_bureau.groupby("ID")["total_loan_days"].last()
    )
    df["total_loan_days_range"] = df["total_loan_days_max"] - df["total_loan_days_min"]

    df["DisbursalDate - MaturityDAte"] = (
        df["DisbursalDate"] - df["MaturityDAte"]
    ).dt.days
    for c in ["ACCT-TYPE", "DISBURSED-AMT/HIGH CREDIT"]:
        df[c + "_first"] = df["ID"].map(df_bureau.groupby("ID")[c].first())
        df[c + "_last"] = df["ID"].map(df_bureau.groupby("ID")[c].last())
        df[c + "_second_last"] = df["ID"].map(df_bureau.groupby("ID")[c].nth(-2))

    df["ACCT-TYPE"] = df["ID"].map(
        df_bureau[df_bureau["DISBURSED-DT"] > df_bureau["app_dd"]]
        .groupby("ID")["ACCT-TYPE"]
        .first()
    )
    fltr = df_bureau["DISBURSED-DT"] > df_bureau["app_dd"]
    tmp = df_bureau[fltr]
    for c in df_bureau["SELF-INDICATOR"].unique():
        df_bureau["tmp"] = (df_bureau["SELF-INDICATOR"] == c) * 1
        df[f"mean_{c}_all"] = df["ID"].map(df_bureau.groupby("ID")["tmp"].mean())
        df[f"sum_{c}_all"] = df["ID"].map(df_bureau.groupby("ID")["tmp"].sum())

    cat_cols = df.head().select_dtypes("object").columns.tolist()
    cat_cols = [
        c for c in cat_cols if c not in ["DisbursalDate", "MaturityDAte", "AuthDate"]
    ]
    print(cat_cols)
    for c in cat_cols:
        df[c] = pd.factorize(df[c])[0]

    fts = [
        c
        for c in df.columns
        if c not in [ID_COL, TARGET_COL, "DisbursalDate", "MaturityDAte", "AuthDate"]
    ]
    print(len(fts))

    cols = df[fts].columns[(df[fts].max() < 62000) & (df[fts].min() > -62000)]
    for i, col in enumerate(cols):
        df[col] = df[col].astype("float32")
        if i % 100 == 0:
            print(i)

    df[["ID"] + fts].to_pickle("nikhil_feats_3.pkl")


def feature_creation2(DATA_DIR):
    train_bureau = pd.read_csv(DATA_DIR + "train_bureau.csv")
    test_bureau = pd.read_csv(DATA_DIR + "test_bureau.csv")
    train = pd.read_csv(DATA_DIR + "train_Data.csv")
    test = pd.read_csv(DATA_DIR + "test_Data.csv")
    ID_COL, TARGET_COL = "ID", "Top-up Month"
    train[TARGET_COL] = train[TARGET_COL].map(target_mapper)
    df = train.append(test).reset_index(drop=True)
    date_cols = ["DisbursalDate", "MaturityDAte", "AuthDate"]
    df[date_cols] = df[date_cols].apply(lambda x: pd.to_datetime(x))
    del df["AssetID"]
    df_bureau = train_bureau.append(test_bureau).reset_index(drop=True)
    df_bureau = df_bureau.drop_duplicates(["ID", "DISBURSED-DT"], keep="first")
    date_cols = ["DATE-REPORTED", "DISBURSED-DT", "CLOSE-DT", "LAST-PAYMENT-DATE"]
    df_bureau[date_cols] = df_bureau[date_cols].apply(
        lambda x: pd.to_datetime(x, errors="coerce")
    )
    df_bureau = df_bureau.sort_values(by=["ID", "DISBURSED-DT"]).reset_index(drop=True)
    df_bureau["app_dd"] = df_bureau["ID"].map(df.set_index("ID")["DisbursalDate"])
    df_bureau["DISBURSED-AMT/HIGH CREDIT"] = df_bureau[
        "DISBURSED-AMT/HIGH CREDIT"
    ].apply(lambda x: clean(x))
    df_bureau["DISBURSED-DT_days_since_start"] = (
        df_bureau["DISBURSED-DT"] - df_bureau["DISBURSED-DT"].min()
    ).dt.days
    for a in tqdm(use_acct_types):
        fltr = df_bureau["ACCT-TYPE"] == a
        df[f"ACCT-TYPE_{a}_DISBURSED_DT_max"] = df["ID"].map(
            df_bureau[fltr].groupby("ID")["DISBURSED-DT_days_since_start"].max()
        )

        df["tmp"] = df["ID"].map(df_bureau[fltr].groupby("ID")["DISBURSED-DT"].max())
        df[f"ACCT-TYPE_{a}_DISBURSED_DT_max - DisbursalDate"] = (
            df["tmp"] - df["DisbursalDate"]
        ).dt.days

        df[f"ACCT-TYPE_{a}_DISBURSED-AMT/HIGH CREDIT_last"] = df["ID"].map(
            df_bureau[fltr].groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].last()
        )
        df[f"ACCT-TYPE_{a}_DISBURSED-AMT/HIGH CREDIT_max"] = df["ID"].map(
            df_bureau[fltr].groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].max()
        )
        df[f"ACCT-TYPE_{a}_DISBURSED-AMT/HIGH CREDIT_last - DisbursalAmount"] = (
            df[f"ACCT-TYPE_{a}_DISBURSED-AMT/HIGH CREDIT_last"] - df["DisbursalAmount"]
        )

    del df["tmp"]
    for a in tqdm(use_acct_types):
        fltr = df_bureau["ACCT-TYPE"] == a
        df[f"ACCT-TYPE_{a}_DISBURSED_DT_max"] = df["ID"].map(
            df_bureau[fltr].groupby("ID")["DISBURSED-DT_days_since_start"].max()
        )
        fltr2 = fltr & (df_bureau["DISBURSED-DT"] > df_bureau["app_dd"])

        df[f"ACCT-TYPE_{a}_DISBURSED_DT_next_after_loan"] = df["ID"].map(
            df_bureau[fltr2].groupby("ID")["DISBURSED-DT_days_since_start"].first()
        )
        df["tmp"] = df["ID"].map(
            df_bureau[df_bureau["DISBURSED-DT"] == df_bureau["app_dd"]].set_index("ID")[
                "DISBURSED-DT_days_since_start"
            ]
        )

        f = f"ACCT-TYPE_{a}_DISBURSED_DT_next_after_loan - DisbursalDate"
        df[f] = df[f"ACCT-TYPE_{a}_DISBURSED_DT_next_after_loan"] - df["tmp"]

        df["tmp"] = df["ID"].map(df_bureau[fltr].groupby("ID")["DISBURSED-DT"].max())
        df[f"ACCT-TYPE_{a}_DISBURSED_DT_max - DisbursalDate"] = (
            df["tmp"] - df["DisbursalDate"]
        ).dt.days

        df["tmp"] = df["ID"].map(df_bureau[fltr].groupby("ID")["DISBURSED-DT"].max())
        df[f"ACCT-TYPE_{a}_DISBURSED_DT_max - DisbursalDate"] = (
            df["tmp"] - df["DisbursalDate"]
        ).dt.days

        df[f"ACCT-TYPE_{a}_DISBURSED-AMT/HIGH CREDIT_last"] = df["ID"].map(
            df_bureau[fltr].groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].last()
        )
        df[f"ACCT-TYPE_{a}_DISBURSED-AMT/HIGH CREDIT_max"] = df["ID"].map(
            df_bureau[fltr].groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].max()
        )
        df[f"ACCT-TYPE_{a}_DISBURSED-AMT/HIGH CREDIT_last - DisbursalAmount"] = (
            df[f"ACCT-TYPE_{a}_DISBURSED-AMT/HIGH CREDIT_last"] - df["DisbursalAmount"]
        )

    del df["tmp"]
    gc.collect()
    tmp = df_bureau[df_bureau["DISBURSED-DT"] > df_bureau["app_dd"]]

    f = "TOTAL_DISBURSED-AMT/HIGH CREDIT_after_DisbursalDate"
    df[f] = df["ID"].map(tmp.groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].sum())

    f = "MAX_DISBURSED-AMT/HIGH CREDIT_after_DisbursalDate"
    df[f] = df["ID"].map(tmp.groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].max())

    f = "LAST_DISBURSED-AMT/HIGH CREDIT"
    df[f] = df["ID"].map(df_bureau.groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].last())
    df_bureau["first_DISBURSED-DT"] = df_bureau.groupby("ID")["DISBURSED-DT"].transform(
        "first"
    )
    df_bureau["DISBURSED-DT_days_since_first_loan"] = (
        df_bureau["DISBURSED-DT"] - df_bureau["first_DISBURSED-DT"]
    ).dt.days
    df_bureau["DISBURSED-DT_days_since_DisbursalDate"] = (
        df_bureau["DISBURSED-DT"] - df_bureau["app_dd"]
    ).dt.days
    df_bureau["DISBURSED-DT_days_since_DisbursalDate2"] = (
        df_bureau.groupby("ID")["DISBURSED-DT"].shift(-1) - df_bureau["app_dd"]
    ).dt.days
    for a in tqdm(df_bureau["SELF-INDICATOR"].unique()):
        fltr = df_bureau["SELF-INDICATOR"] == a
        df[f"SELF-INDICATOR-TYPE_{a}_DISBURSED_DT_max"] = df["ID"].map(
            df_bureau[fltr].groupby("ID")["DISBURSED-DT_days_since_start"].max()
        )

        df["tmp"] = df["ID"].map(df_bureau[fltr].groupby("ID")["DISBURSED-DT"].max())
        df[f"ACCT-TYPE_{a}_SELF-INDICATOR_{a}_DISBURSED_DT_max - DisbursalDate"] = (
            df["tmp"] - df["DisbursalDate"]
        ).dt.days

        df[f"SELF-INDICATOR_{a}_DISBURSED-AMT/HIGH CREDIT_last"] = df["ID"].map(
            df_bureau[fltr].groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].last()
        )

    del df["tmp"]
    for s in tqdm(df_bureau["SELF-INDICATOR"].unique()):
        for c in tqdm(
            ["Personal Loan", "Gold Loan", "Kisan Credit Card", "Consumer Loan"]
        ):
            fltr = (df_bureau["SELF-INDICATOR"] == s) & (df_bureau["ACCT-TYPE"] == c)
            df[
                f"SELF-INDICATOR-TYPE_{a}_ACCT-TYPE_{c}_DISBURSED_DT_max_days_since_first_loan"
            ] = df["ID"].map(
                df_bureau[fltr]
                .groupby("ID")["DISBURSED-DT_days_since_first_loan"]
                .max()
            )
            df[
                f"SELF-INDICATOR-TYPE_{a}_ACCT-TYPE_{c}_DISBURSED_DT_max_days_since_DisbursalDate"
            ] = df["ID"].map(
                df_bureau[fltr]
                .groupby("ID")["DISBURSED-DT_days_since_DisbursalDate"]
                .max()
            )

            fltr_add = (df_bureau["DISBURSED-DT"] > df_bureau["app_dd"]) & fltr

            df[
                f"SELF-INDICATOR-TYPE_{a}_ACCT-TYPE_{c}_DISBURSED_DT_just_after_DisbursalDate"
            ] = df["ID"].map(
                df_bureau[fltr_add]
                .groupby("ID")["DISBURSED-DT_days_since_DisbursalDate"]
                .nth(0)
            )

            df[
                f"SELF-INDICATOR-TYPE_{a}_ACCT-TYPE_{c}_DISBURSED_DT_2nd_last_days_since_DisbursalDate"
            ] = df["ID"].map(
                df_bureau[fltr]
                .groupby("ID")["DISBURSED-DT_days_since_DisbursalDate"]
                .nth(-2)
            )

            df[
                f"SELF-INDICATOR-TYPE_{a}_ACCT-TYPE_{c}_DISBURSED_DT_3rd_last_days_since_DisbursalDate"
            ] = df["ID"].map(
                df_bureau[fltr]
                .groupby("ID")["DISBURSED-DT_days_since_DisbursalDate"]
                .nth(-3)
            )

            df[
                f"SELF-INDICATOR-TYPE_{a}_ACCT-TYPE_{c}_DISBURSED_DT_4th_last_days_since_DisbursalDate"
            ] = df["ID"].map(
                df_bureau[fltr]
                .groupby("ID")["DISBURSED-DT_days_since_DisbursalDate"]
                .nth(-4)
            )

            df[
                f"SELF-INDICATOR-TYPE_{a}_ACCT-TYPE_{c}_DISBURSED_DT_5th_last_days_since_DisbursalDate"
            ] = df["ID"].map(
                df_bureau[fltr]
                .groupby("ID")["DISBURSED-DT_days_since_DisbursalDate"]
                .nth(-5)
            )

            df[
                f"SELF-INDICATOR-TYPE_{a}_ACCT-TYPE_{c}_last_DISBURSED-AMT/HIGH CREDIT"
            ] = df["ID"].map(
                df_bureau[fltr].groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].last()
            )

    for i in range(1, 3):
        fltr = df_bureau["DISBURSED-DT"] >= df_bureau["app_dd"]

        df["tmp"] = df["ID"].map(df_bureau[fltr].groupby("ID")["DISBURSED-DT"].nth(i))
        df[f"DisbursalDT - DISBURSED-DT_next_{i}"] = (
            df["DisbursalDate"] - df["tmp"]
        ).dt.days

        df["tmp"] = df["ID"].map(df_bureau[fltr].groupby("ID")["DISBURSED-DT"].nth(-i))
        df[f"DisbursalDT - DISBURSED-DT_last_{i}"] = (
            df["DisbursalDate"] - df["tmp"]
        ).dt.days

        df["tmp"] = df["ID"].map(
            df_bureau[fltr].groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].nth(-i)
        )
        df[f"DisbursalAmount - DISBURSED-AMT/HIGH CREDIT_last_{i}"] = (
            df["DisbursalAmount"] - df["tmp"]
        )

    del df["tmp"]

    tmp = df_bureau[df_bureau["DISBURSED-DT"] > df_bureau["app_dd"]]
    for c in tqdm(
        [
            "Personal Loan",
            "Gold Loan",
            "Kisan Credit Card",
            "Consumer Loan",
            "Business Loan",
        ]
    ):
        df[f"ACCT-TYPE_{c}_after_DisbursalDate_DISBURSED-AMT/HIGH CREDIT_mean"] = df[
            "ID"
        ].map(
            tmp[tmp["ACCT-TYPE"] == c].groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].mean()
        )
        df[f"ACCT-TYPE_{c}_after_DisbursalDate_DISBURSED-AMT/HIGH CREDIT_max"] = df[
            "ID"
        ].map(
            tmp[tmp["ACCT-TYPE"] == c].groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].max()
        )
        df[f"ACCT-TYPE_{c}_after_DisbursalDate_DISBURSED-AMT/HIGH CREDIT_first"] = df[
            "ID"
        ].map(
            tmp[tmp["ACCT-TYPE"] == c]
            .groupby("ID")["DISBURSED-AMT/HIGH CREDIT"]
            .first()
        )
        df[f"ACCT-TYPE_{c}_after_DisbursalDate_DISBURSED-AMT/HIGH CREDIT_second"] = df[
            "ID"
        ].map(
            tmp[tmp["ACCT-TYPE"] == c].groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].nth(1)
        )
        df[
            f"ACCT-TYPE_{c}_after_DisbursalDate_DISBURSED-AMT/HIGH CREDIT_max - DisbursalAmount"
        ] = (
            df[f"ACCT-TYPE_{c}_after_DisbursalDate_DISBURSED-AMT/HIGH CREDIT_max"]
            - df["DisbursalAmount"]
        )
        df[
            f"ACCT-TYPE_{c}_after_DisbursalDate_DISBURSED-AMT/HIGH CREDIT_first - DisbursalAmount"
        ] = (
            df[f"ACCT-TYPE_{c}_after_DisbursalDate_DISBURSED-AMT/HIGH CREDIT_first"]
            - df["DisbursalAmount"]
        )
        df[
            f"ACCT-TYPE_{c}_after_DisbursalDate_DISBURSED-AMT/HIGH CREDIT_second - DisbursalAmount"
        ] = (
            df[f"ACCT-TYPE_{c}_after_DisbursalDate_DISBURSED-AMT/HIGH CREDIT_second"]
            - df["DisbursalAmount"]
        )
        df[f"ACCT-TYPE_{c}_after_DisbursalDate_total_loans"] = df["ID"].map(
            tmp[tmp["ACCT-TYPE"] == c].groupby("ID").size()
        )

    fltr = df_bureau["app_dd"] == df_bureau["DISBURSED-DT"]
    df_bureau["total_loan_days"] = (
        df_bureau["CLOSE-DT"] - df_bureau["DISBURSED-DT"]
    ).dt.days
    df["total_loan_days"] = df["ID"].map(
        df_bureau[fltr].set_index("ID")["total_loan_days"]
    )
    df["total_loan_days_max"] = df["ID"].map(
        df_bureau.groupby("ID")["total_loan_days"].max()
    )
    df["total_loan_days_min"] = df["ID"].map(
        df_bureau.groupby("ID")["total_loan_days"].min()
    )
    df["total_loan_days_first"] = df["ID"].map(
        df_bureau.groupby("ID")["total_loan_days"].first()
    )
    df["total_loan_days_last"] = df["ID"].map(
        df_bureau.groupby("ID")["total_loan_days"].last()
    )
    df["total_loan_days_range"] = df["total_loan_days_max"] - df["total_loan_days_min"]
    df["DisbursalDate - MaturityDAte"] = (
        df["DisbursalDate"] - df["MaturityDAte"]
    ).dt.days

    for c in ["ACCT-TYPE", "DISBURSED-AMT/HIGH CREDIT"]:
        df[c + "_first"] = df["ID"].map(df_bureau.groupby("ID")[c].first())
        df[c + "_last"] = df["ID"].map(df_bureau.groupby("ID")[c].last())
        df[c + "_second_last"] = df["ID"].map(df_bureau.groupby("ID")[c].nth(-2))

    df["ACCT-TYPE"] = df["ID"].map(
        df_bureau[df_bureau["DISBURSED-DT"] > df_bureau["app_dd"]]
        .groupby("ID")["ACCT-TYPE"]
        .first()
    )
    fltr = df_bureau["DISBURSED-DT"] > df_bureau["app_dd"]
    tmp = df_bureau[fltr]
    for c in df_bureau["SELF-INDICATOR"].unique():
        df_bureau["tmp"] = (df_bureau["SELF-INDICATOR"] == c) * 1
        df[f"mean_{c}_all"] = df["ID"].map(df_bureau.groupby("ID")["tmp"].mean())
        df[f"sum_{c}_all"] = df["ID"].map(df_bureau.groupby("ID")["tmp"].sum())

    cat_cols = df.head().select_dtypes("object").columns.tolist()
    cat_cols = [
        c for c in cat_cols if c not in ["DisbursalDate", "MaturityDAte", "AuthDate"]
    ]
    print(cat_cols)
    for c in cat_cols:
        df[c] = pd.factorize(df[c])[0]

    df["ID_freq"] = df["ID"].map(df_bureau["ID"].value_counts())
    df["ID_freq"] = df["ID_freq"] == 1
    fts = [
        c
        for c in df.columns
        if c not in [ID_COL, TARGET_COL, "DisbursalDate", "MaturityDAte", "AuthDate"]
    ]
    print(len(fts))

    cols = df[fts].columns[(df[fts].max() < 32000) & (df[fts].min() > -32000)]
    for i, col in enumerate(cols):
        df[col] = df[col].astype("float32")
        if i % 100 == 0:
            print(i)
    df[["ID"] + fts].to_pickle("nikhil_feats_2.pkl")


def feature_creation1(DATA_DIR):
    train_bureau = pd.read_csv(DATA_DIR + "train_bureau.csv")
    test_bureau = pd.read_csv(DATA_DIR + "test_bureau.csv")
    train = pd.read_csv(DATA_DIR + "train_Data.csv")
    test = pd.read_csv(DATA_DIR + "test_Data.csv")
    ID_COL, TARGET_COL = "ID", "Top-up Month"

    train[TARGET_COL] = train[TARGET_COL].map(target_mapper)
    df = train.append(test).reset_index(drop=True)
    date_cols = ["DisbursalDate", "MaturityDAte", "AuthDate"]
    df[date_cols] = df[date_cols].apply(lambda x: pd.to_datetime(x))

    del df["AssetID"]
    orig_fts = [c for c in df.columns if c not in ["ID", "Top-up Month"]]
    df[[c + "_freq" for c in orig_fts]] = df[orig_fts].apply(
        lambda x: x.map(x.value_counts())
    )
    df_bureau = train_bureau.append(test_bureau).reset_index(drop=True)
    df_bureau = df_bureau.drop_duplicates(["ID", "DISBURSED-DT"], keep="first")
    date_cols = ["DATE-REPORTED", "DISBURSED-DT", "CLOSE-DT", "LAST-PAYMENT-DATE"]
    df_bureau[date_cols] = df_bureau[date_cols].apply(
        lambda x: pd.to_datetime(x, errors="coerce")
    )
    df_bureau = df_bureau.sort_values(by=["ID", "DISBURSED-DT"]).reset_index(drop=True)
    df_bureau["app_dd"] = df_bureau["ID"].map(df.set_index("ID")["DisbursalDate"])
    fltr = df_bureau["CLOSE-DT"].isnull()
    df_bureau.loc[fltr, "CLOSE-DT"] = df_bureau.loc[fltr, "LAST-PAYMENT-DATE"]
    df_bureau["accounts_closed"] = (~df_bureau["CLOSE-DT"].isnull()) * 1
    df_bureau["total_accounts_closed"] = df_bureau.groupby("ID")[
        "accounts_closed"
    ].transform("sum")
    df_bureau["avg_no_accounts_closed"] = df_bureau.groupby("ID")[
        "accounts_closed"
    ].transform("mean")
    df_bureau["total_accounts_closed_ACCT-TYPE"] = df_bureau.groupby(
        ["ID", "ACCT-TYPE"]
    )["accounts_closed"].transform("sum")
    df_bureau["avg_no_accounts_closed_ACCT-TYPE"] = df_bureau.groupby(
        ["ID", "ACCT-TYPE"]
    )["accounts_closed"].transform("mean")
    df_bureau["total_accounts_closed_CONTRIBUTOR-TYPE"] = df_bureau.groupby(
        ["ID", "CONTRIBUTOR-TYPE"]
    )["accounts_closed"].transform("sum")
    df_bureau["avg_no_accounts_closed_CONTRIBUTOR-TYPE"] = df_bureau.groupby(
        ["ID", "CONTRIBUTOR-TYPE"]
    )["accounts_closed"].transform("mean")
    fltr = df_bureau["app_dd"] == df_bureau["DISBURSED-DT"]
    for f in [
        "total_accounts_closed_ACCT-TYPE",
        "avg_no_accounts_closed_ACCT-TYPE",
        "total_accounts_closed_CONTRIBUTOR-TYPE",
        "avg_no_accounts_closed_CONTRIBUTOR-TYPE",
    ]:
        df[f] = df["ID"].map(df_bureau[fltr].set_index("ID")[f])
    df_bureau["ba"] = pd.factorize(
        df_bureau["ACCT-TYPE"].astype("str")
        + df_bureau["CONTRIBUTOR-TYPE"].astype("str")
    )[0]
    df_bureau["CURRENT-BAL"] = df_bureau["CURRENT-BAL"].apply(lambda x: clean(x))
    df_bureau["INSTALLMENT-AMT"] = df_bureau["INSTALLMENT-AMT"].apply(
        lambda x: clean(x)
    )
    df_bureau["CREDIT-LIMIT/SANC AMT"] = df_bureau["CREDIT-LIMIT/SANC AMT"].apply(
        lambda x: clean(x)
    )

    for c in tqdm(["DATE-REPORTED", "DISBURSED-DT", "CLOSE-DT"]):
        df_bureau[c + "_days_since_start"] = (df_bureau[c] - df_bureau[c].min()).dt.days
        df_bureau[c + "_days_from_end"] = (df_bureau[c].max() - df_bureau[c]).dt.days

        df_bureau[c + "_ACCT_TYPE_min"] = df_bureau.groupby("ACCT-TYPE")[
            c + "_days_since_start"
        ].transform("min")
        df_bureau[c + "_ACCT_TYPE_max"] = df_bureau.groupby("ACCT-TYPE")[
            c + "_days_since_start"
        ].transform("max")

        df_bureau[c + "_max"] = df_bureau.groupby("ID")[c].transform("max")
        df_bureau[c + "_min"] = df_bureau.groupby("ID")[c].transform("min")
        df_bureau[c + "_2nd_max"] = df_bureau.groupby("ID")[c].transform("nth", -2)
        df_bureau[c + "_2nd_min"] = df_bureau.groupby("ID")[c].transform("nth", 1)

        df_bureau[c + "_range_in_days"] = (
            df_bureau[c + "_max"] - df_bureau[c + "_min"]
        ).dt.days

        df_bureau[c + "_next"] = df_bureau.groupby("ID")[c].shift(-1)
        df_bureau[c + "_prev"] = df_bureau.groupby("ID")[c].shift(1)

        df_bureau[c + "_2nd_next"] = df_bureau.groupby("ID")[c].shift(-2)
        df_bureau[c + "_2nd_prev"] = df_bureau.groupby("ID")[c].shift(2)

        df_bureau[c + "_diff_to_next"] = (df_bureau[c + "_next"] - df_bureau[c]).dt.days
        df_bureau[c + "_diff_to_prev"] = (df_bureau[c] - df_bureau[c + "_prev"]).dt.days
        df_bureau[c + "_diff_prev_to_next"] = (
            df_bureau[c + "_next"] - df_bureau[c + "_prev"]
        ).dt.days
        df_bureau[c + "_diff_to_2nd_next"] = (
            df_bureau[c + "_2nd_next"] - df_bureau[c]
        ).dt.days
        df_bureau[c + "_diff_to_2nd_prev"] = (
            df_bureau[c] - df_bureau[c + "_2nd_prev"]
        ).dt.days

        df_bureau[c + "_diff_to_max"] = (df_bureau[c + "_max"] - df_bureau[c]).dt.days
        df_bureau[c + "_diff_to_min"] = (df_bureau[c] - df_bureau[c + "_min"]).dt.days
        df_bureau[c + "_diff_to_2nd_max"] = (
            df_bureau[c + "_2nd_max"] - df_bureau[c]
        ).dt.days
        df_bureau[c + "_diff_to_2nd_min"] = (
            df_bureau[c] - df_bureau[c + "_2nd_min"]
        ).dt.days

        df_bureau[c + "_diff_to_next_acct_type"] = (
            df_bureau[c] - df_bureau.groupby(["ACCT-TYPE", "ID"])[c].shift(-1)
        ).dt.days
        df_bureau[c + "_diff_to_prev_acct_type"] = (
            df_bureau[c] - df_bureau.groupby(["ACCT-TYPE", "ID"])[c].shift(1)
        ).dt.days

        df_bureau[c + "_diff_to_next_CONTRIBUTOR-TYPE"] = (
            df_bureau[c] - df_bureau.groupby(["CONTRIBUTOR-TYPE", "ID"])[c].shift(-1)
        ).dt.days
        df_bureau[c + "_diff_to_prev_CONTRIBUTOR-TYPE"] = (
            df_bureau[c] - df_bureau.groupby(["CONTRIBUTOR-TYPE", "ID"])[c].shift(1)
        ).dt.days

        df_bureau[f"max_{c}_ACCT-TYPE"] = df_bureau.groupby(["ACCT-TYPE", "ID"])[
            c
        ].transform("max")
        df_bureau[f"min_{c}_ACCT-TYPE"] = df_bureau.groupby(["ACCT-TYPE", "ID"])[
            c
        ].transform("min")
        df_bureau[f"range_{c}_ACCT_TYPE"] = (
            df_bureau[f"max_{c}_ACCT-TYPE"] - df_bureau[f"min_{c}_ACCT-TYPE"]
        ).dt.days
        df_bureau[f"diff_to_max_{c}_ACCT-TYPE"] = (
            df_bureau[c] - df_bureau[f"max_{c}_ACCT-TYPE"]
        ).dt.days
        df_bureau[f"diff_to_min_{c}_ACCT-TYPE"] = (
            df_bureau[c] - df_bureau[f"min_{c}_ACCT-TYPE"]
        ).dt.days

        fltr = df_bureau["app_dd"] == df_bureau["DISBURSED-DT"]
        for f in [
            c + "_diff_to_next",
            c + "_diff_to_prev",
            c + "_diff_prev_to_next",
            c + "_range_in_days",
            c + "_diff_to_max",
            c + "_diff_to_min",
            c + "_diff_to_2nd_max",
            c + "_diff_to_2nd_min",
            c + "_diff_to_2nd_next",
            c + "_diff_to_2nd_prev",
            c + "_diff_to_next_acct_type",
            c + "_diff_to_prev_acct_type",
            f"diff_to_max_{c}_ACCT-TYPE",
            f"diff_to_min_{c}_ACCT-TYPE",
            c + "_diff_to_next_CONTRIBUTOR-TYPE",
            c + "_diff_to_prev_CONTRIBUTOR-TYPE",
            "total_accounts_closed",
            "avg_no_accounts_closed",
            "ACCT-TYPE",
            "CONTRIBUTOR-TYPE",
            c + "_days_since_start",
            c + "_days_from_end",
            c + "_ACCT_TYPE_max",
            c + "_ACCT_TYPE_min",
        ]:
            df[f] = df["ID"].map(df_bureau[fltr].set_index("ID")[f])

    df["ID_freq"] = df["ID"].map(df_bureau["ID"].value_counts())

    for c in tqdm(["DATE-REPORTED", "CLOSE-DT"]):
        df_bureau[f"{c} - DISBURSED-DT"] = (
            df_bureau[c] - df_bureau["DISBURSED-DT"]
        ).dt.days
        df_bureau[f"{c} - DISBURSED-DT_max"] = (
            df_bureau[c] - df_bureau["DISBURSED-DT_max"]
        ).dt.days
        df_bureau[f"{c} - DISBURSED-DT_min"] = (
            df_bureau[c] - df_bureau["DISBURSED-DT_min"]
        ).dt.days

    df_bureau["DATE-REPORTED - CLOSE-DT"] = (
        df_bureau["DATE-REPORTED"] - df_bureau["DISBURSED-DT"]
    ).dt.days
    for f in ["DATE-REPORTED - CLOSE-DT"]:
        df[f] = df["ID"].map(df_bureau[fltr].set_index("ID")[f])
    df_bureau["DISBURSED-AMT/HIGH CREDIT"] = df_bureau[
        "DISBURSED-AMT/HIGH CREDIT"
    ].apply(lambda x: clean(x))
    fltr = df_bureau["app_dd"] == df_bureau["DISBURSED-DT"]
    df_bureau["disbursed_amt_diff"] = df_bureau.groupby("ID")[
        "DISBURSED-AMT/HIGH CREDIT"
    ].shift(0) - df_bureau.groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].shift(-1)
    for i in tqdm(range(-7, 7)):
        if i == 0:
            continue
        df_bureau[f"disbursed_amt_next_{i}"] = df_bureau.groupby("ID")[
            "DISBURSED-AMT/HIGH CREDIT"
        ].shift(i)
        df_bureau[f"disbursed_amt_prev_{i}"] = df_bureau.groupby("ID")[
            "DISBURSED-AMT/HIGH CREDIT"
        ].shift(-i)
        df_bureau[f"disbursed_amt_next_ACCT-TYPE_{i}"] = df_bureau.groupby(
            ["ID", "ACCT-TYPE"]
        )["DISBURSED-AMT/HIGH CREDIT"].shift(i)
        df_bureau[f"disbursed_amt_prev_ACCT-TYPE_{i}"] = df_bureau.groupby(
            ["ID", "ACCT-TYPE"]
        )["DISBURSED-AMT/HIGH CREDIT"].shift(-i)

        df_bureau[f"disbursed_amt_diff_to_next_{i}"] = (
            df_bureau["DISBURSED-AMT/HIGH CREDIT"]
            - df_bureau[f"disbursed_amt_next_{i}"]
        )
        df_bureau[f"disbursed_amt_diff_to_prev_{i}"] = (
            df_bureau["DISBURSED-AMT/HIGH CREDIT"]
            - df_bureau[f"disbursed_amt_prev_{i}"]
        )
        for f in [
            f"disbursed_amt_next_{i}",
            f"disbursed_amt_prev_{i}",
            f"disbursed_amt_diff_to_next_{i}",
            f"disbursed_amt_diff_to_prev_{i}",
            "disbursed_amt_diff",
            f"disbursed_amt_next_ACCT-TYPE_{i}",
            f"disbursed_amt_prev_ACCT-TYPE_{i}",
        ]:

            df[f] = df["ID"].map(df_bureau[fltr].set_index("ID")[f])

    fltr = df_bureau["app_dd"] == df_bureau["DISBURSED-DT"]
    df["ACCT-TYPE"] = df["ID"].map(df_bureau[fltr].set_index("ID")["ACCT-TYPE"])
    df["ACCT-TYPE"] = pd.factorize(df["ACCT-TYPE"])[0]
    df["TENURE"] = df["ID"].map(df_bureau[fltr].set_index("ID")["TENURE"])
    fltr = df_bureau["DISBURSED-DT"] == df_bureau["app_dd"]
    df_bureau["app_close_dt"] = df_bureau["ID"].map(
        df_bureau[fltr].set_index("ID")["CLOSE-DT"]
    )

    fltr = df_bureau["DISBURSED-DT"] > df_bureau["app_dd"]
    df_bureau["tmp"] = df_bureau["DISBURSED-DT"] < df_bureau["app_close_dt"]
    df["n_loans_before_app_close_dt"] = df["ID"].map(
        df_bureau[fltr].groupby("ID")["tmp"].sum()
    )

    fltr = (df_bureau["DISBURSED-DT"] > df_bureau["app_dd"]) & (
        df_bureau["DISBURSED-DT"] < df_bureau["app_close_dt"]
    )
    df["n_loans_after_curr_loan_before_app_close_dt"] = (
        df["ID"]
        .map(df_bureau[fltr].groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].sum())
        .fillna(-1)
    )

    for c in [
        "CURRENT-BAL",
        "INSTALLMENT-AMT",
        "CREDIT-LIMIT/SANC AMT",
        "ACCT-TYPE",
        "DISBURSED-AMT/HIGH CREDIT",
        "SELF-INDICATOR",
        "MATCH-TYPE",
        "ba",
    ]:
        df[c + "_first"] = df["ID"].map(df_bureau.groupby("ID")[c].first())
        df[c + "_last"] = df["ID"].map(df_bureau.groupby("ID")[c].last())
        df[c + "_second_first"] = df["ID"].map(df_bureau.groupby("ID")[c].nth(1))
        df[c + "_second_last"] = df["ID"].map(df_bureau.groupby("ID")[c].nth(-2))
        df[c + "_third_first"] = df["ID"].map(df_bureau.groupby("ID")[c].nth(2))
        df[c + "_third_last"] = df["ID"].map(df_bureau.groupby("ID")[c].nth(-3))
        df[c + "_fourth_first"] = df["ID"].map(df_bureau.groupby("ID")[c].nth(3))
        df[c + "_fourth_last"] = df["ID"].map(df_bureau.groupby("ID")[c].nth(-4))
        df[c + "_fifth_first"] = df["ID"].map(df_bureau.groupby("ID")[c].nth(4))
        df[c + "_fifth_last"] = df["ID"].map(df_bureau.groupby("ID")[c].nth(-5))

    df_bureau["tmp"] = df_bureau["DISBURSED-DT"] > df_bureau["app_dd"]
    df["n_loans_after_curr_loan"] = df["ID"].map(df_bureau.groupby("ID")["tmp"].sum())

    df_bureau["tmp2"] = df_bureau["DISBURSED-AMT/HIGH CREDIT"] * df_bureau["tmp"]
    df["total_sum_disbursed_after_loan_applied"] = df["ID"].map(
        df_bureau.groupby("ID")["tmp2"].sum()
    )

    df_bureau["tmp2"] = df_bureau["DISBURSED-AMT/HIGH CREDIT"] * (1 - df_bureau["tmp"])
    df["total_sum_disbursed_before_loan_applied"] = df["ID"].map(
        df_bureau.groupby("ID")["tmp2"].sum()
    )

    del df_bureau["tmp"], df_bureau["tmp2"]
    gc.collect()

    df["mean_sum_disbursed_after_loan_applied"] = (
        df["total_sum_disbursed_after_loan_applied"] / df["n_loans_after_curr_loan"]
    )

    fltr = df_bureau["app_dd"] == df_bureau["DISBURSED-DT"]
    df_bureau["total_loan_days"] = (
        df_bureau["CLOSE-DT"] - df_bureau["DISBURSED-DT"]
    ).dt.days
    df["total_loan_days"] = df["ID"].map(
        df_bureau[fltr].set_index("ID")["total_loan_days"]
    )
    df["total_loan_days_max"] = df["ID"].map(
        df_bureau.groupby("ID")["total_loan_days"].max()
    )
    df["total_loan_days_min"] = df["ID"].map(
        df_bureau.groupby("ID")["total_loan_days"].min()
    )
    df["total_loan_days_first"] = df["ID"].map(
        df_bureau.groupby("ID")["total_loan_days"].first()
    )
    df["total_loan_days_last"] = df["ID"].map(
        df_bureau.groupby("ID")["total_loan_days"].last()
    )
    df["total_loan_days_range"] = df["total_loan_days_max"] - df["total_loan_days_min"]
    for c in ["DisbursalDate", "MaturityDAte", "AuthDate"]:
        print(c)
        df[c + "_days_since_start"] = (df[c] - df[c].min()).dt.days
        df[c + "_days_from_end"] = (df[c].max() - df[c]).dt.days

    df["DisbursalDate - MaturityDAte"] = (
        df["DisbursalDate"] - df["MaturityDAte"]
    ).dt.days
    for c in [
        "BranchID",
        "Area",
        "ManufacturerID",
        "SupplierID",
        "City",
        "State",
        "ZiPCODE",
    ]:
        df[c + "dd_min"] = df.groupby(c)["DisbursalDate_days_since_start"].transform(
            "max"
        )
        df[c + "_dd_max"] = df.groupby(c)["DisbursalDate_days_since_start"].transform(
            "min"
        )
    cat_cols = df.head().select_dtypes("object").columns.tolist()
    cat_cols = [
        c for c in cat_cols if c not in ["DisbursalDate", "MaturityDAte", "AuthDate"]
    ]
    print(cat_cols)
    for c in cat_cols:
        df[c] = pd.factorize(df[c])[0]

    tmp = df_bureau[["ID", "ACCT-TYPE"]]
    tmp = pd.get_dummies(tmp, columns=["ACCT-TYPE"])
    tmp = tmp.groupby("ID").sum().reset_index()
    tmp.columns = ["ID"] + [c + "_total_for_user" for c in tmp.columns[1:]]
    df = pd.merge(df, tmp, on="ID", how="left")

    tmp = df_bureau[df_bureau["DISBURSED-DT"] > df_bureau["app_dd"]][
        ["ID", "ACCT-TYPE"]
    ]
    tmp = pd.get_dummies(tmp, columns=["ACCT-TYPE"])
    tmp = tmp.groupby("ID").sum().reset_index()
    tmp.columns = ["ID"] + [c + "_total_for_user_2" for c in tmp.columns[1:]]
    df = pd.merge(df, tmp, on="ID", how="left")

    tmp = df_bureau[df_bureau["DISBURSED-DT"] > df_bureau["app_dd"]]
    for c in tqdm(tmp["CONTRIBUTOR-TYPE"].unique()):
        df[f"CONTRIBUTOR-TYPE_{c}_max_amount_after_loan"] = np.nan

    tmp = df_bureau[df_bureau["DISBURSED-DT"] > df_bureau["app_dd"]][
        ["ID", "ACCT-TYPE"]
    ]
    for c in tqdm(tmp["ACCT-TYPE"].unique()):
        df[f"kharacha_{c}"] = df["ID"].map(
            df_bureau[df_bureau["ACCT-TYPE"] == c]
            .groupby("ID")["DISBURSED-AMT/HIGH CREDIT"]
            .sum()
        )
        df[f"kharacha_{c}_max"] = np.nan
    tmp = df_bureau[df_bureau["DISBURSED-DT"] > df_bureau["app_dd"]]
    for c in tqdm(tmp["ACCT-TYPE"].unique()):
        df[f"kharacha_2_{c}"] = df["ID"].map(
            tmp[tmp["ACCT-TYPE"] == c].groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].sum()
        )
        df[f"kharacha_2_{c}_max"] = df["ID"].map(
            tmp[tmp["ACCT-TYPE"] == c].groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].max()
        )
        df[f"kharacha_2_{c}_min"] = df["ID"].map(
            tmp[tmp["ACCT-TYPE"] == c].groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].mean()
        )
        df[f"kharacha_2_{c}_days_min"] = np.nan
        df[f"kharcha_1_2_{c}_diff"] = df[f"kharacha_{c}"] - df[f"kharacha_2_{c}"]

    tmp = df_bureau[df_bureau["DISBURSED-DT"] <= df_bureau["app_dd"]]
    for c in tqdm(tmp["ACCT-TYPE"].unique()):
        df[f"kharacha_3_{c}"] = df["ID"].map(
            tmp[tmp["ACCT-TYPE"] == c].groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].max()
        )

    df["days_diff_to_first_Disbursal_Date"] = (
        df["DisbursalDate"]
        - df["ID"].map(df_bureau.groupby("ID")["DISBURSED-DT"].min())
    ).dt.days
    df["days_diff_to_last_Disbursal_Date"] = (
        df["DisbursalDate"]
        - df["ID"].map(df_bureau.groupby("ID")["DISBURSED-DT"].max())
    ).dt.days

    fltr = df_bureau["DISBURSED-DT"] > df_bureau["app_dd"]
    df["CLOSING-DT-diff_to_next_loan"] = df["ID"].map(
        (
            df_bureau[fltr].groupby("ID")["CLOSE-DT"].first()
            - df_bureau[fltr].groupby("ID")["CLOSE-DT"].nth(1)
        ).dt.days
    )
    df["CLOSING-DT-diff_to_next_loan"]

    fltr = df_bureau["DISBURSED-DT"] >= df_bureau["app_dd"]
    tmp = df_bureau[fltr]
    tmp["cola"] = tmp.groupby(["ID", "ACCT-TYPE"])[
        "DISBURSED-AMT/HIGH CREDIT"
    ].transform("first") - df_bureau[fltr].groupby(["ID", "ACCT-TYPE"])[
        "DISBURSED-AMT/HIGH CREDIT"
    ].transform(
        "nth", 1
    )
    df["cola"] = tmp.groupby("ID")["cola"].first()

    fltr = df_bureau["DISBURSED-DT"] >= df_bureau["app_dd"]
    tmp = df_bureau[fltr]
    tmp["cola"] = tmp.groupby(["ID", "ACCT-TYPE"])[
        "DISBURSED-AMT/HIGH CREDIT"
    ].transform("first") - df_bureau[fltr].groupby(["ID", "ACCT-TYPE"])[
        "DISBURSED-AMT/HIGH CREDIT"
    ].transform(
        "nth", 2
    )
    df["cola2"] = tmp.groupby("ID")["cola"].first()

    tmp["cola"] = tmp.groupby(["ID", "ACCT-TYPE"])[
        "DISBURSED-AMT/HIGH CREDIT"
    ].transform("first") - df_bureau[fltr].groupby(["ID", "ACCT-TYPE"])[
        "DISBURSED-AMT/HIGH CREDIT"
    ].transform(
        "nth", 3
    )
    df["cola3"] = tmp.groupby("ID")["cola"].first()

    tmp["cola"] = tmp.groupby(["ID", "ACCT-TYPE"])[
        "DISBURSED-AMT/HIGH CREDIT"
    ].transform("first") - df_bureau[~fltr].groupby(["ID", "ACCT-TYPE"])[
        "DISBURSED-AMT/HIGH CREDIT"
    ].transform(
        "nth", -1
    )
    df["cola4"] = tmp.groupby("ID")["cola"].first()
    df_bureau["loan_closed"] = (df_bureau["LAST-PAYMENT-DATE"].isnull()) * 1
    fltr = df_bureau["DISBURSED-DT"] <= df_bureau["app_dd"]
    df["loans_closed_before_disbursal_mean"] = df["ID"].map(
        df_bureau[fltr].groupby("ID")["loan_closed"].mean()
    )

    fltr = df_bureau["DISBURSED-DT"] > df_bureau["app_dd"]
    df["loans_closed_after_disbursal_mean"] = df["ID"].map(
        df_bureau[fltr].groupby("ID")["loan_closed"].mean()
    )
    df["TENURE"] = df["TENURE"].fillna(-1)
    df["mean_TENURE"] = df["ID"].map(df_bureau.groupby("ID")["TENURE"].mean())
    df["max_TENURE"] = df["ID"].map(df_bureau.groupby("ID")["TENURE"].max())
    df["min_TENURE"] = df["ID"].map(df_bureau.groupby("ID")["TENURE"].min())
    df["DisbursalDate_freq"] = df["DisbursalDate"].map(
        df["DisbursalDate"].value_counts()
    )
    df["mean_self_indicator"] = df["ID"].map(
        df_bureau.groupby("ID")["SELF-INDICATOR"].mean()
    )
    df["sum_self_indicator"] = df["ID"].map(
        df_bureau.groupby("ID")["SELF-INDICATOR"].sum()
    )
    fltr = df_bureau["DISBURSED-DT"] > df_bureau["app_dd"]
    tmp = df_bureau[fltr]
    df["mean_self_indicator_after_loan"] = df["ID"].map(
        tmp.groupby("ID")["SELF-INDICATOR"].mean()
    )
    df["sum_self_indicator_after_loan"] = df["ID"].map(
        tmp.groupby("ID")["SELF-INDICATOR"].sum()
    )

    for c in df_bureau["OWNERSHIP-IND"].unique():
        df_bureau["tmp"] = (df_bureau["OWNERSHIP-IND"] == c) * 1
        df[f"mean_{c}_all"] = df["ID"].map(df_bureau.groupby("ID")["tmp"].mean())
        df[f"sum_{c}_all"] = df["ID"].map(df_bureau.groupby("ID")["tmp"].sum())

    for c in df_bureau["OWNERSHIP-IND"].unique():
        df_bureau["tmp"] = (df_bureau["OWNERSHIP-IND"] == c) * 1
        df[f"mean_{c}_all"] = df["ID"].map(df_bureau.groupby("ID")["tmp"].mean())
        df[f"sum_{c}_all"] = df["ID"].map(df_bureau.groupby("ID")["tmp"].sum())

    for c in df_bureau["ACCOUNT-STATUS"].unique():
        df_bureau["tmp"] = (df_bureau["ACCOUNT-STATUS"] == c) * 1
        df[f"mean_{c}_all"] = df["ID"].map(df_bureau.groupby("ID")["tmp"].mean())
        df[f"sum_{c}_all"] = df["ID"].map(df_bureau.groupby("ID")["tmp"].sum())

    for a in tqdm(df_bureau["ACCT-TYPE"].unique()):
        for c in df_bureau["CONTRIBUTOR-TYPE"].unique():
            fltr = (df_bureau["ACCT-TYPE"] == a) & (df_bureau["CONTRIBUTOR-TYPE"] == c)
            if (fltr.sum() / fltr.shape[0]) > 0.2:
                df[f"sum_DISBURSED-AMT/HIGH CREDIT_{a}_{c}_all_accounts"] = df[
                    "ID"
                ].map(df_bureau[fltr].groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].sum())
    fltr = df_bureau["DISBURSED-DT"] > df_bureau["app_dd"]
    tmp = df_bureau[fltr]
    for c in df_bureau["SELF-INDICATOR"].unique():
        df_bureau["tmp"] = (df_bureau["SELF-INDICATOR"] == c) * 1
        df[f"mean_{c}_all"] = df["ID"].map(tmp.groupby("ID")["tmp"].mean())
        df[f"sum_{c}_all"] = df["ID"].map(tmp.groupby("ID")["tmp"].sum())

    fltr_a = df_bureau["DISBURSED-DT"] > df_bureau["app_dd"]
    tmp = df_bureau[fltr]
    for c in df_bureau["SELF-INDICATOR"].unique():
        fltr_b = df_bureau["SELF-INDICATOR"] == c
        fltr_net = fltr_a & fltr_b
        df[f"mean_{c}_high_DISBURSED-AMT/HIGH CREDIT"] = df["ID"].map(
            df_bureau[fltr_net].groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].mean()
        )
        df[f"sum_{c}_DISBURSED-AMT/HIGH CREDIT"] = df["ID"].map(
            df_bureau[fltr_net].groupby("ID")["DISBURSED-AMT/HIGH CREDIT"].sum()
        )

    df = df.fillna(-9999)

    for a in tqdm(df_bureau["ACCT-TYPE"].unique()):
        fltr = df_bureau["ACCT-TYPE"] == a
        df[f"ACCT-TYPE_{a}_DISBURSED_DT_max"] = df["ID"].map(
            df_bureau[fltr].groupby("ID")["DISBURSED-DT_days_since_start"].max()
        )
        df[f"ACCT-TYPE_{a}_DISBURSED_DT_min"] = df["ID"].map(
            df_bureau[fltr].groupby("ID")["DISBURSED-DT_days_since_start"].min()
        )
        df[f"ACCT-TYPE_{a}_DISBURSED_DT_range"] = (
            df[f"ACCT-TYPE_{a}_DISBURSED_DT_max"]
            - df[f"ACCT-TYPE_{a}_DISBURSED_DT_min"]
        )

        df["tmp"] = df["ID"].map(df_bureau[fltr].groupby("ID")["DISBURSED-DT"].max())
        df[f"ACCT-TYPE_{a}_DISBURSED_DT_max - DisbursalDate"] = (
            df["tmp"] - df["DisbursalDate"]
        ).dt.days

    del df["tmp"]
    gc.collect()

    for a in tqdm(df_bureau["SELF-INDICATOR"].unique()):
        fltr = df_bureau["SELF-INDICATOR"] == a
        df[f"SELF-INDICATOR_{a}_DISBURSED_DT_max"] = df["ID"].map(
            df_bureau[fltr].groupby("ID")["DISBURSED-DT_days_since_start"].max()
        )
        df[f"SELF-INDICATOR_{a}_DISBURSED_DT_2nd_last"] = df["ID"].map(
            df_bureau[fltr].groupby("ID")["DISBURSED-DT_days_since_start"].nth(-2)
        )
        f = f"SELF-INDICATOR_{a}_DISBURSED_DT_max - SELF-INDICATOR_{a}_DISBURSED_DT_2nd_last"
        df[f] = (
            df[f"SELF-INDICATOR_{a}_DISBURSED_DT_max"]
            - df[f"SELF-INDICATOR_{a}_DISBURSED_DT_2nd_last"]
        )
        del df[f"SELF-INDICATOR_{a}_DISBURSED_DT_2nd_last"]
        df[f"SELF-INDICATOR_{a}_DISBURSED_DT_min"] = df["ID"].map(
            df_bureau[fltr].groupby("ID")["DISBURSED-DT_days_since_start"].min()
        )
        df[f"SELF-INDICATOR_{a}_DISBURSED_DT_range"] = (
            df[f"SELF-INDICATOR_{a}_DISBURSED_DT_max"]
            - df[f"SELF-INDICATOR_{a}_DISBURSED_DT_min"]
        )

        df["tmp"] = df["ID"].map(df_bureau[fltr].groupby("ID")["DISBURSED-DT"].max())
        df[f"SELF-INDICATOR_{a}_DISBURSED_DT_max - DisbursalDate"] = (
            df["tmp"] - df["DisbursalDate"]
        ).dt.days

    del df["tmp"]
    df = df.fillna(-1)
    threshold = 0.8
    null_percent = (df.isnull().sum() / df.shape[0]).sort_values(ascending=False)
    drop_fts_2 = null_percent[null_percent > threshold].index.tolist()
    fts = [
        c
        for c in df.columns
        if c not in [ID_COL, TARGET_COL] + ignore_fts + drop_fts + drop_fts_2
    ]
    print(len(fts))
    df[["ID"] + fts].to_pickle("nikhil_feats_1.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", help="pass the path where the data is stored", type=str
    )
    args = parser.parse_args()
    path = args.data_path
    feature_creation1(path)
    feature_creation2(path)
    feature_creation3(path)
