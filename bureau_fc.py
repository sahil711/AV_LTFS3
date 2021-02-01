import numpy as np
import pandas as pd
import datetime as dt

def get_bureau_feats_2(grp):
    _id,account_df = grp
    account_df['disb_ratio'] = account_df['correctedDISBURSED-AMT/HIGH CREDIT']/account_df['DisbursalAmount']
    _dict = {}
    _dict["individual_accounts"] = (account_df["OWNERSHIP-IND"] == "Individual").sum()
    _dict["joint_accounts"] = (account_df["OWNERSHIP-IND"] == "Joint").sum()
    _dict["guarantor_accounts"] = (account_df["OWNERSHIP-IND"] == "Guarantor").sum()
    _dict["curr_bal_grtr_0"] = (account_df["correctedCURRENT-BAL"] > 0).sum()
    _dict["num_accounts"] = len(account_df)
    _dict.update(
        get_stats(
            account_df["correctedDISBURSED-AMT/HIGH CREDIT"],
            "correctedDISBURSED-AMT/HIGH CREDIT",
        )
    )
    _dict.update(get_stats(account_df["correctedCURRENT-BAL"], "correctedCURRENT-BAL"))
    _dict.update(get_stats(account_df["correctedOVERDUE-AMT"], "correctedOVERDUE-AMT"))
    _dict["num_closed_accounts"] = (account_df["ACCOUNT-STATUS"] == "Closed").sum()
    _dict["num_open_accounts"] = (account_df["ACCOUNT-STATUS"] == "Active").sum()
    _dict["num_delinq_accounts"] = (account_df["ACCOUNT-STATUS"] == "Delinquent").sum()
    _dict["total_written_off_amount"] = account_df["WRITE-OFF-AMT"].sum()
    perc_paid_off = (
        1
        - account_df["correctedCURRENT-BAL"]
        / account_df["correctedDISBURSED-AMT/HIGH CREDIT"]
    )
    _dict.update(get_stats(perc_paid_off, "percent_paid_off", include_sum=False))
    _dict.update(get_stats(account_df['disb_ratio'], "disb_ratio"))    
    
    _dict["overall_percent_paid_off"] = (
        1
        - account_df["correctedCURRENT-BAL"].sum()
        / account_df["correctedDISBURSED-AMT/HIGH CREDIT"].sum()
    )
    tenors = ((account_df["DATE-REPORTED"] - account_df["DISBURSED-DT"]).dt.days).clip(
        upper=7300
    )
    _dict["median_tenor"] = tenors.median()
    _dict["max_tenor"] = tenors.max()
    _dict["min_tenor"] = tenors.min()
    
    #### ltfs filter
    is_ltfs = account_df['SELF-INDICATOR']
    temp = account_df[is_ltfs]
    _dict["num_accounts_{}".format('is_ltfs')] = len(temp)
    _dict["total_sanctioned_amount_{}".format('is_ltfs')] = temp[
        "correctedDISBURSED-AMT/HIGH CREDIT"
    ].sum()
    _dict['max_tenor_is_ltfs'] = tenors[is_ltfs].max()
    _dict['min_tenor_is_ltfs'] = tenors[is_ltfs].min()    
    _dict["total_curr_bal_{}".format('is_ltfs')] = temp["correctedCURRENT-BAL"].sum()
    _dict["overall_percentage_paid_off_{}".format('is_ltfs')] = (
        1
        - temp["correctedCURRENT-BAL"].sum()
        / temp["correctedDISBURSED-AMT/HIGH CREDIT"].sum()
    )
    

#     loan_ids = [
#         "Tractor Loan",
#         "Gold Loan",
#         "Overdraft"
#     ]
#     for loan in loan_ids:
#         _filter = account_df["ACCT-TYPE"] == loan
#         temp = account_df[_filter]
#         _dict["num_accounts_{}".format(loan)] = len(temp)
#         _dict["total_sanctioned_amount_{}".format(loan)] = temp[
#             "correctedDISBURSED-AMT/HIGH CREDIT"
#         ].sum()
#         _dict["total_curr_bal_{}".format(loan)] = temp["correctedCURRENT-BAL"].sum()
#         _dict["overall_percentage_paid_off_{}".format(loan)] = (
#             1
#             - temp["correctedCURRENT-BAL"].sum()
#             / temp["correctedDISBURSED-AMT/HIGH CREDIT"].sum()
#         )

    days_diff = (account_df["DISBURSED-DT"] - account_df["DisbursalDate"]).dt.days
    _dict.update(
        get_stats(
            days_diff[days_diff > 0],
            "day_start_day_diff_app_vs_other",
            include_sum=False,
        )
    )
    
    days_diff = (account_df["MaturityDAte"]-account_df["DISBURSED-DT"]).dt.days
    _dict.update(
        get_stats(
            days_diff[days_diff > 0],
            "time_delta_anchor_close_vs_start",
            include_sum=False,
        )
    )
    _dict['num_ltfs_loans'] = account_df['SELF-INDICATOR'].sum()
    
#     _dict.update(
#         get_stats(
#             (account_df["DISBURSED-DT"] - account_df["DISBURSED-DT"].shift(1)).dt.days,
#             "days_bw_loans",
#         )
#     )
    _dict["ID"] = _id
    return _dict

def get_loan_history(data):
    dates = data["date_var"]
    bal = data["cur_bal_var"]
    dpds = data["dpd_strin_var"]
    _min = np.min([len(dates), len(bal), len(dpds)])
    dates = dates[-_min:]
    bal = bal[-_min:]
    dpds = dpds[-_min:]
    temp = np.vstack([dates, bal, dpds]).T
    return temp


def get_loan_profile_feats(grp):
    _id, account_df = grp
    temp = pd.DataFrame(
        np.concatenate([get_loan_history(x) for i, x in account_df.iterrows()], axis=0),
        columns=["dates", "cur_bal", "dpd"],
    )
    temp.cur_bal = temp.cur_bal.fillna("0").replace("", "0").astype("float")
    temp["dpd"] = temp["dpd"].apply(
        lambda x: 0 if (x == "XXX" or x == "DDD") else int(x)
    )
    temp["dates"] = pd.to_datetime(temp["dates"].str[:6], format="%Y%m")
    temp = (
        temp.groupby("dates")[["cur_bal", "dpd"]]
        .sum()
        .reset_index()
        .merge(temp.groupby("dates").size().reset_index(name="count"), on="dates")
    )
    temp = temp[
        (temp["dates"] >= account_df["DisbursalDate"].iloc[0])
        & (
            temp["dates"]
            <= (account_df["MaturityDAte"].iloc[0] + dt.timedelta(2 * 365))
        )
    ]  # considering dates upto 2 years after the loan period
    temp["curr_bal_dec"] = (
        (temp["cur_bal"] - temp["cur_bal"].shift(1)).fillna(0) < 0
    ).astype("int")
    temp["dpd_dec"] = ((temp["dpd"] - temp["dpd"].shift(1)).fillna(0) < 0).astype("int")
    times = [
        account_df["DisbursalDate"].iloc[0] + dt.timedelta(360 * i)
        for i in [0, 1, 2, 3, 4, 10]
    ]
    _dict = {}
    for i in range(len(times) - 1):
        a = temp[(temp.dates >= times[i]) & (temp.dates <= times[i + 1])]
        _dict.update(get_stats(a.dpd, "dpd_feat_{}".format(i), include_sum=False))
        _dict.update(
            get_stats(a.cur_bal, "cur_bal_feat_{}".format(i), include_sum=False)
        )
        _dict["cur_bal_inc_feat_{}".format(i)] = sum(a["curr_bal_dec"] == 0)
        _dict["cur_bal_dec_feat_{}".format(i)] = sum(a["curr_bal_dec"] == 1)
        _dict["dpd_inc_feat_{}".format(i)] = sum(a["dpd_dec"] == 0)
        _dict["dpd_dec_feat_{}".format(i)] = sum(a["dpd_dec"] == 1)
    _dict["ID"] = _id
    return _dict


# def get_bureau_feats(account_df):
#     _dict = {}
#     _dict["individual_accounts"] = (account_df["OWNERSHIP-IND"] == "Individual").sum()
#     _dict["joint_accounts"] = (account_df["OWNERSHIP-IND"] == "Joint").sum()
#     _dict["guarantor_accounts"] = (account_df["OWNERSHIP-IND"] == "Guarantor").sum()
#     _dict["curr_bal_grtr_0"] = (account_df["correctedCURRENT-BAL"] > 0).sum()
#     _dict["num_accounts"] = len(account_df)
#     _dict.update(
#         get_stats(
#             account_df["correctedDISBURSED-AMT/HIGH CREDIT"],
#             "correctedDISBURSED-AMT/HIGH CREDIT",
#         )
#     )
#     _dict.update(get_stats(account_df["correctedCURRENT-BAL"], "correctedCURRENT-BAL"))
#     _dict.update(get_stats(account_df["correctedOVERDUE-AMT"], "correctedOVERDUE-AMT"))
#     _dict["num_closed_accounts"] = (account_df["ACCOUNT-STATUS"] == "Closed").sum()
#     _dict["num_open_accounts"] = (account_df["ACCOUNT-STATUS"] == "Active").sum()
#     _dict["num_delinq_accounts"] = (account_df["ACCOUNT-STATUS"] == "Delinquent").sum()
#     _dict["total_written_off_amount"] = account_df["WRITE-OFF-AMT"].sum()
#     perc_paid_off = (
#         1
#         - account_df["correctedCURRENT-BAL"]
#         / account_df["correctedDISBURSED-AMT/HIGH CREDIT"]
#     )
#     account_df['perc_paid_off'] = perc_paid_off
#     _dict.update(get_stats(perc_paid_off, "percent_paid_off", include_sum=False))
#     _dict["overall_percent_paid_off"] = (
#         1
#         - account_df["correctedCURRENT-BAL"].sum()
#         / account_df["correctedDISBURSED-AMT/HIGH CREDIT"].sum()
#     )
#     account_df['tenor'] = ((account_df["DATE-REPORTED"] - account_df["DISBURSED-DT"]).dt.days).clip(
#         upper=7300
#     )

#     account_df['emi_proxy']=account_df['correctedDISBURSED-AMT/HIGH CREDIT']/account_df['tenor']
#     _dict.update(
#         get_stats(
#             account_df["emi_proxy"],
#             "emi_proxy",
#         )
#     )
        
#     account_df['payment_measure_1']=account_df['perc_paid_off']/account_df['tenor']
#     _dict.update(
#         get_stats(
#             account_df["payment_measure_1"],
#             "payment_measure_1",
#         )
#     )
        
#     _dict["median_tenor"] = account_df['tenor'].median()
#     _dict["max_tenor"] = account_df['tenor'].max()
#     _dict["min_tenor"] = account_df['tenor'].min()
    
    

#     _dict.update(
#         create_payment_history_variables(
#             [x for m in account_df["dpd_strin_var"].tolist() for x in m]
#         )
#     )
#     _dict.update(
#         create_payment_history_variables(account_df.iloc[0], "_application_loan")
#     )
#     _dict.update(
#         get_stats(
#             get_int_dpd_str(
#                 [x for m in account_df["dpd_strin_var"].tolist() for x in m]
#             ),
#             "dpd_str",
#             include_sum=False,
#         )
#     )
#     _dict.update(
#         get_stats(
#             get_int_dpd_str(account_df["dpd_strin_var"].iloc[0]),
#             "dpd_str_application_loan",
#             include_sum=False,
#         )
#     )

#     gap = (account_df["DISBURSED-DT"] - account_df["DisbursalDate"]).dt.days
#     period = [0, 1, 2, 3, 4, 10]
#     for i in range(len(period) - 1):
#         up = 365 * period[i]
#         down = 365 * period[i + 1]
#         col_name = "_between_{}_and_{}_days".format(up, down)
#         _filter = (gap >= up) & (gap < down)
#         temp = account_df[_filter]
#         _dict["num_accounts_{}".format(col_name)] = len(temp)
#         _dict["median_tenor_{}".format(col_name)] = temp['tenor'].median()
#         _dict["max_tenor".format(col_name)] = temp['tenor'].max()
#         _dict["min_tenor".format(col_name)] = temp['tenor'].min()
        
#         _dict.update(
#             get_stats(
#                 temp["emi_proxy"],
#                 "emi_proxy_{}".format(col_name),
#             )
#         )
        
#         _dict.update(
#             get_stats(
#                 temp["perc_paid_off"],
#                 "perc_paid_off_{}".format(col_name),
#             )
#         )
        
#         _dict.update(
#             get_stats(
#                 temp["payment_measure_1"],
#                 "payment_measure_1_{}".format(col_name),
#             )
#         )
        

#         _dict.update(
#             get_stats(
#                 temp["correctedDISBURSED-AMT/HIGH CREDIT"],
#                 "correctedDISBURSED-AMT/HIGH CREDIT_{}".format(col_name),
#             )
#         )
        
#         _dict.update(
#             get_stats(
#                 temp["correctedCURRENT-BAL"],
#                 "correctedCURRENT-BAL_{}".format(col_name),
#             )
#         )
        
        

#     loan_ids = [
#         "Tractor Loan",
#         "Gold Loan",
#         "Business Loan Priority Sector  Agriculture",
#         "Kisan Credit Card",
#         "Auto Loan (Personal)",
#         "Personal Loan",
#         "Other",
#         "Overdraft",
#     ]
#     for loan in loan_ids:
#         _filter = account_df["ACCT-TYPE"] == loan
#         _dict["num_accounts_{}".format(col_name)] = len(temp)
#         _dict["median_tenor_{}".format(col_name)] = temp['tenor'].median()
#         _dict["max_tenor".format(col_name)] = temp['tenor'].max()
#         _dict["min_tenor".format(col_name)] = temp['tenor'].min()
        
#         _dict.update(
#             get_stats(
#                 temp["emi_proxy"],
#                 "emi_proxy_{}".format(col_name),
#             )
#         )
        
#         _dict.update(
#             get_stats(
#                 temp["perc_paid_off"],
#                 "perc_paid_off_{}".format(col_name),
#             )
#         )
        
#         _dict.update(
#             get_stats(
#                 temp["payment_measure_1"],
#                 "payment_measure_1_{}".format(col_name),
#             )
#         )
        

#         _dict.update(
#             get_stats(
#                 temp["correctedDISBURSED-AMT/HIGH CREDIT"],
#                 "correctedDISBURSED-AMT/HIGH CREDIT_{}".format(col_name),
#             )
#         )
        
#         _dict.update(
#             get_stats(
#                 temp["correctedCURRENT-BAL"],
#                 "correctedCURRENT-BAL_{}".format(col_name),
#             )
#         )
        

#     days_diff = (account_df["DISBURSED-DT"] - account_df["DisbursalDate"]).dt.days
#     _dict.update(
#         get_stats(
#             days_diff[days_diff > 0],
#             "day_start_day_diff_app_vs_other",
#             include_sum=False,
#         )
#     )
#     _dict.update(
#         get_stats(
#             (account_df["DISBURSED-DT"] - account_df["DISBURSED-DT"].shift(1)).dt.days,
#             "days_bw_loans",
#         )
#     )
#     _dict["ID"] = account_df["ID"].iloc[0]
#     return _dict

def get_bureau_feats(account_df):
    _dict = {}
    _dict["individual_accounts"] = (account_df["OWNERSHIP-IND"] == "Individual").sum()
    _dict["joint_accounts"] = (account_df["OWNERSHIP-IND"] == "Joint").sum()
    _dict["guarantor_accounts"] = (account_df["OWNERSHIP-IND"] == "Guarantor").sum()
    _dict["curr_bal_grtr_0"] = (account_df["correctedCURRENT-BAL"] > 0).sum()
    _dict["num_accounts"] = len(account_df)
    _dict.update(
        get_stats(
            account_df["correctedDISBURSED-AMT/HIGH CREDIT"],
            "correctedDISBURSED-AMT/HIGH CREDIT",
        )
    )
    _dict.update(get_stats(account_df["correctedCURRENT-BAL"], "correctedCURRENT-BAL"))
    _dict.update(get_stats(account_df["correctedOVERDUE-AMT"], "correctedOVERDUE-AMT"))
    _dict["num_closed_accounts"] = (account_df["ACCOUNT-STATUS"] == "Closed").sum()
    _dict["num_open_accounts"] = (account_df["ACCOUNT-STATUS"] == "Active").sum()
    _dict["num_delinq_accounts"] = (account_df["ACCOUNT-STATUS"] == "Delinquent").sum()
    _dict["total_written_off_amount"] = account_df["WRITE-OFF-AMT"].sum()
    perc_paid_off = (
        1
        - account_df["correctedCURRENT-BAL"]
        / account_df["correctedDISBURSED-AMT/HIGH CREDIT"]
    )
    _dict.update(get_stats(perc_paid_off, "percent_paid_off", include_sum=False))
    _dict["overall_percent_paid_off"] = (
        1
        - account_df["correctedCURRENT-BAL"].sum()
        / account_df["correctedDISBURSED-AMT/HIGH CREDIT"].sum()
    )
    tenors = ((account_df["DATE-REPORTED"] - account_df["DISBURSED-DT"]).dt.days).clip(
        upper=7300
    )
    _dict["median_tenor"] = tenors.median()
    _dict["max_tenor"] = tenors.max()
    _dict["min_tenor"] = tenors.min()

    _dict.update(
        create_payment_history_variables(
            [x for m in account_df["dpd_strin_var"].tolist() for x in m]
        )
    )
    _dict.update(
        create_payment_history_variables(account_df.iloc[0], "_application_loan")
    )
    _dict.update(
        get_stats(
            get_int_dpd_str(
                [x for m in account_df["dpd_strin_var"].tolist() for x in m]
            ),
            "dpd_str",
            include_sum=False,
        )
    )
    _dict.update(
        get_stats(
            get_int_dpd_str(account_df["dpd_strin_var"].iloc[0]),
            "dpd_str_application_loan",
            include_sum=False,
        )
    )

    gap = (account_df["DISBURSED-DT"] - account_df["DisbursalDate"]).dt.days
    period = [0, 1, 2, 3, 4, 10]
    for i in range(len(period) - 1):
        up = 365 * period[i]
        down = 365 * period[i + 1]
        col_name = "_between_{}_and_{}_days".format(up, down)
        _filter = (gap >= up) & (gap < down)
        temp = account_df[_filter]
        _dict["num_accounts_{}".format(col_name)] = len(temp)
        _dict["total_sanctioned_amount_{}".format(col_name)] = temp[
            "correctedDISBURSED-AMT/HIGH CREDIT"
        ].sum()
        _dict["total_curr_bal_{}".format(col_name)] = temp["correctedCURRENT-BAL"].sum()
        _dict["overall_percentage_paid_off_{}".format(col_name)] = (
            1
            - temp["correctedCURRENT-BAL"].sum()
            / temp["correctedDISBURSED-AMT/HIGH CREDIT"].sum()
        )

    loan_ids = [
        "Tractor Loan",
        "Gold Loan",
        "Business Loan Priority Sector  Agriculture",
        "Kisan Credit Card",
        "Auto Loan (Personal)",
        "Personal Loan",
        "Other",
        "Overdraft",
    ]
    for loan in loan_ids:
        _filter = account_df["ACCT-TYPE"] == loan
        temp = account_df[_filter]
        _dict["num_accounts_{}".format(loan)] = len(temp)
        _dict["total_sanctioned_amount_{}".format(loan)] = temp[
            "correctedDISBURSED-AMT/HIGH CREDIT"
        ].sum()
        _dict["total_curr_bal_{}".format(loan)] = temp["correctedCURRENT-BAL"].sum()
        _dict["overall_percentage_paid_off_{}".format(loan)] = (
            1
            - temp["correctedCURRENT-BAL"].sum()
            / temp["correctedDISBURSED-AMT/HIGH CREDIT"].sum()
        )

    days_diff = (account_df["DISBURSED-DT"] - account_df["DisbursalDate"]).dt.days
    _dict.update(
        get_stats(
            days_diff[days_diff > 0],
            "day_start_day_diff_app_vs_other",
            include_sum=False,
        )
    )
    _dict.update(
        get_stats(
            (account_df["DISBURSED-DT"] - account_df["DISBURSED-DT"].shift(1)).dt.days,
            "days_bw_loans",
        )
    )
    _dict["ID"] = account_df["ID"].iloc[0]
    return _dict


def get_stats(val, name, include_sum=True):
    _dict = {}
    if len(val) > 0:
        _dict["mean_" + name] = val.mean()
        _dict["min_" + name] = val.min()
        _dict["max_" + name] = val.max()
        _dict["std_" + name] = val.std()        
        if include_sum:
            _dict["sum_" + name] = val.sum()
    else:
        _dict["mean_" + name] = 0
        _dict["min_" + name] = 0
        _dict["max_" + name] = 0
        _dict["std_" + name] = 0        
        if include_sum:
            _dict["sum_" + name] = 0

    return _dict


def create_payment_history_variables(payment_hist, prefix=None):
    _dict = {
        "std_count": 0,
        "ddd_count": 0,
        "xxx_count": 0,
        "late_count": 0,
        "_30_count": 0,
        "_60_count": 0,
        "_90_count": 0,
        "_180_count": 0,
        "total_count": 0,
    }
    for p_hist in payment_hist:
        _dict["total_count"] += 1
        try:
            p_hist = int(p_hist)
            if p_hist == 0:
                _dict["std_count"] += 1
            if p_hist >= 1:
                _dict["late_count"] += 1
            if p_hist >= 30:
                _dict["_30_count"] += 1
            if p_hist >= 60:
                _dict["_60_count"] += 1
            if p_hist >= 90:
                _dict["_90_count"] += 1
            if p_hist >= 180:
                _dict["_180"] += 1
        except Exception:
            if p_hist == "STD":
                _dict["std_count"] += 1
            if p_hist == "DDD":
                _dict["ddd_count"] += 1
            if p_hist == "XXX":
                _dict["xxx_count"] += 1
    if prefix is not None:
        old_keys = list(_dict.keys())
        for k in old_keys:
            _dict[k + prefix] = _dict.pop(k)
    return _dict


def get_int_dpd_str(dpd_str):
    int_dpd = []
    for dpd in dpd_str:
        try:
            dpd = int(dpd)
        except Exception:
            if (dpd == "XXX") | (dpd == "DDD"):
                dpd = 0
        int_dpd.append(dpd)
    #     int_dpd=np.array(int_dpd)
    #     np.max(int_dpd)
    #     np.mean(int_dpd)
    #     sum(np.where(int_dpd>0,1,0))
    #     sum(np.where(int_dpd>30,1,0))
    #     sum(int_dpd>60),sum(int_dpd>90)
    #     np.mean(int_dpd>0)
    #     np.mean(int_dpd>30)
    #     np.mean(int_dpd>60)
    #     np.mean(int_dpd>90)
    #     int(sum(int_dpd>0)>0)
    #     int(sum(int_dpd>30)>0)
    #     int(sum(int_dpd>60)>0)
    #     int(sum(int_dpd>90)>0)
    return np.array(int_dpd)
