import numpy as np


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
        if include_sum:
            _dict["sum_" + name] = val.sum()
    else:
        _dict["mean_" + name] = 0
        _dict["min_" + name] = 0
        _dict["max_" + name] = 0
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
