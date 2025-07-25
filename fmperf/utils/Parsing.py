import pandas as pd

pd.set_option("future.no_silent_downcasting", True)


def parse_results(results, print_df=False, print_csv=False):
    df = pd.DataFrame.from_dict(results, orient="columns")
    df = df.set_index("timestamp").sort_index()

    df_out = pd.DataFrame(index=df["exp_num_users"].unique())

    # count total number of requests
    for u in df_out.index:
        df_out.at[u, "n_requests"] = df[df["exp_num_users"] == u].value_counts(subset=["worker_idx", "request_idx"]).shape[0]
    df_out["n_requests"] = df_out["n_requests"].astype(int)

    df_fail = df[~df["ok"]].copy()
    df = df[df["ok"]].copy()

    df_exclude = df[df["exclude"]].copy()
    df = df[~df["exclude"]].copy()

    df_out["n_fail"] = df_fail.groupby(["exp_num_users"]).size()
    df_out["n_fail"] = df_out["n_fail"].fillna(0).astype(int)

    df_fail["oom"] = df_fail["error"].apply(lambda x: "out of memory" in x)
    df_out["n_oom"] = df_fail.groupby(["exp_num_users"])["oom"].sum()
    df_out["n_oom"] = df_out["n_oom"].fillna(0).astype(int)

    df_out["n_toks"] = df.groupby(["exp_num_users"])["n_tokens"].sum()
    df_out["n_toks"] = df_out["n_toks"].fillna(0).astype(int)

    # check how many requests ran over
    for u in df_out.index:
        df_out.at[u, "n_exclude"] = (
            df_exclude[df_exclude["exp_num_users"] == u].value_counts(subset=["worker_idx", "request_idx"]).shape[0]
        )
    df_out["n_exclude"] = df_out["n_exclude"].astype(int)

    # if there are no valid outputs; return
    if df.shape[0] == 0:
        return df_out

    # percetnage of consistent responses
    df["consistent"] = df["consistent"].astype(int)
    df_out["consistent_pct"] = 100 * df.groupby(["exp_num_users"])["consistent"].mean()

    df_out["throughput"] = df.groupby(["exp_num_users"])["n_tokens"].sum() / df.iloc[0]["exp_duration"]

    df_prefill = df[df["response_idx"] == 0]
    df_nexttoken = df[df["response_idx"] > 0]

    df_out["latency_prefill_ms"] = df_prefill.groupby(["exp_num_users"])["duration_ms"].mean()
    df_out["latency_nexttoken_ms"] = df_nexttoken.groupby(["exp_num_users"])["duration_ms"].mean()
    df_out["latency_e2e_ms"] = (
        df.groupby(["exp_num_users", "worker_idx", "request_idx"])["duration_ms"].sum().groupby("exp_num_users").mean()
    )

    with pd.option_context(
        "display.float_format",
        "{:7.3f}".format,
        "display.max_columns",
        None,
        "display.max_rows",
        None,
    ):
        if print_df:
            print(df_out)
        if print_csv:
            print(df_out.to_csv())

    return df_out
