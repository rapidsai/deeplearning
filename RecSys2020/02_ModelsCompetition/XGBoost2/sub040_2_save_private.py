"""
Copyright (c) 2020, NVIDIA CORPORATION.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat Jun  6 08:23:09 2020

@author: Kazuki

exp048

"""

import gc
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from dask.distributed import Client
from sklearn.metrics import auc, log_loss, precision_recall_curve
from sklearn.model_selection import KFold

import cudf
import cupy
import dask_cudf
from dask_cuda import LocalCUDACluster

start = time.time()


# =============================================================================
# setting
# =============================================================================

SUB_NAME = "sub040"

# input
FILE_NAME_train = "../preprocessings/train-1.parquet"
FILE_NAME_test = "../preprocessings/test-1.parquet"
FILE_NAME_text = "../preprocessings/text-processings-1_v3.parquet"
FILE_NAME_user = "../preprocessings/a_count_combined-final.parquet"

# output
FILE_NAME_X_test = f"data/X_test_{SUB_NAME}_pvt.pkl"

# =============================================================================
# Feature Engineering
# =============================================================================
# RAPIDS DOESNT IMPLEMENT UINT
def convert2int(df, i=0):
    print("Converting uint to int...")
    for c in df.columns[i:]:
        c_type = str(df[c].dtype)
        if (c_type[:4] == "uint") | (c_type[:3] == "int"):
            mx = df[c].max()
            mn = df[c].min()
            print(c, "max =", mx, ", min =", mn, ", ", end="")
            if (mn >= -127) & (mx <= 127):
                df[c] = df[c].astype("int8")
            elif (mn >= -32_767) & (mx <= 32_767):
                df[c] = df[c].astype("int16")
            elif (mn >= -2_147_483_647) & (mx <= 2_147_483_647):
                df[c] = df[c].astype("int32")
            else:
                df[c] = df[c].astype("int64")


def split_time(df):
    gf = cudf.from_pandas(df[["timestamp"]])
    df["dt_dow"] = gf["timestamp"].dt.weekday.to_array()
    df["dt_hour"] = gf["timestamp"].dt.hour.to_array()
    df["dt_minute"] = gf["timestamp"].dt.minute.to_array()
    df["dt_second"] = gf["timestamp"].dt.second.to_array()
    return


def add_freq_tweet(train, valid):
    gf1 = cudf.from_pandas(train[["a_user_id", "b_user_id", "tweet_id"]]).reset_index(
        drop=True
    )
    gf2 = cudf.from_pandas(valid[["a_user_id", "b_user_id", "tweet_id"]]).reset_index(
        drop=True
    )
    gf1["idx"] = gf1.index
    gf2["idx"] = gf2.index

    gf = cudf.concat([gf1, gf2], axis=0)
    gf_unique = gf[["a_user_id", "tweet_id"]].drop_duplicates()

    gf_unique = gf_unique.groupby(["a_user_id"]).count().reset_index()
    gf_unique.columns = ["a_user_id_tmp", "no_tweet"]
    gf1 = gf1.merge(
        gf_unique[["a_user_id_tmp", "no_tweet"]],
        how="left",
        left_on="b_user_id",
        right_on="a_user_id_tmp",
    )
    gf2 = gf2.merge(
        gf_unique[["a_user_id_tmp", "no_tweet"]],
        how="left",
        left_on="b_user_id",
        right_on="a_user_id_tmp",
    )
    gf1 = gf1.sort_values("idx")
    gf2 = gf2.sort_values("idx")

    train["no_tweet"] = gf1["no_tweet"].fillna(0).astype("int32").to_array()
    valid["no_tweet"] = gf2["no_tweet"].fillna(0).astype("int32").to_array()


def diff_time(train, valid):
    gf1 = cudf.from_pandas(
        train[["timestamp", "a_user_id", "b_user_id", "tweet_id", "no_tweet"]]
    ).reset_index(drop=True)
    gf2 = cudf.from_pandas(
        valid[["timestamp", "a_user_id", "b_user_id", "tweet_id", "no_tweet"]]
    ).reset_index(drop=True)
    gf = cudf.concat([gf1, gf2], axis=0)
    gf = dask_cudf.from_cudf(gf, npartitions=16)
    gf["timestamp"] = gf["timestamp"].astype("int64") / 1e9
    gf_unique = gf[["timestamp", "a_user_id", "tweet_id"]].drop_duplicates()
    gf_unique.columns = ["tmp_timestamp", "tmp_a_user_id", "tmp_tweet_id"]
    gf = gf[gf["no_tweet"] != 0]
    gf = gf.drop("no_tweet", axis=1)
    gf = gf.drop("a_user_id", axis=1)
    gf = gf.merge(gf_unique, how="left", left_on="b_user_id", right_on="tmp_a_user_id")
    gf = gf[gf["tweet_id"] != gf["tmp_tweet_id"]]
    gf = gf[~gf["tmp_a_user_id"].isna()]

    gf["diff_timestamp_prev"] = gf["timestamp"] - gf["tmp_timestamp"]
    gf["diff_timestamp_after"] = gf["tmp_timestamp"] - gf["timestamp"]

    gf["diff_timestamp_after"] = gf.diff_timestamp_after.where(
        gf["diff_timestamp_after"] > 0, 15 * 24 * 3600
    )
    gf["diff_timestamp_prev"] = gf.diff_timestamp_prev.where(
        gf["diff_timestamp_prev"] > 0, 15 * 24 * 3600
    )

    gf = (
        gf[["tweet_id", "b_user_id", "diff_timestamp_prev", "diff_timestamp_after"]]
        .groupby(["tweet_id", "b_user_id"])
        .min()
        .reset_index()
    )

    gf.to_parquet("/tmp/gf")
    del gf
    del gf_unique
    del gf1
    del gf2
    gc.collect()

    gf = cudf.read_parquet("/tmp/gf/part.0.parquet")
    gf1 = cudf.from_pandas(train[["b_user_id", "tweet_id"]]).reset_index(drop=True)
    gf1["idx"] = gf1.index
    gf1 = gf1.merge(
        gf,
        how="left",
        left_on=["tweet_id", "b_user_id"],
        right_on=["tweet_id", "b_user_id"],
    )
    gf1 = gf1.sort_values("idx")
    train["diff_timestamp_prev"] = (
        gf1["diff_timestamp_prev"].fillna(15 * 24 * 3600).astype("int32").to_array()
    )
    train["diff_timestamp_after"] = (
        gf1["diff_timestamp_after"].fillna(15 * 24 * 3600).astype("int32").to_array()
    )
    del gf1
    gc.collect()

    gf1 = cudf.from_pandas(valid[["b_user_id", "tweet_id"]]).reset_index(drop=True)
    gf1["idx"] = gf1.index
    gf1 = gf1.merge(
        gf,
        how="left",
        left_on=["tweet_id", "b_user_id"],
        right_on=["tweet_id", "b_user_id"],
    )
    gf1 = gf1.sort_values("idx")
    valid["diff_timestamp_prev"] = (
        gf1["diff_timestamp_prev"].fillna(15 * 24 * 3600).astype("int32").to_array()
    )
    valid["diff_timestamp_after"] = (
        gf1["diff_timestamp_after"].fillna(15 * 24 * 3600).astype("int32").to_array()
    )


def add_diff_user1(train, valid, col):
    gf1 = cudf.from_pandas(train[[col, "b_user_id", "tweet_id"]]).reset_index(drop=True)
    gf2 = cudf.from_pandas(valid[[col, "b_user_id", "tweet_id"]]).reset_index(drop=True)
    gf1["idx"] = gf1.index
    gf2["idx"] = gf2.index

    gf = cudf.concat([gf1, gf2], axis=0)
    gf_lang = gf[["b_user_id", col, "tweet_id"]]  # .drop_duplicates()
    gf_lang = gf_lang[gf_lang[col] != 0]
    gf_lang = gf_lang.groupby(["b_user_id", col]).count()
    gf_lang = gf_lang[gf_lang > 3].reset_index()
    gf_lang = gf_lang.sort_values(["b_user_id", "tweet_id"], ascending=False)
    gf_lang["b_user_id_shifted"] = gf_lang["b_user_id"].shift(1)
    gf_lang = gf_lang[gf_lang["b_user_id_shifted"] != gf_lang["b_user_id"]]
    gf_lang.columns = ["b_user_id_lang", "top_" + col, "drop1", "drop2"]
    gf1 = gf1.merge(
        gf_lang[["b_user_id_lang", "top_" + col, "drop1", "drop2"]],
        how="left",
        left_on="b_user_id",
        right_on="b_user_id_lang",
    )
    gf2 = gf2.merge(
        gf_lang[["b_user_id_lang", "top_" + col, "drop1", "drop2"]],
        how="left",
        left_on="b_user_id",
        right_on="b_user_id_lang",
    )

    gf1 = gf1.sort_values("idx")
    gf2 = gf2.sort_values("idx")

    gf1["same_" + col] = gf1[col] == gf1["top_" + col]
    gf1["diff_" + col] = gf1[col] != gf1["top_" + col]
    gf1["nan_" + col] = 0
    gf1.loc[gf1["top_" + col].isna(), "same_" + col] = 0
    gf1.loc[gf1["top_" + col].isna(), "diff_" + col] = 0
    gf1.loc[gf1["top_" + col].isna(), "nan_" + col] = 1

    gf2["same_" + col] = gf2[col] == gf2["top_" + col]
    gf2["diff_" + col] = gf2[col] != gf2["top_" + col]
    gf2["nan_" + col] = 0
    gf2.loc[gf2["top_" + col].isna(), "same_" + col] = 0
    gf2.loc[gf2["top_" + col].isna(), "diff_" + col] = 0
    gf2.loc[gf2["top_" + col].isna(), "nan_" + col] = 1

    train["same_" + col] = gf1["same_" + col].fillna(0).astype("int32").to_array()
    train["diff_" + col] = gf1["diff_" + col].fillna(0).astype("int32").to_array()
    train["nan_" + col] = gf1["nan_" + col].fillna(0).astype("int32").to_array()

    valid["same_" + col] = gf2["same_" + col].fillna(0).astype("int32").to_array()
    valid["diff_" + col] = gf2["diff_" + col].fillna(0).astype("int32").to_array()
    valid["nan_" + col] = gf2["nan_" + col].fillna(0).astype("int32").to_array()


def add_diff_user1_fixed(train, valid, col):
    col = "tw_hash0"
    gf1 = cudf.from_pandas(
        train[[col, "tw_hash1", "b_user_id", "tweet_id"]]
    ).reset_index(drop=True)
    gf2 = cudf.from_pandas(
        valid[[col, "tw_hash1", "b_user_id", "tweet_id"]]
    ).reset_index(drop=True)
    gf1["idx"] = gf1.index
    gf2["idx"] = gf2.index

    gf_lang = cudf.concat(
        [
            gf1[["tw_hash0", "b_user_id", "tweet_id"]],
            gf1[["tw_hash1", "b_user_id", "tweet_id"]],
            gf2[["tw_hash0", "b_user_id", "tweet_id"]],
            gf2[["tw_hash1", "b_user_id", "tweet_id"]],
        ],
        axis=0,
    )
    gf_lang = gf_lang[["b_user_id", col, "tweet_id"]].drop_duplicates()
    gf_lang = gf_lang[gf_lang[col] != 0]
    gf_lang = gf_lang.groupby(["b_user_id", col]).count()
    gf_lang = gf_lang[gf_lang > 3].reset_index()
    gf_lang = gf_lang.sort_values(["b_user_id", "tweet_id"], ascending=False)
    gf_lang["b_user_id_shifted"] = gf_lang["b_user_id"].shift(1)
    gf_lang = gf_lang[gf_lang["b_user_id_shifted"] != gf_lang["b_user_id"]]
    gf_lang.columns = ["b_user_id_lang", "top_" + col, "drop1", "drop2"]
    gf1 = gf1.merge(
        gf_lang[["b_user_id_lang", "top_" + col, "drop1", "drop2"]],
        how="left",
        left_on="b_user_id",
        right_on="b_user_id_lang",
    )
    gf2 = gf2.merge(
        gf_lang[["b_user_id_lang", "top_" + col, "drop1", "drop2"]],
        how="left",
        left_on="b_user_id",
        right_on="b_user_id_lang",
    )

    gf1 = gf1.sort_values("idx")
    gf2 = gf2.sort_values("idx")

    gf1["same_" + col] = (gf1[col] == gf1["top_" + col]) | (
        gf1["tw_hash1"] == gf1["top_" + col]
    )
    gf1["diff_" + col] = (gf1[col] != gf1["top_" + col]) & (
        gf1["tw_hash1"] != gf1["top_" + col]
    )
    gf1["nan_" + col] = 0
    gf1.loc[gf1["top_" + col].isna(), "same_" + col] = 0
    gf1.loc[gf1["top_" + col].isna(), "diff_" + col] = 0
    gf1.loc[gf1["top_" + col].isna(), "nan_" + col] = 1

    gf2["same_" + col] = (gf2[col] == gf2["top_" + col]) | (
        gf2["tw_hash1"] == gf2["top_" + col]
    )
    gf2["diff_" + col] = (gf2[col] != gf2["top_" + col]) & (
        gf2["tw_hash1"] != gf2["top_" + col]
    )
    gf2["nan_" + col] = 0
    gf2.loc[gf2["top_" + col].isna(), "same_" + col] = 0
    gf2.loc[gf2["top_" + col].isna(), "diff_" + col] = 0
    gf2.loc[gf2["top_" + col].isna(), "nan_" + col] = 1

    train["same_" + col] = gf1["same_" + col].fillna(0).astype("int32").to_array()
    train["diff_" + col] = gf1["diff_" + col].fillna(0).astype("int32").to_array()
    train["nan_" + col] = gf1["nan_" + col].fillna(0).astype("int32").to_array()

    valid["same_" + col] = gf2["same_" + col].fillna(0).astype("int32").to_array()
    valid["diff_" + col] = gf2["diff_" + col].fillna(0).astype("int32").to_array()
    valid["nan_" + col] = gf2["nan_" + col].fillna(0).astype("int32").to_array()


def add_timeshift(train, valid, shift=1):
    gf1 = cudf.from_pandas(train[["timestamp", "b_user_id"]]).reset_index(drop=True)
    gf2 = cudf.from_pandas(valid[["timestamp", "b_user_id"]]).reset_index(drop=True)
    gf1["idx"] = gf1.index
    gf2["idx"] = gf2.index
    gf1["type"] = 1
    gf2["type"] = 2
    gf = cudf.concat([gf1, gf2], axis=0)

    gf = gf.sort_values(["b_user_id", "timestamp"])
    gf["timestamp"] = gf["timestamp"].astype("int64") / 1e9
    gf["b_user_id_shifted"] = gf["b_user_id"].shift(shift)
    gf["b_timestamp_shifted"] = gf["timestamp"].shift(shift)
    gf["b_timestamp_1"] = (gf["timestamp"] - gf["b_timestamp_shifted"]).abs()
    gf.loc[gf["b_user_id"] != gf["b_user_id_shifted"], "b_timestamp_1"] = 15 * 24 * 3600
    gf = gf.sort_values(["idx"])

    train["b_timestamp_" + str(shift)] = (
        gf.loc[gf["type"] == 1, "b_timestamp_1"].fillna(0).astype("int32").to_array()
    )
    valid["b_timestamp_" + str(shift)] = (
        gf.loc[gf["type"] == 2, "b_timestamp_1"].fillna(0).astype("int32").to_array()
    )


def target_encode_cudf_v3(
    train,
    valid,
    col,
    tar,
    n_folds=5,
    min_ct=0,
    smooth=20,
    seed=42,
    shuffle=False,
    t2=None,
    v2=None,
    x=-1,
):
    #
    # col = column to target encode (or if list of columns then multiple groupby)
    # tar = tar column encode against
    # if min_ct>0 then all classes with <= min_ct are consider in new class "other"
    # smooth = Bayesian smooth parameter
    # seed = for 5 Fold if shuffle==True
    # if x==-1 result appended to train and valid
    # if x>=0 then result returned in column x of t2 and v2
    #

    # SINGLE OR MULTIPLE COLUMN
    if not isinstance(col, list):
        col = [col]
    if (min_ct > 0) & (len(col) > 1):
        print("WARNING: Setting min_ct=0 with multiple columns. Not implemented")
        min_ct = 0
    name = "_".join(col)

    # FIT ALL TRAIN
    gf = cudf.from_pandas(train[col + [tar]]).reset_index(drop=True)
    gf["idx"] = gf.index  # needed because cuDF merge returns out of order
    if min_ct > 0:  # USE MIN_CT?
        other = gf.groupby(col[0]).size()
        other = other[other <= min_ct].index
        save = gf[col[0]].values.copy()
        gf.loc[gf[col[0]].isin(other), col[0]] = -1
    te = gf.groupby(col)[[tar]].agg(["mean", "count"]).reset_index()
    te.columns = col + ["m", "c"]
    mn = gf[tar].mean().astype("float32")
    te["smooth"] = ((te["m"] * te["c"]) + (mn * smooth)) / (te["c"] + smooth)
    if min_ct > 0:
        gf[col[0]] = save.copy()

    # PREDICT VALID
    gf2 = cudf.from_pandas(valid[col]).reset_index(drop=True)
    gf2["idx"] = gf2.index
    if min_ct > 0:
        gf2.loc[gf2[col[0]].isin(other), col[0]] = -1
    gf2 = gf2.merge(te[col + ["smooth"]], on=col, how="left", sort=False).sort_values(
        "idx"
    )
    if x == -1:
        valid[f"TE_{name}_{tar}"] = (
            gf2["smooth"].fillna(mn).astype("float32").to_array()
        )
    elif x >= 0:
        v2[:, x] = gf2["smooth"].fillna(mn).astype("float32").to_array()

    # KFOLD ON TRAIN
    tmp = cupy.zeros((train.shape[0]), dtype="float32")
    gf["fold"] = 0
    if shuffle:  # shuffling is 2x slower
        kf = KFold(n_folds, random_state=seed, shuffle=shuffle)
        for k, (idxT, idxV) in enumerate(kf.split(train)):
            gf.loc[idxV, "fold"] = k
    else:
        fsize = train.shape[0] // n_folds
        gf["fold"] = cupy.clip(gf.idx.values // fsize, 0, n_folds - 1)
    for k in range(n_folds):
        if min_ct > 0:  # USE MIN CT?
            if k < n_folds - 1:
                save = gf[col[0]].values.copy()
            other = gf.loc[gf.fold != k].groupby(col[0]).size()
            other = other[other <= min_ct].index
            gf.loc[gf[col[0]].isin(other), col[0]] = -1
        te = (
            gf.loc[gf.fold != k]
            .groupby(col)[[tar]]
            .agg(["mean", "count"])
            .reset_index()
        )
        te.columns = col + ["m", "c"]
        mn = gf.loc[gf.fold != k, tar].mean().astype("float32")
        te["smooth"] = ((te["m"] * te["c"]) + (mn * smooth)) / (te["c"] + smooth)
        gf = gf.merge(te[col + ["smooth"]], on=col, how="left", sort=False).sort_values(
            "idx"
        )
        tmp[(gf.fold.values == k)] = (
            gf.loc[gf.fold == k, "smooth"].fillna(mn).astype("float32").values
        )
        gf.drop_column("smooth")
        if (min_ct > 0) & (k < n_folds - 1):
            gf[col[0]] = save.copy()
    if x == -1:
        train[f"TE_{name}_{tar}"] = cupy.asnumpy(tmp.astype("float32"))
    elif x >= 0:
        t2[:, x] = cupy.asnumpy(tmp.astype("float32"))


# =============================================================================
# Count Encode
# =============================================================================
def count_encode_cudf_v2(train, valid, col, t2=None, v2=None, x=-1):
    #
    # col = column to count encode
    # if x==-1 then result appended to train and valid
    # if x>=0 then result returned in numpy arrays t2 and v2
    #    make sure x is even because it returns in x and x+1 column
    #
    # COUNT TRAIN SEPARATELY
    gf = cudf.from_pandas(train[[col]]).reset_index(drop=True)
    gf["idx"] = gf.index
    te = gf.groupby(col)[["idx"]].agg("count").rename({"idx": "ct"})
    gf = gf.merge(te, left_on=col, right_index=True, how="left").sort_values("idx")
    if x == -1:
        train[f"CE_{col}_norm"] = (gf.ct / len(gf)).astype("float32").to_array()
    elif x >= 0:
        t2[:, x] = (gf.ct / len(gf)).astype("float32").to_array()

    # COUNT VALID SEPARATELY
    gf2 = cudf.from_pandas(valid[[col]]).reset_index(drop=True)
    gf2["idx"] = gf2.index
    te = gf2.groupby(col)[["idx"]].agg("count").rename({"idx": "ct"})
    gf2 = gf2.merge(te, left_on=col, right_index=True, how="left").sort_values("idx")
    if x == -1:
        valid[f"CE_{col}_norm"] = (gf2.ct / len(gf2)).astype("float32").to_array()
    elif x >= 0:
        v2[:, x] = (gf2.ct / len(gf2)).astype("float32").to_array()

    # COUNT TRAIN VALID TOGETHER
    gf3 = cudf.concat([gf, gf2], axis=0)
    te = gf3.groupby(col)[["idx"]].agg("count").rename({"idx": "ct2"})
    gf = gf.merge(te, left_on=col, right_index=True, how="left").sort_values("idx")
    gf2 = gf2.merge(te, left_on=col, right_index=True, how="left").sort_values("idx")
    if x == -1:
        train[f"CE_{col}"] = gf.ct2.astype("float32").to_array()
        valid[f"CE_{col}"] = gf2.ct2.astype("float32").to_array()
    elif x >= 0:
        t2[:, x + 1] = gf.ct2.astype("float32").to_array()
        v2[:, x + 1] = gf2.ct2.astype("float32").to_array()


# =============================================================================
# DE
# =============================================================================
def diff_encode_cudf_v1(train, col, tar, sort_col=None, sft=1, t2=None, x=0):
    if sort_col is None:
        gf = cudf.from_pandas(train[[col, tar]]).reset_index(drop=True)
        gf["idx"] = gf.index
        gf = gf.sort_values([col])
    else:
        gf = cudf.from_pandas(train[[col, tar, sort_col]]).reset_index(drop=True)
        gf["idx"] = gf.index
        gf = gf.sort_values([col, sort_col])
    gf[col + "_sft"] = gf[col].shift(sft)
    gf[tar + "_sft"] = gf[tar].shift(sft)
    gf[tar + "_diff"] = gf[tar] - gf[tar + "_sft"]
    gf.loc[gf[col] != gf[col + "_sft"], tar + "_diff"] = 0
    gf = gf.sort_values(["idx"])
    if t2 is None:
        train[tar + "_diff"] = gf[tar + "_diff"].fillna(0).astype("float32").to_array()
    else:
        t2[:, x] = gf[tar + "_diff"].fillna(0).astype("float32").to_array()


# =============================================================================
# DL
# =============================================================================
def add_diff_language(train, valid):
    gf1 = cudf.from_pandas(
        train[["a_user_id", "language", "b_user_id", "tweet_id"]]
    ).reset_index(drop=True)
    gf2 = cudf.from_pandas(
        valid[["a_user_id", "language", "b_user_id", "tweet_id"]]
    ).reset_index(drop=True)
    gf1["idx"] = gf1.index
    gf2["idx"] = gf2.index
    gf = cudf.concat([gf1, gf2], axis=0)
    gf_lang = gf[["a_user_id", "language", "tweet_id"]].drop_duplicates()
    gf_lang = gf_lang.groupby(["a_user_id", "language"]).count().reset_index()
    gf_lang = gf_lang.sort_values(["a_user_id", "tweet_id"], ascending=False)
    gf_lang["a_user_shifted"] = gf_lang["a_user_id"].shift(1)
    gf_lang = gf_lang[gf_lang["a_user_shifted"] != gf_lang["a_user_id"]]
    gf_lang.columns = ["a_user_id_lang", "top_tweet_language", "drop1", "drop2"]
    gf1 = gf1.merge(
        gf_lang[["a_user_id_lang", "top_tweet_language"]],
        how="left",
        left_on="b_user_id",
        right_on="a_user_id_lang",
    )
    gf2 = gf2.merge(
        gf_lang[["a_user_id_lang", "top_tweet_language"]],
        how="left",
        left_on="b_user_id",
        right_on="a_user_id_lang",
    )
    gf1 = gf1.sort_values("idx")
    gf2 = gf2.sort_values("idx")
    gf1["same_language"] = gf1["language"] == gf1["top_tweet_language"]
    gf1["diff_language"] = gf1["language"] != gf1["top_tweet_language"]
    gf1["nan_language"] = 0
    gf1.loc[gf1["top_tweet_language"].isna(), "same_language"] = 0
    gf1.loc[gf1["top_tweet_language"].isna(), "diff_language"] = 0
    gf1.loc[gf1["top_tweet_language"].isna(), "nan_language"] = 1
    gf2["same_language"] = gf2["language"] == gf2["top_tweet_language"]
    gf2["diff_language"] = gf2["language"] != gf2["top_tweet_language"]
    gf2["nan_language"] = 0
    gf2.loc[gf2["top_tweet_language"].isna(), "same_language"] = 0
    gf2.loc[gf2["top_tweet_language"].isna(), "diff_language"] = 0
    gf2.loc[gf2["top_tweet_language"].isna(), "nan_language"] = 1
    train["same_language"] = gf1["same_language"].fillna(0).astype("int32").to_array()
    train["diff_language"] = gf1["diff_language"].fillna(0).astype("int32").to_array()
    train["nan_language"] = gf1["nan_language"].fillna(0).astype("int32").to_array()
    valid["same_language"] = gf2["same_language"].fillna(0).astype("int32").to_array()
    valid["diff_language"] = gf2["diff_language"].fillna(0).astype("int32").to_array()
    valid["nan_language"] = gf2["nan_language"].fillna(0).astype("int32").to_array()


def compute_prauc(pred, gt):
    prec, recall, thresh = precision_recall_curve(gt, pred)
    prauc = auc(recall, prec)
    return prauc


def calculate_ctr(gt):
    positive = len([x for x in gt if x == 1])
    ctr = positive / float(len(gt))
    return ctr


def compute_rce(pred, gt):
    cross_entropy = log_loss(gt, pred)
    data_ctr = calculate_ctr(gt)
    strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
    return (1.0 - cross_entropy / strawman_cross_entropy) * 100.0


# FAST METRIC FROM GIBA
def compute_rce_fast(pred, gt):
    cross_entropy = log_loss(gt, pred)
    yt = np.mean(gt)
    strawman_cross_entropy = -(yt * np.log(yt) + (1 - yt) * np.log(1 - yt))
    return (1.0 - cross_entropy / strawman_cross_entropy) * 100.0


def to_pkl_gzip(df, path):
    df.to_pickle(path)
    #os.system("rm " + path + ".gz")
    #os.system("gzip " + path)
    return


# =============================================================================
# xgb
# =============================================================================


def __get_imp__(model):
    for i in ["weight", "gain", "cover"]:
        imp_ = model.get_score(importance_type=i)
        imp_ = pd.DataFrame(list(imp_.items()))
        imp_.columns = ["col", i]
        if i == "weight":
            imp = imp_
        else:
            imp = pd.merge(imp, imp_, on="col", how="outer")
    imp.sort_values("gain", ascending=False, inplace=True)
    imp.set_index("col", inplace=True)
    return imp.fillna(0)


def get_imp(models):
    """
    models: list of model
    [model1, model2,...]
    or model

    return:
    averaged importance
    """
    if isinstance(models, list):
        for i, m in enumerate(models):
            if i == 0:
                imp = __get_imp__(m)
            else:
                imp += __get_imp__(m)
        imp /= i + 1
    else:
        imp = __get_imp__(models)

    imp.sort_values("gain", ascending=False, inplace=True)
    imp.reset_index(inplace=True)
    imp.columns = ["feature", "weight", "gain", "cover"]
    return imp


# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    cluster = LocalCUDACluster()
    client = Client(cluster)
    label_names = ["reply", "retweet", "retweet_comment", "like"]

    train = pd.read_parquet(FILE_NAME_train)
    test = pd.read_parquet(FILE_NAME_test)
    text = pd.read_parquet(FILE_NAME_text)

    train["count_ats"] = text["count_ats"]
    train["count_words"] = text["count_words"]
    train["count_char"] = text["count_char"]
    train["tw_hash0"] = text["tw_hash0"]
    train["tw_hash1"] = text["tw_hash1"]
    train["tw_rt_uhash"] = text["tw_rt_uhash"]
    test.set_index("id", inplace=True)
    test["count_ats"] = text["count_ats"]
    test["count_words"] = text["count_words"]
    test["count_char"] = text["count_char"]
    test["tw_hash0"] = text["tw_hash0"]
    test["tw_hash1"] = text["tw_hash1"]
    test["tw_rt_uhash"] = text["tw_rt_uhash"]
    del text
    gc.collect()

    user = pd.read_parquet(FILE_NAME_user)
    train["user_flag"] = user["a_count_combined"]
    test["user_flag"] = user["a_count_combined"]
    del user
    gc.collect()
    test.reset_index(inplace=True)

    cols_drop = ["links", "hashtags", "id"]
    train.drop(cols_drop, inplace=True, axis=1)
    test.drop(cols_drop, inplace=True, axis=1)

    convert2int(train)
    convert2int(test)
    train["timestamp"] = train["timestamp"].map(datetime.utcfromtimestamp)
    test["timestamp"] = test["timestamp"].map(datetime.utcfromtimestamp)
    split_time(train)
    split_time(test)
    train["timestamp"] = train["timestamp"].astype("int64") / 1e9
    test["timestamp"] = test["timestamp"].astype("int64") / 1e9

    for c in label_names:
        train.loc[train[c] == 0, c] = np.nan
    train["engage_time"] = train[label_names].min(1)
    gf = cudf.from_pandas(train[["engage_time", "timestamp"]])
    gf.loc[gf.engage_time.isnull(), "engage_time"] = np.nan
    gf["elapsed_time"] = gf["engage_time"] - gf["timestamp"]
    train["elapsed_time"] = gf.elapsed_time.astype("float32").to_array()
    train[label_names] = (train[label_names] > 0) * 1
    print(train[label_names].mean())
    del gf
    gc.collect()

    add_freq_tweet(train, test)
    add_timeshift(train, test, shift=1)
    add_timeshift(train, test, shift=-1)
    add_diff_user1(train, test, "tw_rt_uhash")
    add_diff_user1_fixed(train, test, "tw_hash0")
    diff_time(train, test)

    train.loc[train["tw_hash0"] == 0, "diff_tw_hash0"] = 0
    train.loc[train["tw_hash0"] == 0, "same_tw_hash0"] = 0
    test.loc[test["tw_hash0"] == 0, "diff_tw_hash0"] = 0
    test.loc[test["tw_hash0"] == 0, "same_tw_hash0"] = 0

    train.loc[train["tw_rt_uhash"] == 0, "diff_tw_rt_uhash"] = 0
    train.loc[train["tw_rt_uhash"] == 0, "same_tw_rt_uhash"] = 0
    test.loc[test["tw_rt_uhash"] == 0, "diff_tw_rt_uhash"] = 0
    test.loc[test["tw_rt_uhash"] == 0, "same_tw_rt_uhash"] = 0

    train2 = np.zeros((train.shape[0], 32), dtype="float32")
    test2 = np.zeros((test.shape[0], 32), dtype="float32")
    idx = 0
    cols = []
    for c in [
        "media",
        "tweet_type",
        "language",
        "a_user_id",
        "b_user_id",
        "tw_hash0",
        "tw_rt_uhash",
        "user_flag",
    ]:
        for t in ["reply", "retweet", "retweet_comment", "like"]:
            st = time.time()
            target_encode_cudf_v3(
                train,
                test,
                col=c,
                tar=t,
                smooth=20,
                min_ct=0,
                t2=train2,
                v2=test2,
                x=idx,
                shuffle=False,
            )
            end = time.time()
            idx += 1
            cols.append(f"TE_{c}_{t}")
            print("TE", c, t, "%.1f seconds" % (end - st))
    train = pd.concat([train, pd.DataFrame(train2, columns=cols)], axis=1)
    del train2
    x = gc.collect()
    test = pd.concat([test, pd.DataFrame(test2, columns=cols)], axis=1)
    del test2
    x = gc.collect()

    train2 = np.zeros((train.shape[0], 4), dtype="float32")
    test2 = np.zeros((test.shape[0], 4), dtype="float32")
    idx = 0
    cols = []
    c = ["domains", "language", "b_follows_a", "tweet_type", "media", "a_is_verified"]
    for t in ["reply", "retweet", "retweet_comment", "like"]:
        st = time.time()
        target_encode_cudf_v3(
            train,
            test,
            col=c,
            tar=t,
            smooth=20,
            min_ct=0,
            t2=train2,
            v2=test2,
            x=idx,
            shuffle=False,
        )
        end = time.time()
        idx += 1
        cols.append(f"TE_mult_{t}")
        print("TE", "mult", t, "%.1f seconds" % (end - st))
    train = pd.concat([train, pd.DataFrame(train2, columns=cols)], axis=1)
    del train2
    x = gc.collect()
    test = pd.concat([test, pd.DataFrame(test2, columns=cols)], axis=1)
    del test2
    x = gc.collect()

    train2 = np.zeros((train.shape[0], 4), dtype="float32")
    test2 = np.zeros((test.shape[0], 4), dtype="float32")
    idx = 0
    cols = []
    c = [
        "domains",
        "language",
        "b_follows_a",
        "tweet_type",
        "media",
        "a_is_verified",
        "user_flag",
    ]
    for t in ["reply", "retweet", "retweet_comment", "like"]:
        st = time.time()
        target_encode_cudf_v3(
            train,
            test,
            col=c,
            tar=t,
            smooth=20,
            min_ct=0,
            t2=train2,
            v2=test2,
            x=idx,
            shuffle=False,
        )
        end = time.time()
        idx += 1
        cols.append(f"TE_mult2_{t}")
        print("TE", "mult", t, "%.1f seconds" % (end - st))
    train = pd.concat([train, pd.DataFrame(train2, columns=cols)], axis=1)
    del train2
    x = gc.collect()
    test = pd.concat([test, pd.DataFrame(test2, columns=cols)], axis=1)
    del test2
    x = gc.collect()

    train2 = np.zeros((train.shape[0], 6), dtype="float32")
    test2 = np.zeros((test.shape[0], 6), dtype="float32")
    idx = 0
    cols = []
    for c in ["media", "tweet_type", "language", "a_user_id", "b_user_id", "user_flag"]:
        for t in ["elapsed_time"]:
            st = time.time()
            target_encode_cudf_v3(
                train,
                test,
                col=c,
                tar=t,
                smooth=20,
                min_ct=0,
                t2=train2,
                v2=test2,
                x=idx,
                shuffle=False,
            )
            end = time.time()
            idx += 1
            cols.append(f"TE_{c}_{t}")
            print("TE", c, t, "%.1f seconds" % (end - st))
    train = pd.concat([train, pd.DataFrame(train2, columns=cols)], axis=1)
    del train2
    x = gc.collect()
    test = pd.concat([test, pd.DataFrame(test2, columns=cols)], axis=1)
    del test2
    x = gc.collect()

    train2 = np.zeros((train.shape[0], 10), dtype="float32")
    test2 = np.zeros((test.shape[0], 10), dtype="float32")
    idx = 0
    cols = []
    for c in ["media", "tweet_type", "language", "a_user_id", "b_user_id"]:
        st = time.time()
        count_encode_cudf_v2(train, test, col=c, t2=train2, v2=test2, x=idx)
        end = time.time()
        idx += 2
        cols.append(f"CE_{c}_norm")
        cols.append(f"CE_{c}")
        print("CE", c, "%.1f seconds" % (end - st))
    train = pd.concat([train, pd.DataFrame(train2, columns=cols)], axis=1)
    del train2
    x = gc.collect()
    test = pd.concat([test, pd.DataFrame(test2, columns=cols)], axis=1)
    del test2
    x = gc.collect()

    train2 = np.zeros((train.shape[0], 6), dtype="float32")
    test2 = np.zeros((test.shape[0], 6), dtype="float32")
    idx = 0
    cols = []
    sc = "timestamp"
    for c in ["b_user_id"]:
        for t in ["b_follower_count", "b_following_count", "language"]:
            for s in [1, -1]:
                st = time.time()
                diff_encode_cudf_v1(
                    train, col=c, tar=t, sft=s, sort_col=sc, t2=train2, x=idx
                )
                diff_encode_cudf_v1(
                    test, col=c, tar=t, sft=s, sort_col=sc, t2=test2, x=idx
                )
                end = time.time()
                idx += 1
                cols.append(f"DE_{c}_{t}_{s}")
                print("DE", c, t, s, "%.1f seconds" % (end - st))
    train = pd.concat([train, pd.DataFrame(train2, columns=cols)], axis=1)
    del train2
    x = gc.collect()
    test = pd.concat([test, pd.DataFrame(test2, columns=cols)], axis=1)
    del test2
    x = gc.collect()

    add_diff_language(train, test)

    # =============================================================================
    # ff rate
    # =============================================================================
    train["a_ff_rate"] = (
        train["a_following_count"] / train["a_follower_count"]
    ).astype("float32")
    train["b_ff_rate"] = (
        train["b_follower_count"] / train["b_following_count"]
    ).astype("float32")
    test["a_ff_rate"] = (test["a_following_count"] / test["a_follower_count"]).astype(
        "float32"
    )
    test["b_ff_rate"] = (test["b_follower_count"] / test["b_following_count"]).astype(
        "float32"
    )

    # =============================================================================
    #
    # =============================================================================

    label_names = ["reply", "retweet", "retweet_comment", "like"]
    DONT_USE = [
        "timestamp",
        "a_account_creation",
        "b_account_creation",
        "engage_time",
        "tweet_id",
        "b_user_id",
        "a_user_id",
        "dt_dow",
        "a_account_creation",
        "b_account_creation",
        "elapsed_time",
        "domains",
        "tw_hash0",
        "tw_hash1",
        "tw_rt_uhash",
        "user_flag",
    ]
    DONT_USE += label_names

    import xgboost as xgb

    print("XGB Version", xgb.__version__)

    RMV = [c for c in DONT_USE if c in train.columns]

    X_train = train.drop(RMV, axis=1)
    Y_train = (train[label_names] > 0) * 1
    del train
    gc.collect()

    X_test = test[X_train.columns]
    del test
    gc.collect()

    if X_train.columns.duplicated().sum() > 0:
        raise Exception(f"duplicated!: {X_train.columns[X_train.columns.duplicated()]}")
    print("no dup :) ")
    print(f"X_train.shape {X_train.shape}")
    print(f"X_test.shape {X_test.shape}")

    to_pkl_gzip(X_test, FILE_NAME_X_test)
