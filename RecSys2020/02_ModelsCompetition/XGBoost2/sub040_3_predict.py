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

import argparse
import os

import pandas as pd
import xgboost as xgb

# =============================================================================
#
# =============================================================================

SUB_NAME = "sub040"

xgb_parms = {
    "max_depth": 8,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.3,
    "eval_metric": "logloss",
    "objective": "binary:logistic",
    "tree_method": "gpu_hist",
    "nthread": 40,
}

LOOP1 = 1
NROUNDS = [600, 700, 350, 700]
VERBOSE_EVAL = 50

TRAIN_SAMPLE_RATIO = 0.1

label_names = ["reply", "retweet", "retweet_comment", "like"]

parser = argparse.ArgumentParser(description=SUB_NAME)
parser.add_argument("--seed", help="seed for sampling", type=int)
args = parser.parse_args()
print("seed", args.seed)

# =============================================================================
#
# =============================================================================

X_train = pd.read_pickle(f"data/X_train_{SUB_NAME}.pkl").sample(
    frac=TRAIN_SAMPLE_RATIO, random_state=args.seed
)
Y_train = pd.read_pickle(f"data/Y_train_{SUB_NAME}.pkl").sample(
    frac=TRAIN_SAMPLE_RATIO, random_state=args.seed
)

dtest_pub = xgb.DMatrix(data=pd.read_pickle(f"data/X_test_{SUB_NAME}_pub.pkl"))
dtest_pvt = xgb.DMatrix(data=pd.read_pickle(f"data/X_test_{SUB_NAME}_pvt.pkl"))


def to_pkl_gzip(df, path):
    df.to_pickle(path)
    #os.system("rm " + path + ".gz")
    #os.system("gzip " + path)
    return


# =============================================================================
#
# =============================================================================
sub_pub = pd.read_csv("../preprocessings/sample_submission_public.csv")
sub_pub = sub_pub[["tweet_id", "b_user_id"]].rename(
    columns={"b_user_id": "enaging_user_id"}
)

sub_pvt = pd.read_csv("../preprocessings/sample_submission_private.csv")
sub_pvt = sub_pvt[["tweet_id", "b_user_id"]].rename(
    columns={"b_user_id": "enaging_user_id"}
)


for i, name in enumerate(label_names):

    print("#" * 25)
    print("###", name)
    print("#" * 25)

    sub_pub[name] = 0
    sub_pvt[name] = 0

    dtrain = xgb.DMatrix(data=X_train, label=Y_train.iloc[:, i].values)

    for j in range(LOOP1):
        xgb_parms["seed"] = j

        model = xgb.train(
            xgb_parms,
            dtrain=dtrain,
            evals=[(dtrain, "train")],
            num_boost_round=NROUNDS[i],
            verbose_eval=VERBOSE_EVAL,
        )

        sub_pub[name] += model.predict(dtest_pub)
        sub_pvt[name] += model.predict(dtest_pvt)

sub_pub.iloc[:, 2:] /= LOOP1
sub_pvt.iloc[:, 2:] /= LOOP1

to_pkl_gzip(sub_pub, f"output/{SUB_NAME}_sub_pub_s{args.seed}.pkl")
to_pkl_gzip(sub_pvt, f"output/{SUB_NAME}_sub_pvt_s{args.seed}.pkl")
