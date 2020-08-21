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

import os
from datetime import datetime
from glob import glob

import pandas as pd

SUB_NAME = "sub040"

label_names = ["reply", "retweet", "retweet_comment", "like"]


# =============================================================================
# public
# =============================================================================

test = pd.read_parquet("../preprocessings/test-0.parquet")
test["timestamp"] = test["timestamp"].map(datetime.utcfromtimestamp)

files = sorted(glob(f"output/{SUB_NAME}_sub_pub_s*"))

for i, x in enumerate(files):
    if i == 0:
        sub = pd.read_pickle(x)
    else:
        sub.iloc[:, 2:] += pd.read_pickle(x).iloc[:, 2:]
sub.iloc[:, 2:] /= len(files)

sub["dt_day"] = test["timestamp"].dt.day

sub.loc[sub.dt_day == 18, "like"] *= 0.85
sub.loc[sub.dt_day == 18, "reply"] *= 0.85
sub.loc[sub.dt_day == 18, "retweet"] *= 0.85
sub.loc[sub.dt_day == 18, "retweet_comment"] *= 0.85

os.system(f"mkdir ../output/{SUB_NAME}")
for l in label_names:
    cols = ["tweet_id", "enaging_user_id", l]
    sub[cols].to_csv(
        f"output/{SUB_NAME}/pub_{l}_x0.85.csv", index=False, header=None
    )

# =============================================================================
# private
# =============================================================================

test = pd.read_parquet("../preprocessings/test-1.parquet")
test["timestamp"] = test["timestamp"].map(datetime.utcfromtimestamp)

files = sorted(glob(f"output/{SUB_NAME}_sub_pvt_s*"))

for i, x in enumerate(files):
    if i == 0:
        sub = pd.read_pickle(x)
    else:
        sub.iloc[:, 2:] += pd.read_pickle(x).iloc[:, 2:]
sub.iloc[:, 2:] /= len(files)

sub["dt_day"] = test["timestamp"].dt.day

sub.loc[sub.dt_day == 18, "like"] *= 0.85
sub.loc[sub.dt_day == 18, "reply"] *= 0.85
sub.loc[sub.dt_day == 18, "retweet"] *= 0.85
sub.loc[sub.dt_day == 18, "retweet_comment"] *= 0.85

for l in label_names:
    cols = ["tweet_id", "enaging_user_id", l]
    sub[cols].to_csv(
        f"output/{SUB_NAME}/pvt_{l}_x0.85.csv", index=False, header=None
    )
