# basic python packages
from copy import deepcopy
import itertools
from collections import defaultdict, OrderedDict
import datetime as dt
import glob
import os
import re
import subprocess
import tempfile
import time
import scipy
import math

# For data processing in cpu
import pandas as pd
import numpy as np

# For metrics
from sklearn.metrics import auc, precision_recall_curve
from sklearn.metrics import roc_auc_score

# Torch modules and functions for model definition
import torch.nn.functional as F
import torch.optim as torch_optim
from torch.utils import data as torch_data
import torch
import torch.nn as nn

# Fast ai
from fastai.callbacks import *
from fastai import *
from fastai.tabular import *
from fastai.text import *
from fastai.metrics import accuracy
from fastai.tabular import *
from fastai.callbacks import SaveModelCallback

# For distributed pre-processing
from numba import cuda
import cudf
import cudf as gd
import nvstrings
from librmm_cffi import librmm
import dask
from dask.delayed import delayed
from dask.distributed import as_completed, Client, wait
from dask_cuda import LocalCUDACluster
import pyarrow.parquet as pq
import pyarrow as pa


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# the class concat dataset allows us to concatenate different data types in order to use the resulting indexation in our combined data loader
class ConcatDataset2Inputs(Dataset):
    def __init__(self, x1, x2, y): self.x1, self.x2, self.y = x1, x2, y

    def __len__(self): return len(self.y)

    def __getitem__(self, i): return (self.x1[i], self.x2[i]), self.y[i]


# the class concat dataset allows us to concatenate different data types in order to use the resulting indexation in our combined data loader
class ConcatDataset3Inputs(Dataset):
    def __init__(self, x1, x2, y): self.x1, self.x2, self.x3, self.y = x1, x2, x3, y

    def __len__(self): return len(self.y)

    def __getitem__(self, i): return (self.x1[i], self.x2[i], self.x3[i]), self.y[i]


## Define a module that combine two different models, for our example, we are combining rnn_seq_session encoder and tabular module.
class ConcatModel(nn.Module):
    def __init__(self, mod_tab, mod_nlp, layers, drops):
        super().__init__()
        self.mod_tab = mod_tab
        self.mod_nlp = mod_nlp
        lst_layers = []
        activs = [nn.ReLU(inplace=True), ] * (len(layers) - 2) + [None]
        for n_in, n_out, p, actn in zip(layers[:-1], layers[1:], drops, activs):
            lst_layers += bn_drop_lin(n_in, n_out, p=p, actn=actn)
        self.layers = nn.Sequential(*lst_layers)

    def forward(self, *x):
        x_tab = self.mod_tab(*x[2])
        x_nlp = self.mod_nlp(x[0], x[1])
        x = torch.cat([x_tab, x_nlp], dim=1)
        return self.layers(x)

#########################
#                       #
# Metrics and callbacks #
#                       #
#########################

def get_mean_reciprocal_rank(sub):
    # sub is a pandas dataframe
    # sub should have the following columns:
    # 'row_id', 'prob', 'reference', 'item_id'
    # sorted by prob in descending order for each group
    sub = gd.from_pandas(sub)

    def get_order_in_group(prob, row_id, order):
        for i in range(cuda.threadIdx.x, len(prob), cuda.blockDim.x):
            order[i] = i

    dg = sub.groupby('row_id', method="cudf").apply_grouped(get_order_in_group, incols=['prob', 'row_id'],
                                                            outcols={'order': np.int32},
                                                            tpb=32)

    dg = dg.to_pandas()
    dg['order'] = 1.0 / (1 + dg['order'])
    dg = dg[dg['reference'] == dg['item_id']]
    return dg['order'].mean()


def auroc_score(input, target):
    input, target = input.cpu().numpy()[:, 1], target.cpu().numpy()
    return roc_auc_score(target, input)


# Callback to calculate AUC at the end of each epoch
class AUROC(Callback):
    _order = -20  # Needs to run before the recorder

    def __init__(self, learn, **kwargs):
        self.learn = learn

    def on_train_begin(self, **kwargs):
        self.learn.recorder.add_metric_names(['AUROC'])

    def on_epoch_begin(self, **kwargs):
        self.output, self.target = [], []

    def on_batch_end(self, last_target, last_output, train, **kwargs):
        if not train:
            self.output.append(last_output)
            self.target.append(last_target)

    def on_epoch_end(self, last_metrics, **kwargs):
        if len(self.output) > 0:
            output = torch.cat(self.output)
            target = torch.cat(self.target)
            preds = F.softmax(output, dim=1)
            metric = auroc_score(preds, target)
            return add_metrics(last_metrics, [metric])


### CUDF pre-processing

def on_gpu(words, func, arg=None, dtype=np.int32):
    res = librmm.device_array(words.size(), dtype=dtype)
    if arg is None:
        cmd = 'words.%s(res.device_ctypes_pointer.value)' % (func)
    else:
        cmd = 'words.%s(arg,res.device_ctypes_pointer.value)' % (func)
    eval(cmd)
    return res


#########################
#                       #
# chunked parquet files #
#                       #
#########################


def get_chunk_dataset(dataframe, nchunks):
    split_frames = np.array_split(dataframe, nchunks)
    return split_frames


def save(tables_dict, base_output_dir):
    for key, item in tables_dict.items():
        base = os.path.join(base_output_dir, key)
        assert os.path.exists(base), "Output directory {} does not exist!".format(out_dirs[k])

        chunk_tables = get_chunk_dataset(item[0], item[1])

        print("Store data for:", key)
        print(len(chunk_tables))

        for i, table in enumerate(chunk_tables):
            path = os.path.join(base, 'session_%s' % i + ".parquet")
            assert not os.path.exists(path), "Output path already exists at {}!".format(path)
            print("write %s rows in chunk %s" % (len(table), i))
            pq.write_table(pa.Table.from_pandas(table), path, compression='snappy')
            
#########################
#                       #
#  Metadata properties  #
#     processing        #
#                       #
#########################

#regex for rating 
find_rating = re.compile("\|*[A-Za-z]* *[1-9]+ [A-Za-z]*")

#process special characters 
def process_special_charcters(x):
    x = x.replace("24/7", "anytime")
    x = x.replace('-', ' ')
    special_charcters = re.compile("\&|\/|\(|\)|\'")
    x = special_charcters.sub("", x)
    return x 

# Get the rate integer
find_number_star = re.compile("[A-Za-z ]* *([1-9]+) [A-Za-z]*")
map_stars_to_string = { 1 : 'one star', 2: 'two star', 3: 'three star', 4: 'four star', 5: 'five star'}

# get the unique star rating of the hotel as text variable  
def get_unique_star_rating(x): 
    from_list = []
    star_list = []
    
    if x == []:
        return ""
    else: 
        for pattern in x: 
            if pattern.startswith("|From"): 
                from_list.append(int(find_number_star.findall(pattern)[0]))
            else: 
                star_list.append(int(find_number_star.findall(pattern)[0]))

        if star_list == []: 
                #print("No star info, taking the argmax of 'From' filter")
                return map_stars_to_string[np.max(from_list)]
        else: 
            #assert star_list[0] >= np.max(from_list) : I comment this line because there are some lines where there is inconsistency between the argmax of the From filter (greater than) and the actual Star rating of the hotel  
            return map_stars_to_string[star_list[0]]
        
# Build the text sequence of tokens per hotel 
def get_seq_token(x): 
    # remove special 
    x = process_special_charcters(x)
    patterns = find_rating.findall(x)
    rating = get_unique_star_rating(patterns)
    x = find_rating.sub('| '+rating, x)
    tmp = list(set(x.split("|")))
    tmp.sort()
    x = " xxnext ".join(tmp).lower().split(" ")
    return [p for p in x if p]

