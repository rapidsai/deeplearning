import pandas as pd
import numpy as np
import pickle
import re 
import matplotlib.pyplot as plt
import nvstrings, nvcategory
import warnings
import cudf
import cupy as cp
import pyarrow.parquet as pq
import pdb
import torch
import os
import glob


from fastai import *
from fastai.basic_data import *
from fastai.core import *

from time import time
from torch import tensor

from torch.utils.dlpack import from_dlpack
from cuml.preprocessing import LabelEncoder
from sys import getsizeof
from numba import cuda
from sklearn.metrics import roc_auc_score
from datetime import date


cpu = torch.device("cpu")

def _enforce_str(y: cudf.Series) -> cudf.Series:
    """
    Ensure that nvcategory is being given strings
    """
    if y.dtype != "object":
        return y.astype("str")
    return y


class MyLabelEncoder(object):
    
    def __init__(self, *args, **kwargs):
        self._cats: nvcategory.nvcategory = None
        self._dtype = None
        self._fitted: bool = False

    def _check_is_fitted(self):
        if not self._fitted:
            raise TypeError("Model must first be .fit()")

    def fit(self, y: cudf.Series) -> "LabelEncoder":
        self._dtype = y.dtype

        y = _enforce_str(y)

        self._cats = nvcategory.from_strings(y.data)
        self._fitted = True
        return self

    def transform(self, y: cudf.Series) -> cudf.Series:
        self._check_is_fitted()
        y = _enforce_str(y)
        encoded = cudf.Series(
            nvcategory.from_strings(y.data)
            .set_keys(self._cats.keys())
            .values()
        )
        return encoded.replace(-1, 0)

    def fit_transform(self, y: cudf.Series) -> cudf.Series:
        self._dtype = y.dtype

        # Convert y to nvstrings series, if it isn't one
        y = _enforce_str(y)

        # Bottleneck is here, despite everything being done on the device
        self._cats = nvcategory.from_strings(y.data)

        self._fitted = True
        arr: cp.array = cp.array(
            y.data.size(), dtype=np.int32
        )
        self._cats.values(devptr=arr.device_ctypes_pointer.value)
        return cudf.Series(arr)

    def inverse_transform(self, y: cudf.Series) -> cudf.Series:
        raise NotImplementedError
        
MEDIAN = "median"
CONSTANT = "constant"
TRAIN = 'train'
VALID = 'valid'
TEST = 'test'

class PreprocessDF():
    fill_strategy = MEDIAN
    add_col = False
    fill_val = 0
    category_encoders = {}

    def __init__(self, cat_names, cont_names, label_name, mode=TRAIN, fill_strategy=MEDIAN, to_cpu=True):
        self.cat_names, self.cont_names = cat_names, cont_names
        self.fill_strategy = fill_strategy
        self.label_name = label_name
        self.to_cpu = to_cpu 

    def preproc_dataframe(self, gdf: cudf.DataFrame, mode):
        start = time()
        self.gdf = gdf
        self.mode = mode
        self.categorify()
        self.fill_missing()
        self.normalize()
        if is_listy(self.label_name):
            for n in self.label_name: self.gdf[n] = self.gdf[n].astype('float32')
        else: 
            self.gdf[self.label_name] = self.gdf[self.label_name].astype('float32')
#         print(f"preprocessing used {time()-start:.2f} seconds.")
        start = time()
        # int64 in cudf may not be equivalent to that in pytorch
        cats = from_dlpack(self.gdf[self.cat_names].to_dlpack()).long()
        conts = from_dlpack(self.gdf[self.cont_names].to_dlpack())
        label = from_dlpack(self.gdf[self.label_name].to_dlpack())
#         print(f"convert from cudf to cuda tensor used {time()-start:.2f} seconds.")
        if self.to_cpu:
            start = time()
            result = (cats.to(cpu), conts.to(cpu)), label.to(cpu)
#             print(f"convert from cuda tensor to cpu tensor used {time()-start:.2f} seconds.")
            return result
        return (cats, conts), label

    def normalize(self):
        if self.mode == TRAIN:
            gdf_cont = self.gdf[self.cont_names]
            self.means, self.stds = gdf_cont.mean(), gdf_cont.std()
        for i, name in enumerate(self.cont_names):
            self.gdf[name] = (self.gdf[name]-self.means[i])/(1e-7+self.stds[i])
            self.gdf[name] = self.gdf[name].astype('float32')

    def get_median(self, col: cudf.Series):
        col = col.dropna().reset_index(drop=True)
        return col.sort_values()[len(col)//2]

    def add_col_(self, cont_names_na):
        for name in cont_names_na:
            name_na = name + "_na"
            self.gdf[name_na] = self.gdf[name].isna()
            if name_na not in self.cat_names: self.cat_names.append(name_na)

    def fill_missing(self):
        if self.mode == TRAIN:
            self.train_cont_names_na = [name for name in self.cont_names if self.gdf[name].isna().sum()]
            if self.fill_strategy == MEDIAN:
                self.filler = {name: self.get_median(self.gdf[name]) for name in self.train_cont_names_na}
            elif self.fill_strategy == CONSTANT:
                self.filler = {name: self.fill_val for name in self.train_cont_names_na}
            else:
                self.filler = {name: self.gdf[name].value_counts().index[0] for name in self.train_cont_names_na}
            if self.add_col: 
                self.add_col_(self.train_cont_names_na)
            self.gdf[self.train_cont_names_na].fillna(self.filler, inplace=True)
        else:
            cont_names_na = [name for name in self.cont_names if self.gdf[name].isna().sum()]
            if not set(cont_names_na).issubset(set(self.train_cont_names_na)):
                 raise Exception(f"""There are nan values in field {cont_names_na} but there were none in the training set. 
                 Please fix those manually.""")
            if self.add_col: self.add_col_(cont_names_na)
            self.gdf[self.train_cont_names_na].fillna(self.filler, inplace=True)

    def categorify(self):
        for name in self.cat_names:
            if self.mode == TRAIN:
                self.category_encoders[name] = MyLabelEncoder()
                self.gdf[name] = self.category_encoders[name].fit_transform(self.gdf[name].append([None]))[:-1]
            else: self.gdf[name] = self.category_encoders[name].transform(self.gdf[name].append([None]))[:-1]
            self.gdf[name] = self.gdf[name].astype('int64')
