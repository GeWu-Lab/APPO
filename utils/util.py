import os
import sys
import pytz
import json
import torch
from torch import nn
import shutil
import pathlib
import time
import pickle
import logging
import string
import numpy as np
from contextlib import contextmanager
from dataclasses import dataclass
from transformers.tokenization_utils import PreTrainedTokenizer
from datetime import datetime
from collections import OrderedDict
from typing import Optional, List, Dict, Any, Mapping, Iterable, Union
import transformers
import jsonlines
import matplotlib.pyplot as plt


def prepare_sample(data: Union[torch.Tensor, Any], device='cuda', dtype=None) -> Union[torch.Tensor, Any]:
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: prepare_sample(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_sample(v) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = {"device": device}
        if dtype is not None:
            data = data.to(dtype)
        # if isinstance(data.dtype,torch.FloatTensor) and dtype is not None:
        #     # kwargs.update({"dtype": dtype})
        #     data = data.to(dtype=dtype)
        return data.to(**kwargs)
    return data




def write2txt(fp, info, mode = 'a'):
    with open(fp, mode = mode) as f:
        f.write(info)


def write2jsonl(fp, dict_data, mode = 'a'):
    with jsonlines.open(fp, mode = mode) as f:
        f.write(dict_data)


def write2json(fp, data, mode = 'w'):
    with open(fp, mode = mode) as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False))
    
    
    
def set_seed(seed=42):
    """
    Set the random seed for reproducible results.

    :param seed: An integer value to be used as the random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def prepare_sample(data: Union[torch.Tensor, Any], device='cuda', dtype=None) -> Union[torch.Tensor, Any]:
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: prepare_sample(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_sample(v) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = {"device": device}
        if dtype is not None:
            data = data.to(dtype)
        # if isinstance(data.dtype,torch.FloatTensor) and dtype is not None:
        #     # kwargs.update({"dtype": dtype})
        #     data = data.to(dtype=dtype)
        return data.to(**kwargs)
    return data



def normalize(arr, axis = 0):
    min_value = arr.min(axis = axis)
    max_value = arr.max(axis = axis)
    return (arr - min_value) / (max_value - min_value)


def plot(list_data, save_path, bin_width = 1, xlabel='x',ylabel='y',title='title'):
    max_value = max(list_data)
    min_value = min(list_data)
    bins = np.arange(min_value, max_value + bin_width, bin_width)
    plt.hist(list_data,bins=bins,edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    # plt.savefig(save_path)
    plt.savefig(save_path, dpi=300)
    plt.close()




