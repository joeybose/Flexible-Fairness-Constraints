from comet_ml import Experiment
from sklearn.metrics import mean_squared_error
from collections import defaultdict
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.nn.init import xavier_normal, xavier_uniform
from torch.distributions import Categorical
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn import preprocessing
import numpy as np
import random
import argparse
import pickle
import json
import logging
import sys, os
import subprocess
from tqdm import tqdm
tqdm.monitor_interval = 0
from utils import *
from preprocess_movie_lens import make_dataset
import joblib
from collections import Counter
import ipdb
sys.path.append('../')
import gc
from collections import OrderedDict
from model import *

def collate_fn(batch):
    if isinstance(batch, np.ndarray) or (isinstance(batch, list) and isinstance(batch[0], np.ndarray)):
        return torch.LongTensor(batch).contiguous()
    else:
        return torch.stack(batch).contiguous()

def test_reddit(dataset, args, modelD, subsample=1):
    l_ranks, r_ranks = [], []
    test_loader = DataLoader(dataset,batch_size=1, num_workers=1, collate_fn=collate_fn)
    cst_inds_user = np.arange(args.num_users, dtype=np.int64)[:,None]
    cst_inds_sr = np.arange(args.num_sr, dtype=np.int64)[:,None]
    data_itr = tqdm(enumerate(test_loader))

    for idx, data_ in data_itr:
        if idx % subsample != 0:
            continue

        lhs, rhs = data_[:,0], data_[:,1]

        if len(lhs) == 1:
            lhs_temp = np.expand_dims(np.array([rhs.cpu().numpy()]),0)[0]
            rhs_temp = np.expand_dims(np.array([lhs.cpu().numpy()]),1)[0]
        else:
            raise NotImplementedError

        l_batch = np.concatenate([cst_inds_user,\
            lhs_temp.repeat(args.num_users, axis=0)], axis=1)
        r_batch = np.concatenate([rhs_temp.repeat(args.num_sr,axis=0),\
                cst_inds_sr], axis=1)

        l_batch = torch.LongTensor(l_batch).contiguous()
        r_batch = torch.LongTensor(r_batch).contiguous()

        l_batch = Variable(l_batch).cuda()
        r_batch = Variable(r_batch).cuda()

        d_ins = torch.cat([l_batch, r_batch], dim=0)
        d_outs = modelD(d_ins)
        l_enrgs = d_outs[:len(l_batch)]
        r_enrgs = d_outs[len(l_batch):]

        l_rank = compute_rank(v2np(l_enrgs), lhs)
        r_rank = compute_rank(v2np(r_enrgs), rhs)

        l_ranks.append(l_rank)
        r_ranks.append(r_rank)

    l_ranks = np.array(l_ranks)
    r_ranks = np.array(r_ranks)
    l_mean = l_ranks.mean()
    r_mean = r_ranks.mean()
    l_mrr = (1. / l_ranks).mean()
    r_mrr = (1. / r_ranks).mean()
    l_h10 = (l_ranks <= 10).mean()
    r_h10 = (r_ranks <= 10).mean()
    l_h5 = (l_ranks <= 5).mean()
    r_h5 = (r_ranks <= 5).mean()
    avg_mr = (l_mean + r_mean)/2
    avg_mrr = (l_mrr+r_mrr)/2
    avg_h10 = (l_h10+r_h10)/2
    avg_h5 = (l_h5+r_h5)/2

    return l_ranks, r_ranks, l_mrr, r_mrr, avg_mr, avg_mrr, avg_h10, avg_h5
