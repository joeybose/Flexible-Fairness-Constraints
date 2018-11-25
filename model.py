import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.nn.init import xavier_normal, xavier_uniform
from torch.distributions import Categorical
from tensorboard_logger import Logger as tfLogger
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
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
from utils import create_or_append, compute_rank
from preprocess_movie_lens import make_dataset
import joblib
from collections import Counter
import ipdb
sys.path.append('../')
import gc
from collections import OrderedDict

''' Some Helpful Globals '''
ftensor = torch.FloatTensor
ltensor = torch.LongTensor
v2np = lambda v: v.data.cpu().numpy()
USE_SPARSE_EMB = True

class TransE(nn.Module):
    def __init__(self, num_ent, num_rel, embed_dim, p):
        super(TransE, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.embed_dim = embed_dim
        self.p = p

        r = 6 / np.sqrt(self.embed_dim)
        self.ent_embeds = nn.Embedding(self.num_ent, self.embed_dim, max_norm=1, norm_type=2, sparse=USE_SPARSE_EMB)
        self.rel_embeds = nn.Embedding(self.num_rel, self.embed_dim, max_norm=1, norm_type=2, sparse=USE_SPARSE_EMB)

        self.ent_embeds.weight.data.uniform_(-r, r)#.renorm_(p=2, dim=1, maxnorm=1)
        self.rel_embeds.weight.data.uniform_(-r, r)#.renorm_(p=2, dim=1, maxnorm=1)

    def forward(self, triplets):

        lhs_idxs = triplets[:, 0]
        rel_idxs = triplets[:, 1]
        rhs_idxs = triplets[:, 2]
        lhs_es = self.ent_embeds(lhs_idxs)
        rel_es = self.rel_embeds(rel_idxs)
        rhs_es = self.ent_embeds(rhs_idxs)

        enrgs = (lhs_es + rel_es - rhs_es).norm(p=self.p, dim=1)
        return enrgs

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))

class TransD(nn.Module):
    def __init__(self, num_ent, num_rel, embed_dim, p):
        super(TransD, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.embed_dim = embed_dim
        self.p = p

        r = 6 / np.sqrt(self.embed_dim)

        self._ent_embeds = nn.Embedding(self.num_ent, self.embed_dim, max_norm=1, norm_type=2, sparse=USE_SPARSE_EMB)
        self.rel_embeds = nn.Embedding(self.num_rel, self.embed_dim, max_norm=1, norm_type=2, sparse=USE_SPARSE_EMB)

        self.ent_transfer = nn.Embedding(self.num_ent, self.embed_dim, max_norm=1, norm_type=2, sparse=USE_SPARSE_EMB)
        self.rel_transfer = nn.Embedding(self.num_rel, self.embed_dim, max_norm=1, norm_type=2, sparse=USE_SPARSE_EMB)

        self._ent_embeds.weight.data.uniform_(-r, r)#.renorm_(p=2, dim=1, maxnorm=1)
        self.rel_embeds.weight.data.uniform_(-r, r)#.renorm_(p=2, dim=1, maxnorm=1)

    def transfer(self, emb, e_transfer, r_transfer):
        return emb + (emb * e_transfer).sum(dim=1, keepdim=True) * r_transfer

    #@profile
    def ent_embeds(self, idx, rel_idx):
        es = self._ent_embeds(idx)
        ts = self.ent_transfer(idx)

        rel_ts = self.rel_transfer(rel_idx)
        proj_es = self.transfer(es, ts, rel_ts)
        return proj_es

    def forward(self, triplets, return_ent_emb=False):
        lhs_idxs = triplets[:, 0]
        rel_idxs = triplets[:, 1]
        rhs_idxs = triplets[:, 2]

        rel_es = self.rel_embeds(rel_idxs)

        lhs = self.ent_embeds(lhs_idxs, rel_idxs)
        rhs = self.ent_embeds(rhs_idxs, rel_idxs)

        if not return_ent_emb:
            enrgs = (lhs + rel_es - rhs).norm(p=self.p, dim=1)
            return enrgs
        else:
            enrgs = (lhs + rel_es - rhs).norm(p=self.p, dim=1)
            return enrgs,lhs,rhs,rel_es

    def get_embed(self, ents, rel_idxs):
        ent_embed = self.ent_embeds(ents, rel_idxs)
        return ent_embed

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.loaded = True
        self.load_state_dict(torch.load(fn))

class TransD_BiDecoder(nn.Module):
    def __init__(self, num_ent, num_rel, embed_dim, p):
        super(TransD_BiDecoder, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.embed_dim = embed_dim
        self.p = p

        r = 6 / np.sqrt(self.embed_dim)

        ''' Encoder '''
        self._ent_embeds = nn.Embedding(self.num_ent, self.embed_dim, max_norm=1, norm_type=2, sparse=USE_SPARSE_EMB)
        self.rel_embeds = nn.Embedding(self.num_rel, self.embed_dim, max_norm=1, norm_type=2, sparse=USE_SPARSE_EMB)

        self.ent_transfer = nn.Embedding(self.num_ent, self.embed_dim, max_norm=1, norm_type=2, sparse=USE_SPARSE_EMB)
        self.rel_transfer = nn.Embedding(self.num_rel, self.embed_dim, max_norm=1, norm_type=2, sparse=USE_SPARSE_EMB)

        self._ent_embeds.weight.data.uniform_(-r, r)#.renorm_(p=2, dim=1, maxnorm=1)
        self.rel_embeds.weight.data.uniform_(-r, r)#.renorm_(p=2, dim=1, maxnorm=1)

        ''' Decoder '''
        self.decoder = nn.Embedding(self.embed_dim, self.embed_dim, max_norm=1, norm_type=2, sparse=USE_SPARSE_EMB)

    def transfer(self, emb, e_transfer, r_transfer):
        return emb + (emb * e_transfer).sum(dim=1, keepdim=True) * r_transfer

    #@profile
    def ent_embeds(self, idx, rel_idx):
        es = self._ent_embeds(idx)
        ts = self.ent_transfer(idx)

        rel_ts = self.rel_transfer(rel_idx)
        proj_es = self.transfer(es, ts, rel_ts)
        return proj_es

    def forward(self, triplets, return_ent_emb=False):
        lhs_idxs = triplets[:, 0]
        rel_idxs = triplets[:, 1]
        rhs_idxs = triplets[:, 2]

        rel_es = self.rel_embeds(rel_idxs)

        lhs = self.ent_embeds(lhs_idxs, rel_idxs)
        rhs = self.ent_embeds(rhs_idxs, rel_idxs)

        if not return_ent_emb:
            enrgs = (lhs + rel_es - rhs).norm(p=self.p, dim=1)
            return enrgs
        else:
            enrgs = (lhs + rel_es - rhs).norm(p=self.p, dim=1)
            return enrgs,lhs,rhs

    def get_embed(self, ents, rel_idxs):
        ent_embed = self.ent_embeds(ents, rel_idxs)
        return ent_embed

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))

class AttributeFilter(nn.Module):
    def __init__(self, embed_dim, attribute='gender'):
        super(AttributeFilter, self).__init__()
        self.embed_dim = int(embed_dim)
        self.attribute = attribute
        self.W1 = nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True)
        self.W2 = nn.Linear(int(self.embed_dim * 2), int(self.embed_dim), bias=True)

    def forward(self, ents_emb):
        h1 = F.leaky_relu(self.W1(ents_emb))
        h2 = F.leaky_relu(self.W2(h1))
        return h2

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.loaded = True
        self.load_state_dict(torch.load(fn))

class DemParDisc(nn.Module):
    def __init__(self, embed_dim, attribute_data,attribute='gender',use_cross_entropy=True):
        super(DemParDisc, self).__init__()
        self.embed_dim = int(embed_dim)
        self.attribute_data = attribute_data
        if use_cross_entropy:
            self.cross_entropy = True
        else:
            self.cross_entropy = False
        if attribute == 'gender':
            users_sex = attribute_data[0]['sex']
            users_sex = [0 if i == 'M' else 1 for i in users_sex]
            self.users_sensitive = np.ascontiguousarray(users_sex)
            self.out_dim = 1
        elif attribute == 'occupation':
            users_occupation = attribute_data[0]['occupation']
            users_occupation_list = sorted(set(users_occupation))
            occ_to_idx = {}
            for i, occ in enumerate(users_occupation_list):
                occ_to_idx[occ] = i
            users_occupation = [occ_to_idx[occ] for occ in users_occupation]
            self.users_sensitive = np.ascontiguousarray(users_occupation)
            self.out_dim = len(users_occupation_list)
        elif attribute == 'random':
            users_random = attribute_data[0]['rand']
            self.users_sensitive = np.ascontiguousarray(users_random)
            self.out_dim = 1
        else:
            users_age = attribute_data[0]['age'].values
            users_age_list = sorted(set(users_age))
            bins = np.linspace(5, 75, num=15, endpoint=True)
            inds = np.digitize(users_age, bins) - 1
            self.users_sensitive = np.ascontiguousarray(inds)
            self.out_dim = len(bins)

        self.W1 = nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True)
        self.W2 = nn.Linear(int(self.embed_dim * 2), int(self.embed_dim), bias=True)
        self.W3 = nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True)
        self.W4 = nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
        self.attribute = attribute

    def forward(self, ents_emb, ents):
        h1 = F.leaky_relu(self.W1(ents_emb))
        h2 = F.leaky_relu(self.W2(h1))
        h3 = F.leaky_relu(self.W3(h2))
        scores = self.W4(h3)
        if self.attribute == 'gender' or self.attribute == 'random':
            A_labels = Variable(torch.Tensor(self.users_sensitive[ents])).cuda()
            A_labels = A_labels.unsqueeze(1)
            if self.cross_entropy:
                fair_penalty = F.binary_cross_entropy_with_logits(scores,\
                        A_labels)
            else:
                probs = torch.sigmoid(scores)
                fair_penalty = F.l1_loss(probs,A_labels,reduction='elementwise_mean')
        else:
            A_labels = Variable(torch.LongTensor(self.users_sensitive[ents])).cuda()
            if self.cross_entropy:
                fair_penalty = F.cross_entropy(scores,A_labels)
            else:
                probs = torch.softmax(scores,dim=1)
                fair_penalty = F.multi_margin_loss(probs,A_labels,reduction='elementwise_mean')

        return fair_penalty

    def predict(self, ents_emb, ents, return_preds=False):
        h1 = F.leaky_relu(self.W1(ents_emb))
        h2 = F.leaky_relu(self.W2(h1))
        h3 = F.leaky_relu(self.W3(h2))
        scores = self.W4(h3)
        if self.attribute == 'gender':
            A_labels = Variable(torch.Tensor(self.users_sensitive[ents])).cuda()
            A_labels = A_labels.unsqueeze(1)
            probs = torch.sigmoid(scores)
            preds = (probs > torch.Tensor([0.5]).cuda()).float() * 1
        else:
            A_labels = Variable(torch.LongTensor(self.users_sensitive[ents])).cuda()
            log_probs = F.log_softmax(scores, dim=1)
            probs = torch.exp(log_probs)
            preds = log_probs.max(1, keepdim=True)[1] # get the index of the max
            correct = preds.eq(A_labels.view_as(preds)).sum().item()
        if return_preds:
            return preds, A_labels, probs
        else:
            return correct

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.loaded = True
        self.load_state_dict(torch.load(fn))

class FBDemParDisc(nn.Module):
    def __init__(self, embed_dim, a_idx, attribute='0', attribute_data=None,\
            use_cross_entropy=True):
        super(FBDemParDisc, self).__init__()
        self.embed_dim = int(embed_dim)
        self.a_idx = a_idx
        self.attribute = attribute
        if use_cross_entropy:
            self.cross_entropy = True
            self.out_dim = 1
        else:
            self.cross_entropy = False
            self.out_dim = 1
        self.W1 = nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True)
        self.W2 = nn.Linear(int(self.embed_dim * 2), int(self.embed_dim), bias=True)
        self.W3 = nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True)
        self.W4 = nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)

        if attribute_data is not None:
            self.attr_mat = np.array(pickle.load(open(attribute_data[0],'rb')))
            with open(attribute_data[1]) as f:
                self.ent_to_idx = json.load(f)
            f.close()
            with open(attribute_data[2]) as f:
                self.attr_to_idx = json.load(f)
            f.close()
            with open(attribute_data[3]) as f:
                self.reindex_to_idx = json.load(f)
            f.close()
            with open(attribute_data[4]) as f:
                self.attr_count = Counter(json.load(f))
            f.close()
            self.inv_attr_map = {v: k for k, v in self.attr_to_idx.items()}
            self.most_common = self.attr_count.most_common(50)
            self.sensitive_weight = 1-float(self.most_common[self.a_idx][1]) / sum(self.attr_count.values())
            self.weights = torch.Tensor((1-self.sensitive_weight,self.sensitive_weight)).cuda()

    def forward(self, ents_emb, ents, force_ce=False):
        h1 = F.leaky_relu(self.W1(ents_emb))
        h2 = F.leaky_relu(self.W2(h1))
        h3 = F.leaky_relu(self.W3(h2))
        scores = self.W4(h3)
        A_labels = Variable(torch.Tensor(self.attr_mat[ents][:,self.a_idx])).cuda()
        A_labels = A_labels.unsqueeze(1)
        if self.cross_entropy or force_ce:
            fair_penalty = F.binary_cross_entropy_with_logits(scores,\
                    A_labels,weight=self.weights)
        else:
            probs = torch.sigmoid(scores)
            fair_penalty = F.l1_loss(probs,A_labels,\
                    reduction='elementwise_mean')
        return fair_penalty

    def predict(self, ents_emb, ents, return_preds=False):
        h1 = F.leaky_relu(self.W1(ents_emb))
        h2 = F.leaky_relu(self.W2(h1))
        h3 = F.leaky_relu(self.W3(h2))
        scores = self.W4(h3)
        A_labels = Variable(torch.Tensor(self.attr_mat[ents][:,self.a_idx])).cuda()
        A_labels = A_labels.unsqueeze(1)
        probs = torch.sigmoid(scores)
        preds = (probs > torch.Tensor([0.5]).cuda()).float() * 1
        correct = preds.eq(A_labels.view_as(preds)).sum().item()
        if return_preds:
            return preds, A_labels
        else:
            return correct

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.loaded = True
        self.load_state_dict(torch.load(fn))

