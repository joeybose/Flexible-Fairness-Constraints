import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import math
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
USE_SPARSE_EMB = False

def apply_filters_gcmc(p_lhs_emb,masked_filter_set):
    ''' Doesnt Have Masked Filters yet '''
    filter_l_emb, filter_r_emb = 0,0
    for filter_ in masked_filter_set:
        if filter_ is not None:
            filter_l_emb += filter_(p_lhs_emb)
    return filter_l_emb

def apply_filters_single_node(p_lhs_emb,masked_filter_set):
    ''' Doesnt Have Masked Filters yet '''
    filter_l_emb, filter_r_emb = 0,0
    for filter_ in masked_filter_set:
        if filter_ is not None:
            filter_l_emb += filter_(p_lhs_emb)
    return filter_l_emb

def apply_filters_reddit(p_lhs_emb,masked_filter_set):
    ''' Doesnt Have Masked Filters yet '''
    filter_l_emb = 0
    for filter_ in masked_filter_set:
        if filter_ is not None:
            filter_l_emb += filter_(p_lhs_emb)
    return filter_l_emb

def apply_filters_transd(p_lhs_emb,p_rhs_emb,masked_filter_set):
    ''' Doesnt Have Masked Filters yet '''
    filter_l_emb = 0
    filter_r_emb = 0
    for filter_ in masked_filter_set:
        if filter_ is not None:
            filter_l_emb += filter_(p_lhs_emb)
            filter_r_emb += filter_(p_rhs_emb)
    return filter_l_emb,filter_r_emb

class RedditEncoder(nn.Module):
    def __init__(self, num_users, num_sr, embed_dim, p):
        super(RedditEncoder, self).__init__()
        self.num_users = num_users
        self.num_sr = num_sr
        self.embed_dim = embed_dim
        self.p = p

        r = 6 / np.sqrt(self.embed_dim)
        self.user_embeds = nn.Embedding(self.num_users,self.embed_dim, \
                max_norm=1, norm_type=2, sparse=USE_SPARSE_EMB)
        self.sr_embeds = nn.Embedding(self.num_sr,self.embed_dim, \
                max_norm=1, norm_type=2, sparse=USE_SPARSE_EMB)

        self.user_embeds.weight.data.uniform_(-r, r)
        self.sr_embeds.weight.data.uniform_(-r, r)

    def forward(self, batch, return_ent_emb=False, filters=None):
        users, sr = batch[:,0],batch[:,1]
        users_embed, sr_embed = self.encode(users,sr,filters)
        enrgs = -1*(users_embed*sr_embed).sum(dim=1)
        if not return_ent_emb:
            return enrgs
        else:
            return enrgs,users_embed,sr_embed

    def get_embed(self, users, filters=None):
        with torch.no_grad():
            user_embed = self.user_embeds(users)
            if filters is not None:
                constant = len(filters) - filters.count(None)
                if constant !=0:
                    user_embed = apply_filters_reddit(user_embed,filters)
        return user_embed

    def encode(self, users, sr, filters=None):
        user_embed = self.user_embeds(users)
        if filters is not None:
            constant = len(filters) - filters.count(None)
            if constant !=0:
                user_embed = apply_filters_reddit(user_embed,filters)
        sr_embed = self.sr_embeds(sr)
        return user_embed, sr_embed

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))

class RedditDiscriminator(nn.Module):
    def __init__(self,G,embed_dim,sensitive_sr,u_to_idx,use_cross_entropy=True):
        super(RedditDiscriminator, self).__init__()
        self.embed_dim = int(embed_dim)
        self.u_to_idx = u_to_idx
        self.G_neighbors = set(G.neighbors(sensitive_sr))
        self.attribute = sensitive_sr
        if use_cross_entropy:
            self.cross_entropy = True
        else:
            self.cross_entropy = False

        self.users_sensitive = self.create_sensitive_labels(self.G_neighbors,self.u_to_idx)
        self.out_dim = 1
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        self.num_correct = 0

        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 2 ), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim*4),bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 4), int(self.embed_dim*4),bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 4), int(self.embed_dim*2),bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim*2),bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim*2), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim), int(self.embed_dim/2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim / 2), self.out_dim,bias=True)
            )

    def create_sensitive_labels(self,neighbor_users,u_to_idx):
        filtered_neighbors = [n for n in neighbor_users if n.split('_')[0] == 'U']
        sensitive_users = [u_to_idx[user] for user in filtered_neighbors]
        all_users_attr = np.zeros(len(u_to_idx))
        all_users_attr[np.asarray(sensitive_users)] = 1
        return all_users_attr

    def forward(self, ents_emb, ents, return_loss=False):
        scores = self.net(ents_emb)
        output = self.sigmoid(scores)
        A_labels = Variable(torch.FloatTensor(self.users_sensitive[ents.cpu()])).cuda()
        if return_loss:
            loss = self.criterion(output.squeeze(), A_labels)
            return loss
        else:
            return output.squeeze(),A_labels

    def predict(self, ents_emb, ents, return_preds=False):
        with torch.no_grad():
            scores = self.net(ents_emb)
            output = self.sigmoid(scores)
            A_labels = Variable(torch.FloatTensor(self.users_sensitive[ents.cpu()])).cuda()
            preds = (output > torch.Tensor([0.5]).cuda()).float() * 1
        if return_preds:
            return output.squeeze(),A_labels,preds
        else:
            return output.squeeze(),A_labels

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))

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

    def forward(self, triplets, return_ent_emb=False):

        lhs_idxs = triplets[:, 0]
        rel_idxs = triplets[:, 1]
        rhs_idxs = triplets[:, 2]
        lhs_es = self.ent_embeds(lhs_idxs)
        rel_es = self.rel_embeds(rel_idxs)
        rhs_es = self.ent_embeds(rhs_idxs)

        enrgs = (lhs_es + rel_es - rhs_es).norm(p=self.p, dim=1)
        if not return_ent_emb:
            enrgs = (lhs_es + rel_es - rhs_es).norm(p=self.p, dim=1)
            return enrgs
        else:
            enrgs = (lhs_es + rel_es - rhs_es).norm(p=self.p, dim=1)
            return enrgs,lhs_es,rhs_es,rel_es
        return enrgs

    def get_embed(self, ents, rel_idxs=None):
        ent_embed = self.ent_embeds(ents)
        return ent_embed

    def encode(self, ents):
        ent_embed = self.ent_embeds(ents)
        return ent_embed

    def predict(self,heads,tails):
        with torch.no_grad():
            embeds1 = self.encode(heads)
            embeds2 = self.encode(tails)
            energs_list = []
            for i in range(0,self.num_rel):
                index = Variable(torch.LongTensor([i])).cuda()
                rel_es = self.rel_embeds(index)
                enrgs = (embeds1 + rel_es - embeds2).norm(p=self.p, dim=1)
                energs_list.append(enrgs)
            enrgs_outputs = torch.stack(energs_list,dim=1)
            outputs = F.log_softmax(enrgs_outputs,dim=1)
            probs = torch.exp(outputs)
            preds = outputs.max(1, keepdim=True)[1] # get the index of the max
            weighted_preds = 0
            for j in range(0,self.num_rel):
                index = Variable(torch.LongTensor([j])).cuda()
                ''' j+1 because of zero index '''
                weighted_preds+= (j+1)*torch.exp(torch.index_select(outputs, 1,index))
        return preds,weighted_preds,probs

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

    def forward(self, triplets, return_ent_emb=False, filters=None):
        lhs_idxs = triplets[:, 0]
        rel_idxs = triplets[:, 1]
        rhs_idxs = triplets[:, 2]

        rel_es = self.rel_embeds(rel_idxs)

        lhs = self.ent_embeds(lhs_idxs, rel_idxs)
        rhs = self.ent_embeds(rhs_idxs, rel_idxs)
        if filters is not None:
            constant = len(filters) - filters.count(None)
            if constant !=0:
                lhs,rhs = apply_filters_transd(lhs,rhs,filters)

        if not return_ent_emb:
            enrgs = (lhs + rel_es - rhs).norm(p=self.p, dim=1)
            return enrgs
        else:
            enrgs = (lhs + rel_es - rhs).norm(p=self.p, dim=1)
            return enrgs,lhs,rhs

    def get_embed(self, ents, rel_idxs, filters=None):
        with torch.no_grad():
            ent_embed = self.ent_embeds(ents, rel_idxs)
            if filters is not None:
                constant = len(filters) - filters.count(None)
                if constant !=0:
                    ents_embed = apply_filters_single_node(ents_embed,filters)
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
        self.batchnorm = nn.BatchNorm1d(self.embed_dim)

    def forward(self, ents_emb):
        h1 = F.leaky_relu(self.W1(ents_emb))
        h2 = F.leaky_relu(self.W2(h1))
        h2 = self.batchnorm(h2)
        return h2

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.loaded = True
        self.load_state_dict(torch.load(fn))

class BilinearDecoder(nn.Module):
    """
    Decoder where the relationship score is given by a bilinear form
    between the embeddings (i.e., one learned matrix per relationship type).
    """

    def __init__(self, num_relations, embed_dim):
        super(BilinearDecoder, self).__init__()
        self.rel_embeds = nn.Embedding(num_relations, embed_dim*embed_dim)
        self.embed_dim = embed_dim

    def forward(self, embeds1, embeds2, rels):
        rel_mats = self.rel_embeds(rels).reshape(-1, self.embed_dim, self.embed_dim)
        embeds1 = torch.matmul(embeds1, rel_mats)
        return (embeds1 * embeds2).sum(dim=1)

class SharedBilinearDecoder(nn.Module):
    """
    Decoder where the relationship score is given by a bilinear form
    between the embeddings (i.e., one learned matrix per relationship type).
    """

    def __init__(self, num_relations, num_weights, embed_dim):
        super(SharedBilinearDecoder, self).__init__()
        self.rel_embeds = nn.Embedding(num_weights, embed_dim*embed_dim)
        self.weight_scalars = nn.Parameter(torch.Tensor(num_weights,num_relations))
        stdv = 1. / math.sqrt(self.weight_scalars.size(1))
        self.weight_scalars.data.uniform_(-stdv, stdv)
        self.embed_dim = embed_dim
        self.num_weights = num_weights
        self.num_relations = num_relations
        self.nll = nn.NLLLoss()
        self.mse = nn.MSELoss()

    def predict(self,embeds1,embeds2):
        basis_outputs = []
        for i in range(0,self.num_weights):
            index = Variable(torch.LongTensor([i])).cuda()
            rel_mat = self.rel_embeds(index).reshape(self.embed_dim,\
                    self.embed_dim)
            u_Q = torch.matmul(embeds1, rel_mat)
            u_Q_v = (u_Q*embeds2).sum(dim=1)
            basis_outputs.append(u_Q_v)
        basis_outputs = torch.stack(basis_outputs,dim=1)
        logit = torch.matmul(basis_outputs,self.weight_scalars)
        outputs = F.log_softmax(logit,dim=1)
        preds = 0
        for j in range(0,self.num_relations):
            index = Variable(torch.LongTensor([j])).cuda()
            ''' j+1 because of zero index '''
            preds += (j+1)*torch.exp(torch.index_select(outputs, 1,index))
        return preds

    def forward(self, embeds1, embeds2, rels):
        basis_outputs = []
        for i in range(0,self.num_weights):
            index = Variable(torch.LongTensor([i])).cuda()
            rel_mat = self.rel_embeds(index).reshape(self.embed_dim,\
                    self.embed_dim)
            u_Q = torch.matmul(embeds1, rel_mat)
            u_Q_v = (u_Q*embeds2).sum(dim=1)
            basis_outputs.append(u_Q_v)
        basis_outputs = torch.stack(basis_outputs,dim=1)
        logit = torch.matmul(basis_outputs,self.weight_scalars)
        outputs = F.log_softmax(logit,dim=1)
        log_probs = torch.gather(outputs,1,rels.unsqueeze(1))
        loss = self.nll(outputs,rels)
        preds = 0
        for j in range(0,self.num_relations):
            index = Variable(torch.LongTensor([j])).cuda()
            ''' j+1 because of zero index '''
            preds += (j+1)*torch.exp(torch.index_select(outputs, 1,index))
        return loss,preds

class SimpleGCMC(nn.Module):
    def __init__(self, decoder, embed_dim, num_ent, p , encoder=None, attr_filter=None):
        super(SimpleGCMC, self).__init__()
        self.attr_filter = attr_filter
        self.decoder = decoder
        self.num_ent = num_ent
        self.embed_dim = embed_dim
        self.batchnorm = nn.BatchNorm1d(self.embed_dim)
        self.p = p
        if encoder is None:
            r = 6 / np.sqrt(self.embed_dim)
            self.encoder = nn.Embedding(self.num_ent, self.embed_dim,\
                    max_norm=1, norm_type=2)
            self.encoder.weight.data.uniform_(-r, r).renorm_(p=2, dim=1, maxnorm=1)
        else:
            self.encoder = encoder

    def encode(self, nodes, filters=None):
        embs = self.encoder(nodes)
        embs = self.batchnorm(embs)
        if filters is not None:
            constant = len(filters) - filters.count(None)
            if constant !=0:
                embs = apply_filters_gcmc(embs,filters)
        return embs

    def predict_rel(self,heads,tails_embed,filters=None):
        with torch.no_grad():
            head_embeds = self.encode(heads)
            if filters is not None:
                constant = len(filters) - filters.count(None)
                if constant !=0:
                    head_embeds = apply_filters_gcmc(head_embeds,filters)
            preds = self.decoder.predict(head_embeds,tail_embeds)
        return preds

    def forward(self, pos_edges, weights=None, return_embeds=False, filters=None):
        pos_head_embeds = self.encode(pos_edges[:,0])
        if filters is not None:
            constant = len(filters) - filters.count(None)
            if constant !=0:
                pos_head_embeds = apply_filters_gcmc(pos_head_embeds,filters)
        pos_tail_embeds = self.encode(pos_edges[:,-1])
        rels = pos_edges[:,1]
        loss, preds = self.decoder(pos_head_embeds, pos_tail_embeds, rels)
        if return_embeds:
            return loss, preds, pos_head_embeds, pos_tail_embeds
        else:
            return loss, preds

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))

class RandomDiscriminator(nn.Module):
    def __init__(self,use_1M,embed_dim,attribute_data,attribute,use_cross_entropy=True):
        super(RandomDiscriminator, self).__init__()
        self.embed_dim = int(embed_dim)
        self.attribute_data = attribute_data
        self.attribute = attribute
        if use_cross_entropy:
            self.cross_entropy = True
        else:
            self.cross_entropy = False

        users_random = attribute_data[0]['rand']
        self.users_sensitive = np.ascontiguousarray(users_random)
        self.out_dim = 1
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 2 ), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim*4),bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 4), int(self.embed_dim*4),bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 4), int(self.embed_dim*2),bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim*2),bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim*2), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim), int(self.embed_dim/2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim / 2), self.out_dim,bias=True)
            )

    def forward(self, ents_emb, ents, return_loss=False):
        scores = self.net(ents_emb)
        output = self.sigmoid(scores)
        A_labels = Variable(torch.FloatTensor(self.users_sensitive[ents.cpu()])).cuda()
        if return_loss:
            loss = self.criterion(output.squeeze(), A_labels)
            return loss
        else:
            return output.squeeze(),A_labels

    def predict(self, ents_emb, ents, return_preds=False):
        with torch.no_grad():
            scores = self.net(ents_emb)
            output = self.sigmoid(scores)
            A_labels = Variable(torch.FloatTensor(self.users_sensitive[ents.cpu()])).cuda()
            preds = (output > torch.Tensor([0.5]).cuda()).float() * 1
        if return_preds:
            return output.squeeze(),A_labels,preds
        else:
            return output.squeeze(),A_labels

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))

class GenderDiscriminator(nn.Module):
    def __init__(self,use_1M,embed_dim,attribute_data,attribute,use_cross_entropy=True):
        super(GenderDiscriminator, self).__init__()
        self.embed_dim = int(embed_dim)
        self.attribute_data = attribute_data
        self.attribute = attribute
        if use_cross_entropy:
            self.cross_entropy = True
        else:
            self.cross_entropy = False

        users_sex = attribute_data[0]['sex']
        users_sex = [0 if i == 'M' else 1 for i in users_sex]
        self.users_sensitive = np.ascontiguousarray(users_sex)
        self.out_dim = 1
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 2 ), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim*4),bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 4), int(self.embed_dim*4),bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 4), int(self.embed_dim*2),bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim*2),bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim*2), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim), int(self.embed_dim/2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim / 2), self.out_dim,bias=True)
            )

    def forward(self, ents_emb, ents, return_loss=False):
        scores = self.net(ents_emb)
        output = self.sigmoid(scores)
        A_labels = Variable(torch.FloatTensor(self.users_sensitive[ents.cpu()])).cuda()
        if return_loss:
            loss = self.criterion(output.squeeze(), A_labels)
            return loss
        else:
            return output.squeeze(),A_labels

    def predict(self, ents_emb, ents, return_preds=False):
        with torch.no_grad():
            scores = self.net(ents_emb)
            output = self.sigmoid(scores)
            A_labels = Variable(torch.FloatTensor(self.users_sensitive[ents.cpu()])).cuda()
            preds = (output > torch.Tensor([0.5]).cuda()).float() * 1
        if return_preds:
            return output.squeeze(),A_labels,preds
        else:
            return output.squeeze(),A_labels

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))

class AgeDiscriminator(nn.Module):
    def __init__(self,use_1M,embed_dim,attribute_data,attribute,use_cross_entropy=True):
        super(AgeDiscriminator, self).__init__()
        self.embed_dim = int(embed_dim)
        self.attribute_data = attribute_data
        self.attribute = attribute
        self.criterion = nn.NLLLoss()
        if use_cross_entropy:
            self.cross_entropy = True
        else:
            self.cross_entropy = False

        users_age = attribute_data[0]['age'].values
        users_age_list = sorted(set(users_age))
        if not use_1M:
            bins = np.linspace(5, 75, num=15, endpoint=True)
            inds = np.digitize(users_age, bins) - 1
            self.users_sensitive = np.ascontiguousarray(inds)
            self.out_dim = len(bins)
        else:
            reindex = {1:0, 18:1, 25:2, 35:3, 45:4, 50:5, 56:6}
            inds = [reindex.get(n, n) for n in users_age]
            self.users_sensitive = np.ascontiguousarray(inds)
            self.out_dim = 7

        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 2 ), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim*4),bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 4), int(self.embed_dim*2),bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim*2),bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim*2), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim), int(self.embed_dim/2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim / 2), self.out_dim,bias=True)
            )

    def forward(self, ents_emb, ents, return_loss=False):
        scores = self.net(ents_emb)
        output = F.log_softmax(scores, dim=1)
        A_labels = Variable(torch.LongTensor(self.users_sensitive[ents.cpu()])).cuda()
        if return_loss:
            loss = self.criterion(output.squeeze(), A_labels)
            return loss
        else:
            return output.squeeze(),A_labels

    def predict(self, ents_emb, ents, return_preds=False):
        with torch.no_grad():
            scores = self.net(ents_emb)
            output = F.log_softmax(scores, dim=1)
            A_labels = Variable(torch.LongTensor(self.users_sensitive[ents.cpu()])).cuda()
            preds = output.max(1, keepdim=True)[1] # get the index of the max
        if return_preds:
            return output.squeeze(),A_labels,preds
        else:
            return output.squeeze(),A_labels

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))

class OccupationDiscriminator(nn.Module):
    def __init__(self,use_1M,embed_dim,attribute_data,attribute,use_cross_entropy=True):
        super(OccupationDiscriminator, self).__init__()
        self.embed_dim = int(embed_dim)
        self.attribute_data = attribute_data
        self.attribute = attribute
        self.criterion = nn.NLLLoss()
        if use_cross_entropy:
            self.cross_entropy = True
        else:
            self.cross_entropy = False

        users_occupation = attribute_data[0]['occupation']
        if use_1M:
            self.users_sensitive = np.ascontiguousarray(users_occupation.values)
            self.out_dim = 21
        else:
            users_occupation_list = sorted(set(users_occupation))
            occ_to_idx = {}
            for i, occ in enumerate(users_occupation_list):
                occ_to_idx[occ] = i
            users_occupation = [occ_to_idx[occ] for occ in users_occupation]
            self.users_sensitive = np.ascontiguousarray(users_occupation)
            self.out_dim = len(users_occupation_list)

        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim *2 ), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim*4),bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 4), int(self.embed_dim*2),bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim*2), int(self.embed_dim*2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim*2), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim), int(self.embed_dim/2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim / 2), self.out_dim,bias=True)
            )

    def forward(self, ents_emb, ents, return_loss=False):
        scores = self.net(ents_emb)
        output = F.log_softmax(scores, dim=1)
        A_labels = Variable(torch.LongTensor(self.users_sensitive[ents.cpu()])).cuda()
        if return_loss:
            loss = self.criterion(output.squeeze(), A_labels)
            return loss
        else:
            return output.squeeze(),A_labels

    def predict(self, ents_emb, ents, return_preds=False):
        with torch.no_grad():
            scores = self.net(ents_emb)
            output = F.log_softmax(scores, dim=1)
            A_labels = Variable(torch.LongTensor(self.users_sensitive[ents.cpu()])).cuda()
            preds = output.max(1, keepdim=True)[1] # get the index of the max
        if return_preds:
            return output.squeeze(),A_labels,preds
        else:
            return output.squeeze(),A_labels

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))

class DemParDisc2(nn.Module):
    def __init__(self, use_1M,  embed_dim, attribute_data,attribute='gender',use_cross_entropy=True):
        super(DemParDisc2, self).__init__()
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
            self.out_dim = 2
        elif attribute == 'occupation':
            users_occupation = attribute_data[0]['occupation']
            if use_1M:
                self.users_sensitive = np.ascontiguousarray(users_occupation.values)
                self.out_dim = 21
            else:
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
            self.out_dim = 2
        else:
            users_age = attribute_data[0]['age'].values
            users_age_list = sorted(set(users_age))
            if not use_1M:
                bins = np.linspace(5, 75, num=15, endpoint=True)
                inds = np.digitize(users_age, bins) - 1
                self.users_sensitive = np.ascontiguousarray(inds)
                self.out_dim = len(bins)
            else:
                reindex = {1:0, 18:1, 25:2, 35:3, 45:4, 50:5, 56:6}
                inds = [reindex.get(n, n) for n in users_age]
                self.users_sensitive = np.ascontiguousarray(inds)
                self.out_dim = 7
        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 4), bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(int(self.embed_dim * 4), int(self.embed_dim*2),bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim),bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(int(self.embed_dim), int(self.embed_dim/2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(int(self.embed_dim / 2), self.out_dim,bias=True)
            )
        self.attribute = attribute

    def forward(self, ents_emb, ents):
        scores = self.net(ents_emb)
        output = F.log_softmax(scores, dim=1)
        A_labels = Variable(torch.LongTensor(self.users_sensitive[ents])).cuda()
        loss = F.nll_loss(output, A_labels)
        # if self.attribute == 'gender' or self.attribute == 'random':
            # A_labels = Variable(torch.Tensor(self.users_sensitive[ents])).cuda()
            # A_labels = A_labels.unsqueeze(1)
            # loss = F.binary_cross_entropy_with_logits(scores,A_labels)
        # else:
            # output = F.log_softmax(scores, dim=1)
            # A_labels = Variable(torch.LongTensor(self.users_sensitive[ents])).cuda()
            # loss = F.nll_loss(output, A_labels)
        return loss

    def predict(self, ents_emb, ents, return_preds=False):
        scores = self.net(ents_emb)
        A_labels = Variable(torch.LongTensor(self.users_sensitive[ents])).cuda()
        output = F.log_softmax(scores, dim=1)
        probs = torch.exp(output)
        preds = output.max(1, keepdim=True)[1] # get the index of the max
        # if self.attribute == 'gender':
            # A_labels = A_labels.unsqueeze(1)
            # probs = F.sigmoid(scores)
            # preds = (probs > torch.Tensor([0.5]).cuda()).float() * 1
        # else:
            # output = F.log_softmax(scores, dim=1)
            # probs = torch.exp(output)
            # preds = output.max(1, keepdim=True)[1] # get the index of the max
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

class DemParDisc(nn.Module):
    def __init__(self, use_1M,  embed_dim, attribute_data,attribute='gender',use_cross_entropy=True):
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
            if not use_1M:
                bins = np.linspace(5, 75, num=15, endpoint=True)
                inds = np.digitize(users_age, bins) - 1
                self.users_sensitive = np.ascontiguousarray(inds)
                self.out_dim = len(bins)
            else:
                reindex = {1:0, 18:1, 25:2, 35:3, 45:4, 50:5, 56:6}
                inds = [reindex.get(n, n) for n in users_age]
                self.users_sensitive = np.ascontiguousarray(inds)
                self.out_dim = 7

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
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

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

    def forward(self, ents_emb, ents, return_loss=False):
        h1 = F.leaky_relu(self.W1(ents_emb))
        h2 = F.leaky_relu(self.W2(h1))
        h3 = F.leaky_relu(self.W3(h2))
        scores = self.W4(h3)
        output = self.sigmoid(scores)
        A_labels = Variable(torch.Tensor(self.attr_mat[ents][:,self.a_idx])).cuda()
        # A_labels = A_labels.unsqueeze(1)
        if return_loss:
            loss = self.criterion(output.squeeze(), A_labels)
            return loss
        else:
            return output.squeeze(),A_labels
        # if self.cross_entropy or force_ce:
            # # fair_penalty = F.binary_cross_entropy_with_logits(scores,\
                    # # A_labels)#,weight=self.weights)
            # loss = self.criterion(output.squeeze(), A_labels)
            # return loss
        # else:
            # probs = torch.sigmoid(scores)
            # fair_penalty = F.l1_loss(probs,A_labels,\
                    # reduction='elementwise_mean')
        # return fair_penalty

    def predict(self, ents_emb, ents, return_preds=False):
        with torch.no_grad():
            h1 = F.leaky_relu(self.W1(ents_emb))
            h2 = F.leaky_relu(self.W2(h1))
            h3 = F.leaky_relu(self.W3(h2))
            scores = self.W4(h3)
            output = self.sigmoid(scores)
            A_labels = Variable(torch.Tensor(self.attr_mat[ents][:,self.a_idx])).cuda()
            # A_labels = A_labels.unsqueeze(1)
            preds = (output > torch.Tensor([0.5]).cuda()).float() * 1
        correct = preds.eq(A_labels.view_as(preds)).sum().item()
        if return_preds:
            return preds, A_labels, output.squeeze()
        else:
            return correct

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.loaded = True
        self.load_state_dict(torch.load(fn))

