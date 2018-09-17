"""
An implementation of TransE model in PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.nn.init import xavier_normal, xavier_uniform
from torch.distributions import Categorical

import numpy as np
import random
import argparse
import pickle
import json
import logging
import sys, os
import subprocess
from tqdm import tqdm
from utils import create_or_append, compute_rank
import joblib
import ipdb
sys.path.append('../')
import gc

from collections import OrderedDict

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

    #@profile
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

    def forward(self, triplets):

        lhs_idxs = triplets[:, 0]
        rel_idxs = triplets[:, 1]
        rhs_idxs = triplets[:, 2]


        rel_es = self.rel_embeds(rel_idxs)

        lhs = self.ent_embeds(lhs_idxs, rel_idxs)
        rhs = self.ent_embeds(rhs_idxs, rel_idxs)

        enrgs = (lhs + rel_es - rhs).norm(p=self.p, dim=1)
        return enrgs

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))

class MarginRankingLoss(nn.Module):
    def __init__(self, margin):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin

    ##@profile
    def forward(self, p_enrgs, n_enrgs, weights=None):
        scores = (self.margin + p_enrgs - n_enrgs).clamp(min=0)

        if weights is not None:
            scores = scores * weights / weights.mean()

        return scores.mean(), scores

class VanillaReinforceLoss(nn.Module):
    def forward(self, raw_penalty, log_a_p, weights=None):

        vv = log_a_p * raw_penalty
        if weights is not None:
            vv = vv * weights

        reinforce_loss = torch.mean(vv)
        return reinforce_loss

class IWReinforceLoss(nn.Module):

    def forward(self, nce_penalty, log_a_q, log_a_p, weights=None):

        ratio = torch.exp((log_a_p.detach() - log_a_q))

        #logging.info(ratio.mean().data.cpu().numpy()[0])
        vv = nce_penalty * log_a_p * ratio

        if weights is not None:
            reinforce_loss = (vv * weights) / weights.sum()
        else:
            reinforce_loss = torch.mean(vv)

        return reinforce_loss


class KBDataset(Dataset):
    def __init__(self, path, prefetch_to_gpu=False):
        self.prefetch_to_gpu = prefetch_to_gpu

        if isinstance(path, str):
            self.dataset = np.ascontiguousarray(np.array(pickle.load(open(path, 'rb'))))
        elif isinstance(path, np.ndarray):
            self.dataset = np.ascontiguousarray(path)
        else:
            raise ValueError()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def shuffle(self):
        if self.dataset.is_cuda:
            self.dataset = self.dataset.cpu()

        data = self.dataset.numpy()
        np.random.shuffle(data)
        data = np.ascontiguousarray(data)
        self.dataset = ltensor(data)

        if self.prefetch_to_gpu:
            self.dataset = self.dataset.cuda().contiguous()


def collate_fn(batch):
    if isinstance(batch, np.ndarray) or (isinstance(batch, list) and isinstance(batch[0], np.ndarray)):
        return ltensor(batch).contiguous()
    else:
        return torch.stack(batch).contiguous()


_cb_var = []
def corrupt_batch(batch, num_ent):
    # batch: ltensor type, contains positive triplets
    batch_size, _ = batch.size()

    corrupted = batch.clone()

    if len(_cb_var) == 0:
        _cb_var.append(ltensor(batch_size//2).cuda())

    q_samples_l = _cb_var[0].random_(0, num_ent)
    q_samples_r = _cb_var[0].random_(0, num_ent)

    corrupted[:batch_size//2, 0] = q_samples_l
    corrupted[batch_size//2:, 2] = q_samples_r

    return corrupted.contiguous(), torch.cat([q_samples_l, q_samples_r])


class TransEGen(nn.Module):

    def __init__(self, D_model, num_ent, num_rel, Z_dim=100, \
                 embedding_size=50, embedding_size_g=5, share_D_embedding=False, \
                 sparse=USE_SPARSE_EMB, entropy_reg=0., **kwargs):

        super(TransEGen, self).__init__(**kwargs)

        self.D_model = [D_model]
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.Z_dim = Z_dim
        self.embedding_size = embedding_size
        self.embedding_size_g = embedding_size_g
        self.sparse = sparse
        self.entropy_reg = entropy_reg
        self.share_D_embedding = share_D_embedding

        if not self.share_D_embedding:
            self.ent_embeds = nn.Embedding(self.num_ent, self.embedding_size, max_norm=1, norm_type=2)
            self.rel_embeds = nn.Embedding(self.num_rel, self.embedding_size)

            r = 6 / np.sqrt(self.embedding_size)
            self.ent_embeds.weight.data.uniform_(-r, r)#.renorm_(p=2, dim=1, maxnorm=1)
            self.rel_embeds.weight.data.uniform_(-r, r)#.renorm_(p=2, dim=1, maxnorm=1)

        self.G_W1 = nn.Linear(self.Z_dim+self.embedding_size, self.embedding_size_g, bias=True)
        self.G_W2 = nn.Linear(self.embedding_size_g, self.num_ent, bias=True)


        self.use_cuda = torch.cuda.is_available()
        self.softmax = nn.Softmax(dim=1)
        self.h_nonlinearity = nn.LeakyReLU()
        self._inds = None

    ##@profile
    def _sample(self, z, ents, rels, q_samples=None, mode=1, n_samples=1):

        if self.share_D_embedding:

            if isinstance(self.D_model[0], TransE):
                ent_es = self.D_model[0].ent_embeds(ents).detach()
                rel_es = self.D_model[0].rel_embeds(rels).detach()
            elif isinstance(self.D_model[0], TransD):
                ent_es = self.D_model[0].ent_embeds(ents, rels).detach()
                rel_es = self.D_model[0].rel_embeds(rels).detach()

        else:
            ent_es = self.ent_embeds(ents)
            rel_es = self.rel_embeds(rels)

        if mode == 1:
            inputs = torch.cat([z, ent_es - rel_es], 1)
        else:
            inputs = torch.cat([z, ent_es + rel_es], 1)

        G_h = self.h_nonlinearity(self.G_W1(inputs))
        G_h = self.G_W2(G_h)

        gen_probs = self.softmax(G_h)
        gen_probs = gen_probs + 1e-3 / self.num_ent
        gen_probs = gen_probs / gen_probs.sum(dim=-1, keepdim=True)

        assert (gen_probs > 0).all()

        m = Categorical(gen_probs)
        g_samples = m.sample()

        _, greedy_samples = gen_probs.max(dim=-1)

        if n_samples > 1:
            g_samples = torch.stack([m.sample() for _ in xrange(n_samples)])
            return None, None, None,  g_samples, None, greedy_samples, None
        else:
            log_a_p = m.log_prob(g_samples)

            if self._inds is None:
                self._inds = torch.LongTensor(np.arange(len(z))).cuda()

            if q_samples is not None:
                log_action_q = m.log_prob(q_samples)
                e_qs = -torch.log(gen_probs[self._inds, q_samples])
            else:
                log_action_q = None
                e_qs = None

            e_ps = -torch.log(gen_probs[self._inds, g_samples])

            if self.entropy_reg:
                neg_entropy = (gen_probs * torch.log(gen_probs)).sum(dim=-1).mean()
            else:
                neg_entropy = 0.

            return log_a_p, log_action_q, e_qs,  g_samples, e_ps, greedy_samples, neg_entropy

    ##@profile
    def forward(self, z, batch, q_samples=None, n_samples=1):

        batch_size, _ = batch.size()
        corrupted = batch.detach().clone()
        s = batch_size//2

        ents_l = corrupted[:s, 0]
        ents_r = corrupted[s:, 2]

        if q_samples is None:
            q_samples_l = q_samples_r = q_samples
        else:
            q_samples_l = q_samples[:s]
            q_samples_r = q_samples[s:]

        results_l = self._sample(z[:s,:], ents_r, corrupted[:s,1], mode=1, q_samples=q_samples_l, n_samples=n_samples)
        results_r = self._sample(z[s:,:], ents_l, corrupted[s:,1], mode=2, q_samples=q_samples_r, n_samples=n_samples)

        log_a_probs_l, log_a_q_l, e_q_l, g_samples_l, e_p_l, greedy_samples_l, neg_ent_l = results_l
        log_a_probs_r, log_a_q_r, e_q_r, g_samples_r, e_p_r, greedy_samples_r, neg_ent_r = results_r

        greedy_corrupted = corrupted.clone()
        greedy_corrupted[:s, 0]  = greedy_samples_l
        greedy_corrupted[s:, 2]  = greedy_samples_r




        if n_samples > 1:
            corrupted2 = corrupted.repeat(n_samples,1).resize_(n_samples,batch_size,3)
            corrupted2[:,:s,0] = g_samples_l
            corrupted2[:,s:,2] = g_samples_r

            return None, None, None, corrupted2, None, greedy_corrupted, None
        else:
            e_ps = torch.cat([e_p_l, e_p_r], 0)
            log_probs = torch.cat([log_a_probs_l, log_a_probs_r], 0)

            if q_samples is not None:
                log_qs = torch.cat([log_a_q_l, log_a_q_r], 0)
                e_qs = torch.cat([e_q_l, e_q_r], 0)
            else:
                log_qs = None
                e_qs = None

            corrupted[:s, 0]  = g_samples_l #.data
            corrupted[s:, 2]  = g_samples_r #.data

            neg_entropy = (neg_ent_l + neg_ent_r) / 2
            return log_probs, log_qs, e_qs, corrupted, e_ps, greedy_corrupted, neg_entropy

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))


class TransEGenV2(nn.Module):

    def __init__(self, D_model, num_ent, num_rel, \
                 embedding_size=50, embedding_size_g=50, share_D_embedding=False, \
                 sparse=USE_SPARSE_EMB, entropy_reg=0., **kwargs):

        super(TransEGenV2, self).__init__(**kwargs)

        self.D_model = [D_model]
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.Z_dim = embedding_size
        self.embedding_size = embedding_size
        self.conv_stride = embedding_size // 25
        self.conv_filter = self.conv_stride * 4 + 1

        self.embedding_size_g = embedding_size
        self.sparse = sparse
        self.entropy_reg = entropy_reg
        self.share_D_embedding = share_D_embedding

        if not self.share_D_embedding:
            raise NotImplementedError()

            self.ent_embeds = nn.Embedding(self.num_ent, self.embedding_size, max_norm=1, norm_type=2)
            self.rel_embeds = nn.Embedding(self.num_rel, self.embedding_size)

            r = 6 / np.sqrt(self.embedding_size)
            self.ent_embeds.weight.data.uniform_(-r, r)#.renorm_(p=2, dim=1, maxnorm=1)
            self.rel_embeds.weight.data.uniform_(-r, r).renorm_(p=2, dim=1, maxnorm=1)

        self.conv1 = torch.nn.Conv1d(2, 64, self.conv_filter, self.conv_stride, 0, bias=True)
        self.lstm = nn.LSTM(64, 32, 1, batch_first=True, bidirectional=True)
        self.init_state_dict = {}

        #self.G_W1 = nn.Linear(self.Z_dim+2*self.embedding_size, self.embedding_size_g, bias=True)
        self.fc = nn.Linear(64, self.num_ent, bias=True)


        self.use_cuda = torch.cuda.is_available()
        self.softmax = nn.Softmax(dim=1)
        self.h_nonlinearity = nn.LeakyReLU()
        self._inds = None

    def init(self):

        xavier_normal(self.conv1.weight.data)

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)

        xavier_normal(self.fc.weight)

    #@profile
    def _sample(self, z, ents, rels, q_samples=None, mode=1, n_samples=1):

        if self.share_D_embedding:
            if isinstance(self.D_model[0], TransE):
                ent_es = self.D_model[0].ent_embeds(ents).detach()
                rel_es = self.D_model[0].rel_embeds(rels).detach()
            elif isinstance(self.D_model[0], TransD):
                ent_es = self.D_model[0].ent_embeds(ents, rels).detach()
                rel_es = self.D_model[0].rel_embeds(rels).detach()
        else:
            ent_es = self.ent_embeds(ents)
            rel_es = self.rel_embeds(rels)

        if mode == 1:
            inputs = torch.cat([z, ent_es - rel_es], 1)
        else:
            inputs = torch.cat([z, ent_es + rel_es], 1)

        inputs = inputs.view(-1, 2, self.embedding_size)
        x = self.conv1(inputs)
        x = self.h_nonlinearity(x)

        pre_lstm_x = x
        x = x.permute((0,2,1))

        if x.shape[0] not in self.init_state_dict:
            h0 = Variable(torch.zeros(2, x.size(0), 32))
            c0 = Variable(torch.zeros(2, x.size(0), 32))

            h0 = h0.cuda()
            c0 = c0.cuda()

            self.init_state_dict[x.shape[0]] = (h0, c0)

        h0, c0 = self.init_state_dict[x.shape[0]]
        out, _ = self.lstm(x, (h0, c0))

        lstm_max = out.mean(dim=1)
        conv_max = pre_lstm_x.mean(dim=2)
        x =  lstm_max + conv_max

        G_h = self.fc(x)

        gen_probs = self.softmax(G_h)
        gen_probs = gen_probs + 1e-3 / self.num_ent
        gen_probs = gen_probs / gen_probs.sum(dim=-1, keepdim=True)

        assert (gen_probs > 0).all()

        m = Categorical(gen_probs)
        g_samples = m.sample()

        _, greedy_samples = gen_probs.max(dim=-1)

        if n_samples > 1:
            g_samples = torch.stack([m.sample() for _ in xrange(n_samples)])
            return None, None, None,  g_samples, None, greedy_samples, None
        else:
            log_a_p = m.log_prob(g_samples)

            if self._inds is None:
                self._inds = torch.LongTensor(np.arange(len(z))).cuda()

            if q_samples is not None:
                log_action_q = m.log_prob(q_samples)
                e_qs = -torch.log(gen_probs[self._inds, q_samples])
            else:
                log_action_q = None
                e_qs = None

            e_ps = -torch.log(gen_probs[self._inds, g_samples])

            if self.entropy_reg:
                neg_entropy = (gen_probs * torch.log(gen_probs)).sum(dim=-1).mean()
            else:
                neg_entropy = 0.

            return log_a_p, log_action_q, e_qs,  g_samples, e_ps, greedy_samples, neg_entropy

    #@profile
    def forward(self, z, batch, q_samples=None, n_samples=1):

        batch_size, _ = batch.size()
        corrupted = batch.detach().clone()
        s = batch_size//2

        ents_l = corrupted[:s, 0]
        ents_r = corrupted[s:, 2]

        if q_samples is None:
            q_samples_l = q_samples_r = q_samples
        else:
            q_samples_l = q_samples[:s]
            q_samples_r = q_samples[s:]

        results_l = self._sample(z[:s,:], ents_r, corrupted[:s,1], mode=1, q_samples=q_samples_l, n_samples=n_samples)
        results_r = self._sample(z[s:,:], ents_l, corrupted[s:,1], mode=2, q_samples=q_samples_r, n_samples=n_samples)

        log_a_probs_l, log_a_q_l, e_q_l, g_samples_l, e_p_l, greedy_samples_l, neg_ent_l = results_l
        log_a_probs_r, log_a_q_r, e_q_r, g_samples_r, e_p_r, greedy_samples_r, neg_ent_r = results_r

        greedy_corrupted = corrupted.clone()
        greedy_corrupted[:s, 0]  = greedy_samples_l
        greedy_corrupted[s:, 2]  = greedy_samples_r


        e_ps = torch.cat([e_p_l, e_p_r], 0)
        log_probs = torch.cat([log_a_probs_l, log_a_probs_r], 0)

        if n_samples > 1:
            corrupted2 = corrupted.repeat(n_samples,1).resize_(n_samples,batch_size,3)
            corrupted2[:,:s,0] = g_samples_l
            corrupted2[:,s:,2] = g_samples_r

            return log_probs, None, None, corrupted2, None, greedy_corrupted, None
        else:

            if q_samples is not None:
                log_qs = torch.cat([log_a_q_l, log_a_q_r], 0)
                e_qs = torch.cat([e_q_l, e_q_r], 0)
            else:
                log_qs = None
                e_qs = None

            corrupted[:s, 0]  = g_samples_l #.data
            corrupted[s:, 2]  = g_samples_r #.data

            neg_entropy = (neg_ent_l + neg_ent_r) / 2
            return log_probs, log_qs, e_qs, corrupted, e_ps, greedy_corrupted, neg_entropy

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))



class TransEGenV3(nn.Module):

    def __init__(self, D_model, num_ent, num_rel, \
                 embedding_size=50, embedding_size_g=50, share_D_embedding=False, \
                 sparse=USE_SPARSE_EMB, entropy_reg=0., **kwargs):

        super(TransEGenV3, self).__init__(**kwargs)

        self.D_model = [D_model]
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.Z_dim = 128
        self.embedding_size = embedding_size
        self.conv_stride = embedding_size // 25
        self.conv_filter = self.conv_stride * 2 + 1

        self.embedding_size_g = embedding_size_g
        self.sparse = sparse
        self.entropy_reg = entropy_reg
        self.share_D_embedding = share_D_embedding

        if not self.share_D_embedding:
            self.ent_embeds = nn.Embedding(self.num_ent, self.embedding_size, max_norm=1, norm_type=2)
            self.rel_embeds = nn.Embedding(self.num_rel, self.embedding_size)

            r = 6 / np.sqrt(self.embedding_size)
            self.ent_embeds.weight.data.uniform_(-r, r)#.renorm_(p=2, dim=1, maxnorm=1)
            self.rel_embeds.weight.data.uniform_(-r, r).renorm_(p=2, dim=1, maxnorm=1)

        self.G_W1 = nn.Linear(2*self.embedding_size, self.embedding_size, bias=True)
        self.G_W2 = nn.Linear(self.embedding_size, self.embedding_size_g, bias=True)
        self.G_W3 = nn.Linear(self.embedding_size_g, self.num_ent, bias=True)

        self.drop = torch.nn.Dropout(.1)

        self.use_cuda = torch.cuda.is_available()
        self.softmax = nn.Softmax(dim=1)
        self.h_nonlinearity = nn.LeakyReLU()
        self._inds = None
        self.z_var = None

    def init(self):

        xavier_normal(self.conv1.weight.data)
        xavier_normal(self.conv2_c.weight.data)
        xavier_normal(self.conv2_xy.weight.data)

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 1.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)

        xavier_normal(self.fc.weight)

    #@profile
    def _sample(self, ents, rels, q_samples=None, mode=1, n_samples=1):

        if self.share_D_embedding:
            if isinstance(self.D_model[0], TransE):
                ent_es = self.D_model[0].ent_embeds(ents).detach()
                rel_es = self.D_model[0].rel_embeds(rels).detach()
            elif isinstance(self.D_model[0], TransD):
                ent_es = self.D_model[0].ent_embeds(ents, rels).detach()
                rel_es = self.D_model[0].rel_embeds(rels).detach()
        else:
            ent_es = self.ent_embeds(ents)
            rel_es = self.rel_embeds(rels)

        if mode == 1:
            inputs = torch.cat([ent_es, ent_es - rel_es], dim=1)
        else:
            inputs = torch.cat([ent_es, ent_es + rel_es], dim=1)

        x = self.G_W1(inputs)
        x = self.h_nonlinearity(x)
        x = self.drop(x)

        x = self.G_W2(x)
        x = self.h_nonlinearity(x)
        x = self.drop(x)

        x = self.G_W3(x)
        x = self.h_nonlinearity(x)
        G_h = self.drop(x)

        if self.z_var is None:
            self.z_var = Variable(torch.zeros(*G_h.shape)).cuda()

        G_h = G_h + self.z_var.normal(0., .1*G_h.std().detach())

        gen_probs = self.softmax(G_h)
        gen_probs = gen_probs + 1e-3 / self.num_ent
        gen_probs = gen_probs / gen_probs.sum(dim=-1, keepdim=True)

        assert (gen_probs > 0).all()

        m = Categorical(gen_probs)
        g_samples = m.sample()

        _, greedy_samples = gen_probs.max(dim=-1)

        if n_samples > 1:
            g_samples = torch.stack([m.sample() for _ in xrange(n_samples)])
            return None, None, None,  g_samples, None, greedy_samples, None
        else:
            log_a_p = m.log_prob(g_samples)

            if self._inds is None:
                self._inds = torch.LongTensor(np.arange(len(ents))).cuda()

            if q_samples is not None:
                log_action_q = m.log_prob(q_samples)
                e_qs = -torch.log(gen_probs[self._inds, q_samples])
            else:
                log_action_q = None
                e_qs = None

            e_ps = -torch.log(gen_probs[self._inds, g_samples])

            if self.entropy_reg:
                neg_entropy = (gen_probs * torch.log(gen_probs)).sum(dim=-1).mean()
            else:
                neg_entropy = 0.

            return log_a_p, log_action_q, e_qs,  g_samples, e_ps, greedy_samples, neg_entropy

    #@profile
    def forward(self, z, batch, q_samples=None, n_samples=1):

        batch_size, _ = batch.size()
        corrupted = batch.detach().clone()
        s = batch_size//2

        ents_l = corrupted[:s, 0]
        ents_r = corrupted[s:, 2]

        if q_samples is None:
            q_samples_l = q_samples_r = q_samples
        else:
            q_samples_l = q_samples[:s]
            q_samples_r = q_samples[s:]

        results_l = self._sample(ents_r, corrupted[:s,1], mode=1, q_samples=q_samples_l, n_samples=n_samples)
        results_r = self._sample(ents_l, corrupted[s:,1], mode=2, q_samples=q_samples_r, n_samples=n_samples)

        log_a_probs_l, log_a_q_l, e_q_l, g_samples_l, e_p_l, greedy_samples_l, neg_ent_l = results_l
        log_a_probs_r, log_a_q_r, e_q_r, g_samples_r, e_p_r, greedy_samples_r, neg_ent_r = results_r

        greedy_corrupted = corrupted.clone()
        greedy_corrupted[:s, 0]  = greedy_samples_l
        greedy_corrupted[s:, 2]  = greedy_samples_r




        if n_samples > 1:
            corrupted2 = corrupted.repeat(n_samples,1).resize_(n_samples,batch_size,3)
            corrupted2[:,:s,0] = g_samples_l
            corrupted2[:,s:,2] = g_samples_r

            return None, None, None, corrupted2, None, greedy_corrupted, None
        else:
            e_ps = torch.cat([e_p_l, e_p_r], 0)
            log_probs = torch.cat([log_a_probs_l, log_a_probs_r], 0)

            if q_samples is not None:
                log_qs = torch.cat([log_a_q_l, log_a_q_r], 0)
                e_qs = torch.cat([e_q_l, e_q_r], 0)
            else:
                log_qs = None
                e_qs = None

            corrupted[:s, 0]  = g_samples_l #.data
            corrupted[s:, 2]  = g_samples_r #.data

            neg_entropy = (neg_ent_l + neg_ent_r) / 2
            return log_probs, log_qs, e_qs, corrupted, e_ps, greedy_corrupted, neg_entropy

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))




class TransEGenV4(nn.Module):

    def __init__(self, D_model, num_ent, num_rel, \
                 embedding_size=50, embedding_size_g=50, share_D_embedding=False, \
                 sparse=USE_SPARSE_EMB, entropy_reg=0., **kwargs):

        super(TransEGenV4, self).__init__(**kwargs)

        self.D_model = [D_model]
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.Z_dim = 128
        self.embedding_size = embedding_size
        # self.conv_stride = embedding_size // 25
        # self.conv_filter = self.conv_stride * 2 + 1

        self.embedding_size_g = embedding_size_g
        self.sparse = sparse
        self.entropy_reg = entropy_reg
        self.share_D_embedding = share_D_embedding

        if not self.share_D_embedding:
            self.ent_embeds = nn.Embedding(self.num_ent, self.embedding_size, max_norm=1, norm_type=2)
            self.rel_embeds = nn.Embedding(self.num_rel, self.embedding_size)

            r = 6 / np.sqrt(self.embedding_size)
            self.ent_embeds.weight.data.uniform_(-r, r)#.renorm_(p=2, dim=1, maxnorm=1)
            self.rel_embeds.weight.data.uniform_(-r, r).renorm_(p=2, dim=1, maxnorm=1)

        self.G_W1 = nn.Linear(2*self.embedding_size, self.embedding_size_g, bias=True)
        self.G_W2 = nn.Linear(self.embedding_size_g, self.num_ent, bias=True)

        self.drop = torch.nn.Dropout(.5)
        #self.bn2 = torch.nn.BatchNorm1d(128)

        self.use_cuda = torch.cuda.is_available()
        self.softmax = nn.Softmax(dim=1)
        self.h_nonlinearity = nn.LeakyReLU()
        self._inds = None
        self.z_var = None

    def init(self):
        xavier_normal(self.G_W1.weight.data)
        xavier_normal(self.G_W2.weight.data)

    #@profile
    def _sample(self, ents, rels, q_samples=None, mode=1, n_samples=1):

        if self.share_D_embedding:
            if isinstance(self.D_model[0], TransE):
                ent_es = self.D_model[0].ent_embeds(ents).detach()
                rel_es = self.D_model[0].rel_embeds(rels).detach()
            elif isinstance(self.D_model[0], TransD):
                ent_es = self.D_model[0].ent_embeds(ents, rels).detach()
                rel_es = self.D_model[0].rel_embeds(rels).detach()
        else:
            ent_es = self.ent_embeds(ents)
            rel_es = self.rel_embeds(rels)

        if mode == 1:
            inputs = torch.cat([ent_es, ent_es - rel_es], dim=1)
        else:
            inputs = torch.cat([ent_es, ent_es + rel_es], dim=1)

        x = self.G_W1(inputs)
        x = self.h_nonlinearity(x)
        x = self.drop(x)

        x = self.G_W2(x)
        x = self.h_nonlinearity(x)
        G_h = x #self.drop(x)

        if self.z_var is None:
            self.z_var = Variable(torch.zeros(*G_h.shape)).cuda()

        G_h = G_h + self.z_var.normal(0., .1*G_h.std().detach())

        gen_probs = self.softmax(G_h)
        gen_probs = gen_probs + 1e-3 / self.num_ent
        gen_probs = gen_probs / gen_probs.sum(dim=-1, keepdim=True)

        assert (gen_probs > 0).all()

        m = Categorical(gen_probs)
        g_samples = m.sample()

        _, greedy_samples = gen_probs.max(dim=-1)

        if n_samples > 1:
            g_samples = torch.stack([m.sample() for _ in xrange(n_samples)])
            return None, None, None,  g_samples, None, greedy_samples, None
        else:
            log_a_p = m.log_prob(g_samples)

            if self._inds is None:
                self._inds = torch.LongTensor(np.arange(len(ents))).cuda()

            if q_samples is not None:
                log_action_q = m.log_prob(q_samples)
                e_qs = -torch.log(gen_probs[self._inds, q_samples])
            else:
                log_action_q = None
                e_qs = None

            e_ps = -torch.log(gen_probs[self._inds, g_samples])

            if self.entropy_reg:
                neg_entropy = (gen_probs * torch.log(gen_probs)).sum(dim=-1).mean()
            else:
                neg_entropy = 0.

            return log_a_p, log_action_q, e_qs,  g_samples, e_ps, greedy_samples, neg_entropy

    #@profile
    def forward(self, z, batch, q_samples=None, n_samples=1):

        batch_size, _ = batch.size()
        corrupted = batch.detach().clone()
        s = batch_size//2

        ents_l = corrupted[:s, 0]
        ents_r = corrupted[s:, 2]

        if q_samples is None:
            q_samples_l = q_samples_r = q_samples
        else:
            q_samples_l = q_samples[:s]
            q_samples_r = q_samples[s:]

        results_l = self._sample(ents_r, corrupted[:s,1], mode=1, q_samples=q_samples_l, n_samples=n_samples)
        results_r = self._sample(ents_l, corrupted[s:,1], mode=2, q_samples=q_samples_r, n_samples=n_samples)

        log_a_probs_l, log_a_q_l, e_q_l, g_samples_l, e_p_l, greedy_samples_l, neg_ent_l = results_l
        log_a_probs_r, log_a_q_r, e_q_r, g_samples_r, e_p_r, greedy_samples_r, neg_ent_r = results_r

        greedy_corrupted = corrupted.clone()
        greedy_corrupted[:s, 0]  = greedy_samples_l
        greedy_corrupted[s:, 2]  = greedy_samples_r

        if n_samples > 1:
            corrupted2 = corrupted.repeat(n_samples,1).resize_(n_samples,batch_size,3)
            corrupted2[:,:s,0] = g_samples_l
            corrupted2[:,s:,2] = g_samples_r

            return None, None, None, corrupted2, None, greedy_corrupted, None
        else:
            e_ps = torch.cat([e_p_l, e_p_r], 0)
            log_probs = torch.cat([log_a_probs_l, log_a_probs_r], 0)

            if q_samples is not None:
                log_qs = torch.cat([log_a_q_l, log_a_q_r], 0)
                e_qs = torch.cat([e_q_l, e_q_r], 0)
            else:
                log_qs = None
                e_qs = None

            corrupted[:s, 0]  = g_samples_l #.data
            corrupted[s:, 2]  = g_samples_r #.data

            neg_entropy = (neg_ent_l + neg_ent_r) / 2
            return log_probs, log_qs, e_qs, corrupted, e_ps, greedy_corrupted, neg_entropy

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--show_tqdm', type=int, default=0, help='')
    parser.add_argument('--dataset', type=str, default='FB15k', help='Knowledge base version (default: WN)')
    parser.add_argument('--save_dir', type=str, default='./results/', help="output path")
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs (default: 500)')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size (default: 512)')
    parser.add_argument('--valid_freq', type=int, default=20, help='Validate frequency in epochs (default: 50)')
    parser.add_argument('--print_freq', type=int, default=5, help='Print frequency in epochs (default: 5)')
    parser.add_argument('--embed_dim', type=int, default=50, help='Embedding dimension (default: 50)')
    parser.add_argument('--embed_dim_g', type=int, default=50, help='Embedding dimension for G(default: 5)')
    parser.add_argument('--z_dim', type=int, default=1, help='noise Embedding dimension (default: 5)')
    parser.add_argument('--lr', type=float, default=0.008, help='Learning rate (default: 0.001)')
    parser.add_argument('--margin', type=float, default=3, help='Loss margin (default: 1)')
    parser.add_argument('--p', type=int, default=1, help='P value for p-norm (default: 1)')
    parser.add_argument('--ace', type=int, default=1, help="do ace training (otherwise just NCE)")
    parser.add_argument('--ace_iw', type=int, default=0, help="importance reweighting on ace samples")
    parser.add_argument('--prefetch_to_gpu', type=int, default=0, help="")
    parser.add_argument('--share_D_embedding', type=int, default=1, help="")
    parser.add_argument('--D_ace_weight', type=float, default=1, help="D ace term weight")
    parser.add_argument('--D_nce_weight', type=float, default=1, help="D nce term weight")
    parser.add_argument('--full_loss_penalty', type=int, default=0, help="")
    parser.add_argument('--G_model', type=str, default='v3', help="")
    parser.add_argument('--g_base_reinforce_weight', type=float, default=10, help="")
    parser.add_argument('--filter_false_negs', type=int, default=1, help="filter out sampled false negatives")
    parser.add_argument('--false_neg_penalty', type=float, default=1., help="false neg penalty for G")
    parser.add_argument('--mb_reward_normalization', type=int, default=0, help="minibatch based reward normalization")
    parser.add_argument('--g_off_policy', type=float, default=.0, help="Weight for the off-policy learning of G by importance re-weighting NCE samples")
    parser.add_argument('--nce_baseline_vr', type=int, default=1, help="Use NCE values as baseline for ACE Reinforce")
    parser.add_argument('--sc_baseline_vr', type=int, default=4, help="self critical baseline")
    parser.add_argument('--reward_nce_normalization', type=int, default=0, help="")

    parser.add_argument('--entropy_reg', type=float, default=10., help="weight for entropy regularizer")
    parser.add_argument('--entropy_top_num', type=int, default=50, help="used to define cutoff for entropy penalty")

    parser.add_argument('--g_weight_decay', type=float, default=1e-6, help="")
    parser.add_argument('--g_ematching', type=float, default=1., help="")
    parser.add_argument('--n_proposal_samples', type=int, default=10, help="")

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--decay_lr', type=str, default='halving_step100', help='lr decay mode')
    parser.add_argument('--n_g_steps', type=int, default=1, help='num of G updates per iter')
    parser.add_argument('--optim_mode', type=str, default='adam_hyp2', help='optimizer')

    parser.add_argument('--ematching_mode', type=str, default='cosine', help='')

    parser.add_argument('--namestr', type=str, default='', help='additional info in output filename to help identify experiments')


    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    args.z_dim = args.embed_dim

    if args.dataset == 'WN' or args.dataset == 'FB15k':
        path = './data/' + args.dataset + '-%s.pkl'

        args.num_ent = len(json.load(open('./data/%s-ent_to_idx.json' % args.dataset, 'r')))
        args.num_rel = len(json.load(open('./data/%s-rel_to_idx.json' % args.dataset, 'r')))
        args.data_path = path

    elif args.dataset in ('FB15k-237', 'kinship', 'nations', 'umls', 'WN18RR', 'YAGO3-10'):
        path = './data/' + args.dataset + '.pkl'
        args.data_path = path

        S = joblib.load(path)
        args.num_ent = max([S['train_data'][:,0].max(),
                            S['train_data'][:,2].max(),
                            S['val_data'][:,0].max(),
                            S['val_data'][:,2].max(),
                            S['test_data'][:,0].max(),
                            S['test_data'][:,2].max()]) + 1

        args.num_rel = max([S['train_data'][:,1].max(),
                            S['val_data'][:,1].max(),
                            S['test_data'][:,1].max()]) + 1

    else:
        raise Exception("Argument 'dataset' can only be 'WN' or 'FB15k'.")



    args.outname_base = os.path.join(args.save_dir,
                                     'Paper_{}'.format(args.dataset))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    logging.info('========= Configuration =============')

    logging.info('=====================================')
    logging.info(args)

    ##############################################################

    return args

def optimizer(params, mode, *args, **kwargs):

    if mode == 'SGD':
        opt = optim.SGD(params, *args, momentum=0., **kwargs)
    elif mode.startswith('nesterov'):
        momentum = float(mode[len('nesterov'):])
        opt = optim.SGD(params, *args, momentum=momentum, nesterov=True, **kwargs)
    elif mode.lower() == 'adam':
        betas = kwargs.pop('betas', (.9, .999))
        opt = optim.Adam(params, *args, betas=betas, amsgrad=True, **kwargs)
    elif mode.lower() == 'adam_hyp2':
        betas = kwargs.pop('betas', (.5, .99))
        opt = optim.Adam(params, *args, betas=betas, amsgrad=True, **kwargs)
    elif mode.lower() == 'adam_hyp3':
        betas = kwargs.pop('betas', (0., .99))
        opt = optim.Adam(params, *args, betas=betas, amsgrad=True, **kwargs)
    elif mode.lower() == 'adam_sparse':
        betas = kwargs.pop('betas', (.9, .999))
        opt = optim.SparseAdam(params, *args, betas=betas)
    elif mode.lower() == 'adam_sparse_hyp2':
        betas = kwargs.pop('betas', (.5, .99))
        opt = optim.SparseAdam(params, *args, betas=betas)
    elif mode.lower() == 'adam_sparse_hyp3':
        betas = kwargs.pop('betas', (.0, .99))
        opt = optim.SparseAdam(params, *args, betas=betas)
    else:
        raise NotImplementedError()
    return opt

def lr_scheduler(optimizer, decay_lr, num_epochs):

    if decay_lr in ('ms1', 'ms2', 'ms3'):
        decay_lr = int(decay_lr[-1])
        lr_milestones = [2 ** x for x in xrange(10-decay_lr, 10) if 2 ** x < num_epochs]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)

    elif decay_lr.startswith('step_exp_'):
        gamma = float(decay_lr[len('step_exp_'):])
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    elif decay_lr.startswith('halving_step'):
        step_size = int(decay_lr[len('halving_step'):])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)

    elif decay_lr.startswith('ReduceLROnPlateau'):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=10, threshold=1e-3, factor=0.1, min_lr=1e-7, verbose=True)

    elif decay_lr == '':
        scheduler = None
    else:
        raise NotImplementedError()

    return scheduler
###@profile
def main(args):

    if args.dataset in ('FB15k-237', 'kinship', 'nations', 'umls', 'WN18RR', 'YAGO3-10'):
        S = joblib.load(args.data_path)
        train_set = KBDataset(S['train_data'], args.prefetch_to_gpu)
        valid_set = KBDataset(S['val_data'])
        test_set = KBDataset(S['test_data'])
    else:
        train_set = KBDataset(args.data_path % 'train', args.prefetch_to_gpu)
        valid_set = KBDataset(args.data_path % 'valid')
        test_set = KBDataset(args.data_path % 'test')

    if args.prefetch_to_gpu:
        train_hash = set([r.tobytes() for r in train_set.dataset.cpu().numpy()])
    else:
        train_hash = set([r.tobytes() for r in train_set.dataset])

    all_hash = train_hash.copy()
    all_hash.update(set([r.tobytes() for r in valid_set.dataset]))
    all_hash.update(set([r.tobytes() for r in test_set.dataset]))

    modelD = TransD(args.num_ent, args.num_rel, args.embed_dim, args.p)
    if args.ace > 0:
        if args.G_model == 'v1':
            modelG = TransEGen(modelD, args.num_ent, args.num_rel, args.z_dim, args.embed_dim, args.embed_dim_g,
                               entropy_reg=args.entropy_reg, share_D_embedding=args.share_D_embedding)

        elif args.G_model == 'v2':
            modelG = TransEGenV2(modelD, args.num_ent, args.num_rel, args.embed_dim, args.embed_dim_g,
                                 entropy_reg=args.entropy_reg, share_D_embedding=args.share_D_embedding)

        elif args.G_model == 'v3':
            modelG = TransEGenV3(modelD, args.num_ent, args.num_rel, args.embed_dim, args.embed_dim_g,
                                 entropy_reg=args.entropy_reg, share_D_embedding=args.share_D_embedding)

        elif args.G_model == 'v4':
            modelG = TransEGenV4(modelD, args.num_ent, args.num_rel, args.embed_dim, args.embed_dim_g,
                                 entropy_reg=args.entropy_reg, share_D_embedding=args.share_D_embedding)
        else:
            raise NotImplementedError()

    if args.use_cuda:
        modelD.cuda()

        if args.ace > 0:
            modelG.cuda()

    D_monitor = OrderedDict()
    G_monitor = OrderedDict()
    test_val_monitor = OrderedDict()

    optimizerD = optimizer(modelD.parameters(), 'adam_sparse_hyp3', args.lr)
    schedulerD = lr_scheduler(optimizerD, args.decay_lr, args.num_epochs)

    if args.ace > 0:
        optimizerG = optimizer(modelG.parameters(), args.optim_mode, args.lr, weight_decay=args.g_weight_decay)
        schedulerG = lr_scheduler(optimizerG, args.decay_lr, args.num_epochs)


    loss_func = MarginRankingLoss(args.margin)


    g_reinforce_loss = VanillaReinforceLoss()
    g_reinforce_loss_IS = IWReinforceLoss()

    if args.ace > 0 and args.entropy_reg:
        entropy_threshold = float(args.num_ent)/args.entropy_top_num  * (-np.log(args.entropy_top_num))


    _cst_inds = torch.LongTensor(np.arange(args.num_ent, dtype=np.int64)[:,None]).cuda().repeat(1, args.batch_size//2)

    _cst_s = torch.LongTensor(np.arange(args.batch_size//2)).cuda()
    _cst_s_nb = torch.LongTensor(np.arange(args.batch_size//2,args.batch_size)).cuda()
    _cst_nb = torch.LongTensor(np.arange(args.batch_size)).cuda()
    def gibbs_resample(extended_batch, T=1, false_negs=None, deterministic=False):
        ''' extended_batch: (candidates,minibatch,3)
        '''
        nc, nb, _ = extended_batch.size()
        s = nb // 2
        extended_batch = extended_batch.resize_(nc*nb,3).contiguous().cuda()
        eval_batch = extended_batch

        enrgs = modelD(eval_batch)

        enrgs = enrgs + enrgs.max() * false_negs
        enrgs = enrgs.resize_(nc, nb)

        if deterministic:
            vals, sample_idx = enrgs.min(dim=0)
        else:
            enrgs = enrgs / enrgs.mean(dim=0, keepdim=True)
            probs = F.softmax(-enrgs/T, dim=0)
            m = Categorical(probs.t())
            sample_idx = m.sample()

        extended_batch = extended_batch.resize_(nc,nb,3)
        out_batch = extended_batch[0,:,:].squeeze().clone().contiguous()
        out_batch[:s,0] = extended_batch[sample_idx[:s],_cst_s,0]
        out_batch[s:,2] = extended_batch[sample_idx[s:],_cst_s_nb,2]

        false_negs = false_negs.resize_(nc, nb)
        false_negs = false_negs[sample_idx, _cst_nb]

        return out_batch, false_negs

    def gibbs_corrupt(batch, T=1):

        s = len(batch) // 2

        corrupted = batch.clone()#.detach()
        corrupted2 = corrupted.repeat(args.num_ent,1).resize_(args.num_ent,len(batch),3)

        corrupted2[:,:s,0] = _cst_inds
        corrupted2[:,s:,2] = _cst_inds

        corrupted2 = corrupted2.resize_(args.num_ent*len(batch),3).contiguous().cuda()

        eval_batch = Variable(corrupted2)
        enrgs = modelD(eval_batch)
        enrgs = enrgs.resize(args.num_ent,len(batch))
        enrgs = enrgs / enrgs.mean(dim=0, keepdim=True)

        probs = F.softmax(-enrgs/T, dim=0)

        m = Categorical(probs.t())
        sample = m.sample()

        corrupted[:s,0] = sample.data[:s]
        corrupted[s:,2] = sample.data[s:]

        return Variable(corrupted), None

    cosine = nn.CosineSimilarity(dim=0, eps=1e-6)
    #@profile
    def train(data_loader):

        lossesD = []
        lossesG = []

        monitor_grads = []
        if args.show_tqdm:
            data_itr = tqdm(enumerate(data_loader))
        else:
            data_itr = enumerate(data_loader)

        for idx, p_batch in data_itr:
            nce_batch, q_samples = corrupt_batch(p_batch, args.num_ent)

            if idx == 0 and args.ace:
                _z = ftensor(p_batch.size()[0], args.z_dim)
                if args.use_cuda:
                    _z = _z.cuda()

            if args.ace > 0:
                Z_sample = Variable(_z.uniform_(-6/(args.z_dim**.5), 6/(args.z_dim**.5)))

            if args.filter_false_negs:
                if args.prefetch_to_gpu:
                    nce_np = nce_batch.cpu().numpy()
                else:
                    nce_np = nce_batch.numpy()

                nce_falseNs = ftensor(np.array([int(x.tobytes() in train_hash) for x in nce_np], dtype=np.float32))
                nce_falseNs = Variable(nce_falseNs.cuda()) if args.use_cuda else Variable(nce_falseNs)
            else:
                nce_falseNs = None

            if args.use_cuda:
                p_batch = p_batch.cuda()
                nce_batch = nce_batch.cuda()
                q_samples = q_samples.cuda()

            optimizerD.zero_grad()

            if args.ace > 0:
                optimizerG.zero_grad()

            p_batch = Variable(p_batch)
            nce_batch = Variable(nce_batch)
            q_samples = Variable(q_samples)


            if args.ace == 0:
                d_ins = torch.cat([p_batch, nce_batch], dim=0).contiguous()
                d_outs = modelD(d_ins)

                p_enrgs = d_outs[:len(p_batch)]
                nce_enrgs = d_outs[len(p_batch):(len(p_batch)+len(nce_batch))]
                nce_term, nce_term_scores  = loss_func(p_enrgs, nce_enrgs, weights=(1.-nce_falseNs))
                lossD = args.D_nce_weight*nce_term

                create_or_append(D_monitor, 'D_nce_loss', nce_term, v2np)

            elif args.ace > 0:
                with torch.no_grad():
                    log_a_p, _, _, g_samples_proposals, _, _, _ = modelG.forward(Z_sample, p_batch, n_samples=args.n_proposal_samples)

                    if args.filter_false_negs:
                        if args.n_proposal_samples > 1:
                            g_samples_proposals = g_samples_proposals.resize_(args.n_proposal_samples*args.batch_size,3)

                        ace_falseNs = ftensor(np.array([int(x.tobytes() in train_hash) for x in v2np(g_samples_proposals)], dtype=np.float32))
                        ace_falseNs = Variable(ace_falseNs.cuda()) if args.use_cuda else Variable(ace_falseNs)

                        if args.n_proposal_samples > 1:
                            g_samples_proposals = g_samples_proposals.resize_(args.n_proposal_samples,args.batch_size,3)
                    else:
                        ace_falseNs = None

                    if args.n_proposal_samples > 1:
                        g_samples, ace_falseNs = gibbs_resample(g_samples_proposals, false_negs=ace_falseNs, deterministic=False)
                    else:
                        g_samples = g_samples_proposals

                d_ins = torch.cat([p_batch, nce_batch, g_samples], dim=0).contiguous()
                d_outs = modelD(d_ins)

                p_enrgs = d_outs[:len(p_batch)]
                nce_enrgs = d_outs[len(p_batch):(len(p_batch)+len(nce_batch))]
                adv_enrgs = d_outs[(len(p_batch)+len(nce_batch)):]

                nce_term, nce_term_scores  = loss_func(p_enrgs, nce_enrgs, weights=(1.-nce_falseNs))
                lossD = args.D_nce_weight*nce_term

                create_or_append(D_monitor, 'D_nce_loss', nce_term, v2np)

                if args.ace_iw:
                    iw_weights = (1. / args.num_ent) / torch.exp(log_a_p)
                    ace_weights = (1.-ace_falseNs)
                    _, adv_term_scores = loss_func(p_enrgs, adv_enrgs, weights=None)

                    adv_scores_filtered = adv_term_scores * ace_weights / ace_weights.sum()

                    ace_weights = (1.-ace_falseNs) * iw_weights
                    ace_weights = ace_weights.detach()

                    adv_term = ((adv_term_scores * ace_weights) / ace_weights.sum()).sum() #/ ace_weights.sum()).sum()


                    lossD = lossD + args.D_ace_weight * adv_term

                    create_or_append(D_monitor, 'iw_ratios', iw_weights.mean(), v2np)
                    create_or_append(D_monitor, 'D_ace_loss (no IW)', adv_scores_filtered.sum(), v2np)
                    create_or_append(D_monitor, 'D_ace_loss (IW)', adv_term, v2np)

                else:
                    ace_weights = (1.-ace_falseNs)
                    adv_term, adv_term_scores = loss_func(p_enrgs, adv_enrgs, weights=ace_weights)
                    lossD = lossD + args.D_ace_weight * adv_term

                    create_or_append(D_monitor, 'D_ace_loss', adv_term, v2np)

            elif args.ace < 0:
                g_samples,_ = gibbs_corrupt(p_batch.data, 1./abs(args.ace))

                if args.filter_false_negs:
                    ace_falseNs = ftensor(np.array([int(x.tobytes() in train_hash) for x in v2np(g_samples)], dtype=np.float32))
                    ace_falseNs = Variable(ace_falseNs.cuda()) if args.use_cuda else Variable(ace_falseNs)
                else:
                    ace_falseNs = None

                d_ins = torch.cat([p_batch, nce_batch, g_samples], dim=0).contiguous()
                d_outs = modelD(d_ins)

                p_enrgs = d_outs[:len(p_batch)]
                nce_enrgs = d_outs[len(p_batch):(len(p_batch)+len(nce_batch))]
                adv_enrgs = d_outs[(len(p_batch)+len(nce_batch)):]

                adv_term, adv_term_scores = loss_func(p_enrgs, adv_enrgs, weights=(1.-ace_falseNs))
                lossD = lossD + args.D_ace_weight * adv_term
                create_or_append(D_monitor, 'D_ace_loss', adv_term, v2np)

            create_or_append(D_monitor, 'D_loss', lossD, v2np)

            lossD.backward()

            optimizerD.step()
            optimizerD.zero_grad()


            if args.ace > 0:
                ''' G update
                '''

                # nce_enrgs = nce_enrgs.detach()
                for _ in xrange(args.n_g_steps):

                    optimizerG.zero_grad()
                    Z_sample = Variable(_z.uniform_(-6/(args.z_dim**.5), 6/(args.z_dim**.5)))


                    if False and not args.g_off_policy and not args.g_ematching:
                        q_samples = None

                    log_a_p, log_a_p_at_qs, e_qs, g_samples, e_ps, greedy_samples, neg_entropy = modelG.forward(Z_sample, p_batch, q_samples=q_samples)

                    with torch.no_grad():

                        g_ins = torch.cat([g_samples, greedy_samples], dim=0).contiguous()

                        d_outs = modelD(g_ins)

                        adv_enrgs = d_outs[:len(g_samples)]
                        greedy_enrgs = d_outs[len(g_samples):]

                        # greedy_enrgs = modelD(greedy_samples)

                        #penalty = adv_enrgs.detach()

                        if args.full_loss_penalty:
                            _, penalty = loss_func(p_enrgs, adv_enrgs)
                            penalty = - penalty

                            _, nce_penalty_raw = loss_func(p_enrgs, nce_enrgs)
                            nce_penalty_raw = - nce_penalty_raw

                            _, greedy_penalty_raw = loss_func(p_enrgs, greedy_enrgs)
                            greedy_penalty_raw = - greedy_penalty_raw

                        else:
                            penalty = adv_enrgs
                            nce_penalty_raw = nce_enrgs
                            greedy_penalty_raw = greedy_enrgs

                        create_or_append(G_monitor, 'G_penalty_raw_avg', penalty.mean(), v2np)
                        create_or_append(G_monitor, 'G_penalty_raw_std', penalty.std(), v2np)

                        if args.nce_baseline_vr and args.sc_baseline_vr:
                            w1 = args.nce_baseline_vr / float(args.nce_baseline_vr + args.sc_baseline_vr)
                            baseline = w1 * nce_penalty_raw + (1-w1) * greedy_penalty_raw

                        elif args.nce_baseline_vr:
                            baseline = nce_penalty_raw

                        elif args.sc_baseline_vr:
                            baseline = greedy_penalty_raw
                        else:
                            baseline = Variable(torch.FloatTensor(np.zeros(args.batch_size, dtype=np.float32)).cuda())

                        create_or_append(G_monitor, 'G_baseline_avg', baseline.mean(), v2np)
                        create_or_append(G_monitor, 'G_baseline_std', baseline.std(), v2np)

                        penalty = penalty - baseline

                        false_negs = ftensor(np.array([int(x.tobytes() in train_hash) for x in v2np(g_samples)], dtype=np.float32))
                        if args.use_cuda:
                            false_negs = false_negs.cuda()
                        false_negs = Variable(false_negs)
                        penalty = penalty * (1-false_negs)

                        create_or_append(G_monitor, 'G_sample_fn_ratio', false_negs.mean(), v2np)

                        if args.false_neg_penalty:
                            penalty = penalty + args.false_neg_penalty * false_negs

                        #create_or_append(G_monitor, 'G_falseN_penalty_avg', false_negs.mean())

                        if args.mb_reward_normalization:
                            penalty = (penalty - penalty.mean()) / penalty.std().clamp(min=1e-8)

                    reinforce_loss = args.g_base_reinforce_weight * g_reinforce_loss(penalty.detach(), log_a_p)
                    create_or_append(G_monitor, 'G_base_reinforce_loss', reinforce_loss, v2np)


                    if args.g_off_policy:

                        if args.mb_reward_normalization or args.reward_nce_normalization:
                            nce_penalty = (nce_penalty_raw - nce_penalty_raw.mean()) / nce_penalty_raw.std().clamp(min=1e-8)
                        else:
                            nce_penalty = nce_penalty_raw

                        g_iw_loss = g_reinforce_loss_IS((nce_penalty - baseline).detach(), -np.log(args.num_ent), log_a_p_at_qs)


                        create_or_append(G_monitor, 'G_iw_loss', g_iw_loss, v2np)
                        reinforce_loss = reinforce_loss + args.g_off_policy * g_iw_loss

                    create_or_append(G_monitor, 'G_reinforce_loss', reinforce_loss, v2np)

                    lossG = reinforce_loss
                    if args.entropy_reg:
                        entropy_loss = neg_entropy.clamp(min=entropy_threshold) #- entropy_threshold
                        lossG = lossG + args.entropy_reg * entropy_loss
                        create_or_append(G_monitor, 'G_neg_entropy_raw', neg_entropy, v2np)
                        create_or_append(G_monitor, 'G_neg_entropy_loss', entropy_loss, v2np)

                    if True or args.g_ematching:
                        if args.ematching_mode == 'l2':
                            #p_ematching_loss =  ((e_ps - adv_enrgs.detach()) ** 2).mean()
                            Ge = torch.cat([e_ps.squeeze(), e_qs.squeeze()])
                            De = torch.cat([adv_term_scores.squeeze(), nce_term_scores.squeeze()])
                            Ge = Ge - Ge.mean()
                            Ge = Ge / Ge.std()

                            De = De - De.mean()
                            De = De / De.std()

                            q_ematching_loss =  ((Ge - De.detach()) ** 2).mean()
                        else:
                            #p_ematching_loss =  (1-cosine(e_ps, adv_enrgs.detach())).mean() #((e_ps - adv_enrgs.detach()) ** 2).mean()
                            Ge = torch.cat([e_ps.squeeze(), e_qs.squeeze()])
                            De = torch.cat([p_enrgs.squeeze() - adv_enrgs.squeeze(), p_enrgs.squeeze() - nce_enrgs.squeeze()])
                            Ge = Ge - Ge.mean()
                            Ge = Ge / Ge.std()

                            De = De - De.mean()
                            De = De / De.std()

                            q_ematching_loss =  1-cosine(Ge, De.detach())

                        lossG = lossG + args.g_ematching * q_ematching_loss

                        #create_or_append(G_monitor, 'G_p_ematching_loss', p_ematching_loss)
                        create_or_append(G_monitor, 'G_q_ematching_loss', q_ematching_loss, v2np)

                    lossG.backward()

                    optimizerG.step()
                    create_or_append(G_monitor, 'G_loss', lossG, v2np)

    def test(dataset, subsample=1):
        l_ranks, r_ranks = [], []
        test_loader = DataLoader(dataset, num_workers=1, collate_fn=collate_fn)

        cst_inds = np.arange(args.num_ent, dtype=np.int64)[:,None]

        if args.show_tqdm:
            data_itr = tqdm(enumerate(test_loader))
        else:
            data_itr = enumerate(test_loader)

        for idx, triplet in data_itr:
            if idx % subsample != 0:
                continue

            lhs, rel, rhs = triplet.view(-1)

            l_batch = np.concatenate([cst_inds, np.array([[rel, rhs]]).repeat(args.num_ent, axis=0)], axis=1)
            r_batch = np.concatenate([np.array([[lhs, rel]]).repeat(args.num_ent, axis=0), cst_inds], axis=1)

            l_fns = np.array([int(x.tobytes() in all_hash) for x in l_batch], dtype=np.float32)
            r_fns = np.array([int(x.tobytes() in all_hash) for x in r_batch], dtype=np.float32)

            l_batch = ltensor(l_batch).contiguous()
            r_batch = ltensor(r_batch).contiguous()

            if args.use_cuda:
                l_batch = l_batch.cuda()
                r_batch = r_batch.cuda()

            l_batch = Variable(l_batch)
            r_batch = Variable(r_batch)

            d_ins = torch.cat([l_batch, r_batch], dim=0)
            d_outs = modelD(d_ins)
            l_enrgs = d_outs[:len(l_batch)]
            r_enrgs = d_outs[len(l_batch):]

            l_rank = compute_rank(v2np(l_enrgs), lhs, mask_observed=l_fns)
            r_rank = compute_rank(v2np(r_enrgs), rhs, mask_observed=r_fns)

            l_ranks.append(l_rank)
            r_ranks.append(r_rank)

        l_ranks = np.array(l_ranks)
        r_ranks = np.array(r_ranks)

        return l_ranks, r_ranks

    if args.prefetch_to_gpu:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                  num_workers=0, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                  num_workers=4, pin_memory=True, collate_fn=collate_fn)


    for epoch in tqdm(range(1, args.num_epochs + 1)):

        train(train_loader)
        gc.collect()
        if epoch % args.print_freq == 0:


            logging.info("~~~~~~ Epoch {} ~~~~~~".format(epoch))
            for k in D_monitor:
                if k.endswith('_epoch_avg'):
                    logging.info("{:<30} {:10.5f}".format(k, D_monitor[k][-1]))

            logging.info("****")
            for k in G_monitor:
                if k.endswith('_epoch_avg'):
                    logging.info("{:<30} {:10.5f}".format(k, G_monitor[k][-1]))

        if args.decay_lr:
            if args.decay_lr == 'ReduceLROnPlateau':
                schedulerD.step(monitor['D_loss_epoch_avg'])
            else:
                schedulerD.step()

            if args.ace > 0:
                assert args.decay_lr != 'ReduceLROnPlateau', 'ACE mode does not support ReduceLROnPlateau'
                schedulerG.step()

        if epoch % args.valid_freq == 0:

            with torch.no_grad():
                l_ranks, r_ranks = test(valid_set, subsample=10)

                l_mean = l_ranks.mean()
                r_mean = r_ranks.mean()
                l_mrr = (1. / l_ranks).mean()
                r_mrr = (1. / r_ranks).mean()
                l_h10 = (l_ranks <= 10).mean()
                r_h10 = (r_ranks <= 10).mean()
                l_h5 = (l_ranks <= 5).mean()
                r_h5 = (r_ranks <= 5).mean()

            create_or_append(test_val_monitor, 'validation l_avg_rank', l_mean)
            create_or_append(test_val_monitor, 'validation r_avg_rank', r_mean)
            create_or_append(test_val_monitor, 'validation avg_rank', (l_mean+r_mean)/2)
            create_or_append(test_val_monitor, 'validation l_mrr', l_mrr)
            create_or_append(test_val_monitor, 'validation r_mrr', r_mrr)
            create_or_append(test_val_monitor, 'validation mrr', (l_mrr+r_mrr)/2)
            create_or_append(test_val_monitor, 'validation l_h10', l_h10)
            create_or_append(test_val_monitor, 'validation r_h10', r_h10)
            create_or_append(test_val_monitor, 'validation h10', (l_h10+r_h10)/2)
            create_or_append(test_val_monitor, 'validation l_h5', l_h5)
            create_or_append(test_val_monitor, 'validation r_h5', r_h5)
            create_or_append(test_val_monitor, 'validation h5', (l_h5+r_h5)/2)

            logging.info("#######################################")
            for k in test_val_monitor:
                if k.startswith('validation'):
                    logging.info("{:<30} {:10.5f}".format(k, test_val_monitor[k][-1]))
            logging.info("#######################################")
            joblib.dump({'l_ranks':l_ranks, 'r_ranks':r_ranks}, args.outname_base+'epoch{}_validation_ranks.pkl'.format(epoch), compress=9)

            modelD.save(args.outname_base+'D_epoch{}.pts'.format(epoch))
            if args.ace > 0:
                modelG.save(args.outname_base+'G_epoch{}.pts'.format(epoch))

        if epoch % (args.valid_freq * 5) == 0:
            l_ranks, r_ranks = test(test_set)
            l_mean = l_ranks.mean()
            r_mean = r_ranks.mean()
            l_mrr = (1. / l_ranks).mean()
            r_mrr = (1. / r_ranks).mean()
            l_h10 = (l_ranks <= 10).mean()
            r_h10 = (r_ranks <= 10).mean()
            l_h5 = (l_ranks <= 5).mean()
            r_h5 = (r_ranks <= 5).mean()

            create_or_append(test_val_monitor, 'test l_avg_rank', l_mean)
            create_or_append(test_val_monitor, 'test r_avg_rank', r_mean)
            create_or_append(test_val_monitor, 'test avg_rank', (l_mean+r_mean)/2)
            create_or_append(test_val_monitor, 'test l_mrr', l_mrr)
            create_or_append(test_val_monitor, 'test r_mrr', r_mrr)
            create_or_append(test_val_monitor, 'test mrr', (l_mrr+r_mrr)/2)
            create_or_append(test_val_monitor, 'test l_h10', l_h10)
            create_or_append(test_val_monitor, 'test r_h10', r_h10)
            create_or_append(test_val_monitor, 'test h10', (l_h10+r_h10)/2)
            create_or_append(test_val_monitor, 'test l_h5', l_h5)
            create_or_append(test_val_monitor, 'test r_h5', r_h5)
            create_or_append(test_val_monitor, 'test h5', (l_h5+r_h5)/2)

            logging.info("=======================================")
            for k in test_val_monitor:
                if k.startswith('test'):
                    logging.info("{:<30} {:10.5f}".format(k, test_val_monitor[k][-1]))
            logging.info("=======================================")

    l_ranks, r_ranks = test(test_set)
    l_mean = l_ranks.mean()
    r_mean = r_ranks.mean()
    l_mrr = (1. / l_ranks).mean()
    r_mrr = (1. / r_ranks).mean()
    l_h10 = (l_ranks <= 10).mean()
    r_h10 = (r_ranks <= 10).mean()
    l_h5 = (l_ranks <= 5).mean()
    r_h5 = (r_ranks <= 5).mean()

    create_or_append(test_val_monitor, 'test l_avg_rank', l_mean)
    create_or_append(test_val_monitor, 'test r_avg_rank', r_mean)
    create_or_append(test_val_monitor, 'test avg_rank', (l_mean+r_mean)/2)
    create_or_append(test_val_monitor, 'test l_mrr', l_mrr)
    create_or_append(test_val_monitor, 'test r_mrr', r_mrr)
    create_or_append(test_val_monitor, 'test mrr', (l_mrr+r_mrr)/2)
    create_or_append(test_val_monitor, 'test l_h10', l_h10)
    create_or_append(test_val_monitor, 'test r_h10', r_h10)
    create_or_append(test_val_monitor, 'test h10', (l_h10+r_h10)/2)
    create_or_append(test_val_monitor, 'test l_h5', l_h5)
    create_or_append(test_val_monitor, 'test r_h5', r_h5)
    create_or_append(test_val_monitor, 'test h5', (l_h5+r_h5)/2)

    logging.info("=======================================")
    for k in test_val_monitor:
        if k.startswith('test'):
            logging.info("{:<30} {:10.5f}".format(k, test_val_monitor[k][-1]))
    logging.info("=======================================")

    modelD.save(args.outname_base+'D_final.pts')
    if args.ace > 0:
        modelG.save(args.outname_base+'G_final.pts')

    joblib.dump({'l_ranks':l_ranks, 'r_ranks':r_ranks}, args.outname_base+'test_ranks.pkl', compress=9)


    logging.info("COMPLETE!!!")

if __name__ == '__main__':
    main(parse_args())

