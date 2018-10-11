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
from preprocess_movie_lens import make_dataset
import joblib
from collections import Counter
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
            return enrgs,lhs,rhs

    def get_embed(self, ents, rel_idxs):
        ent_embed = self.ent_embeds(ents, rel_idxs)
        return ent_embed

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
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

class DemParDisc(nn.Module):
    def __init__(self, embed_dim, attribute_data,attribute='gender',use_cross_entropy=True):
        super(DemParDisc, self).__init__()
        self.embed_dim = int(embed_dim)
        if use_cross_entropy:
            self.cross_entropy = True
        else:
            self.cross_entropy = False
        if attribute == 'gender':
            users_sex = attribute_data[0]['sex']
            users_sex = [0 if i == 'M' else 1 for i in users_sex]
            self.users_sensitive = np.ascontiguousarray(users_sex)
            self.out_dim = 1
        else:
            users_occupation = attribute_data[0]['occupation']
            users_occupation_list = sorted(set(users_occupation))
            occ_to_idx = {}
            for i, occ in enumerate(users_occupation_list):
                occ_to_idx[occ] = i
            users_occupation = [occ_to_idx[occ] for occ in users_occupation]
            self.users_sensitive = np.ascontiguousarray(users_occupation)
            self.out_dim = len(users_occupation_list)

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
        if self.attribute == 'gender':
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
        # if self.cross_entropy:
            # probs = F.binary_cross_entropy_with_logits(scores,A_labels,reduction='none')
            # preds = (probs > torch.Tensor([0.5]).cuda()).float() * 1
        # else:
        if self.attribute == 'gender':
            A_labels = Variable(torch.Tensor(self.users_sensitive[ents])).cuda()
            A_labels = A_labels.unsqueeze(1)
            probs = torch.sigmoid(scores)
            preds = (probs > torch.Tensor([0.5]).cuda()).float() * 1
        else:
            A_labels = Variable(torch.LongTensor(self.users_sensitive[ents])).cuda()
            log_probs = F.log_softmax(scores, dim=1)
            preds = log_probs.max(1, keepdim=True)[1] # get the index of the max
            correct = preds.eq(A_labels.view_as(preds)).sum().item()
        if return_preds:
            return preds, A_labels
        else:
            return correct

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))

class MarginRankingLoss(nn.Module):
    def __init__(self, margin):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, p_enrgs, n_enrgs, weights=None):
        scores = (self.margin + p_enrgs - n_enrgs).clamp(min=0)

        if weights is not None:
            scores = scores * weights / weights.mean()

        return scores.mean(), scores

class KBDataset(Dataset):
    def __init__(self,data_split,prefetch_to_gpu=False):
        self.prefetch_to_gpu = prefetch_to_gpu
        self.dataset = np.ascontiguousarray(data_split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def shuffle(self):
        if self.dataset.is_cuda:
            self.dataset = self.dataset.cpu()

        data = self.dataset
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

'''Monitor Norm of gradients'''
def monitor_grad_norm(model):
    parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

'''Monitor Norm of weights'''
def monitor_weight_norm(model):
    parameters = list(filter(lambda p: p is not None, model.parameters()))
    total_norm = 0
    for p in parameters:
        param_norm = p.data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--show_tqdm', type=int, default=0, help='')
    parser.add_argument('--save_dir', type=str, default='./results/MovieLens/', help="output path")
    parser.add_argument('--model_save_dir', type=str, default='./results/', help="output path")
    parser.add_argument('--do_log', action='store_true', help="whether to log to csv")
    parser.add_argument('--load_transD', action='store_true', help="Load TransD")
    parser.add_argument('--freeze_transD', action='store_true', help="Load TransD")
    parser.add_argument('--remove_old_run', action='store_true', help="remove old run")
    parser.add_argument('--use_cross_entropy', action='store_true', help="DemPar Discriminators Loss as CE")
    parser.add_argument('--data_dir', type=str, default='./data/', help="Contains Pickle files")
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs (default: 500)')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size (default: 512)')
    parser.add_argument('--gamma', type=int, default=0.01, help='Tradeoff for Adversarial Penalty')
    parser.add_argument('--valid_freq', type=int, default=20, help='Validate frequency in epochs (default: 50)')
    parser.add_argument('--print_freq', type=int, default=5, help='Print frequency in epochs (default: 5)')
    parser.add_argument('--embed_dim', type=int, default=50, help='Embedding dimension (default: 50)')
    parser.add_argument('--z_dim', type=int, default=100, help='noise Embedding dimension (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--margin', type=float, default=3, help='Loss margin (default: 1)')
    parser.add_argument('--p', type=int, default=1, help='P value for p-norm (default: 1)')
    parser.add_argument('--prefetch_to_gpu', type=int, default=0, help="")
    parser.add_argument('--full_loss_penalty', type=int, default=0, help="")
    parser.add_argument('--filter_false_negs', type=int, default=1, help="filter out sampled false negatives")
    parser.add_argument('--ace', type=int, default=0, help="do ace training (otherwise just NCE)")
    parser.add_argument('--false_neg_penalty', type=float, default=1., help="false neg penalty for G")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--use_attr', type=bool, default=False, help='Use Attribute Matrix')
    parser.add_argument('--use_multi_attr', type=bool, default=False, help='Use Multi Attribute Matrix')
    parser.add_argument('--use_occ_attr', type=bool, default=False, help='Use Occ_Attribute Matrix')
    parser.add_argument('--use_age_attr', type=bool, default=False, help='Use Age Attribute Matrix')
    parser.add_argument('--sample_mask', type=bool, default=False, help='Sample a binary mask for discriminators to use')
    parser.add_argument('--decay_lr', type=str, default='halving_step100', help='lr decay mode')
    parser.add_argument('--optim_mode', type=str, default='adam', help='optimizer')
    parser.add_argument('--fairD_optim_mode', type=str, default='adam_hyp2',help='optimizer for Fairness Discriminator')
    parser.add_argument('--namestr', type=str, default='', help='additional info in output filename to help identify experiments')

    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    args.train_ratings,args.test_ratings,args.users,args.movies = make_dataset(True)
    args.num_ent = len(args.users) + len(args.movies)
    args.num_rel = 5
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.saved_path = os.path.join(args.model_save_dir,'MovieLens_resultsD_final.pts')
    args.outname_base = os.path.join(args.save_dir,args.namestr+'_MovieLens_results')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

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

def freeze_model(model):
    model.eval()
    for params in model.parameters():
        params.requires_grad = False

def train(data_loader, counter, args, train_hash, modelD, optimizerD,\
        schedulerD, tflogger, fairD_gender=None, optimizer_fairD_gender=None,\
        fairD_occupation=None, optimizer_fairD_occupation=None):

    lossesD = []
    monitor_grads = []
    precision_list = []
    recall_list = []
    fscore_list = []
    correct = 0
    occ_correct = 0
    total_ent = 0
    loss_func = MarginRankingLoss(args.margin)

    if args.sample_mask:
        mask = np.random.choice([0, 1], size=(3,))

    if args.show_tqdm:
        data_itr = tqdm(enumerate(data_loader))
    else:
        data_itr = enumerate(data_loader)

    for idx, p_batch in data_itr:
        nce_batch, q_samples = corrupt_batch(p_batch, args.num_ent)

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

        p_batch_var = Variable(p_batch)
        nce_batch = Variable(nce_batch)
        q_samples = Variable(q_samples)

        if args.ace == 0:
            d_ins = torch.cat([p_batch_var, nce_batch], dim=0).contiguous()
            if (args.use_attr or args.use_multi_attr or args.use_occ_attr) and not args.freeze_transD:
                d_outs,lhs_emb,rhs_emb = modelD(d_ins,True)
                with torch.no_grad():
                    p_lhs_emb = lhs_emb[:len(p_batch_var)]
                    # p_rhs_emb = rhs_emb[:len(p_batch_var)]
                    if args.use_multi_attr:
                        l_penalty = fairD_gender(p_lhs_emb,p_batch[:,0])
                        l_penalty += fairD_occupation(p_lhs_emb,p_batch[:,0])
                    elif args.use_attr:
                        l_penalty = fairD_gender(p_lhs_emb,p_batch[:,0])
                    elif args.use_occ_attr:
                        l_penalty = fairD_occupation(p_lhs_emb,p_batch[:,0])
                    # r_penalty = fairD_gender(p_rhs_emb,p_batch[:,2])
                    if not args.use_cross_entropy:
                        if args.use_multi_attr:
                            fair_penalty = 2 - l_penalty
                        else:
                            fair_penalty = 1 - l_penalty
                    else:
                        fair_penalty = l_penalty
            else:
                d_outs = modelD(d_ins)
                fair_penalty = Variable(torch.zeros(1)).cuda()

            p_enrgs = d_outs[:len(p_batch_var)]
            nce_enrgs = d_outs[len(p_batch_var):(len(p_batch_var)+len(nce_batch))]
            nce_term, nce_term_scores  = loss_func(p_enrgs, nce_enrgs, weights=(1.-nce_falseNs))
            lossD = nce_term + args.gamma*fair_penalty

        if not args.freeze_transD:
            lossD.backward()
            optimizerD.step()

        if args.use_attr or args.use_multi_attr:
            optimizer_fairD_gender.zero_grad()
            with torch.no_grad():
                d_outs,lhs_emb,rhs_emb = modelD(d_ins,True)
                p_lhs_emb = lhs_emb[:len(p_batch)]
                # p_rhs_emb = rhs_emb[:len(p_batch)]
            l_loss = fairD_gender(p_lhs_emb,p_batch[:,0])
            # r_loss = fairD_gender(p_rhs_emb,p_batch[:,2])
            if not args.use_cross_entropy:
                fairD_gender_loss = -1*(1 - l_loss)
            else:
                fairD_gender_loss = l_loss
            # fairD_gender_loss = -1*fairD_gender_loss
            fairD_gender_loss.backward()
            optimizer_fairD_gender.step()
            l_preds,l_A_labels = fairD_gender.predict(p_lhs_emb,p_batch[:,0],return_preds=True)
            # r_preds, r_A_labels = fairD_gender.predict(p_rhs_emb,p_batch[:,2],return_preds=True)
            l_correct = l_preds.eq(l_A_labels.view_as(l_preds)).sum().item()
            # r_correct = r_preds.eq(r_A_labels.view_as(r_preds)).sum().item()
            correct += l_correct #+ r_correct
            total_ent += len(p_batch)
            l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                    average='binary')
            # r_precision,r_recall,r_fscore,_ = precision_recall_fscore_support(r_A_labels, r_preds,\
                    # average='binary')
            precision = l_precision #+ r_precision) / 2
            recall = l_recall #+ r_recall) / 2
            fscore = l_fscore #+ r_fscore) / 2
            precision_list.append(precision)
            recall_list.append(recall)
            fscore_list.append(fscore)

        if args.use_occ_attr or args.use_multi_attr:
            optimizer_fairD_occupation.zero_grad()
            with torch.no_grad():
                d_outs,lhs_emb,rhs_emb = modelD(d_ins,True)
                p_lhs_emb = lhs_emb[:len(p_batch)]
            l_loss = fairD_occupation(p_lhs_emb,p_batch[:,0])
            if not args.use_cross_entropy:
                fairD_occupation_loss = -1*(1 - l_loss)
            else:
                fairD_occupation_loss = l_loss
            fairD_occupation_loss.backward()
            optimizer_fairD_occupation.step()
            l_preds,l_A_labels = fairD_occupation.predict(p_lhs_emb,p_batch[:,0],return_preds=True)
            l_correct = l_preds.eq(l_A_labels.view_as(l_preds)).sum().item()
            occ_correct += l_correct #+ r_correct
            # total_ent += len(p_batch)
            # l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                    # average='binary')
            # r_precision,r_recall,r_fscore,_ = precision_recall_fscore_support(r_A_labels, r_preds,\
                    # average='binary')
            # precision = l_precision #+ r_precision) / 2
            # recall = l_recall #+ r_recall) / 2
            # fscore = l_fscore #+ r_fscore) / 2
            # precision_list.append(precision)
            # recall_list.append(recall)
            # fscore_list.append(fscore)

    if args.do_log and args.use_attr: # Tensorboard logging
        acc = 100. * correct / total_ent
        mean_precision = np.mean(np.asarray(precision_list))
        mean_recall = np.mean(np.asarray(recall_list))
        mean_fscore = np.mean(np.asarray(fscore_list))
        tflogger.scalar_summary('Train Fairness Discriminator,\
                Accuracy',float(acc),counter)
        # tflogger.scalar_summary('Train Fairness Discriminator,\
                # Precision',float(mean_precision),counter)
        # tflogger.scalar_summary('Train Fairness Discriminator,\
                # Recall',float(mean_recall),counter)
        # tflogger.scalar_summary('Train Fairness Discriminator,\
                # F-score',float(mean_fscore),counter)

    ''' Logging for end of epoch '''
    if args.use_attr or args.use_multi_attr:
        fairD_gender_grad_norm = monitor_grad_norm(fairD_gender)
        fairD_gender_weight_norm = monitor_weight_norm(fairD_gender)

    if args.do_log: # Tensorboard logging
        tflogger.scalar_summary('TransD Loss',float(lossD),counter)
        if args.use_attr or args.use_multi_attr:
            tflogger.scalar_summary('Fair Gender Disc Loss',float(fairD_gender_loss),counter)
        if args.use_occ_attr or args.use_multi_attr:
            tflogger.scalar_summary('Fair Occupation Disc Loss',float(fairD_occupation_loss),counter)

def test_fairness(dataset,args,all_hash,modelD,tflogger,fairD,attribute,epoch):
    test_loader = DataLoader(dataset, num_workers=1, batch_size=4096, collate_fn=collate_fn)
    correct = 0
    total_ent = 0
    precision_list = []
    recall_list = []
    fscore_list = []
    if args.show_tqdm:
        data_itr = tqdm(enumerate(test_loader))
    else:
        data_itr = enumerate(test_loader)

    for idx, triplet in data_itr:
        lhs, rel, rhs = triplet[:,0], triplet[:,1],triplet[:,2]
        l_batch = Variable(lhs).cuda()
        r_batch = Variable(rhs).cuda()
        rel_batch = Variable(rel).cuda()
        lhs_emb = modelD.get_embed(l_batch,rel_batch)
        rhs_emb = modelD.get_embed(r_batch,rel_batch)
        l_preds,l_A_labels = fairD.predict(lhs_emb,lhs,return_preds=True)
        # r_preds, r_A_labels = fairD_gender.predict(rhs_emb,rhs,return_preds=True)
        l_correct = l_preds.eq(l_A_labels.view_as(l_preds)).sum().item()
        # r_correct = r_preds.eq(r_A_labels.view_as(r_preds)).sum().item()
        if attribute == 'gender':
            l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                    average='binary')
        else:
            l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                    average='micro')
        # r_precision,r_recall,r_fscore,_ = precision_recall_fscore_support(r_A_labels, r_preds,\
                # average='binary')
        precision = l_precision #+ r_precision) / 2
        recall = l_recall #+ r_recall) / 2
        fscore = l_fscore #+ r_fscore) / 2
        precision_list.append(precision)
        recall_list.append(recall)
        fscore_list.append(fscore)
        correct += l_correct #+ r_correct
        total_ent += len(lhs_emb) #+ len(rhs_emb)

    if args.do_log:
        acc = 100. * correct / total_ent
        mean_precision = np.mean(np.asarray(precision_list))
        mean_recall = np.mean(np.asarray(recall_list))
        mean_fscore = np.mean(np.asarray(fscore_list))
        tflogger.scalar_summary(attribute +'_Valid Fairness Discriminator Accuracy',float(acc),epoch)
        # tflogger.scalar_summary(attribute + '_Valid Fairness Discriminator Precision',float(mean_precision),epoch)
        # tflogger.scalar_summary(attribute + '_Valid Fairness Discriminator Recall',float(mean_recall),epoch)
        tflogger.scalar_summary(attribute + '_Valid Fairness Discriminator F-score',float(mean_fscore),epoch)

def test(dataset, args, all_hash, modelD, tflogger,subsample=1):
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

def main(args):
    train_set = KBDataset(args.train_ratings, args.prefetch_to_gpu)
    test_set = KBDataset(args.test_ratings, args.prefetch_to_gpu)

    if args.prefetch_to_gpu:
        train_hash = set([r.tobytes() for r in train_set.dataset.cpu().numpy()])
    else:
        train_hash = set([r.tobytes() for r in train_set.dataset])

    all_hash = train_hash.copy()
    all_hash.update(set([r.tobytes() for r in test_set.dataset]))
    logdir = args.outname_base + '_logs' + '/'
    if args.remove_old_run:
        shutil.rmtree(logdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    tflogger = tfLogger(logdir)

    modelD = TransD(args.num_ent, args.num_rel, args.embed_dim, args.p)

    if args.use_attr:
        attr_data = [args.users,args.movies]
        fairD_gender = DemParDisc(args.embed_dim,attr_data,use_cross_entropy=args.use_cross_entropy)
        fairD_gender.cuda()
        optimizer_fairD_gender = optimizer(fairD_gender.parameters(),'adam', args.lr)
        fairD_occupation = None
        optimizer_fairD_occupation = None
    elif args.use_occ_attr:
        attr_data = [args.users,args.movies]
        fairD_occupation = DemParDisc(args.embed_dim,attr_data,\
                attribute='occupation',use_cross_entropy=args.use_cross_entropy)
        fairD_occupation.cuda()
        optimizer_fairD_occupation = optimizer(fairD_occupation.parameters(),'adam',args.lr)
        fairD_gender = None
        optimizer_fairD_gender = None
    elif args.use_multi_attr or args.use_occ_attr:
        attr_data = [args.users,args.movies]
        fairD_gender = DemParDisc(args.embed_dim,attr_data,use_cross_entropy=args.use_cross_entropy)
        fairD_gender.cuda()
        optimizer_fairD_gender = optimizer(fairD_gender.parameters(),'adam', args.lr)
        fairD_occupation = DemParDisc(args.embed_dim,attr_data,\
                attribute='occupation',use_cross_entropy=args.use_cross_entropy)
        fairD_occupation.cuda()
        optimizer_fairD_occupation = optimizer(fairD_occupation.parameters(),'adam',args.lr)
    else:
        fairD_gender = None
        optimizer_fairD_gender = None
        fairD_occupation = None
        optimizer_fairD_occupation = None

    if args.load_transD:
        modelD.load(args.saved_path)

    if args.use_cuda:
        modelD.cuda()

    D_monitor = OrderedDict()
    test_val_monitor = OrderedDict()

    optimizerD = optimizer(modelD.parameters(), 'adam_sparse_hyp3', args.lr)
    schedulerD = lr_scheduler(optimizerD, args.decay_lr, args.num_epochs)

    _cst_inds = torch.LongTensor(np.arange(args.num_ent, dtype=np.int64)[:,None]).cuda().repeat(1, args.batch_size//2)
    _cst_s = torch.LongTensor(np.arange(args.batch_size//2)).cuda()
    _cst_s_nb = torch.LongTensor(np.arange(args.batch_size//2,args.batch_size)).cuda()
    _cst_nb = torch.LongTensor(np.arange(args.batch_size)).cuda()

    if args.prefetch_to_gpu:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                  num_workers=0, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                  num_workers=4, pin_memory=True, collate_fn=collate_fn)

    if args.freeze_transD:
        freeze_model(modelD)

    for epoch in tqdm(range(1, args.num_epochs + 1)):
        train(train_loader,epoch,args,train_hash,modelD,optimizerD,\
                schedulerD,tflogger,fairD_gender,optimizer_fairD_gender,\
                fairD_occupation, optimizer_fairD_occupation)
        gc.collect()
        if args.decay_lr:
            if args.decay_lr == 'ReduceLROnPlateau':
                schedulerD.step(monitor['D_loss_epoch_avg'])
            else:
                schedulerD.step()
                # scheduler_fairD.step()

        if epoch % args.valid_freq == 0:
            with torch.no_grad():
                l_ranks, r_ranks = test(test_set, args, all_hash,\
                        modelD,tflogger,subsample=10)
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

            if args.use_attr:
                test_fairness(test_set,args, all_hash, modelD,tflogger,\
                        fairD_gender, attribute='gender',epoch=epoch)
            elif args.use_occ_attr:
                test_fairness(test_set,args, all_hash, modelD,tflogger,\
                        fairD_occupation,attribute='occupation', epoch=epoch)
            elif args.use_multi_attr:
                test_fairness(test_set,args, all_hash, modelD,tflogger,\
                        fairD_gender, attribute='gender',epoch=epoch)
                test_fairness(test_set,args, all_hash, modelD,tflogger,\
                        fairD_occupation,attribute='occupation', epoch=epoch)
            else:
                pass

            joblib.dump({'l_ranks':l_ranks, 'r_ranks':r_ranks}, args.outname_base+'epoch{}_validation_ranks.pkl'.format(epoch), compress=9)
            if args.do_log: # Tensorboard logging
                tflogger.scalar_summary('Mean Rank',float(avg_mr),epoch)
                tflogger.scalar_summary('Mean Reciprocal Rank',float(avg_mrr),epoch)
                tflogger.scalar_summary('Hit @10',float(avg_h10),epoch)
                tflogger.scalar_summary('Hit @5',float(avg_h5),epoch)

            modelD.save(args.outname_base+'D_epoch{}.pts'.format(epoch))

        if epoch % (args.valid_freq * 5) == 0:
            l_ranks, r_ranks = test(test_set,args, all_hash,\
                    modelD,tflogger)
            l_mean = l_ranks.mean()
            r_mean = r_ranks.mean()
            l_mrr = (1. / l_ranks).mean()
            r_mrr = (1. / r_ranks).mean()
            l_h10 = (l_ranks <= 10).mean()
            r_h10 = (r_ranks <= 10).mean()
            l_h5 = (l_ranks <= 5).mean()
            r_h5 = (r_ranks <= 5).mean()

    l_ranks, r_ranks = test(test_set,args, all_hash, modelD,tflogger)
    l_mean = l_ranks.mean()
    r_mean = r_ranks.mean()
    l_mrr = (1. / l_ranks).mean()
    r_mrr = (1. / r_ranks).mean()
    l_h10 = (l_ranks <= 10).mean()
    r_h10 = (r_ranks <= 10).mean()
    l_h5 = (l_ranks <= 5).mean()
    r_h5 = (r_ranks <= 5).mean()

    modelD.save(args.outname_base+'D_final.pts')
    joblib.dump({'l_ranks':l_ranks, 'r_ranks':r_ranks}, args.outname_base+'test_ranks.pkl', compress=9)

if __name__ == '__main__':
    main(parse_args())
