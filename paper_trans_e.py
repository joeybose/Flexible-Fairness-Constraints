from comet_ml import Experiment
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
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import f1_score
import numpy as np
import random
import argparse
import pickle
import json
import logging
import sys, os
import subprocess
from model import FBDemParDisc,AttributeFilter
from tqdm import tqdm
from utils import create_or_append, compute_rank, NodeClassification
import joblib
from collections import Counter
import ipdb
sys.path.append('../')
import gc
from collections import OrderedDict

ftensor = torch.FloatTensor
ltensor = torch.LongTensor

v2np = lambda v: v.data.cpu().numpy()

USE_SPARSE_EMB = False

def apply_filters_single_node(p_lhs_emb,masked_filter_set):
    ''' Doesnt Have Masked Filters yet '''
    filter_l_emb, filter_r_emb = 0,0
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

def mask_fairDiscriminators(discriminators, mask):
    # compress('ABCDEF', [1,0,1,0,1,1]) --> A C E F
    return (d for d, s in zip(discriminators, mask) if s)

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
                    ent_embed = apply_filters_single_node(ent_embed,filters)
        return ent_embed

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.loaded = True
        self.load_state_dict(torch.load(fn))




# class TransD(nn.Module):
    # def __init__(self, num_ent, num_rel, embed_dim, p):
        # super(TransD, self).__init__()
        # self.num_ent = num_ent
        # self.num_rel = num_rel
        # self.embed_dim = embed_dim
        # self.p = p

        # r = 6 / np.sqrt(self.embed_dim)

        # self._ent_embeds = nn.Embedding(self.num_ent, self.embed_dim, max_norm=1, norm_type=2, sparse=USE_SPARSE_EMB)
        # self.rel_embeds = nn.Embedding(self.num_rel, self.embed_dim, max_norm=1, norm_type=2, sparse=USE_SPARSE_EMB)

        # self.ent_transfer = nn.Embedding(self.num_ent, self.embed_dim, max_norm=1, norm_type=2, sparse=USE_SPARSE_EMB)
        # self.rel_transfer = nn.Embedding(self.num_rel, self.embed_dim, max_norm=1, norm_type=2, sparse=USE_SPARSE_EMB)

        # self._ent_embeds.weight.data.uniform_(-r, r)#.renorm_(p=2, dim=1, maxnorm=1)
        # self.rel_embeds.weight.data.uniform_(-r, r)#.renorm_(p=2, dim=1, maxnorm=1)

    # def transfer(self, emb, e_transfer, r_transfer):
        # return emb + (emb * e_transfer).sum(dim=1, keepdim=True) * r_transfer

    # #@profile
    # def ent_embeds(self, idx, rel_idx):
        # es = self._ent_embeds(idx)
        # ts = self.ent_transfer(idx)

        # rel_ts = self.rel_transfer(rel_idx)
        # proj_es = self.transfer(es, ts, rel_ts)
        # return proj_es

    # def forward(self, triplets, return_ent_emb=False):
        # lhs_idxs = triplets[:, 0]
        # rel_idxs = triplets[:, 1]
        # rhs_idxs = triplets[:, 2]

        # rel_es = self.rel_embeds(rel_idxs)

        # lhs = self.ent_embeds(lhs_idxs, rel_idxs)
        # rhs = self.ent_embeds(rhs_idxs, rel_idxs)

        # if not return_ent_emb:
            # enrgs = (lhs + rel_es - rhs).norm(p=self.p, dim=1)
            # return enrgs
        # else:
            # enrgs = (lhs + rel_es - rhs).norm(p=self.p, dim=1)
            # return enrgs,lhs,rhs

    # def get_embed(self, ents, rel_idxs):
        # ent_embed = self.ent_embeds(ents, rel_idxs)
        # return ent_embed

    # def save(self, fn):
        # torch.save(self.state_dict(), fn)

    # def load(self, fn):
        # self.load_state_dict(torch.load(fn))

class DemParDisc(nn.Module):
    def __init__(self, embed_dim, a_idx, attribute_data=None,\
            use_cross_entropy=True):
        super(DemParDisc, self).__init__()
        self.embed_dim = int(embed_dim)
        self.a_idx = a_idx
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
            self.sensitive_weight = 1-float(self.most_common[a_idx[0]][1]) / sum(self.attr_count.values())
            self.weights = torch.Tensor((1-self.sensitive_weight,self.sensitive_weight)).cuda()

    def forward(self, ents_emb, ents):
        h1 = F.leaky_relu(self.W1(ents_emb))
        h2 = F.leaky_relu(self.W2(h1))
        h3 = F.leaky_relu(self.W3(h2))
        scores = self.W4(h3)
        A_labels = Variable(torch.Tensor(self.attr_mat[ents.cpu()][:,self.a_idx])).cuda()
        if self.cross_entropy:
            fair_penalty = F.binary_cross_entropy_with_logits(scores,\
                    A_labels,weight=self.weights)
        else:
            probs = torch.sigmoid(scores)
            fair_penalty = F.l1_loss(probs,A_labels,reduction='elementwise_mean')
        return fair_penalty

    def predict(self, ents_emb, ents, return_preds=False):
        h1 = F.leaky_relu(self.W1(ents_emb))
        h2 = F.leaky_relu(self.W2(h1))
        h3 = F.leaky_relu(self.W3(h2))
        scores = self.W4(h3)
        A_labels = Variable(torch.Tensor(self.attr_mat[ents.cpu()][:,self.a_idx])).cuda()
        # if self.cross_entropy:
            # probs = F.binary_cross_entropy_with_logits(scores,A_labels,reduction='none')
            # preds = (probs > torch.Tensor([0.5]).cuda()).float() * 1
        # else:
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
    def __init__(self, path, attribute_data=None,prefetch_to_gpu=False):
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
    parser.add_argument('--show_tqdm', type=int, default=1, help='')
    parser.add_argument('--use_trained_filters', type=bool, default=False, help='Sample a binary mask for discriminators to use')
    parser.add_argument('--test_new_disc', action='store_true', help="Load TransD")
    parser.add_argument('--sample_mask', type=bool, default=False, help='Sample a binary mask for discriminators to use')
    parser.add_argument('--dataset', type=str, default='FB15k', help='Knowledge base version (default: WN)')
    parser.add_argument('--num_classifier_epochs', type=int, default=50, help='Number of training epochs (default: 500)')
    parser.add_argument('--save_dir', type=str, default='./results/', help="output path")
    parser.add_argument('--do_log', action='store_true', help="whether to log to csv")
    parser.add_argument('--api_key', type=str, default=" ", help="Api key for Comet ml")
    parser.add_argument('--project_name', type=str, default=" ", help="Comet project_name")
    parser.add_argument('--workspace', type=str, default=" ", help="Comet Workspace")
    parser.add_argument('--load_transD', action='store_true', help="Load TransD")
    parser.add_argument('--freeze_transD', action='store_true', help="Load TransD")
    parser.add_argument('--use_cross_entropy', action='store_true', help="DemPar Discriminators Loss as CE")
    parser.add_argument('--D_steps', type=int, default=5, help='Number of D steps')
    parser.add_argument('--remove_old_run', action='store_true', help="remove old run")
    parser.add_argument('--data_dir', type=str, default='./data/', help="Contains Pickle files")
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs (default: 500)')
    parser.add_argument('--batch_size', type=int, default=16000, help='Batch size (default: 512)')
    parser.add_argument('--valid_freq', type=int, default=20, help='Validate frequency in epochs (default: 50)')
    parser.add_argument('--print_freq', type=int, default=5, help='Print frequency in epochs (default: 5)')
    parser.add_argument('--embed_dim', type=int, default=50, help='Embedding dimension (default: 50)')
    parser.add_argument('--z_dim', type=int, default=100, help='noise Embedding dimension (default: 100)')
    parser.add_argument('--gamma', type=int, default=0.1, help='Tradeoff for Adversarial Penalty')
    parser.add_argument('--lr', type=float, default=0.008, help='Learning rate (default: 0.001)')
    parser.add_argument('--margin', type=float, default=3, help='Loss margin (default: 1)')
    parser.add_argument('--p', type=int, default=1, help='P value for p-norm (default: 1)')
    parser.add_argument('--prefetch_to_gpu', type=int, default=0, help="")
    parser.add_argument('--D_nce_weight', type=float, default=1, help="D nce term weight")
    parser.add_argument('--full_loss_penalty', type=int, default=0, help="")
    parser.add_argument('--filter_false_negs', type=int, default=1, help="filter out sampled false negatives")
    parser.add_argument('--ace', type=int, default=0, help="do ace training (otherwise just NCE)")
    parser.add_argument('--false_neg_penalty', type=float, default=1., help="false neg penalty for G")
    parser.add_argument('--mb_reward_normalization', type=int, default=0, help="minibatch based reward normalization")
    parser.add_argument('--n_proposal_samples', type=int, default=10, help="")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--use_attr', type=bool, default=False, help='Use Attribute Matrix')
    parser.add_argument('--use_0_attr', type=bool, default=False, help='Use Only 0 Attribute')
    parser.add_argument('--use_1_attr', type=bool, default=False, help='Use Only 1 Attribute')
    parser.add_argument('--use_2_attr', type=bool, default=False, help='Use Only 2 Attribute')
    parser.add_argument('--decay_lr', type=str, default='halving_step100', help='lr decay mode')
    parser.add_argument('--optim_mode', type=str, default='adam_hyp2', help='optimizer')
    parser.add_argument('--fairD_optim_mode', type=str, default='adam_hyp2',help='optimizer for Fairness Discriminator')
    parser.add_argument('--namestr', type=str, default='', help='additional info in output filename to help identify experiments')

    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()

    if args.dataset == 'WN' or args.dataset == 'FB15k':
        path = './data/' + args.dataset + '-%s.pkl'

        args.num_ent = len(json.load(open('./data/%s-ent_to_idx.json' % args.dataset, 'r')))
        args.num_rel = len(json.load(open('./data/%s-rel_to_idx.json' % args.dataset, 'r')))
        args.data_path = path
    else:
        raise Exception("Argument 'dataset' can only be 'WN' or 'FB15k'.")

    # if args.use_attr:
        # args.attr_mat = os.path.join(args.data_dir,\
                # 'Attributes_FB15k-train.pkl')
        # args.ent_to_idx = os.path.join(args.data_dir,\
                # 'Attributes_FB15k-ent_to_idx.json')
        # args.attr_to_idx = os.path.join(args.data_dir,\
                # 'Attributes_FB15k-attr_to_idx.json')
        # args.reindex_attr_idx = os.path.join(args.data_dir,\
                # 'Attributes_FB15k-reindex_attr_to_idx.json')
        # args.attr_count = os.path.join(args.data_dir,\
                # 'Attributes_FB15k-attr_count.json')
    # else:
        # args.attr_mat = None
        # args.ent_to_idx = None
        # args.attr_to_idx = None
        # args.reindex_attr_idx = None
    args.attr_mat = os.path.join(args.data_dir,\
            'Attributes_FB15k-train.pkl')
    args.ent_to_idx = os.path.join(args.data_dir,\
            'Attributes_FB15k-ent_to_idx.json')
    args.attr_to_idx = os.path.join(args.data_dir,\
            'Attributes_FB15k-attr_to_idx.json')
    args.reindex_attr_idx = os.path.join(args.data_dir,\
            'Attributes_FB15k-reindex_attr_to_idx.json')
    args.attr_count = os.path.join(args.data_dir,\
            'Attributes_FB15k-attr_count.json')
    args.fair_att_0 = 0
    args.fair_att_1 = 1
    args.fair_att_2 = 2
    args.saved_path = os.path.join(args.save_dir,'Updated_Paper_FB15kD_final.pts')
    # args.fair_att = np.random.choice(1,1) # Pick a random sensitive attribute
    args.outname_base = os.path.join(args.save_dir,\
            args.namestr+'FB_results')

    args.filter_0_saved_path = args.outname_base + 'Filter_0.pts'
    args.filter_1_saved_path = args.outname_base + 'Filter_1.pts'
    args.filter_2_saved_path = args.outname_base + 'Filter_2.pts'
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

def main(args):
    if args.dataset in ('FB15k-237', 'kinship', 'nations', 'umls', 'WN18RR', 'YAGO3-10'):
        S = joblib.load(args.data_path)
        train_set = KBDataset(S['train_data'], args.prefetch_to_gpu)
        valid_set = KBDataset(S['val_data'], attr_data)
        test_set = KBDataset(S['test_data'], attr_data)
    else:
        train_set = KBDataset(args.data_path % 'train', args.prefetch_to_gpu)
        valid_set = KBDataset(args.data_path % 'valid')
        test_set = KBDataset(args.data_path % 'test')
        print('50 Most Common Attributes')

    if args.prefetch_to_gpu:
        train_hash = set([r.tobytes() for r in train_set.dataset.cpu().numpy()])
    else:
        train_hash = set([r.tobytes() for r in train_set.dataset])

    cutoff_constant = 0.8
    train_cutoff_row = int(np.round(args.num_ent*cutoff_constant))
    args.cutoff_row = train_cutoff_row
    all_ents = np.arange(args.num_ent)
    np.random.shuffle(all_ents)
    all_hash = train_hash.copy()
    all_hash.update(set([r.tobytes() for r in valid_set.dataset]))
    all_hash.update(set([r.tobytes() for r in test_set.dataset]))
    logdir = args.outname_base + '_logs' + '/'
    if args.remove_old_run:
        shutil.rmtree(logdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    tflogger = tfLogger(logdir)
    train_fairness_set = NodeClassification(all_ents[:args.cutoff_row], args.prefetch_to_gpu)
    test_fairness_set = NodeClassification(all_ents[args.cutoff_row:], args.prefetch_to_gpu)
    ''' Comet Logging '''
    experiment = Experiment(api_key=args.api_key, disabled= not args.do_log
                        ,project_name=args.project_name,workspace=args.workspace)
    experiment.set_name(args.namestr)
    modelD = TransD(args.num_ent, args.num_rel, args.embed_dim, args.p).cuda()
    fairD_0, fairD_1, fairD_2 = None,None,None
    optimizer_fairD_0, optimizer_fairD_1, optimizer_fairD_2 = None,None,None
    filter_0, filter_1, filter_2 = None, None, None

    if args.load_transD:
        modelD.load(args.saved_path)

    if args.use_cuda:
        modelD.cuda()

    if args.use_attr:
        ''' Hard Coded to the most common attribute for now '''
        attr_data = [args.attr_mat,args.ent_to_idx,args.attr_to_idx,\
                args.reindex_attr_idx,args.attr_count]
        fairD_0 = FBDemParDisc(args.embed_dim,args.fair_att_0,'0',attr_data,args.use_cross_entropy)
        fairD_1 = FBDemParDisc(args.embed_dim,args.fair_att_1,'1',attr_data,args.use_cross_entropy)
        fairD_2 = FBDemParDisc(args.embed_dim,args.fair_att_2,'2',attr_data,args.use_cross_entropy)
        most_common_attr = [print(fairD_0.inv_attr_map[int(k)]) for k in \
                fairD_0.reindex_to_idx.keys()]

        ''' Initialize Optimizers '''
        if args.sample_mask:
            filter_0 = AttributeFilter(args.embed_dim,attribute='0')
            filter_1 = AttributeFilter(args.embed_dim,attribute='1')
            filter_2 = AttributeFilter(args.embed_dim,attribute='2')
            filter_0.cuda()
            filter_1.cuda()
            filter_2.cuda()
            optimizer_fairD_0 = optimizer(fairD_0.parameters(),'adam', args.lr)
            optimizer_fairD_1 = optimizer(fairD_1.parameters(),'adam',args.lr)
            optimizer_fairD_2 = optimizer(fairD_2.parameters(),'adam', args.lr)
        elif args.use_trained_filters and not args.sample_mask:
            filter_0 = AttributeFilter(args.embed_dim,attribute='0')
            filter_1 = AttributeFilter(args.embed_dim,attribute='1')
            filter_2 = AttributeFilter(args.embed_dim,attribute='2')
            filter_0.cuda()
            filter_1.cuda()
            filter_2.cuda()
        else:
            optimizer_fairD_0 = optimizer(fairD_0.parameters(),'adam', args.lr)
            optimizer_fairD_1 = optimizer(fairD_1.parameters(),'adam',args.lr)
            optimizer_fairD_2 = optimizer(fairD_2.parameters(),'adam', args.lr)
            filter_0, filter_1, filter_2 = None, None, None

        if args.use_cuda:
            fairD_0.cuda()
            fairD_1.cuda()
            fairD_2.cuda()

    elif args.use_1_attr:
        attr_data = [args.attr_mat,args.ent_to_idx,args.attr_to_idx,\
                args.reindex_attr_idx,args.attr_count]
        fairD_1 = FBDemParDisc(args.embed_dim,args.fair_att_1,'1',attr_data,\
                use_cross_entropy=args.use_cross_entropy)
        fairD_1.cuda()
        optimizer_fairD_1 = optimizer(fairD_1.parameters(),'adam',args.lr)
    elif args.use_0_attr:
        attr_data = [args.attr_mat,args.ent_to_idx,args.attr_to_idx,\
                args.reindex_attr_idx,args.attr_count]
        fairD_0 = FBDemParDisc(args.embed_dim,args.fair_att_0,'0',attr_data,\
               use_cross_entropy=args.use_cross_entropy)
        optimizer_fairD_0 = optimizer(fairD_0.parameters(),'adam', args.lr)
        fairD_0.cuda()
    elif args.use_2_attr:
        attr_data = [args.attr_mat,args.ent_to_idx,args.attr_to_idx,\
                args.reindex_attr_idx,args.attr_count]
        fairD_2 = FBDemParDisc(args.embed_dim,args.fair_att_2,'2',attr_data,\
                use_cross_entropy=args.use_cross_entropy)
        optimizer_fairD_2 = optimizer(fairD_2.parameters(),'adam', args.lr)
        fairD_2.cuda()
    # if args.use_attr:
        # ''' Hard Coded to the most common attribute for now '''
        # attr_data = [args.attr_mat,args.ent_to_idx,args.attr_to_idx,\
                # args.reindex_attr_idx,args.attr_count]
        # fairD = DemParDisc(args.embed_dim,args.fair_att,attr_data,args.use_cross_entropy)
        # fairD.cuda()
        # most_common_attr = [print(fairD.inv_attr_map[int(k)]) for k in \
                # fairD.reindex_to_idx.keys()]
        # optimizer_fairD = optimizer(fairD.parameters(), 'adam', args.lr)
        # scheduler_fairD = lr_scheduler(optimizer_fairD, args.decay_lr, args.num_epochs)

    D_monitor = OrderedDict()
    test_val_monitor = OrderedDict()
    ''' Create Sets '''
    fairD_set = [fairD_0,fairD_1,fairD_2]
    filter_set = [filter_0,filter_1,filter_2]
    optimizer_fairD_set = [optimizer_fairD_0, optimizer_fairD_1,\
            optimizer_fairD_2]

    if args.sample_mask and not args.use_trained_filters:
        optimizerD = optimizer(list(modelD.parameters()) + \
                list(filter_0.parameters()) + \
                list(filter_1.parameters()) + \
                list(filter_2.parameters()), 'adam', args.lr)
    else:
        optimizerD = optimizer(modelD.parameters(), 'adam', args.lr)
    schedulerD = lr_scheduler(optimizerD, args.decay_lr, args.num_epochs)

    loss_func = MarginRankingLoss(args.margin)

    _cst_inds = torch.LongTensor(np.arange(args.num_ent, dtype=np.int64)[:,None]).cuda().repeat(1, args.batch_size//2)
    _cst_s = torch.LongTensor(np.arange(args.batch_size//2)).cuda()
    _cst_s_nb = torch.LongTensor(np.arange(args.batch_size//2,args.batch_size)).cuda()
    _cst_nb = torch.LongTensor(np.arange(args.batch_size)).cuda()


    def train(data_loader, counter, args, train_hash, modelD, optimizerD,\
            tflogger, fairD_set, optimizer_fairD_set, filter_set, experiment):

        correct = 0
        total_ent = 0
        fairD_0_loss, fairD_1_loss, fairD_2_loss = 0,0,0
        loss_func = MarginRankingLoss(args.margin)
        if args.show_tqdm:
            data_itr = tqdm(enumerate(data_loader))
        else:
            data_itr = enumerate(data_loader)
        for idx, p_batch in data_itr:
            ''' Sample Fairness Discriminators '''
            if args.sample_mask:
                mask = np.random.choice([0, 1], size=(3,))
                masked_fairD_set = list(mask_fairDiscriminators(fairD_set,mask))
                masked_optimizer_fairD_set = list(mask_fairDiscriminators(optimizer_fairD_set,mask))
                masked_filter_set = list(mask_fairDiscriminators(filter_set,mask))
            else:
                ''' No mask applied despite the name '''
                masked_fairD_set = fairD_set
                masked_optimizer_fairD_set = optimizer_fairD_set
                masked_filter_set = filter_set

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

            p_batch_var = Variable(p_batch)
            nce_batch = Variable(nce_batch)
            q_samples = Variable(q_samples)

            ''' Number of Active Discriminators '''
            constant = len(masked_fairD_set) - masked_fairD_set.count(None)
            if args.ace == 0:
                d_ins = torch.cat([p_batch_var, nce_batch], dim=0).contiguous()
                if constant != 0:
                    optimizerD.zero_grad()
                    d_outs,lhs_emb,rhs_emb = modelD(d_ins,True,filters=masked_filter_set)
                    l_penalty,r_penalty = 0,0
                    p_lhs_emb = lhs_emb[:len(p_batch_var)]
                    p_rhs_emb = rhs_emb[:len(p_batch_var)]
                    filter_l_emb = p_lhs_emb
                    filter_r_emb = p_rhs_emb

                    ''' Apply Discriminators '''
                    for fairD_disc in masked_fairD_set:
                        if fairD_disc is not None:
                            l_penalty += fairD_disc(filter_l_emb,p_batch[:,0].cpu(),True)
                            # r_penalty += fairD_disc(filter_r_emb,p_batch[:,2].cpu(),True)

                    # if not args.use_cross_entropy:
                        # fair_penalty = constant - 0.5*(l_penalty + r_penalty)
                    # else:
                    # fair_penalty = -1*(l_penalty + r_penalty)
                    fair_penalty = -1*l_penalty

                    if not args.freeze_transD:
                        p_enrgs = d_outs[:len(p_batch_var)]
                        nce_enrgs = d_outs[len(p_batch_var):(len(p_batch_var)+len(nce_batch))]
                        nce_term, nce_term_scores  = loss_func(p_enrgs, nce_enrgs, weights=(1.-nce_falseNs))
                        lossD = nce_term + args.gamma*fair_penalty
                        lossD.backward()
                        optimizerD.step()

                    for k in range(0,args.D_steps):
                        for fairD_disc, fair_optim in zip(masked_fairD_set,\
                                masked_optimizer_fairD_set):
                            l_penalty_2 = 0
                            r_penalty_2 = 0
                            if fairD_disc is not None and fair_optim is not None:
                                fair_optim.zero_grad()
                                l_penalty_2 = fairD_disc(filter_l_emb.detach(),\
                                        p_batch[:,0].cpu(),True)
                                # r_penalty_2 = fairD_disc(filter_r_emb.detach(),\
                                        # p_batch[:,2].cpu(),True)
                                fairD_loss = l_penalty_2 #+ r_penalty_2
                                fairD_loss.backward(retain_graph=False)
                                fair_optim.step()
                # elif args.freeze_transD:
                    # with torch.no_grad():
                        # d_outs,lhs_emb,rhs_emb = modelD(d_ins,True,filters=masked_filter_set)
                        # p_lhs_emb = lhs_emb[:len(p_batch_var)]
                        # p_rhs_emb = rhs_emb[:len(p_batch_var)]
                        # filter_l_emb = p_lhs_emb
                        # filter_r_emb = p_rhs_emb
                    # for fairD_disc, fair_optim in zip(masked_fairD_set,\
                            # masked_optimizer_fairD_set):
                        # l_penalty_2 = 0
                        # r_penalty_2 = 0
                        # if fairD_disc is not None and fair_optim is not None:
                            # fair_optim.zero_grad()
                            # l_penalty_2 = fairD_disc(filter_l_emb.detach(),\
                                    # p_batch[:,0].cpu(),True)
                            # r_penalty_2 = fairD_disc(filter_r_emb.detach(),\
                                    # p_batch[:,2].cpu(),True)
                            # fairD_loss = l_penalty_2 + r_penalty_2
                            # fairD_loss.backward(retain_graph=False)
                            # fair_optim.step()
                else:
                    d_outs = modelD(d_ins)
                    fair_penalty = Variable(torch.zeros(1)).cuda()
                    p_enrgs = d_outs[:len(p_batch_var)]
                    nce_enrgs = d_outs[len(p_batch_var):(len(p_batch_var)+len(nce_batch))]
                    nce_term, nce_term_scores  = loss_func(p_enrgs, nce_enrgs, weights=(1.-nce_falseNs))
                    lossD = nce_term + args.gamma*fair_penalty
                    lossD.backward()
                    optimizerD.step()

            if constant != 0:
                correct = 0
                correct_0,correct_1,correct_2 = 0,0,0
                for fairD_disc in masked_fairD_set:
                    if fairD_disc is not None:
                        with torch.no_grad():
                            d_outs,lhs_emb,rhs_emb = modelD(d_ins,True,filters=masked_filter_set)
                            p_lhs_emb = lhs_emb[:len(p_batch)]
                            p_rhs_emb = rhs_emb[:len(p_batch)]
                            filter_l_emb = p_lhs_emb
                            filter_r_emb = p_rhs_emb

                            l_preds, l_A_labels, l_probs = fairD_disc.predict(filter_l_emb,\
                                    p_batch[:,0].cpu(),return_preds=True)
                            r_preds, r_A_labels, r_probs = fairD_disc.predict(filter_r_emb,\
                                    p_batch[:,2].cpu(),return_preds=True)
                            l_correct = l_preds.eq(l_A_labels.view_as(l_preds)).sum().item()
                            r_correct = r_preds.eq(r_A_labels.view_as(r_preds)).sum().item()
                            correct += l_correct + r_correct
                            total_ent += 2*len(p_batch)
                            l_AUC = roc_auc_score(l_A_labels.cpu().numpy(),l_probs.cpu().numpy(),average="micro")
                            r_AUC = roc_auc_score(r_A_labels.cpu().numpy(),r_probs.cpu().numpy(),average="micro")
                            AUC = (l_AUC + r_AUC) / 2
                            print("Train %s AUC: %f" %(fairD_disc.attribute,AUC))

                            if fairD_disc.attribute == '0':
                                correct_0 += l_correct + r_correct
                            elif fairD_disc.attribute == '1':
                                correct_1 += l_correct + r_correct
                            else:
                                correct_2 += l_correct + r_correct

                            total_ent += 2*len(p_batch)

                    if args.do_log and fairD_disc is not None: # Tensorboard logging
                        acc_0 = 100. * correct_0 / total_ent
                        acc_1 = 100. * correct_1 / total_ent
                        acc_2 = 100. * correct_2 / total_ent
                        attribute = fairD_disc.attribute

        ''' Logging for end of epoch '''
        if args.do_log: # Tensorboard logging
            tflogger.scalar_summary('TransD Loss',float(lossD),counter)
            if fairD_set[0] is not None:
                experiment.log_metric("Train Fairness Disc 0",float(acc_0),step=counter)
            if fairD_set[1] is not None:
                experiment.log_metric("Train Fairness Disc 1",float(acc_1),step=counter)
            if fairD_set[2] is not None:
                experiment.log_metric("Train Fairness Disc 2",float(acc_2),step=counter)

    def test_attr(args,test_dataset,modelD,net,experiment,\
            epoch,attribute,filter_set=None):
        test_loader = DataLoader(test_dataset, num_workers=1, batch_size=4096)
        # test_data_itr = enumerate(test_loader)
        correct = 0
        l_probs_list, r_probs_list = [], []
        l_labels_list, r_labels_list = [], []
        total_ent = 0
        for triplet in test_loader:
            lhs, rel, rhs = triplet[:,0], triplet[:,1],triplet[:,2]
            l_batch = Variable(lhs).cuda()
            r_batch = Variable(rhs).cuda()
            rel_batch = Variable(rel).cuda()
            lhs_emb = modelD.get_embed(l_batch.detach(),rel_batch.detach(),filter_set)
            rhs_emb = modelD.get_embed(r_batch.detach(),rel_batch.detach(),filter_set)
            l_preds,l_A_labels,l_probs = net.predict(lhs_emb,lhs.cpu(),return_preds=True)
            r_preds, r_A_labels,r_probs = net.predict(rhs_emb,rhs.cpu(),return_preds=True)
            l_correct = l_preds.eq(l_A_labels.view_as(l_preds)).sum().item()
            r_correct = r_preds.eq(r_A_labels.view_as(r_preds)).sum().item()
            l_probs_list.append(l_probs)
            r_probs_list.append(r_probs)
            l_labels_list.append(l_A_labels)
            r_labels_list.append(r_A_labels)
            correct += l_correct #+ r_correct
            total_ent += len(lhs_emb) #+ len(rhs_emb)

        cat_l_labels_list = torch.cat(l_labels_list,0).data.cpu().numpy()
        cat_r_labels_list = torch.cat(r_labels_list,0).data.cpu().numpy()
        cat_l_probs_list = torch.cat(l_probs_list,0).data.cpu().numpy()
        cat_r_probs_list = torch.cat(r_probs_list,0).data.cpu().numpy()
        l_AUC = roc_auc_score(cat_l_labels_list,cat_l_probs_list,average="micro")
        r_AUC = roc_auc_score(cat_r_labels_list,cat_r_probs_list,average="micro")
        AUC = (l_AUC + r_AUC) / 2
        AUC = l_AUC
        acc = 100. * correct / total_ent
        print("Test %s Accuracy is: %f AUC: %f" %(attribute,acc,AUC))
        if args.do_log:
            experiment.log_metric("Test "+attribute+" AUC",float(AUC),step=epoch)
            experiment.log_metric("Test "+attribute+" Accuracy",float(acc),step=epoch)

    def train_attr(args,modelD,train_dataset,test_dataset,\
            attr_data,experiment,filter_set=None,attribute=None):
        freeze_model(modelD)
        if attribute == '0':
            attr_data = [args.attr_mat,args.ent_to_idx,args.attr_to_idx,\
                    args.reindex_attr_idx,args.attr_count]
            net = FBDemParDisc(args.embed_dim,args.fair_att_0,'0',attr_data,
                    use_cross_entropy=args.use_cross_entropy)
            net.cuda()
        elif attribute == '1':
            attr_data = [args.attr_mat,args.ent_to_idx,args.attr_to_idx,\
                    args.reindex_attr_idx,args.attr_count]
            net = FBDemParDisc(args.embed_dim,args.fair_att_1,'1',attr_data,\
                    use_cross_entropy=args.use_cross_entropy)
            net.cuda()
        elif attribute == '2':
            attr_data = [args.attr_mat,args.ent_to_idx,args.attr_to_idx,\
                    args.reindex_attr_idx,args.attr_count]
            net = FBDemParDisc(args.embed_dim,args.fair_att_2,'2',attr_data,\
                    use_cross_entropy=args.use_cross_entropy)
            net.cuda()

        modelD.eval()
        opt = optimizer(net.parameters(),'adam', args.lr)
        train_loader = DataLoader(train_dataset, num_workers=1,
                batch_size=16000)
        criterion = nn.BCELoss()

        for epoch in range(1,args.num_classifier_epochs + 1):
            correct = 0
            if epoch % 10 == 0:
                test_attr(args,test_dataset,modelD,net,experiment,epoch,attribute,filter_set)

            for triplet in train_loader:
                lhs, rel, rhs = triplet[:,0], triplet[:,1],triplet[:,2]
                l_batch = Variable(lhs).cuda()
                r_batch = Variable(rhs).cuda()
                rel_batch = Variable(rel).cuda()
                lhs_emb = modelD.get_embed(l_batch,rel_batch,filter_set)
                # rhs_emb = modelD.get_embed(r_batch,rel_batch,filter_set)
                opt.zero_grad()
                l_y_hat, l_y = net(lhs_emb.detach(),lhs)
                l_loss = criterion(l_y_hat, l_y)
                # r_y_hat, r_y = net(rhs_emb.detach(),rhs)
                # r_loss = criterion(r_y_hat, r_y)
                loss = l_loss #+ r_loss
                loss.backward()
                opt.step()
                preds = (l_y_hat > torch.Tensor([0.5]).cuda()).float() * 1
                correct = preds.eq(l_y.view_as(preds)).sum().item()
                acc = 100. * correct / len(lhs)
                AUC = roc_auc_score(l_y.data.cpu().numpy(),\
                        l_y_hat.data.cpu().numpy(),average="micro")
                # f1 = f1_score(l_y.data.cpu().numpy(), preds.data.cpu().numpy(),\
                        # average='binary')

            print("Train %s Attrbitue Loss is %f Accuracy is: %f AUC: %f"\
                    %(attribute,loss,acc,AUC))
            if args.do_log:
                experiment.log_metric("Train "+ attribute+"\
                         AUC",float(AUC),step=epoch)

    def test_fairness(dataset,args,modelD,tflogger,fairD,attribute,\
            epoch,experiment,filter_=None):
        test_loader = DataLoader(dataset, num_workers=4, batch_size=8192, collate_fn=collate_fn)
        correct = 0
        total_ent = 0
        if args.show_tqdm:
            data_itr = tqdm(enumerate(test_loader))
        else:
            data_itr = enumerate(test_loader)

        l_probs_list, r_probs_list = [], []
        l_labels_list, r_labels_list = [], []
        for triplet in test_loader:
            lhs, rel, rhs = triplet[:,0], triplet[:,1],triplet[:,2]
            l_batch = Variable(lhs).cuda()
            r_batch = Variable(rhs).cuda()
            rel_batch = Variable(rel).cuda()
            lhs_emb = modelD.get_embed(l_batch,rel_batch,[filter_])
            rhs_emb = modelD.get_embed(r_batch,rel_batch,[filter_])
            l_preds,l_A_labels,l_probs = fairD.predict(lhs_emb,lhs.cpu(),return_preds=True)
            r_preds, r_A_labels,r_probs = fairD.predict(rhs_emb,rhs.cpu(),return_preds=True)
            l_correct = l_preds.eq(l_A_labels.view_as(l_preds)).sum().item()
            r_correct = r_preds.eq(r_A_labels.view_as(r_preds)).sum().item()
            l_probs_list.append(l_probs)
            r_probs_list.append(r_probs)
            l_labels_list.append(l_A_labels)
            r_labels_list.append(r_A_labels)
            correct += l_correct + r_correct
            total_ent += len(lhs_emb) + len(rhs_emb)

        cat_l_labels_list = torch.cat(l_labels_list,0).data.cpu().numpy()
        cat_r_labels_list = torch.cat(r_labels_list,0).data.cpu().numpy()
        cat_l_probs_list = torch.cat(l_probs_list,0).data.cpu().numpy()
        cat_r_probs_list = torch.cat(r_probs_list,0).data.cpu().numpy()
        l_AUC = roc_auc_score(cat_l_labels_list,cat_l_probs_list,average="micro")
        r_AUC = roc_auc_score(cat_r_labels_list,cat_r_probs_list,average="micro")
        AUC = (l_AUC + r_AUC) / 2
        acc = 100. * correct / total_ent
        print("Test %s Accuracy is: %f AUC: %f" %(attribute,acc,AUC))
        if args.do_log:
            tflogger.scalar_summary(attribute+'_Valid Fairness Discriminator,\
                    Accuracy',float(acc),epoch)
            experiment.log_metric("Test "+attribute+" AUC",float(AUC),step=epoch)
            experiment.log_metric("Test "+attribute+" Accuracy",float(acc),step=epoch)

    def test(dataset, args, all_hash, modelD, tflogger, filter_set, experiment, subsample=1):
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
            d_outs = modelD(d_ins,filters=filter_set)
            l_enrgs = d_outs[:len(l_batch)]
            r_enrgs = d_outs[len(l_batch):]

            l_rank = compute_rank(v2np(l_enrgs), lhs, mask_observed=l_fns)
            r_rank = compute_rank(v2np(r_enrgs), rhs, mask_observed=r_fns)

            l_ranks.append(l_rank)
            r_ranks.append(r_rank)

        l_ranks = np.array(l_ranks)
        r_ranks = np.array(r_ranks)

        return l_ranks, r_ranks

    def retrain_disc(args,train_loader,train_hash,test_set,modelD,optimizerD,tflogger,\
            filter_0,filter_1,filter_2,attribute):

        if args.use_trained_filters:
            print("Retrain New Discriminator with Filter on %s" %(attribute))
        else:
            print("Retrain New Discriminator on %s" %(attribute))

        ''' Reset some flags '''
        args.use_cross_entropy = True
        args.sample_mask = False
        args.freeze_transD = True
        new_fairD_0,new_fairD_1,new_fairD_2 = None,None,None
        new_optimizer_fairD_0,new_optimizer_fairD_1,new_optimizer_fairD_2 = None,None,None

        if attribute == '0':
            args.use_0_attr = True
            args.use_1_attr = False
            args.use_2_attr = False
            args.use_attr = False
        elif attribute =='1':
            args.use_0_attr = False
            args.use_1_attr = True
            args.use_2_attr = False
            args.use_attr = False
        elif attribute =='2':
            args.use_0_attr = False
            args.use_1_attr = False
            args.use_2_attr = True
            args.use_attr = False
        else:
            args.use_0_attr = False
            args.use_1_attr = False
            args.use_2_attr = False
            args.use_attr = True

        '''Retrain Discriminator on Frozen TransD Model '''
        if args.use_1_attr:
            attr_data = [args.attr_mat,args.ent_to_idx,args.attr_to_idx,\
                    args.reindex_attr_idx,args.attr_count]
            new_fairD_1 = FBDemParDisc(args.embed_dim,args.fair_att_1,'1',attr_data,
                    use_cross_entropy=args.use_cross_entropy)
            new_fairD_1.cuda()
            new_optimizer_fairD_1 = optimizer(new_fairD_1.parameters(),'adam')
            fairD_disc = new_fairD_1
            fair_optim = new_optimizer_fairD_1
        elif args.use_0_attr:
            attr_data = [args.attr_mat,args.ent_to_idx,args.attr_to_idx,\
                    args.reindex_attr_idx,args.attr_count]
            new_fairD_0 = FBDemParDisc(args.embed_dim,args.fair_att_0,'0',attr_data,\
                    use_cross_entropy=args.use_cross_entropy)
            new_optimizer_fairD_0 = optimizer(new_fairD_0.parameters(),'adam')
            new_fairD_0.cuda()
            fairD_disc = new_fairD_0
            fair_optim = new_optimizer_fairD_0
        elif args.use_2_attr:
            attr_data = [args.attr_mat,args.ent_to_idx,args.attr_to_idx,\
                    args.reindex_attr_idx,args.attr_count]
            new_fairD_2 = FBDemParDisc(args.embed_dim,args.fair_att_2,'2',attr_data,\
                    use_cross_entropy=args.use_cross_entropy)
            new_optimizer_fairD_2 = optimizer(new_fairD_2.parameters(),'adam')
            new_fairD_2.cuda()
            fairD_disc = new_fairD_2
            fair_optim = new_optimizer_fairD_2

        attr_data = [args.attr_mat,args.ent_to_idx,args.attr_to_idx,\
                args.reindex_attr_idx,args.attr_count]
        new_fairD_set = [new_fairD_0,new_fairD_1,new_fairD_2]
        new_optimizer_fairD_set = [new_optimizer_fairD_0,new_optimizer_fairD_1,new_optimizer_fairD_2]
        if args.use_trained_filters:
            filter_set = [filter_0,filter_1,filter_2]
        else:
            filter_set = [None,None,None]

        ''' Freeze Model + Filters '''
        for filter_ in filter_set:
            if filter_ is not None:
                freeze_model(filter_)
        freeze_model(modelD)

        for epoch in tqdm(range(1, args.num_epochs + 1)):
            train(train_loader,epoch,args,train_hash,modelD,optimizerD,\
                    tflogger,new_fairD_set,new_optimizer_fairD_set,filter_set,experiment)
            gc.collect()
            if args.decay_lr:
                if args.decay_lr == 'ReduceLROnPlateau':
                    schedulerD.step(monitor['D_loss_epoch_avg'])
                else:
                    pass
                    # schedulerD.step()

            if epoch % args.valid_freq == 0:
                if args.use_attr:
                    test_fairness(test_set,args, modelD,tflogger,\
                            new_fairD_0,attribute='0',\
                            epoch=epoch,experiment=experiment,filter_=filter_0)
                    test_fairness(test_set,args,modelD,tflogger,\
                            new_fairD_1,attribute='1',epoch=epoch,\
                            experiment=experiment,filter_=filter_1)
                    test_fairness(test_set,args, modelD,tflogger,\
                            new_fairD_2,attribute='2',epoch=epoch,\
                            experiment=experiment,filter_=filter_2)
                elif args.use_0_attr:
                    test_fairness(test_set,args,modelD,tflogger,\
                            new_fairD_0,attribute='0',epoch=epoch,\
                            experiment=experiment,filter_=filter_0)
                elif args.use_1_attr:
                    test_fairness(test_set,args,modelD,tflogger,\
                            new_fairD_1,attribute='1',epoch=epoch,\
                            experiment=experiment,filter_=filter_1)
                elif args.use_2_attr:
                    test_fairness(test_set,args,modelD,tflogger,\
                            new_fairD_2,attribute='2',epoch=epoch,\
                            experiment=experiment,filter_=filter_2)

    if args.prefetch_to_gpu:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                  num_workers=0, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                  num_workers=4, pin_memory=True, collate_fn=collate_fn)

    if args.freeze_transD:
        freeze_model(modelD)

    with experiment.train():
        for epoch in tqdm(range(1, args.num_epochs + 1)):
            train(train_loader,epoch,args,train_hash,modelD,optimizerD,\
                    tflogger,fairD_set,optimizer_fairD_set,filter_set,experiment)
            gc.collect()
            if args.decay_lr:
                if args.decay_lr == 'ReduceLROnPlateau':
                    schedulerD.step(monitor['D_loss_epoch_avg'])
                else:
                    schedulerD.step()
                    # scheduler_fairD.step()

            if epoch % args.valid_freq == 0:
                with torch.no_grad():
                    l_ranks, r_ranks = test(valid_set,args,all_hash,\
                            modelD,tflogger,filter_set,experiment,subsample=20)
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

                print("Mean Rank is %f" %(float(avg_mr)))
                if args.use_attr:
                    test_fairness(test_set,args, modelD,tflogger,\
                            fairD_0,attribute='0',\
                            epoch=epoch,experiment=experiment,filter_=filter_0)
                    test_fairness(test_set,args,modelD,tflogger,\
                            fairD_1,attribute='1',epoch=epoch,\
                            experiment=experiment,filter_=filter_1)
                    test_fairness(test_set,args, modelD,tflogger,\
                            fairD_2,attribute='2',epoch=epoch,\
                            experiment=experiment,filter_=filter_2)
                elif args.use_0_attr:
                    test_fairness(test_set,args,modelD,tflogger,\
                            fairD_0,attribute='0',epoch=epoch,\
                            experiment=experiment,filter_=filter_0)
                elif args.use_1_attr:
                    test_fairness(test_set,args,modelD,tflogger,\
                            fairD_1,attribute='1',epoch=epoch,\
                            experiment=experiment,filter_=filter_1)
                elif args.use_2_attr:
                    test_fairness(test_set,args,modelD,tflogger,\
                            fairD_2,attribute='2',epoch=epoch,\
                            experiment=experiment,filter_=filter_2)

                if args.do_log: # Tensorboard logging
                    tflogger.scalar_summary('Mean Rank',float(avg_mr),epoch)
                    tflogger.scalar_summary('Mean Reciprocal Rank',float(avg_mrr),epoch)
                    tflogger.scalar_summary('Hit @10',float(avg_h10),epoch)
                    tflogger.scalar_summary('Hit @5',float(avg_h5),epoch)
                    experiment.log_metric("Mean Rank",float(avg_mr),step=epoch)

                modelD.save(args.outname_base+'D_epoch{}.pts'.format(epoch))

        l_ranks, r_ranks = test(test_set,args,all_hash,\
                modelD,tflogger,filter_set,experiment,subsample=1)
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
        print("Mean Rank is %f" %(float(avg_mr)))
        if args.do_log: # Tensorboard logging
            experiment.log_metric("Mean Rank",float(avg_mr),step=args.num_epochs + 2)

    if args.test_new_disc:
        ''' Testing with fresh discriminators '''
        args.use_attr = True
        attr_data = [args.attr_mat,args.ent_to_idx,args.attr_to_idx,\
                args.reindex_attr_idx,args.attr_count]
        with experiment.test():
            logdir_filter = args.outname_base + '_test_2_filter_logs' + '/'
            if args.remove_old_run:
                shutil.rmtree(logdir_filter)
            if not os.path.exists(logdir_filter):
                os.makedirs(logdir_filter)
            tflogger_filter = tfLogger(logdir_filter)

            args.use_trained_filters = True
            ''' Test With Filters '''
            if args.use_attr:
                train_attr(args,modelD,train_set,test_set,\
                        attr_data,experiment,filter_set=None,attribute='2')
                train_attr(args,modelD,train_set,test_set,\
                        attr_data,experiment,filter_set=None,attribute='1')
                train_attr(args,modelD,train_set,test_set,\
                        attr_data,experiment,filter_set=None,attribute='0')

if __name__ == '__main__':
    main(parse_args())

