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
    parser.add_argument('--lr', type=float, default=0.008, help='Learning rate (default: 0.001)')
    parser.add_argument('--margin', type=float, default=3, help='Loss margin (default: 1)')
    parser.add_argument('--p', type=int, default=1, help='P value for p-norm (default: 1)')
    parser.add_argument('--ace', type=int, default=1, help="do ace training (otherwise just NCE)")
    parser.add_argument('--prefetch_to_gpu', type=int, default=0, help="")
    parser.add_argument('--share_D_embedding', type=int, default=1, help="")
    parser.add_argument('--D_nce_weight', type=float, default=1, help="D nce term weight")
    parser.add_argument('--full_loss_penalty', type=int, default=0, help="")
    parser.add_argument('--filter_false_negs', type=int, default=1, help="filter out sampled false negatives")
    parser.add_argument('--false_neg_penalty', type=float, default=1., help="false neg penalty for G")
    parser.add_argument('--mb_reward_normalization', type=int, default=0, help="minibatch based reward normalization")
    parser.add_argument('--n_proposal_samples', type=int, default=10, help="")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--decay_lr', type=str, default='halving_step100', help='lr decay mode')
    parser.add_argument('--optim_mode', type=str, default='adam_hyp2', help='optimizer')
    parser.add_argument('--namestr', type=str, default='', help='additional info in output filename to help identify experiments')

    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    args.z_dim = args.embed_dim

    if args.dataset == 'WN' or args.dataset == 'FB15k':
        path = './data/' + args.dataset + '-%s.pkl'

        args.num_ent = len(json.load(open('./data/%s-ent_to_idx.json' % args.dataset, 'r')))
        args.num_rel = len(json.load(open('./data/%s-rel_to_idx.json' % args.dataset, 'r')))
        args.data_path = path
    else:
        raise Exception("Argument 'dataset' can only be 'WN' or 'FB15k'.")

    args.outname_base = os.path.join(args.save_dir,
                                     'Updated_Paper_{}'.format(args.dataset))

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

    if args.use_cuda:
        modelD.cuda()

    D_monitor = OrderedDict()
    G_monitor = OrderedDict()
    test_val_monitor = OrderedDict()

    optimizerD = optimizer(modelD.parameters(), 'adam_sparse_hyp3', args.lr)
    schedulerD = lr_scheduler(optimizerD, args.decay_lr, args.num_epochs)

    loss_func = MarginRankingLoss(args.margin)

    _cst_inds = torch.LongTensor(np.arange(args.num_ent, dtype=np.int64)[:,None]).cuda().repeat(1, args.batch_size//2)
    _cst_s = torch.LongTensor(np.arange(args.batch_size//2)).cuda()
    _cst_s_nb = torch.LongTensor(np.arange(args.batch_size//2,args.batch_size)).cuda()
    _cst_nb = torch.LongTensor(np.arange(args.batch_size)).cuda()

    def train(data_loader):

        lossesD = []
        monitor_grads = []
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

            lossD.backward()
            optimizerD.step()
            optimizerD.zero_grad()

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

        if args.decay_lr:
            if args.decay_lr == 'ReduceLROnPlateau':
                schedulerD.step(monitor['D_loss_epoch_avg'])
            else:
                schedulerD.step()

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

    joblib.dump({'l_ranks':l_ranks, 'r_ranks':r_ranks}, args.outname_base+'test_ranks.pkl', compress=9)
    logging.info("COMPLETE!!!")

if __name__ == '__main__':
    main(parse_args())

