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
from model import *

ftensor = torch.FloatTensor
ltensor = torch.LongTensor

v2np = lambda v: v.data.cpu().numpy()

USE_SPARSE_EMB = True

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
    parser.add_argument('--load_filters', action='store_true', help="Load TransD")
    parser.add_argument('--freeze_transD', action='store_true', help="Load TransD")
    parser.add_argument('--test_new_disc', action='store_true', help="Load TransD")
    parser.add_argument('--remove_old_run', action='store_true', help="remove old run")
    parser.add_argument('--use_cross_entropy', action='store_true', help="DemPar Discriminators Loss as CE")
    parser.add_argument('--data_dir', type=str, default='./data/', help="Contains Pickle files")
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs (default: 500)')
    parser.add_argument('--batch_size', type=int, default=8192, help='Batch size (default: 512)')
    parser.add_argument('--gamma', type=int, default=10, help='Tradeoff for Adversarial Penalty')
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
    parser.add_argument('--use_attr', type=bool, default=False, help='Initialize all Attribute')
    parser.add_argument('--use_occ_attr', type=bool, default=False, help='Use Only Occ Attribute')
    parser.add_argument('--use_gender_attr', type=bool, default=False, help='Use Only Gender Attribute')
    parser.add_argument('--use_age_attr', type=bool, default=False, help='Use Only Age Attribute')
    parser.add_argument('--dont_train', action='store_true', help='Dont Do Train Loop')
    parser.add_argument('--sample_mask', type=bool, default=False, help='Sample a binary mask for discriminators to use')
    parser.add_argument('--use_trained_filters', type=bool, default=False, help='Sample a binary mask for discriminators to use')
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
    # args.saved_path = os.path.join(args.model_save_dir,'MovieLens_resultsD_final.pts')
    args.outname_base = os.path.join(args.save_dir,args.namestr+'_MovieLens_results')
    args.saved_path = os.path.join(args.save_dir,args.namestr+'_MovieLens_resultsD_final.pts')
    args.gender_filter_saved_path = args.outname_base + 'GenderFilter.pts'
    args.occupation_filter_saved_path = args.outname_base + 'OccupationFilter.pts'
    args.age_filter_saved_path = args.outname_base + 'AgeFilter.pts'

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


def mask_fairDiscriminators(discriminators, mask):
    # compress('ABCDEF', [1,0,1,0,1,1]) --> A C E F
    return (d for d, s in zip(discriminators, mask) if s)

def train(data_loader, counter, args, train_hash, modelD, optimizerD,\
        tflogger, fairD_set, optimizer_fairD_set, filter_set, experiment):

    lossesD = []
    monitor_grads = []
    total_ent = 0
    fairD_gender_loss, fairD_occupation_loss, fairD_age_loss = 0,0,0
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

        optimizerD.zero_grad()
        p_batch_var = Variable(p_batch)
        nce_batch = Variable(nce_batch)
        q_samples = Variable(q_samples)

        ''' Number of Active Discriminators '''
        constant = len(masked_fairD_set) - masked_fairD_set.count(None)

        if args.ace == 0:
            d_ins = torch.cat([p_batch_var, nce_batch], dim=0).contiguous()

            if constant != 0 and not args.freeze_transD:
                d_outs,lhs_emb,rhs_emb = modelD(d_ins,True)
                with torch.no_grad():
                    p_lhs_emb = lhs_emb[:len(p_batch_var)]
                    l_penalty = 0

                    ''' Apply Filter or Not to Embeddings '''
                    if args.sample_mask:
                        filter_emb = 0
                        for filter_ in masked_filter_set:
                            if filter_ is not None:
                                filter_emb += filter_(p_lhs_emb)
                    else:
                        filter_emb = p_lhs_emb

                    ''' Apply Discriminators '''
                    for fairD_disc in masked_fairD_set:
                        if fairD_disc is not None:
                            l_penalty += fairD_disc(filter_emb,p_batch[:,0])

                    if not args.use_cross_entropy:
                        fair_penalty = constant - l_penalty
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

        if constant != 0:
            correct = 0
            gender_correct,occupation_correct,age_correct = 0,0,0
            precision_list = []
            recall_list = []
            fscore_list = []
            correct = 0
            for fairD_disc, fair_optim in zip(masked_fairD_set,masked_optimizer_fairD_set):
                if fairD_disc is not None and fair_optim is not None:
                    fair_optim.zero_grad()
                    with torch.no_grad():
                        d_outs,lhs_emb,rhs_emb = modelD(d_ins,True)
                        p_lhs_emb = lhs_emb[:len(p_batch)]

                    ''' Apply Filter or Not to Embeddings '''
                    if args.sample_mask or args.use_trained_filters:
                        filter_emb = 0
                        for filter_ in masked_filter_set:
                            if filter_ is not None:
                                filter_emb += filter_(p_lhs_emb)
                    else:
                        filter_emb = p_lhs_emb
                    l_loss = fairD_disc(filter_emb,p_batch[:,0])
                    if not args.use_cross_entropy:
                        fairD_loss = -1*(1 - l_loss)
                    else:
                        fairD_loss = l_loss
                    if fairD_disc.attribute == 'gender':
                        fairD_gender_loss = fairD_loss.detach().cpu().numpy()
                    elif fairD_disc.attribute == 'occupation':
                        fairD_occupation_loss = fairD_loss.detach().cpu().numpy()
                    else:
                        fairD_age_loss = fairD_loss.detach().cpu().numpy()

                    fairD_loss.backward(retain_graph=True)
                    fair_optim.step()
                    l_preds, l_A_labels = fairD_disc.predict(filter_emb,p_batch[:,0],return_preds=True)
                    l_correct = l_preds.eq(l_A_labels.view_as(l_preds)).sum().item()
                    if fairD_disc.attribute == 'gender':
                        l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                                average='binary')
                        gender_correct += l_correct #+ r_correct
                    elif fairD_disc.attribute == 'occupation':
                        l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                                average='micro')
                        occupation_correct += l_correct #+ r_correct
                    else:
                        l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                                average='micro')
                        age_correct += l_correct #+ r_correct

                    total_ent += len(p_batch)

                if args.do_log and fairD_disc is not None: # Tensorboard logging
                    gender_acc = 100. * gender_correct / total_ent
                    occ_acc = 100. * occupation_correct / total_ent
                    age_acc = 100. * age_correct / total_ent
                    attribute = fairD_disc.attribute

    ''' Logging for end of epoch '''
    if args.do_log: # Tensorboard logging
        tflogger.scalar_summary('TransD Loss',float(lossD),counter)
        experiment.log_metric("TransD Loss",float(lossD),step=counter)
        if fairD_set[0] is not None:
            tflogger.scalar_summary('Fair Gender Disc Loss',float(fairD_gender_loss),counter)
            experiment.log_metric("Fair Gender Disc Loss",float(fairD_gender_loss),step=counter)
        if fairD_set[1] is not None:
            tflogger.scalar_summary('Fair Occupation Disc Loss',float(fairD_occupation_loss),counter)
            experiment.log_metric("Fair Occupation Disc Loss",float(fairD_occupation_loss),step=counter)
        if fairD_set[2] is not None:
            tflogger.scalar_summary('Fair Age Disc Loss',float(fairD_age_loss),counter)
            experiment.log_metric("Fair Age Disc Loss",float(fairD_age_loss),step=counter)

def test_compositional_fairness(dataset,args,modelD,\
        fairD_set,filter_set,attributes,epoch):
    test_loader = DataLoader(dataset, num_workers=1, batch_size=4096, collate_fn=collate_fn)
    gender_correct, occupation_correct, age_correct = 0,0,0
    total_ent = 0
    precision_list = []
    recall_list = []
    fscore_list = []
    print("Testing Attributes")
    print('\n'.join('{}: {}'.format(*k) for k in enumerate(attributes)))
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

        filter_emb = 0
        for filter_ in filter_set:
            if filter_ is not None:
                filter_emb += filter_(lhs_emb)

        ''' Apply Discriminators '''
        for fairD in fairD_set:
            if fairD is not None:
                l_preds, l_A_labels = fairD.predict(filter_emb,lhs,return_preds=True)
                l_correct = l_preds.eq(l_A_labels.view_as(l_preds)).sum().item()
                if fairD.attribute == 'gender':
                    l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                            average='binary')
                    gender_correct += l_correct #+ r_correct
                elif fairD.attribute == 'occupation':
                    l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                            average='micro')
                    occupation_correct += l_correct #+ r_correct
                else:
                    l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                            average='micro')
                    age_correct += l_correct #+ r_correct

        total_ent += len(lhs_emb)

    gender_acc = 100. * gender_correct / total_ent
    occupation_acc = 100. * occupation_correct / total_ent
    age_acc = 100. * age_correct / total_ent
    print("Gender Accuracy %f " %(gender_acc))
    print("Occupation Accuracy %f " %(occupation_acc))
    print("Age Accuracy %f " %(age_acc))

def test_fairness(dataset,args,modelD,tflogger,experiment,fairD,\
        attribute,epoch,filter_=None,retrain=False):

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

        if filter_ is not None:
            lhs_emb = filter_(lhs_emb)

        l_preds,l_A_labels = fairD.predict(lhs_emb,lhs,return_preds=True)
        l_correct = l_preds.eq(l_A_labels.view_as(l_preds)).sum().item()
        if attribute == 'gender':
            l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                    average='binary')
        else:
            l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                    average='micro')

        precision = l_precision
        recall = l_recall
        fscore = l_fscore
        precision_list.append(precision)
        recall_list.append(recall)
        fscore_list.append(fscore)
        correct += l_correct
        total_ent += len(lhs_emb)

    if args.do_log:
        acc = 100. * correct / total_ent
        mean_precision = np.mean(np.asarray(precision_list))
        mean_recall = np.mean(np.asarray(recall_list))
        mean_fscore = np.mean(np.asarray(fscore_list))
        if retrain:
            attribute = 'Retrained_D_' + attribute
        tflogger.scalar_summary(attribute +' Valid FairD Accuracy',float(acc),epoch)
        tflogger.scalar_summary(attribute + ' Valid FairD F-score',float(mean_fscore),epoch)
        experiment.log_metric(attribute + "Valid FairD Accuracy",float(acc),step=epoch)
        experiment.log_metric(attribute + "Valid FairD F-score",float(mean_fscore),step=epoch)

def test(dataset, args, all_hash, modelD, subsample=1):
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

def retrain_disc(args,train_loader,train_hash,test_set,modelD,optimizerD,tflogger,\
        experiment,gender_filter,occupation_filter,age_filter,attribute):

    if args.use_trained_filters:
        print("Retrain New Discriminator with Filter on %s" %(attribute))
    else:
        print("Retrain New Discriminator on %s" %(attribute))

    ''' Reset some flags '''
    args.use_cross_entropy = True
    args.sample_mask = False
    args.freeze_transD = True
    new_fairD_gender,new_fairD_occupation,new_fairD_age = None,None,None
    new_optimizer_fairD_gender,new_optimizer_fairD_occupation,new_optimizer_fairD_age = None,None,None

    if attribute == 'gender':
        args.use_gender_attr = True
        args.use_occ_attr = False
        args.use_age_attr = False
        args.use_attr = False
    elif attribute =='occupation':
        args.use_gender_attr = False
        args.use_occ_attr = True
        args.use_age_attr = False
        args.use_attr = False
    elif attribute =='age':
        args.use_gender_attr = False
        args.use_occ_attr = False
        args.use_age_attr = True
        args.use_attr = False
    else:
        args.use_gender_attr = False
        args.use_occ_attr = False
        args.use_age_attr = False
        args.use_attr = True

    '''Retrain Discriminator on Frozen TransD Model '''
    if args.use_occ_attr:
        attr_data = [args.users,args.movies]
        new_fairD_occupation = DemParDisc(args.embed_dim,attr_data,\
                attribute='occupation',use_cross_entropy=args.use_cross_entropy)
        new_fairD_occupation.cuda()
        new_optimizer_fairD_occupation = optimizer(new_fairD_occupation.parameters(),'adam')
        fairD_disc = new_fairD_occupation
        fair_optim = new_optimizer_fairD_occupation
    elif args.use_gender_attr:
        attr_data = [args.users,args.movies]
        new_fairD_gender = DemParDisc(args.embed_dim,attr_data,use_cross_entropy=args.use_cross_entropy)
        new_optimizer_fairD_gender = optimizer(new_fairD_gender.parameters(),'adam')
        new_fairD_gender.cuda()
        fairD_disc = new_fairD_gender
        fair_optim = new_optimizer_fairD_gender
    elif args.use_age_attr:
        attr_data = [args.users,args.movies]
        new_fairD_age = DemParDisc(args.embed_dim,attr_data,\
            attribute='age',use_cross_entropy=args.use_cross_entropy)
        new_optimizer_fairD_age = optimizer(new_fairD_age.parameters(),'adam')
        new_fairD_age.cuda()
        fairD_disc = new_fairD_age
        fair_optim = new_optimizer_fairD_age

    attr_data = [args.users,args.movies]
    new_fairD_set = [new_fairD_gender,new_fairD_occupation,new_fairD_age]
    new_optimizer_fairD_set = [new_optimizer_fairD_gender,new_optimizer_fairD_occupation, new_optimizer_fairD_age]
    if args.use_trained_filters:
        filter_set = [gender_filter,occupation_filter,age_filter]
    else:
        filter_set = [None,None,None]

    ''' Freeze Model + Filters '''
    for filter_ in filter_set:
        if filter_ is not None:
            freeze_model(filter_)
    freeze_model(modelD)

    with experiment.test():
        for epoch in tqdm(range(1, args.num_epochs + 1)):
            train(train_loader,epoch,args,train_hash,modelD,optimizerD,\
                    tflogger,new_fairD_set,new_optimizer_fairD_set,filter_set,experiment)
            gc.collect()
            if args.decay_lr:
                if args.decay_lr == 'ReduceLROnPlateau':
                    schedulerD.step(monitor['D_loss_epoch_avg'])
                else:
                    schedulerD.step()

            if epoch % args.valid_freq == 0:
                if args.use_attr:
                    test_fairness(test_set,args, modelD,tflogger,experiment,\
                            fairD_gender, attribute='gender',epoch=epoch,\
                            retrain=True)
                    test_fairness(test_set,args,modelD,tflogger,experiment,\
                            fairD_occupation,attribute='occupation',epoch=epoch,\
                            retrain=True)
                    test_fairness(test_set,args, modelD,tflogger,experiment,\
                            fairD_age,attribute='age',epoch=epoch,retrain=True)
                elif args.use_gender_attr:
                    test_fairness(test_set,args,modelD,tflogger,experiment,\
                            fairD_gender, attribute='gender',epoch=epoch,\
                            retrain=True)
                elif args.use_occ_attr:
                    test_fairness(test_set,args,modelD,tflogger,experiment,\
                            fairD_occupation,attribute='occupation',epoch=epoch,\
                            retrain=True)
                elif args.use_age_attr:
                    test_fairness(test_set,args,modelD,tflogger,experiment,\
                            fairD_age,attribute='age',epoch=epoch,retrain=True)

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

    ''' Comet Logging '''
    experiment = Experiment(api_key="Ht9lkWvTm58fRo9ccgpabq5zV",
                        project_name="graph-fairness", workspace="joeybose")

    # ipdb.set_trace()
    # experiment.log_multiple_params(args)

    tflogger = tfLogger(logdir)
    modelD = TransD(args.num_ent, args.num_rel, args.embed_dim, args.p)

    fairD_gender, fairD_occupation, fairD_age = None,None,None
    optimizer_fairD_gender, optimizer_fairD_occupation, optimizer_fairD_age = None,None,None
    if args.use_attr:
        attr_data = [args.users,args.movies]

        ''' Initialize Discriminators '''
        fairD_gender = DemParDisc(args.embed_dim,attr_data,use_cross_entropy=args.use_cross_entropy)
        fairD_occupation = DemParDisc(args.embed_dim,attr_data,\
                attribute='occupation',use_cross_entropy=args.use_cross_entropy)
        fairD_age = DemParDisc(args.embed_dim,attr_data,\
                attribute='age',use_cross_entropy=args.use_cross_entropy)

        ''' Initialize Optimizers '''
        if args.sample_mask:
            gender_filter = AttributeFilter(args.embed_dim,attribute='gender')
            occupation_filter = AttributeFilter(args.embed_dim,attribute='occupation')
            age_filter = AttributeFilter(args.embed_dim,attribute='age')
            gender_filter.cuda()
            occupation_filter.cuda()
            age_filter.cuda()
            optimizer_fairD_gender = optimizer(list(fairD_gender.parameters()) + \
                    list(gender_filter.parameters()),'adam', args.lr)
            optimizer_fairD_occupation = optimizer(list(fairD_occupation.parameters()) + \
                    list(occupation_filter.parameters()),'adam',args.lr)
            optimizer_fairD_age = optimizer(list(fairD_age.parameters()) + \
                    list(age_filter.parameters()),'adam', args.lr)
        elif args.use_trained_filters and not args.sample_mask:
            gender_filter = AttributeFilter(args.embed_dim,attribute='gender')
            occupation_filter = AttributeFilter(args.embed_dim,attribute='occupation')
            age_filter = AttributeFilter(args.embed_dim,attribute='age')
            gender_filter.cuda()
            occupation_filter.cuda()
            age_filter.cuda()
        else:
            optimizer_fairD_gender = optimizer(fairD_gender.parameters(),'adam', args.lr)
            optimizer_fairD_occupation = optimizer(fairD_occupation.parameters(),'adam',args.lr)
            optimizer_fairD_age = optimizer(fairD_age.parameters(),'adam', args.lr)
            gender_filter, occupation_filter, age_filter = None, None, None

        if args.use_cuda:
            fairD_gender.cuda()
            fairD_occupation.cuda()
            fairD_age.cuda()

    elif args.use_occ_attr:
        attr_data = [args.users,args.movies]
        fairD_occupation = DemParDisc(args.embed_dim,attr_data,\
                attribute='occupation',use_cross_entropy=args.use_cross_entropy)
        fairD_occupation.cuda()
        optimizer_fairD_occupation = optimizer(fairD_occupation.parameters(),'adam',args.lr)
    elif args.use_gender_attr:
        attr_data = [args.users,args.movies]
        fairD_gender = DemParDisc(args.embed_dim,attr_data,use_cross_entropy=args.use_cross_entropy)
        optimizer_fairD_gender = optimizer(fairD_gender.parameters(),'adam', args.lr)
        fairD_gender.cuda()
    elif args.use_age_attr:
        attr_data = [args.users,args.movies]
        fairD_age = DemParDisc(args.embed_dim,attr_data,\
                attribute='age',use_cross_entropy=args.use_cross_entropy)
        optimizer_fairD_age = optimizer(fairD_age.parameters(),'adam', args.lr)
        fairD_age.cuda()

    if args.load_transD:
        modelD.load(args.saved_path)

    if args.use_cuda:
        modelD.cuda()

    if args.load_filters:
        gender_filter.load(args.gender_filter_saved_path)
        occupation_filter.load(args.occupation_filter_saved_path)
        age_filter.load(args.age_filter_saved_path)

    ''' Create Sets '''
    fairD_set = [fairD_gender,fairD_occupation,fairD_age]
    filter_set = [gender_filter,occupation_filter,age_filter]
    optimizer_fairD_set = [optimizer_fairD_gender, optimizer_fairD_occupation,\
            optimizer_fairD_age]

    D_monitor = OrderedDict()
    test_val_monitor = OrderedDict()

    optimizerD = optimizer(modelD.parameters(), 'adam_sparse_hyp3', args.lr)
    schedulerD = lr_scheduler(optimizerD, args.decay_lr, args.num_epochs)

    _cst_inds = torch.LongTensor(np.arange(args.num_ent, \
            dtype=np.int64)[:,None]).cuda().repeat(1, args.batch_size//2)
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

    ''' Joint Training '''
    if not args.dont_train:
        with experiment.train():
            for epoch in tqdm(range(1, args.num_epochs + 1)):
                experiment.log_current_epoch(epoch)
                train(train_loader,epoch,args,train_hash,modelD,optimizerD,\
                        tflogger,fairD_set,optimizer_fairD_set,filter_set,experiment)
                gc.collect()
                if args.decay_lr:
                    if args.decay_lr == 'ReduceLROnPlateau':
                        schedulerD.step(monitor['D_loss_epoch_avg'])
                    else:
                        schedulerD.step()

                if epoch % args.valid_freq == 0:
                    with torch.no_grad():
                        l_ranks, r_ranks = test(test_set, args, all_hash,\
                                modelD,subsample=10)
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
                        test_fairness(test_set,args,modelD,tflogger,experiment,\
                                fairD_gender, attribute='gender',epoch=epoch)
                        test_fairness(test_set,args,modelD,tflogger,experiment,\
                                fairD_occupation,attribute='occupation', epoch=epoch)
                        test_fairness(test_set,args,modelD,tflogger,experiment,\
                                fairD_age,attribute='age', epoch=epoch)
                    elif args.use_gender_attr:
                        test_fairness(test_set,args,modelD,tflogger,experiment,\
                                fairD_gender, attribute='gender',epoch=epoch)
                    elif args.use_occ_attr:
                        test_fairness(test_set,args,modelD,tflogger,experiment,\
                                fairD_occupation,attribute='occupation', epoch=epoch)
                    elif args.use_age_attr:
                        test_fairness(test_set,args,modelD,tflogger,experiment,\
                                fairD_age,attribute='age', epoch=epoch)

                    joblib.dump({'l_ranks':l_ranks, 'r_ranks':r_ranks},args.outname_base+\
                            'epoch{}_validation_ranks.pkl'.format(epoch), compress=9)

                    if args.do_log: # Tensorboard logging
                        tflogger.scalar_summary('Mean Rank',float(avg_mr),epoch)
                        tflogger.scalar_summary('Mean Reciprocal Rank',float(avg_mrr),epoch)
                        tflogger.scalar_summary('Hit @10',float(avg_h10),epoch)
                        tflogger.scalar_summary('Hit @5',float(avg_h5),epoch)
                        experiment.log_metric("Mean Rank",float(avg_mr),step=epoch)
                        experiment.log_metric("Mean Reciprocal Rank",\
                                float(avg_mrr),step=epoch)
                        experiment.log_metric("Hit @10",float(avg_h10),step=epoch)
                        experiment.log_metric("Hit @5",float(avg_h5),step=epoch)

                if epoch % (args.valid_freq * 5) == 0:
                    l_ranks, r_ranks = test(test_set,args, all_hash,\
                            modelD)
                    l_mean = l_ranks.mean()
                    r_mean = r_ranks.mean()
                    l_mrr = (1. / l_ranks).mean()
                    r_mrr = (1. / r_ranks).mean()
                    l_h10 = (l_ranks <= 10).mean()
                    r_h10 = (r_ranks <= 10).mean()
                    l_h5 = (l_ranks <= 5).mean()
                    r_h5 = (r_ranks <= 5).mean()

        l_ranks, r_ranks = test(test_set,args, all_hash, modelD)
        l_mean = l_ranks.mean()
        r_mean = r_ranks.mean()
        l_mrr = (1. / l_ranks).mean()
        r_mrr = (1. / r_ranks).mean()
        l_h10 = (l_ranks <= 10).mean()
        r_h10 = (r_ranks <= 10).mean()
        l_h5 = (l_ranks <= 5).mean()
        r_h5 = (r_ranks <= 5).mean()
        joblib.dump({'l_ranks':l_ranks, 'r_ranks':r_ranks}, args.outname_base+'test_ranks.pkl', compress=9)

        ''' Testing Compositional Fairness '''
        if args.sample_mask:
            for i in range(0,10):
                mask = np.random.choice([0, 1], size=(3,))
                masked_filter_set = list(mask_fairDiscriminators(filter_set,mask))
                attribute_names = list(mask_fairDiscriminators(['gender','occupation','age'],mask))
                test_compositional_fairness(test_set,args,modelD,fairD_set,\
                        filter_set,attribute_names,epoch=epoch)

        modelD.save(args.outname_base+'D_final.pts')
        if args.use_attr or args.use_gender_attr:
            fairD_gender.save(args.outname_base+'GenderFairD_final.pts')
        if args.use_attr or args.use_occ_attr:
            fairD_occupation.save(args.outname_base+'OccupationFairD_final.pts')
        if args.use_attr or args.use_age_attr:
            fairD_age.save(args.outname_base+'AgeFairD_final.pts')

        if args.sample_mask:
            gender_filter.save(args.outname_base+'GenderFilter.pts')
            occupation_filter.save(args.outname_base+'OccupationFilter.pts')
            age_filter.save(args.outname_base+'AgeFilter.pts')

    if args.test_new_disc:
        ''' Testing with fresh discriminators '''
        args.freeze_transD = True
        logdir_filter = args.outname_base + '_test_filter_logs' + '/'
        if args.remove_old_run:
            shutil.rmtree(logdir_filter)
        if not os.path.exists(logdir_filter):
            os.makedirs(logdir_filter)
        tflogger_filter = tfLogger(logdir_filter)

        args.use_trained_filters = True
        ''' Test With Filters '''
        retrain_disc(args,train_loader,train_hash,test_set,modelD,\
                optimizerD,tflogger_filter,experiment,gender_filter,occupation_filter=None,\
                age_filter=None,attribute='gender')
        retrain_disc(args,train_loader,train_hash,test_set,modelD,\
                optimizerD,tflogger_filter,experiment,occupation_filter=occupation_filter,\
                gender_filter=None,age_filter=None,attribute='occupation')
        retrain_disc(args,train_loader,train_hash,test_set,modelD,\
                optimizerD,tflogger_filter,experiment,age_filter=age_filter,gender_filter=None,\
                occupation_filter=None,attribute='age')

        args.use_trained_filters = False
        logdir_no_filter = args.outname_base + '_test_no_2_filter_logs' + '/'
        if args.remove_old_run:
            shutil.rmtree(logdir_no_filter)
        if not os.path.exists(logdir_no_filter):
            os.makedirs(logdir_no_filter)
        tflogger_no_filter = tfLogger(logdir_no_filter)

        '''Test Without Filters '''
        retrain_disc(args,train_loader,train_hash,test_set,modelD,\
                optimizerD,tflogger_no_filter,experiment,gender_filter=None,\
                occupation_filter=None,age_filter=None,attribute='gender')
        retrain_disc(args,train_loader,train_hash,test_set,modelD,\
                optimizerD,tflogger_no_filter,experiment,gender_filter=None,\
                occupation_filter=None,age_filter=None,attribute='occupation')
        retrain_disc(args,train_loader,train_hash,test_set,modelD,\
                optimizerD,tflogger_no_filter,experiment,gender_filter=None,\
                occupation_filter=None,age_filter=None,attribute='age')

if __name__ == '__main__':
    main(parse_args())
