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
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, accuracy_score
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
from eval_movielens import *

ftensor = torch.FloatTensor
ltensor = torch.LongTensor
v2np = lambda v: v.data.cpu().numpy()
USE_SPARSE_EMB = True

def optimizer(params, mode, *args, **kwargs):
    if mode == 'SGD':
        opt = optim.SGD(params, *args, momentum=0., **kwargs)
    elif mode.startswith('nesterov'):
        momentum = float(mode[len('nesterov'):])
        opt = optim.SGD(params, *args, momentum=momentum, nesterov=True, **kwargs)
    elif mode.lower() == 'adam':
        betas = kwargs.pop('betas', (.9, .999))
        opt = optim.Adam(params, *args, betas=betas, amsgrad=True,
                weight_decay=1e-4, **kwargs)
    elif mode.lower() == 'adam_hyp2':
        betas = kwargs.pop('betas', (.5, .99))
        opt = optim.Adam(params, *args, betas=betas, amsgrad=True, **kwargs)
    elif mode.lower() == 'adam_hyp3':
        betas = kwargs.pop('betas', (0., .99))
        opt = optim.Adam(params, *args, betas=betas, amsgrad=True, **kwargs)
    elif mode.lower() == 'adam_sparse':
        betas = kwargs.pop('betas', (.9, .999))
        opt = optim.SparseAdam(params, *args, weight_decay=1e-4, betas=betas)
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

def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):
    #creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        #creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]

        #marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        #using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict

class MarginRankingLoss(nn.Module):
    def __init__(self, margin):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, p_enrgs, n_enrgs, weights=None):
        scores = (self.margin + p_enrgs - n_enrgs).clamp(min=0)

        if weights is not None:
            scores = scores * weights / weights.mean()

        return scores.mean(), scores

_cb_var_user = []
_cb_var_movie = []
def corrupt_batch(batch, num_ent, num_users, num_movies):
    # batch: ltensor type, contains positive triplets
    batch_size, _ = batch.size()

    corrupted = batch.clone()

    if len(_cb_var_user) == 0 and len(_cb_var_movie) == 0:
        _cb_var_user.append(ltensor(batch_size//2).cuda())
        _cb_var_movie.append(ltensor(batch_size//2).cuda())

    q_samples_l = _cb_var_user[0].random_(0, num_users)
    q_samples_r = _cb_var_movie[0].random_(num_users, num_users + num_movies - 1)

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

def collate_fn(batch):
    if isinstance(batch, np.ndarray) or (isinstance(batch, list) and isinstance(batch[0], np.ndarray)):
        return ltensor(batch).contiguous()
    else:
        return torch.stack(batch).contiguous()

def mask_fairDiscriminators(discriminators, mask):
    # compress('ABCDEF', [1,0,1,0,1,1]) --> A C E F
    return (d for d, s in zip(discriminators, mask) if s)

def apply_filters_gcmc(args,p_lhs_emb,masked_filter_set):
    ''' Doesnt Have Masked Filters yet '''
    filter_l_emb, filter_r_emb = 0,0
    if args.sample_mask:
        for filter_ in masked_filter_set:
            if filter_ is not None:
                filter_l_emb += filter_(p_lhs_emb)
    else:
        filter_l_emb = p_lhs_emb
    return filter_l_emb

def apply_filters_nce(args,p_lhs_emb,p_rhs_emb,nce_lhs_emb,nce_rhs_emb,\
        rel_emb,p_batch_var,nce_batch,d_outs):
    ''' Doesnt Have Masked Filters yet '''
    filter_l_emb, filter_r_emb = 0,0
    filter_nce_l_emb, filter_nce_r_emb = 0,0
    if args.sample_mask:
        for filter_ in masked_filter_set:
            if filter_ is not None:
                filter_l_emb += filter_(p_lhs_emb)
                filter_r_emb += filter_(p_rhs_emb)
                filter_nce_l_emb += filter_(nce_lhs_emb)
                filter_nce_r_emb += filter_(nce_rhs_emb)
        p_enrgs = (filter_l_emb + rel_emb[:len(p_batch_var)] -\
                filter_r_emb).norm(p=self.p, dim=1)
        nce_enrgs = (filter_nce_l_emb + rel_emb[len(p_batch_var):(len(p_batch_var)+len(nce_batch))] -\
                filter_nce_r_emb).norm(p=self.p, dim=1)
    else:
        filter_l_emb = p_lhs_emb
        filter_r_emb = p_rhs_emb
        filter_nce_l_emb = nce_lhs_emb
        filter_nce_r_emb = nce_rhs_emb
        p_enrgs = d_outs[:len(p_batch_var)]
        nce_enrgs = d_outs[len(p_batch_var):(len(p_batch_var)+len(nce_batch))]

    return p_enrgs, nce_enrgs, filter_l_emb

def train_nce(data_loader,counter,args,train_hash,modelD,optimizerD,\
        fairD_set, optimizer_fairD_set, filter_set, experiment):

    lossesD = []
    monitor_grads = []
    total_ent = 0
    fairD_gender_loss,fairD_occupation_loss,fairD_age_loss,\
            fairD_random_loss = 0,0,0,0
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

        nce_batch, q_samples = corrupt_batch(p_batch,args.num_ent,\
                args.num_users, args.num_movies)

        if args.filter_false_negs:
            if args.prefetch_to_gpu:
                nce_np = nce_batch.cpu().numpy()
            else:
                nce_np = nce_batch.numpy()

            nce_falseNs = torch.FloatTensor(np.array([int(x.tobytes() in train_hash) for x in nce_np], dtype=np.float32))
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
        d_ins = torch.cat([p_batch_var, nce_batch], dim=0).contiguous()
        ''' Update TransD Model '''
        if constant != 0:
            d_outs,lhs_emb,rhs_emb,rel_emb = modelD(d_ins,True)
            p_lhs_emb = lhs_emb[:len(p_batch_var)]
            p_rhs_emb = rhs_emb[:len(p_batch_var)]
            nce_lhs_emb = lhs_emb[len(p_batch_var):(len(p_batch_var)+len(nce_batch))]
            nce_rhs_emb = rhs_emb[len(p_batch_var):(len(p_batch_var)+len(nce_batch))]
            l_penalty = 0

            ''' Apply Filter or Not to Embeddings '''
            p_enrgs,nce_enrgs,filter_l_emb = apply_filters_nce(args,p_lhs_emb,p_rhs_emb,nce_lhs_emb,\
                    nce_rhs_emb,rel_emb,p_batch_var,nce_batch,d_outs)

            ''' Apply Discriminators '''
            for fairD_disc, fair_optim in zip(masked_fairD_set,masked_optimizer_fairD_set):
                if fairD_disc is not None and fair_optim is not None:
                    l_penalty += fairD_disc(filter_l_emb,p_batch[:,0],True)

            if not args.use_cross_entropy:
                fair_penalty = constant - l_penalty
            else:
                fair_penalty = -1*l_penalty

            if not args.freeze_transD:
                optimizerD.zero_grad()
                nce_term, nce_term_scores = loss_func(p_enrgs, nce_enrgs, weights=(1.-nce_falseNs))
                lossD = nce_term + args.gamma*fair_penalty
                lossD.backward(retain_graph=True)
                optimizerD.step()

            l_penalty_2 = 0
            for fairD_disc, fair_optim in zip(masked_fairD_set,\
                    masked_optimizer_fairD_set):
                if fairD_disc is not None and fair_optim is not None:
                    fair_optim.zero_grad()
                    l_penalty_2 += fairD_disc(filter_l_emb.detach(),p_batch[:,0],True)
                    if not args.use_cross_entropy:
                        fairD_loss = -1*(1 - l_penalty_2)
                    else:
                        fairD_loss = l_penalty_2
                    fairD_loss.backward(retain_graph=True)
                    fair_optim.step()

        else:
            d_outs = modelD(d_ins)
            fair_penalty = Variable(torch.zeros(1)).cuda()
            p_enrgs = d_outs[:len(p_batch_var)]
            nce_enrgs = d_outs[len(p_batch_var):(len(p_batch_var)+len(nce_batch))]
            optimizerD.zero_grad()
            nce_term, nce_term_scores = loss_func(p_enrgs, nce_enrgs, weights=(1.-nce_falseNs))
            lossD = nce_term + args.gamma*fair_penalty
            lossD.backward(retain_graph=False)
            optimizerD.step()

        if constant != 0:
            correct = 0
            gender_correct,occupation_correct,age_correct,random_correct = 0,0,0,0
            precision_list = []
            recall_list = []
            fscore_list = []
            correct = 0
            for fairD_disc, fair_optim in zip(masked_fairD_set,masked_optimizer_fairD_set):
                if fairD_disc is not None and fair_optim is not None:
                    fair_optim.zero_grad()
                    ''' No Gradients Past Here '''
                    with torch.no_grad():
                        d_outs,lhs_emb,rhs_emb,rel_emb = modelD(d_ins,True,filters=masked_filter_set)
                        p_lhs_emb = lhs_emb[:len(p_batch)]

                        # ''' Apply Filter or Not to Embeddings '''
                        # if args.sample_mask or args.use_trained_filters:
                            # filter_emb = 0
                            # for filter_ in masked_filter_set:
                                # if filter_ is not None:
                                    # filter_emb += filter_(p_lhs_emb)
                        # else:
                        filter_emb = p_lhs_emb
                        probs, l_A_labels, l_preds = fairD_disc.predict(filter_emb,p_batch[:,0],True)
                        l_correct = l_preds.eq(l_A_labels.view_as(l_preds)).sum().item()
                        if fairD_disc.attribute == 'gender':
                            fairD_gender_loss = fairD_loss.detach().cpu().numpy()
                            l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                                    average='binary')
                            gender_correct += l_correct #
                        elif fairD_disc.attribute == 'occupation':
                            fairD_occupation_loss = fairD_loss.detach().cpu().numpy()
                            l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                                    average='micro')
                            occupation_correct += l_correct
                        elif fairD_disc.attribute == 'age':
                            fairD_age_loss = fairD_loss.detach().cpu().numpy()
                            l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                                    average='micro')
                            age_correct += l_correct
                        else:
                            fairD_random_loss = fairD_loss.detach().cpu().numpy()
                            l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                                    average='micro')
                            random_correct += l_correct

    ''' Logging for end of epoch '''
    if args.do_log:
        if not args.freeze_transD:
            experiment.log_metric("TransD Loss",float(lossD),step=counter)
        if fairD_set[0] is not None:
            experiment.log_metric("Fair Gender Disc Loss",float(fairD_gender_loss),step=counter)
        if fairD_set[1] is not None:
            experiment.log_metric("Fair Occupation Disc Loss",float(fairD_occupation_loss),step=counter)
        if fairD_set[2] is not None:
            experiment.log_metric("Fair Age Disc Loss",float(fairD_age_loss),step=counter)
        if fairD_set[3] is not None:
            experiment.log_metric("Fair Random Disc Loss",float(fairD_age_loss),step=counter)

def train_gcmc(data_loader,counter,args,train_hash,modelD,optimizerD,\
        fairD_set, optimizer_fairD_set, filter_set, experiment):

    total_ent = 0
    fairD_gender_loss,fairD_occupation_loss,fairD_age_loss,\
            fairD_random_loss = 0,0,0,0

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

        if args.use_cuda:
            p_batch = p_batch.cuda()

        p_batch_var = Variable(p_batch)

        ''' Number of Active Discriminators '''
        constant = len(masked_fairD_set) - masked_fairD_set.count(None)

        ''' Update GCMC Model '''
        if constant != 0:
            task_loss,preds,lhs_emb,rhs_emb = modelD(p_batch_var,\
                    return_embeds=True,filters=masked_filter_set)
            filter_l_emb = lhs_emb[:len(p_batch_var)]
            l_penalty = 0

            # ''' Apply Filter or Not to Embeddings '''
            # filter_l_emb = apply_filters_gcmc(args,p_lhs_emb,masked_filter_set)

            ''' Apply Discriminators '''
            for fairD_disc, fair_optim in zip(masked_fairD_set,masked_optimizer_fairD_set):
                if fairD_disc is not None and fair_optim is not None:
                    l_penalty += fairD_disc(filter_l_emb,p_batch[:,0],True)

            if not args.use_cross_entropy:
                fair_penalty = constant - l_penalty
            else:
                fair_penalty = -1*l_penalty

            if not args.freeze_transD:
                optimizerD.zero_grad()
                full_loss = task_loss + args.gamma*fair_penalty
                full_loss.backward(retain_graph=False)
                optimizerD.step()

            for k in range(0,args.D_steps):
                l_penalty_2 = 0
                for fairD_disc, fair_optim in zip(masked_fairD_set,\
                        masked_optimizer_fairD_set):
                    if fairD_disc is not None and fair_optim is not None:
                        fair_optim.zero_grad()
                        l_penalty_2 += fairD_disc(filter_l_emb.detach(),\
                                p_batch[:,0],True)
                        if not args.use_cross_entropy:
                            fairD_loss = -1*(1 - l_penalty_2)
                        else:
                            fairD_loss = l_penalty_2
                        fairD_loss.backward(retain_graph=True)
                        fair_optim.step()
        else:
            task_loss,preds = modelD(p_batch_var)
            fair_penalty = Variable(torch.zeros(1)).cuda()
            optimizerD.zero_grad()
            full_loss = task_loss + args.gamma*fair_penalty
            full_loss.backward(retain_graph=False)
            optimizerD.step()

        if constant != 0:
            gender_correct,occupation_correct,age_correct,random_correct = 0,0,0,0
            correct = 0
            for fairD_disc in masked_fairD_set:
                if fairD_disc is not None:
                    ''' No Gradients Past Here '''
                    with torch.no_grad():
                        task_loss,preds,lhs_emb,rhs_emb = modelD(p_batch_var,\
                                return_embeds=True,filters=masked_filter_set)
                        p_lhs_emb = lhs_emb[:len(p_batch)]
                        filter_emb = p_lhs_emb
                        probs, l_A_labels, l_preds = fairD_disc.predict(filter_emb,p_batch[:,0],True)
                        l_correct = l_preds.eq(l_A_labels.view_as(l_preds)).sum().item()
                        if fairD_disc.attribute == 'gender':
                            fairD_gender_loss = fairD_loss.detach().cpu().numpy()
                            gender_correct += l_correct #
                        elif fairD_disc.attribute == 'occupation':
                            fairD_occupation_loss = fairD_loss.detach().cpu().numpy()
                            occupation_correct += l_correct
                        elif fairD_disc.attribute == 'age':
                            fairD_age_loss = fairD_loss.detach().cpu().numpy()
                            age_correct += l_correct
                        else:
                            fairD_random_loss = fairD_loss.detach().cpu().numpy()
                            random_correct += l_correct

    ''' Logging for end of epoch '''
    if args.do_log:
        if not args.freeze_transD:
            experiment.log_metric("Task Loss",float(full_loss),step=counter)
        if fairD_set[0] is not None:
            experiment.log_metric("Fair Gender Disc Loss",float(fairD_gender_loss),step=counter)
        if fairD_set[1] is not None:
            experiment.log_metric("Fair Occupation Disc Loss",float(fairD_occupation_loss),step=counter)
        if fairD_set[2] is not None:
            experiment.log_metric("Fair Age Disc Loss",float(fairD_age_loss),step=counter)
        if fairD_set[3] is not None:
            experiment.log_metric("Fair Random Disc Loss",float(fairD_age_loss),step=counter)

def train(data_loader, counter, args, train_hash, modelD, optimizerD,\
         fairD_set, optimizer_fairD_set, filter_set, experiment):

    ''' This Function Does Training based on NCE Sampling, for GCMC switch to
    another train function which does not need NCE Sampling'''
    if args.use_gcmc:
        train_gcmc(data_loader,counter,args,train_hash,modelD,optimizerD,\
                fairD_set, optimizer_fairD_set, filter_set, experiment)
    else:
        train_nce(data_loader,counter,args,train_hash,modelD,optimizerD,\
                fairD_set, optimizer_fairD_set, filter_set, experiment)

