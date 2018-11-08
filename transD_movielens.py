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

def collate_fn(batch):
    if isinstance(batch, np.ndarray) or (isinstance(batch, list) and isinstance(batch[0], np.ndarray)):
        return ltensor(batch).contiguous()
    else:
        return torch.stack(batch).contiguous()

def mask_fairDiscriminators(discriminators, mask):
    # compress('ABCDEF', [1,0,1,0,1,1]) --> A C E F
    return (d for d, s in zip(discriminators, mask) if s)

def train(data_loader, counter, args, train_hash, modelD, optimizerD,\
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
        d_ins = torch.cat([p_batch_var, nce_batch], dim=0).contiguous()

        ''' Update TransD Model '''
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

        ''' Update the Fair Discriminator '''
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
                    elif fairD_disc.attribute == 'age':
                        fairD_age_loss = fairD_loss.detach().cpu().numpy()
                    else:
                        fairD_random_loss = fairD_loss.detach().cpu().numpy()

                    fairD_loss.backward(retain_graph=True)
                    fair_optim.step()
                    l_preds, l_A_labels = fairD_disc.predict(filter_emb,p_batch[:,0],return_preds=True)
                    l_correct = l_preds.eq(l_A_labels.view_as(l_preds)).sum().item()
                    if fairD_disc.attribute == 'gender':
                        l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                                average='binary')
                        gender_correct += l_correct #
                    elif fairD_disc.attribute == 'occupation':
                        l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                                average='micro')
                        occupation_correct += l_correct
                    elif fairD_disc.attribute == 'age':
                        l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                                average='micro')
                        age_correct += l_correct
                    else:
                        l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                                average='micro')
                        random_correct += l_correct

                    total_ent += len(p_batch)

    ''' Logging for end of epoch '''
    if args.do_log:
        experiment.log_metric("TransD Loss",float(lossD),step=counter)
        if fairD_set[0] is not None:
            experiment.log_metric("Fair Gender Disc Loss",float(fairD_gender_loss),step=counter)
        if fairD_set[1] is not None:
            experiment.log_metric("Fair Occupation Disc Loss",float(fairD_occupation_loss),step=counter)
        if fairD_set[2] is not None:
            experiment.log_metric("Fair Age Disc Loss",float(fairD_age_loss),step=counter)
        if fairD_set[3] is not None:
            experiment.log_metric("Fair Random Disc Loss",float(fairD_age_loss),step=counter)

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

def test_fairness(dataset,args,modelD,experiment,fairD,\
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

    return l_ranks, r_ranks, avg_mr, avg_mrr, avg_h10, avg_h5

def retrain_disc(args,train_loader,train_hash,test_set,modelD,optimizerD,\
        experiment,gender_filter,occupation_filter,age_filter,attribute):

    if args.use_trained_filters:
        print("Retrain New Discriminator with Filter on %s" %(attribute))
    else:
        print("Retrain New Discriminator on %s" %(attribute))

    ''' Reset some flags '''
    args.use_cross_entropy = True
    args.sample_mask = False
    args.freeze_transD = True
    new_fairD_gender,new_fairD_occupation,new_fairD_age,new_fair_random = None,None,None,None
    new_optimizer_fairD_gender,new_optimizer_fairD_occupation,\
            new_optimizer_fairD_age,new_optimizer_fairD_random = None,None,None,None

    if attribute == 'gender':
        args.use_gender_attr = True
        args.use_occ_attr = False
        args.use_age_attr = False
        args.use_random_attr = False
        args.use_attr = False
    elif attribute =='occupation':
        args.use_gender_attr = False
        args.use_occ_attr = True
        args.use_age_attr = False
        args.use_random_attr = False
        args.use_attr = False
    elif attribute =='age':
        args.use_gender_attr = False
        args.use_occ_attr = False
        args.use_age_attr = True
        args.use_random_attr = False
        args.use_attr = False
    elif attribute =='random':
        args.use_gender_attr = False
        args.use_occ_attr = False
        args.use_age_attr = False
        args.use_random_attr = True
        args.use_attr = False
    else:
        args.use_gender_attr = False
        args.use_occ_attr = False
        args.use_age_attr = False
        args.use_random_attr = False
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
    elif args.use_random_attr:
        attr_data = [args.users,args.movies]
        new_fairD_random = DemParDisc(args.embed_dim,attr_data,\
            attribute='random',use_cross_entropy=args.use_cross_entropy)
        new_optimizer_fairD_random = optimizer(new_fairD_age.parameters(),'adam')
        new_fairD_random.cuda()
        fairD_disc = new_fairD_random
        fair_optim = new_optimizer_fairD_random

    attr_data = [args.users,args.movies]
    new_fairD_set = [new_fairD_gender,new_fairD_occupation,new_fairD_age,new_fairD_random]
    new_optimizer_fairD_set = [new_optimizer_fairD_gender,new_optimizer_fairD_occupation,\
            new_optimizer_fairD_age,new_optimizer_fairD_random]
    if args.use_trained_filters:
        filter_set = [gender_filter,occupation_filter,age_filter,None]
    else:
        filter_set = [None,None,None,None]

    ''' Freeze Model + Filters '''
    for filter_ in filter_set:
        if filter_ is not None:
            freeze_model(filter_)
    freeze_model(modelD)

    with experiment.test():
        for epoch in tqdm(range(1, args.num_epochs + 1)):
            train(train_loader,epoch,args,train_hash,modelD,optimizerD,\
                    new_fairD_set,new_optimizer_fairD_set,filter_set,experiment)
            gc.collect()
            if args.decay_lr:
                if args.decay_lr == 'ReduceLROnPlateau':
                    schedulerD.step(monitor['D_loss_epoch_avg'])
                else:
                    schedulerD.step()

            if epoch % args.valid_freq == 0:
                if args.use_attr:
                    test_fairness(test_set,args, modelD,experiment,\
                            fairD_gender, attribute='gender',epoch=epoch,\
                            retrain=True)
                    test_fairness(test_set,args,modelD,experiment,\
                            fairD_occupation,attribute='occupation',epoch=epoch,\
                            retrain=True)
                    test_fairness(test_set,args, modelD,experiment,\
                            fairD_age,attribute='age',epoch=epoch,retrain=True)
                elif args.use_gender_attr:
                    test_fairness(test_set,args,modelD,experiment,\
                            fairD_gender, attribute='gender',epoch=epoch,\
                            retrain=True)
                elif args.use_occ_attr:
                    test_fairness(test_set,args,modelD,experiment,\
                            fairD_occupation,attribute='occupation',epoch=epoch,\
                            retrain=True)
                elif args.use_age_attr:
                    test_fairness(test_set,args,modelD,experiment,\
                            fairD_age,attribute='age',epoch=epoch,retrain=True)
                elif args.use_random_attr:
                    test_fairness(test_set,args,modelD,experiment,\
                            fairD_age,attribute='random',epoch=epoch,retrain=True)
