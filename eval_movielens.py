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
from sklearn.metrics import f1_score
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

def multiclass_roc_auc_score(y_test, y_pred, average="micro"):
    y_test = np.asarray(y_test).squeeze()
    y_pred = np.asarray(y_pred).squeeze()
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    return roc_auc_score(y_test, y_pred, average=average)

def test_random(args,test_dataset,modelD,net,experiment,\
        epoch,filter_set=None):
    test_loader = DataLoader(test_dataset, num_workers=1, batch_size=512)
    correct = 0
    preds_list, probs_list, labels_list = [], [],[]
    for p_batch in test_loader:
        p_batch_var = Variable(p_batch).cuda()
        p_batch_emb = modelD.encode(p_batch_var.detach(),filter_set)
        y_hat, y = net.predict(p_batch_emb,p_batch_var)
        preds = (y_hat > torch.Tensor([0.5]).cuda()).float() * 1
        correct += preds.eq(y.view_as(preds)).sum().item()
        preds_list.append(preds)
        probs_list.append(y_hat)
        labels_list.append(y)
    cat_preds_list = torch.cat(preds_list,0).data.cpu().numpy()
    cat_labels_list = torch.cat(labels_list,0).data.cpu().numpy()
    cat_probs_list = torch.cat(probs_list,0).data.cpu().numpy()
    AUC = roc_auc_score(cat_labels_list,cat_probs_list,average="micro")
    acc = 100. * correct / len(test_dataset)
    f1 = f1_score(cat_labels_list,cat_preds_list,average='binary')
    print("Test Random Accuracy is: %f AUC: %f F1: %f" %(acc,AUC,f1))
    if args.do_log:
        experiment.log_metric("Test"+net.attribute+" AUC",float(AUC),step=epoch)
        experiment.log_metric("Test "+net.attribute+" Accuracy",float(acc),step=epoch)
        experiment.log_metric("Test "+net.attribute+" F1",float(f1),step=epoch)

def train_random(args,modelD,train_dataset,test_dataset,\
        attr_data,experiment,filter_set=None):
    modelD.eval()
    net = RandomDiscriminator(args.use_1M,args.embed_dim,attr_data,\
            'random',use_cross_entropy=args.use_cross_entropy).to(args.device)
    opt = optimizer(net.parameters(),'adam', args.lr)
    train_loader = DataLoader(train_dataset, num_workers=1, batch_size=3000)
    train_data_itr = enumerate(train_loader)
    criterion = nn.BCELoss()

    for epoch in range(1,args.num_classifier_epochs):
        correct = 0
        if epoch % 10 == 0:
            test_random(args,test_dataset,modelD,net,experiment,epoch,filter_set)

        for p_batch in train_loader:
            p_batch_var = Variable(p_batch).cuda()
            p_batch_emb = modelD.encode(p_batch_var.detach(),filter_set)
            opt.zero_grad()
            y_hat, y = net(p_batch_emb,p_batch_var)
            loss = criterion(y_hat, y)
            loss.backward()
            opt.step()
            preds = (y_hat > torch.Tensor([0.5]).cuda()).float() * 1
            correct = preds.eq(y.view_as(preds)).sum().item()
            acc = 100. * correct / len(p_batch)
            AUC = roc_auc_score(y.data.cpu().numpy(),\
                    y_hat.data.cpu().numpy(),average="micro")
            f1 = f1_score(y.data.cpu().numpy(), preds.data.cpu().numpy(),\
                    average='binary')
            print("Train Random Loss is %f Accuracy is: %f AUC: %f F1:%f"\
                    %(loss,acc,AUC,f1))
            if args.do_log:
                experiment.log_metric("Train "+ net.attribute+"\
                         AUC",float(AUC),step=epoch)

def test_gender(args,test_dataset,modelD,net,experiment,\
        epoch,filter_set=None):
    test_loader = DataLoader(test_dataset, num_workers=1, batch_size=512)
    correct = 0
    preds_list, probs_list, labels_list = [], [],[]
    for p_batch in test_loader:
        p_batch_var = Variable(p_batch).cuda()
        p_batch_emb = modelD.encode(p_batch_var.detach(),filter_set)
        y_hat, y = net.predict(p_batch_emb,p_batch_var)
        preds = (y_hat > torch.Tensor([0.5]).cuda()).float() * 1
        correct += preds.eq(y.view_as(preds)).sum().item()
        preds_list.append(preds)
        probs_list.append(y_hat)
        labels_list.append(y)
    cat_preds_list = torch.cat(preds_list,0).data.cpu().numpy()
    cat_labels_list = torch.cat(labels_list,0).data.cpu().numpy()
    cat_probs_list = torch.cat(probs_list,0).data.cpu().numpy()
    AUC = roc_auc_score(cat_labels_list,cat_probs_list,average="micro")
    acc = 100. * correct / len(test_dataset)
    f1 = f1_score(cat_labels_list,cat_preds_list,average='binary')
    print("Test Gender Accuracy is: %f AUC: %f F1: %f" %(acc,AUC,f1))
    if args.do_log:
        experiment.log_metric("Test"+net.attribute+" AUC",float(AUC),step=epoch)
        experiment.log_metric("Test "+net.attribute+" Accuracy",float(acc),step=epoch)
        experiment.log_metric("Test "+net.attribute+" F1",float(f1),step=epoch)

def train_gender(args,modelD,train_dataset,test_dataset,\
        attr_data,experiment,filter_set=None):
    modelD.eval()
    net = GenderDiscriminator(args.use_1M,args.embed_dim,attr_data,\
            'gender',use_cross_entropy=args.use_cross_entropy).to(args.device)
    opt = optimizer(net.parameters(),'adam', args.lr)
    train_loader = DataLoader(train_dataset, num_workers=1, batch_size=3000)
    train_data_itr = enumerate(train_loader)
    criterion = nn.BCELoss()

    for epoch in range(1,args.num_classifier_epochs):
        correct = 0
        if epoch % 10 == 0:
            test_gender(args,test_dataset,modelD,net,experiment,epoch,filter_set)

        for p_batch in train_loader:
            p_batch_var = Variable(p_batch).cuda()
            p_batch_emb = modelD.encode(p_batch_var.detach(),filter_set)
            opt.zero_grad()
            y_hat, y = net(p_batch_emb,p_batch_var)
            loss = criterion(y_hat, y)
            loss.backward()
            opt.step()
            preds = (y_hat > torch.Tensor([0.5]).cuda()).float() * 1
            correct = preds.eq(y.view_as(preds)).sum().item()
            acc = 100. * correct / len(p_batch)
            AUC = roc_auc_score(y.data.cpu().numpy(),\
                    y_hat.data.cpu().numpy(),average="micro")
            f1 = f1_score(y.data.cpu().numpy(), preds.data.cpu().numpy(),\
                    average='binary')
            print("Train Gender Loss is %f Accuracy is: %f AUC: %f F1:%f"\
                    %(loss,acc,AUC,f1))
            if args.do_log:
                experiment.log_metric("Train "+ net.attribute+"\
                         AUC",float(AUC),step=epoch)

def test_age(args,test_dataset,modelD,net,experiment,\
        epoch,filter_set=None):
    test_loader = DataLoader(test_dataset, num_workers=1, batch_size=512)
    correct = 0
    preds_list, labels_list, probs_list = [], [],[]
    for p_batch in test_loader:
        p_batch_var = Variable(p_batch).cuda()
        p_batch_emb = modelD.encode(p_batch_var.detach(),filter_set)
        y_hat, y = net.predict(p_batch_emb,p_batch_var)
        preds = y_hat.max(1, keepdim=True)[1] # get the index of the max
        correct += preds.eq(y.view_as(preds)).sum().item()
        preds_list.append(preds)
        probs_list.append(y_hat)
        labels_list.append(y)
    cat_preds_list = torch.cat(preds_list,0).data.cpu().numpy()
    cat_labels_list = torch.cat(labels_list,0).data.cpu().numpy()
    cat_probs_list = torch.cat(probs_list,0).data.cpu().numpy()
    AUC = multiclass_roc_auc_score(cat_labels_list,cat_probs_list)
    acc = 100. * correct / len(test_dataset)
    f1 = f1_score(cat_labels_list, cat_preds_list, average='micro')
    print("Test Age Accuracy is: %f AUC: %f F1: %f" %(acc,AUC,f1))
    if args.do_log:
        experiment.log_metric("Test"+net.attribute+"AUC",float(AUC),step=epoch)
        experiment.log_metric("Test "+net.attribute+" Accuracy",float(acc),step=epoch)
        experiment.log_metric("Test "+net.attribute+" F1",float(f1),step=epoch)

def train_age(args,modelD,train_dataset,test_dataset,attr_data,\
        experiment,filter_set=None):
    modelD.eval()
    net = AgeDiscriminator(args.use_1M,args.embed_dim,attr_data,\
            'age',use_cross_entropy=args.use_cross_entropy).to(args.device)
    opt = optimizer(net.parameters(),'adam', args.lr)
    train_loader = DataLoader(train_dataset, num_workers=1, batch_size=3000)
    train_data_itr = enumerate(train_loader)
    criterion = nn.NLLLoss()

    for epoch in range(1,args.num_classifier_epochs):
        correct = 0
        if epoch % 10 == 0:
            test_age(args,test_dataset,modelD,net,experiment,epoch,filter_set)

        for p_batch in train_loader:
            p_batch_var = Variable(p_batch).cuda()
            p_batch_emb = modelD.encode(p_batch_var.detach(),filter_set)
            opt.zero_grad()
            y_hat, y = net(p_batch_emb,p_batch_var)
            loss = criterion(y_hat, y)
            loss.backward()
            opt.step()
            preds = y_hat.max(1, keepdim=True)[1] # get the index of the max
            correct = preds.eq(y.view_as(preds)).sum().item()
            acc = 100. * correct / len(p_batch)
            AUC = multiclass_roc_auc_score(y.data.cpu().numpy(),y_hat.data.cpu().numpy())
            f1 = f1_score(y.data.cpu().numpy(), preds.data.cpu().numpy(), average='micro')
            print("Train Age Loss is %f Accuracy is: %f AUC: %f F1: %f" \
                    %(loss,acc,AUC,f1))
            if args.do_log:
                experiment.log_metric("Train"+net.attribute+"AUC",float(AUC),step=epoch)

def test_occupation(args,test_dataset,modelD,net,experiment,epoch,filter_set=None):
    test_loader = DataLoader(test_dataset, num_workers=1, batch_size=8000)
    correct = 0
    preds_list, labels_list, probs_list = [], [],[]
    for p_batch in test_loader:
        p_batch_var = Variable(p_batch).cuda()
        p_batch_emb = modelD.encode(p_batch_var.detach(),filter_set)
        y_hat, y = net.predict(p_batch_emb,p_batch_var)
        preds = y_hat.max(1, keepdim=True)[1] # get the index of the max
        correct += preds.eq(y.view_as(preds)).sum().item()
        probs_list.append(y_hat)
        preds_list.append(preds)
        labels_list.append(y)
    cat_preds_list = torch.cat(preds_list,0).data.cpu().numpy()
    cat_labels_list = torch.cat(labels_list,0).data.cpu().numpy()
    cat_probs_list = torch.cat(probs_list,0).data.cpu().numpy()
    try:
        f1 = f1_score(cat_labels_list, cat_preds_list, average='micro')
        AUC = multiclass_roc_auc_score(cat_labels_list,cat_probs_list)
        acc = 100. * correct / len(test_dataset)
        print("Test Occupation Accuracy is: %f AUC: %f F1: %f" %(acc,AUC,f1))
        if args.do_log:
            experiment.log_metric("Test"+net.attribute+" AUC",float(AUC),step=epoch)
            experiment.log_metric("Test "+net.attribute+" Accuracy",float(acc),step=epoch)
            experiment.log_metric("Test "+net.attribute+" F1",float(f1),step=epoch)
    except:
        acc = 100. * correct / len(test_dataset)
        print("Test Occupation Accuracy is: %f" %(acc))
        if args.do_log:
            experiment.log_metric("Test"+net.attribute+" Accuracy",float(acc),step=epoch)
            experiment.log_metric("Test "+net.attribute+" Accuracy",float(acc),step=epoch)
            experiment.log_metric("Test "+net.attribute+" F1",float(acc),step=epoch)

def train_occupation(args,modelD,train_dataset,test_dataset,\
        attr_data,experiment,filter_set=None):
    modelD.eval()
    net = OccupationDiscriminator(args.use_1M,args.embed_dim,attr_data,\
            'occupation',use_cross_entropy=args.use_cross_entropy).to(args.device)
    opt = optimizer(net.parameters(),'adam', args.lr)
    train_loader = DataLoader(train_dataset, num_workers=1, batch_size=8000)
    train_data_itr = enumerate(train_loader)
    criterion = nn.NLLLoss()

    for epoch in range(1,args.num_classifier_epochs):
        correct = 0
        if epoch % 10 == 0:
            test_occupation(args,test_dataset,modelD,net,experiment,epoch,filter_set)

        for p_batch in train_loader:
            p_batch_var = Variable(p_batch).cuda()
            p_batch_emb = modelD.encode(p_batch_var.detach(),filter_set)
            opt.zero_grad()
            y_hat, y = net(p_batch_emb,p_batch_var)
            loss = criterion(y_hat, y)
            loss.backward()
            opt.step()
            preds = y_hat.max(1, keepdim=True)[1] # get the index of the max
            correct = preds.eq(y.view_as(preds)).sum().item()
            acc = 100. * correct / len(p_batch)
            AUC = multiclass_roc_auc_score(y.data.cpu().numpy(),y_hat.data.cpu().numpy())
            f1 = f1_score(y.data.cpu().numpy(), preds.data.cpu().numpy(), average='micro')
            print("Train Occupation Loss is %f Accuracy is: %f AUC: %f F1: %f"\
                    %(loss,acc,AUC,f1))
            if args.do_log:
                experiment.log_metric("Train"+ net.attribute+"AUC",float(AUC),step=epoch)

def collate_fn(batch):
    if isinstance(batch, np.ndarray) or (isinstance(batch, list) and isinstance(batch[0], np.ndarray)):
        return ltensor(batch).contiguous()
    else:
        return torch.stack(batch).contiguous()

def onevsall_bias(vals,pos_index):
    bias = 0
    for i in range(0,len(vals)):
        bias = torch.abs(vals[pos_index] - vals[i])
    weighted_avg_bias = bias / len(vals)
    return weighted_avg_bias
def calc_majority_class(groups,attribute):
    counts = []
    for k in groups.keys():
        counts.append(len(groups[k]))
    counts = np.asarray(counts)
    index = np.argmax(counts)
    prob = 100. * np.max(counts) / counts.sum()
    print("%s Majority Class %s has prob %f" %(attribute,index,prob))

def calc_attribute_bias(mode,args,modelD,experiment,\
        attribute,epoch,filter_=None):
    movie_ids = args.movies['movie_id'].values
    if mode == 'Train':
        dataset = PredBias(args.use_1M,movie_ids,args.users[:args.cutoff_row],\
                attribute, args.prefetch_to_gpu)
    else:
        dataset = PredBias(args.use_1M,movie_ids,args.users[args.cutoff_row:],\
                attribute,args.prefetch_to_gpu)

    data_loader = DataLoader(dataset, num_workers=1, batch_size=4000)
    if args.show_tqdm:
        test_data_itr = tqdm(enumerate(data_loader))
    else:
        test_data_itr = enumerate(data_loader)

    groups = dataset.groups
    group_preds = defaultdict(list)
    group_embeds_list = []
    calc_majority_class(groups,attribute)
    for idx, movies in test_data_itr:
        movies_var = Variable(movies).cuda()
        with torch.no_grad():
            movies_embed = modelD.encode(movies_var)
            for group, vals in groups.items():
                users_var = Variable(torch.LongTensor(vals)).cuda()
                users_embed = modelD.encode(users_var,filter_)
                group_embeds_list.append(users_embed)

    for group_embed in group_embeds_list:
        movies_repeated = movies_embed.repeat(len(group_embed),1,1).permute(1,0,2)
        with torch.no_grad():
            for i in range(0,len(movies_repeated)):
                preds = modelD.decoder.predict(group_embed,movies_repeated[i])
                avg_preds = preds.mean()
                group_preds[i].append(avg_preds)

    bias = 0
    for ind, val in group_preds.items():
        if len(val) == 2:
            bias += torch.abs(val[0] - val[1])
        else:
            weighted_bias = 0
            for i in range(0,len(val)):
                weighted_bias += onevsall_bias(val,i)
            bias += weighted_bias / len(val)
    avg_bias = bias / len(movies)
    print("%s %s Bias is %f" %(mode,attribute,avg_bias))
    if args.do_log:
        experiment.log_metric(mode +" " + attribute + "Bias",float(avg_bias))
    return avg_bias

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

def test_gcmc(dataset, args, modelD,filter_set=None):
    test_loader = DataLoader(dataset, batch_size=4000, num_workers=1, collate_fn=collate_fn)
    cst_inds = np.arange(args.num_ent, dtype=np.int64)[:,None]
    if args.show_tqdm:
        data_itr = tqdm(enumerate(test_loader))
    else:
        data_itr = enumerate(test_loader)

    preds_list= []
    rels_list =[]
    test_loss_list = []
    for idx, p_batch in data_itr:
        p_batch_var = Variable(p_batch).cuda()
        lhs, rel, rhs = p_batch_var[:,0],p_batch_var[:,1],p_batch_var[:,2]
        test_loss,preds = modelD(p_batch_var,filters=filter_set)
        rel += 1
        preds_list.append(preds.squeeze())
        rels_list.append(rel.float())
        test_loss_list.append(test_loss)
    total_preds = torch.cat(preds_list)
    total_rels = torch.cat(rels_list)
    test_loss = torch.mean(torch.stack(test_loss_list))
    rms = torch.sqrt(F.mse_loss(total_preds.squeeze(),total_rels.squeeze()))
    return rms,test_loss

def test_nce(dataset, args, modelD, epoch, experiment):
    test_loader = DataLoader(dataset, batch_size=4000, num_workers=1, collate_fn=collate_fn)
    cst_inds = np.arange(args.num_ent, dtype=np.int64)[:,None]
    if args.show_tqdm:
        data_itr = tqdm(enumerate(test_loader))
    else:
        data_itr = enumerate(test_loader)

    probs_list= []
    preds_list = []
    rels_list =[]
    correct = 0
    for idx, p_batch in data_itr:
        p_batch_var = Variable(p_batch).cuda()
        lhs, rel, rhs = p_batch_var[:,0],p_batch_var[:,1],p_batch_var[:,2]
        preds,weighted_preds,probs = modelD.predict(lhs,rhs)
        probs_list.append(probs.squeeze())
        preds_list.append(preds.squeeze())
        correct += preds.eq(rel.view_as(preds)).sum().item()
        rels_list.append((rel+1).float())
    total_probs = torch.cat(probs_list)
    total_preds = torch.cat(preds_list)
    total_rels = torch.cat(rels_list)
    rms = torch.sqrt(F.mse_loss(total_preds.squeeze().float(),\
            total_rels.squeeze().float()))
    AUC = multiclass_roc_auc_score(total_rels,total_probs)
    acc = 100. * correct / len(dataset)
    print("Test Model Accuracy is: %f AUC: %f" %(acc,AUC))
    if args.do_log:
        experiment.log_metric("Test Model RMSE",float(rms),step=epoch)
        experiment.log_metric("Test Model AUC",float(AUC),step=epoch)
        experiment.log_metric("Test Model Accuracy",float(acc),step=epoch)

def train_fairness_classifier_gcmc(train_dataset,args,modelD,experiment,fairD,\
        fair_optim,epoch,filter_=None,retrain=False,log_freq=2):

    train_loader = DataLoader(train_dataset, num_workers=1, batch_size=8000)#, collate_fn=collate_fn)
    correct = 0
    total_ent = 0
    gender_correct,occupation_correct,age_correct,random_correct = 0,0,0,0
    preds_list = []
    labels_list = []

    # train_data_itr = tqdm(enumerate(train_loader))
    train_data_itr = enumerate(train_loader)

    ''' Training Classifier on Nodes '''
    # for epoch in tqdm(range(1, args.num_classifier_epochs + 1)):
    for idx, p_batch in train_data_itr:
        fair_optim.zero_grad()
        p_batch_var = Variable(p_batch).cuda()
        p_batch_emb = modelD.encode(p_batch_var)
        if filter_ is not None:
            p_batch_emb = filter_(p_batch_emb)
        fairD_loss = fairD(p_batch_emb.detach(),p_batch_var)
        print("%s Classifier has loss %f" %(fairD.attribute,fairD_loss))
        fairD_loss.backward(retain_graph=False)
        fair_optim.step()
        with torch.no_grad():
            l_preds, l_A_labels, probs = fairD.predict(p_batch_emb,\
                    p_batch_var,return_preds=True)
            l_correct = l_preds.eq(l_A_labels.view_as(l_preds)).sum().item()
            if fairD.attribute == 'gender':
                fairD_gender_loss = fairD_loss.detach().cpu().numpy()
                l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                        average='binary')
                gender_correct += l_correct #
            elif fairD.attribute == 'occupation':
                fairD_occupation_loss = fairD_loss.detach().cpu().numpy()
                l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                        average='micro')
                occupation_correct += l_correct
            elif fairD.attribute == 'age':
                fairD_age_loss = fairD_loss.detach().cpu().numpy()
                l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                        average='micro')
                age_correct += l_correct
            else:
                fairD_random_loss = fairD_loss.detach().cpu().numpy()
                l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                        average='micro')
                random_correct += l_correct

            preds_list.append(probs)
            labels_list.append(l_A_labels.view_as(l_preds))
            correct += l_correct
            total_ent += len(p_batch)

        acc = 100. * correct / total_ent
        print('Train Accuracy is %f' %(float(acc)))
        cat_preds_list = torch.cat(preds_list,0).data.cpu().numpy()
        cat_labels_list = torch.cat(labels_list,0).data.cpu().numpy()
        print('Train Accuracy is %f' %(float(acc)))
        if fairD.attribute == 'gender' or fairD.attribute == 'random':
            AUC = roc_auc_score(cat_labels_list, np.argmax(cat_preds_list,1))
            print('Train AUC is %f' %(float(AUC)))
        else:
            AUC = multiclass_roc_auc_score(cat_labels_list,cat_preds_list)
            print('Train AUC is %f' %(float(AUC)))

        ''' Logging '''
        if args.do_log:
            counter = epoch
            acc = 100. * correct / total_ent
            experiment.log_metric(fairD.attribute + " Train FairD Accuracy",float(acc),step=counter)
            if fairD.attribute == 'gender' or fairD.attribute == 'random':
                cat_preds_list = torch.cat(preds_list,0).data.cpu().numpy()
                cat_labels_list = torch.cat(labels_list,0).data.cpu().numpy()
                print('Train Accuracy is %f' %(float(acc)))
                if fairD.attribute == 'gender' or fairD.attribute == 'random':
                    AUC = roc_auc_score(cat_labels_list, np.argmax(cat_preds_list,1))
                    # AUC = roc_auc_score(cat_labels_list, cat_preds_list)
                    print('Train AUC is %f' %(float(AUC)))
                    experiment.log_metric(fairD.attribute + " Train FairD AUC",float(AUC),step=counter)
                # else:
                    # AUC = multiclass_roc_auc_score(cat_labels_list,cat_preds_list)
                    # print('Train AUC is %f' %(float(AUC)))
                    # experiment.log_metric(fairD.attribute + " Train FairD AUC",float(AUC),step=counter)
            if fairD.attribute == 'gender':
                experiment.log_metric("Train Classifier Gender Disc Loss",float(fairD_gender_loss),step=counter)
            if fairD.attribute == 'occupation':
                experiment.log_metric("Train Classifier Occupation Disc Loss",float(fairD_occupation_loss),step=counter)
            if fairD.attribute == 'age':
                experiment.log_metric("Train Classifier Age Disc Loss",float(fairD_age_loss),step=counter)
            if fairD.attribute == 'random':
                experiment.log_metric("Train Classifier  Random Disc Loss",float(fairD_random_loss),step=counter)

def train_fairness_classifier(dataset,args,modelD,experiment,fairD,\
        fair_optim,epoch,filter_=None,retrain=False):

    # if args.use_gcmc:
    train_fairness_classifier_gcmc(dataset,args,modelD,experiment,fairD,\
        fair_optim,epoch,filter_=None,retrain=False,log_freq=2)
    # else:
        # raise NotImplementedError
        # train_fairness_classifier_nce(dataset,args,modelD,experiment,fairD,\
            # fair_optim,attribute,epoch,filter_=None,retrain=False):

def test_fairness_gcmc(test_dataset,args,modelD,experiment,fairD,\
        attribute,epoch,filter_=None,retrain=False):

    test_loader = DataLoader(test_dataset, num_workers=1, batch_size=8000)#, collate_fn=collate_fn)
    correct = 0
    total_ent = 0
    precision_list = []
    recall_list = []
    fscore_list = []
    preds_list = []
    labels_list = []

    if args.show_tqdm:
        test_data_itr = tqdm(enumerate(test_loader))
    else:
        test_data_itr = enumerate(test_loader)

    ''' Test Classifier on Nodes '''
    for idx, p_batch in test_data_itr:
        p_batch_var = Variable(p_batch).cuda()
        p_batch_emb = modelD.encode(p_batch_var)
        ''' If Compositional Add the Attribute Specific Filter '''
        if filter_ is not None:
            p_batch_emb = filter_(p_batch_emb)
        l_preds, l_A_labels, probs = fairD.predict(p_batch_emb,\
                p_batch_var,return_preds=True)
        l_correct = l_preds.eq(l_A_labels.view_as(l_preds)).sum().item()
        correct += l_correct
        if fairD.attribute == 'gender':
            l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                    average='binary')
        elif fairD.attribute == 'occupation':
            l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                    average='micro')
        elif fairD.attribute == 'age':
            l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                    average='micro')
        else:
            l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                    average='micro')

        precision = l_precision
        recall = l_recall
        fscore = l_fscore
        precision_list.append(precision)
        recall_list.append(recall)
        fscore_list.append(fscore)
        preds_list.append(probs)
        labels_list.append(l_A_labels.view_as(l_preds))

    acc = 100. * correct / len(test_dataset)
    cat_preds_list = torch.cat(preds_list,0).data.cpu().numpy()
    cat_labels_list = torch.cat(labels_list,0).data.cpu().numpy()
    print('Classifier Test Accuracy is %f' %(float(acc)))
    if fairD.attribute == 'gender' or fairD.attribute == 'random':
        AUC = roc_auc_score(cat_labels_list, np.argmax(cat_preds_list,1))
        print('Classifier Test AUC is %f' %(float(AUC)))
    else:
        AUC = multiclass_roc_auc_score(cat_labels_list,cat_preds_list)
        print('Classifier Test AUC is %f' %(float(AUC)))

    ''' Logging '''
    if args.do_log:
        acc = 100. * correct / len(test_dataset)
        experiment.log_metric(fairD.attribute + " Test FairD Accuracy",float(acc),step=epoch)
        cat_preds_list = torch.cat(preds_list,0).data.cpu().numpy()
        cat_labels_list = torch.cat(labels_list,0).data.cpu().numpy()
        print('Classifier Test Accuracy is %f' %(float(acc)))
        if fairD.attribute == 'gender' or fairD.attribute == 'random':
            AUC = roc_auc_score(cat_labels_list, np.argmax(cat_preds_list,1))
            # AUC = roc_auc_score(cat_labels_list, cat_preds_list)
            print('Classifier Test AUC is %f' %(float(AUC)))
            experiment.log_metric(fairD.attribute + " Test FairD AUC",float(AUC),step=epoch)
        else:
            AUC = multiclass_roc_auc_score(cat_labels_list,cat_preds_list)
            print('Classifier Test AUC is %f' %(float(AUC)))
            experiment.log_metric(fairD.attribute + " Test FairD AUC",float(AUC),step=epoch)

def test_fairness_nce(dataset,args,modelD,experiment,fairD,\
        attribute,epoch,filter_=None,retrain=False):

    test_loader = DataLoader(dataset, num_workers=1, batch_size=4096, collate_fn=collate_fn)
    correct = 0
    total_ent = 0
    precision_list = []
    recall_list = []
    fscore_list = []
    preds_list = []
    labels_list = []

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

        l_preds,l_A_labels,probs = fairD.predict(lhs_emb,lhs,return_preds=True)
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
        preds_list.append(probs)
        labels_list.append(l_A_labels.view_as(l_preds))
        correct += l_correct
        total_ent += len(lhs_emb)

    if args.do_log:
        acc = 100. * correct / total_ent
        mean_precision = np.mean(np.asarray(precision_list))
        mean_recall = np.mean(np.asarray(recall_list))
        mean_fscore = np.mean(np.asarray(fscore_list))
        preds_list = torch.cat(preds_list,0).data.cpu().numpy()
        labels_list = torch.cat(labels_list,0).data.cpu().numpy()
        if retrain:
            attribute = 'Retrained_D_' + attribute
        experiment.log_metric(attribute + "_Valid FairD Accuracy",float(acc),step=epoch)
        print('Valid Accuracy is %f' %(float(acc)))
        if attribute == 'gender' or attribute == 'random':
            AUC = roc_auc_score(labels_list, preds_list)
            print('Valid AUC is %f' %(float(AUC)))
            experiment.log_metric(attribute + "_Valid FairD AUC",float(AUC),step=epoch)

def test_fairness(dataset,args,modelD,experiment,fairD,\
        attribute,epoch,filter_=None,retrain=False):

    if args.use_gcmc:
        test_fairness_gcmc(dataset,args,modelD,experiment,fairD,\
                attribute,epoch,filter_=None,retrain=False)
    else:
        test_fairness_nce(dataset,args,modelD,experiment,fairD,\
                attribute,epoch,filter_=None,retrain=False)


