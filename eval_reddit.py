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
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.dummy import DummyClassifier
from model import *
from train_reddit import corrupt_reddit_batch,mask_fairDiscriminators

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

def freeze_model(model):
    model.eval()
    for params in model.parameters():
        params.requires_grad = False

def collate_fn(batch):
    if isinstance(batch, np.ndarray) or (isinstance(batch, list) and isinstance(batch[0], np.ndarray)):
        return torch.LongTensor(batch).contiguous()
    else:
        return torch.stack(batch).contiguous()

def test_dummy_reddit(args,test_dataset,modelD,net,dummy,experiment,\
        epoch,strategy,multi_class=False,filter_set=None):
    test_loader = DataLoader(test_dataset, num_workers=1, batch_size=2048)
    correct = 0
    preds_list, probs_list, labels_list = [], [],[]
    sensitive_attr = net.users_sensitive
    for p_batch in test_loader:
        p_batch_var = Variable(p_batch).cuda()
        p_batch_emb = modelD.get_embed(p_batch_var.detach(),filter_set)
        y = sensitive_attr[p_batch]
        preds = dummy.predict(p_batch_emb)
        acc = 100.* accuracy_score(y,preds)
        preds_list.append(preds)
        probs_list.append(dummy.predict_proba(p_batch_emb)[:, 1])
        labels_list.append(y)
    AUC = roc_auc_score(labels_list[0],probs_list[0],average="micro")
    f1 = f1_score(labels_list[0],preds_list[0],average="micro")
    print("Test Dummy %s Accuracy is: %f AUC: %f F1: %f" %(strategy,acc,AUC,f1))
    if args.do_log:
        experiment.log_metric("Test Dummy "+strategy+net.attribute+" AUC",float(AUC),step=epoch)
        experiment.log_metric("Test Dummy "+strategy+net.attribute+" Accuracy",float(acc),step=epoch)
        experiment.log_metric("Test Dummy "+strategy+net.attribute+" F1",float(f1),step=epoch)

def test_sensitive_sr(args,test_dataset,modelD,net,experiment,\
        epoch,filter_set=None):
    test_loader = DataLoader(test_dataset, num_workers=1, batch_size=512)
    correct = 0
    preds_list, probs_list, labels_list  = [], [], []
    for p_batch in test_loader:
        p_batch_var = Variable(p_batch).cuda()
        p_batch_emb = modelD.get_embed(p_batch_var.detach(),filter_set)
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
    f1 = f1_score(cat_labels_list,cat_preds_list,average='micro')
    print("Test %s Accuracy is: %f AUC: %f F1: %f" %(net.attribute,acc,AUC,f1))
    if args.do_log:
        experiment.log_metric("Test "+net.attribute+" AUC",float(AUC),step=epoch)
        experiment.log_metric("Test "+net.attribute+" Accuracy",float(acc),step=epoch)
        experiment.log_metric("Test "+net.attribute+" F1",float(f1),step=epoch)

def train_reddit_classifier(args,modelD,G,attribute,u_to_idx,train_dataset,test_dataset,\
        experiment,filter_set=None):
    modelD.eval()
    net = RedditDiscriminator(G,args.embed_dim,\
            attribute,u_to_idx).to(args.device)
    opt = optimizer(net.parameters(),'adam', args.lr)
    train_loader = DataLoader(train_dataset, num_workers=1, batch_size=3000)
    train_data_itr = enumerate(train_loader)
    criterion = nn.BCELoss()

    for epoch in range(1,args.num_classifier_epochs + 1):
        correct = 0
        if epoch % 10 == 0:
            print("Train %s Loss is %f Accuracy is: %f AUC: %f F1:%f"\
                    %(net.attribute,loss,acc,AUC,f1))
            test_sensitive_sr(args,test_dataset,modelD,net,experiment,epoch,filter_set)
        embs_list, labels_list = [], []
        for p_batch in train_loader:
            p_batch_var = Variable(p_batch).cuda()
            p_batch_emb = modelD.get_embed(p_batch_var.detach(),filter_set)
            opt.zero_grad()
            y_hat, y = net(p_batch_emb,p_batch_var)
            loss = criterion(y_hat, y)
            loss.backward()
            opt.step()
            preds = (y_hat > torch.Tensor([0.5]).cuda()).float() * 1
            correct = preds.eq(y.view_as(preds)).sum().item()
            acc = 100. * correct / len(p_batch)
            try:
                AUC = roc_auc_score(y.data.cpu().numpy(),\
                        y_hat.data.cpu().numpy(),average="micro")
                f1 = f1_score(y.data.cpu().numpy(), preds.data.cpu().numpy(),\
                        average='micro')
            except:
                AUC = 0
                f1 = 0
            if epoch == args.num_classifier_epochs:
                embs_list.append(p_batch_emb)
                labels_list.append(y)
            if args.do_log:
                experiment.log_metric("Train "+ net.attribute+"\
                         AUC",float(AUC),step=epoch)

    cat_labels_list = torch.cat(labels_list,0).data.cpu().numpy()
    cat_embs_list = torch.cat(embs_list,0).data.cpu().numpy()
    ''' Dummy Classifier '''
    for strategy in ['stratified', 'most_frequent', 'uniform']:
        dummy = DummyClassifier(strategy=strategy)
        dummy.fit(cat_embs_list, cat_labels_list)
        test_dummy_reddit(args,test_dataset,modelD,net,dummy,experiment,\
                epoch,strategy,filter_set)

def train_compositional_reddit_classifier(args,modelD,G,sensitive_nodes,\
        u_to_idx,train_dataset,test_dataset,experiment,masks,filter_set=None):
    modelD.eval()
    fairD_set,optimizer_fairD_set = [],[]
    for sens_node in sensitive_nodes:
        D = RedditDiscriminator(G,args.embed_dim,\
                sens_node[0],u_to_idx).to(args.device)
        optimizer_fairD = optimizer(D.parameters(),'adam',args.lr)
        fairD_set.append(D)
        optimizer_fairD_set.append(optimizer_fairD)
    train_loader = DataLoader(train_dataset, num_workers=1, batch_size=3000)
    train_data_itr = enumerate(train_loader)
    criterion = nn.BCELoss()
    print("Starting Training of Compositional Classifiers")
    for epoch in range(1,args.num_classifier_epochs + 1):
        print("Epoch %d" %(epoch))
        if epoch % 10 == 0:
            for i, fairD_disc in enumerate(fairD_set):
                test_sensitive_sr(args,test_dataset,modelD,fairD_disc,\
                        experiment,epoch,[filter_set[i]])
        for p_batch in train_loader:
            ''' Apply Masks '''
            mask = random.choice(masks)
            masked_fairD_set = list(mask_fairDiscriminators(fairD_set,mask))
            masked_optimizer_fairD_set = list(mask_fairDiscriminators(optimizer_fairD_set,mask))
            masked_filter_set = list(mask_fairDiscriminators(filter_set,mask))
            p_batch_var = Variable(p_batch).cuda()
            p_batch_emb = modelD.get_embed(p_batch_var.detach(),filter_set)
            ''' Apply Classifiers '''
            for fairD_disc, fair_optim in zip(masked_fairD_set,\
                    masked_optimizer_fairD_set):
                if fairD_disc is not None and fair_optim is not None:
                    fair_optim.zero_grad()
                    y_hat, y= fairD_disc(p_batch_emb.detach(),\
                            p_batch_var)
                    loss = criterion(y_hat, y)
                    loss.backward(retain_graph=False)
                    fair_optim.step()
                    # preds = (y_hat > torch.Tensor([0.5]).cuda()).float() * 1
                    # correct = preds.eq(y.view_as(preds)).sum().item()
                    # acc = 100. * correct / len(p_batch)
                    # AUC = roc_auc_score(y.data.cpu().numpy(),\
                            # y_hat.data.cpu().numpy(),average="micro")
                            # fair_optim.step()
                    # f1 = f1_score(y.data.cpu().numpy(), preds.data.cpu().numpy(),\
                            # average='micro')

            # if epoch == args.num_classifier_epochs:
                # embs_list.append(p_batch_emb)
                # labels_list.append(y)
            # if args.do_log:
                # experiment.log_metric("Train "+ net.attribute+"\
                         # AUC",float(AUC),step=epoch)

    # cat_labels_list = torch.cat(labels_list,0).data.cpu().numpy()
    # cat_embs_list = torch.cat(embs_list,0).data.cpu().numpy()
    # ''' Dummy Classifier '''
    # for strategy in ['stratified', 'most_frequent', 'uniform']:
        # dummy = DummyClassifier(strategy=strategy)
        # dummy.fit(cat_embs_list, cat_labels_list)
        # test_dummy_reddit(args,test_dataset,modelD,net,dummy,experiment,\
                # epoch,strategy,filter_set)

def test_reddit_nce(dataset, epoch, test_hash, args, modelD, experiment,\
        filters_set=None, subsample=1):
    test_loader = DataLoader(dataset,batch_size=2048, num_workers=4, collate_fn=collate_fn)
    data_itr = tqdm(enumerate(test_loader))
    correct = 0
    labels_list, preds_list = [], []
    for idx, p_batch in data_itr:
        if idx % subsample != 0:
            continue

        lhs, rhs = p_batch[:,0], p_batch[:,1]
        nce_batch = corrupt_reddit_batch(p_batch,args.num_users,args.num_sr)
        p_batch_var = Variable(p_batch).cuda()
        nce_batch_var = Variable(nce_batch).cuda()
        if args.filter_false_negs:
            nce_falseNs = torch.FloatTensor(np.array([int(x.tobytes() in test_hash)\
                    for x in nce_batch.numpy()], dtype=np.float32))
        p_enrgs = modelD(p_batch_var,filters=filters_set)
        nce_enrgs = modelD(nce_batch_var,filters=filters_set)
        ''' Artificially create labels for both classes '''
        if idx % 2 == 0:
            preds = (p_enrgs < nce_enrgs)
            labels_list.append(np.ones(len(p_batch)))
            correct += preds.sum() + 1
        else:
            preds = (p_enrgs > nce_enrgs)
            labels_list.append(np.zeros(len(p_batch)))
            incorrect = preds.sum() + 1
            correct += len(p_enrgs) - incorrect
        preds_list.append(preds)

    cat_preds_list = torch.cat(preds_list,0).data.cpu().numpy()
    cat_labels_list = np.concatenate(labels_list)
    acc = 100. * correct / len(cat_labels_list)
    AUC = roc_auc_score(cat_labels_list,cat_preds_list,average="micro")
    f1 = f1_score(cat_labels_list,cat_preds_list,average='binary')
    print("Test Encoder Accuracy is: %f AUC: %f F1: %f" %(acc,AUC,f1))
    if args.do_log:
        experiment.log_metric("Test AUC",float(AUC),step=epoch)
        experiment.log_metric("Test Accuracy",float(acc),step=epoch)
        experiment.log_metric("Test Encoder F1",float(f1),step=epoch)
