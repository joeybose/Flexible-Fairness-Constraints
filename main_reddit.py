from comet_ml import Experiment
import argparse
import pickle
import json
import logging
import sys, os
import subprocess
from tqdm import tqdm
import concurrent.futures
import pandas as pd
import networkx as nx
import glob
import itertools
from itertools import chain
import ipdb
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import torch.optim as optim
from torch.autograd import Variable
from utils import *
from model import *
from train_reddit import *
from eval_reddit import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filedir',type=str,default='./reddit_data/Reddit_split_2017-11/split_csv/',\
            help='Reddit dump')
    parser.add_argument('--k_core', type=int, default=10, help="K-core for Graph")
    parser.add_argument('--skip_n', type=int, default=5, help="Skip N first SR's")
    parser.add_argument('--num_sensitive', type=int, default=50,\
            help="Number of Sensitive Subreddits")
    parser.add_argument('--fileprefix', type=str,\
            default='Split_RC_2017-11*', help='Split Prefix Reddit dump')
    parser.add_argument('--save_dir_prefix', type=str,\
            default='split_csv/', help="output path")
    parser.add_argument('--save_master', type=str,\
            default='master_G_graph.pkl', help="output path")
    parser.add_argument('--save_master_k_core', type=str,\
            default='master_G_k_core_graph.pkl', help="output path")
    parser.add_argument('--use_gcmc', type=bool, default=False, help='Use a GCMC')
    parser.add_argument('--api_key', type=str, default=" ", help="Api key for Comet ml")
    parser.add_argument('--project_name', type=str, default=" ", help="Comet project_name")
    parser.add_argument('--workspace', type=str, default=" ", help="Comet Workspace")
    parser.add_argument('--D_steps', type=int, default=5, help='Number of D steps')
    parser.add_argument('--test_new_disc', action='store_true', help="Test new classifier")
    parser.add_argument('--do_log', action='store_true', help="whether to log to csv")
    parser.add_argument('--use_trained_filters', type=bool, default=False, help='Sample a binary mask for discriminators to use')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs (default: 500)')
    parser.add_argument('--use_cross_entropy', action='store_true', help="DemPar Discriminators Loss as CE")
    parser.add_argument('--freeze_encoder', action='store_true', help="Freeze Main Model")
    parser.add_argument('--use_multi', action='store_true', help="Use Multi-GPU")
    parser.add_argument('--num_nce', type=int, default=1, help='Number of NCE negatives')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.001)')
    parser.add_argument('--namestr', type=str, default='', help='additional info in output filename to help identify experiments')
    parser.add_argument('--debug', action='store_true', help='Stop before Train Loop')
    parser.add_argument('--use_attr', type=bool, default=False, help='Initialize all Attribute')
    parser.add_argument('--sample_mask', type=bool, default=False, help='Sample a binary mask for discriminators to use')
    parser.add_argument('--held_out_comp', type=bool, default=False, help='Held out set for compositional')
    parser.add_argument('--margin', type=float, default=1, help='Loss margin (default: 1)')
    parser.add_argument('--p', type=int, default=1, help='P value for p-norm (default: 1)')
    parser.add_argument('--gamma', type=int, default=1, help='Tradeoff for Adversarial Penalty')
    parser.add_argument('--prefetch_to_gpu', type=int, default=0, help="")
    parser.add_argument('--valid_freq', type=int, default=5, help='Validate frequency in epochs (default: 50)')
    parser.add_argument('--filter_false_negs', type=bool, default=False, help="filter out sampled false negatives")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--batch_size', type=int, default=42000, help='Batch size (default: 512)')
    parser.add_argument('--embed_dim', type=int, default=50, help='Embedding dimension (default: 50)')
    parser.add_argument('--num_classifier_epochs', type=int, default=100, help='Number of training epochs (default: 500)')
    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.use_cuda else "cpu")
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

def collate_fn(batch):
    if isinstance(batch, np.ndarray) or (isinstance(batch, list) and isinstance(batch[0], np.ndarray)):
        return torch.LongTensor(batch).contiguous()
    else:
        return torch.stack(batch).contiguous()

def main(args):
    ''' Preamble '''
    save_path_base = "./reddit_data/Reddit_split_2017-11/split_csv/"
    save_path_k_core = save_path_base + str(args.k_core) + \
            '_' + args.save_master_k_core
    G = nx.read_gpickle(save_path_k_core)
    top_nodes_G = sorted(G.degree, key=lambda x: x[1], \
            reverse=True)[args.skip_n:101+args.skip_n]
    top_nodes_G = [n for n in top_nodes_G if n[0].split('_')[0] != 'U']
    sensitive_nodes = random.sample(top_nodes_G,args.num_sensitive)
    u_to_idx, sr_to_idx = reddit_mappings(list(G.nodes()))
    args.num_users = len(u_to_idx)
    args.num_sr = len(sr_to_idx)
    cutoff_constant = 0.9
    reddit_check_edges(list(G.edges()))
    train_cutoff_row = int(np.round(len(G.edges())*cutoff_constant))
    users_cutoff_row = int(np.round(len(u_to_idx)*cutoff_constant))
    args.cutoff_row = train_cutoff_row
    args.users_cutoff_row = users_cutoff_row
    all_users = list(u_to_idx)
    random.shuffle(all_users)
    ''' Train/Test Splits '''
    args.users_train = [u_to_idx[user] for user in all_users[:args.users_cutoff_row]]
    args.users_test = [u_to_idx[user] for user in all_users[args.users_cutoff_row:]]
    train_set = RedditDataset(list(G.edges())[:args.cutoff_row],u_to_idx,sr_to_idx)
    test_set = RedditDataset(list(G.edges())[args.cutoff_row:],u_to_idx,sr_to_idx)
    train_fairness_set = NodeClassification(args.users_train,args.prefetch_to_gpu)
    test_fairness_set = NodeClassification(args.users_test,args.prefetch_to_gpu)
    if args.filter_false_negs:
        train_hash = set([(train_set.get_mapping(r)).numpy().tobytes() for r in train_set.dataset])
        all_hash = train_hash.copy()
        all_hash.update(set([(test_set.get_mapping(r)).numpy().tobytes()\
                for r in test_set.dataset]))
    else:
        train_hash = None
        all_hash = None
    all_masks = list(map(list, itertools.product([0, 1],\
        repeat=args.num_sensitive)))
    if args.held_out_comp:
        args.mask_cutoff_row = int(np.round(len(all_masks)*cutoff_constant))
        train_masks =  all_masks[:args.mask_cutoff_row]
        test_masks =  all_masks[args.mask_cutoff_row:]
    else:
        train_masks = all_masks
    print("Training Set size %d" %(len(train_set)))
    print("Test Set size %d" %(len(test_set)))

    ''' Define Models '''
    if args.use_multi:
        modelD = to_multi_gpu(RedditEncoder(args.num_users,args.num_sr,args.embed_dim,\
                args.p))
    else:
        modelD = RedditEncoder(args.num_users,args.num_sr,args.embed_dim,\
                args.p).to(args.device)

    ''' Define Discriminators '''
    if args.use_attr:
        fairD_set,optimizer_fairD_set,filter_set = [],[],[]
        for sens_node in sensitive_nodes:
            D = RedditDiscriminator(G,args.embed_dim,\
                    sens_node[0],u_to_idx).to(args.device)
            optimizer_fairD = optimizer(D.parameters(),'adam',args.lr)
            fairD_set.append(D)
            optimizer_fairD_set.append(optimizer_fairD)

        if not args.sample_mask:
            filter_set = None
        else:
            sr_params = []
            for sens_node in sensitive_nodes:
                sr_filter = AttributeFilter(args.embed_dim,\
                        attribute=sens_node[0]).to(args.device)
                sr_params.append(sr_filter)
                filter_set.append(sr_filter)
    else:
        fairD_set,optimizer_fairD_set,filter_set = [None], None, None

    if args.debug:
        ipdb.set_trace()

    if args.sample_mask and not args.use_trained_filters:
        models = [modelD] + sr_params
        optimizerD = optimizer(itertools.chain.from_iterable(m.parameters() for m in\
            models), 'adam', args.lr)
    else:
        optimizerD = optimizer(modelD.parameters(), 'adam', args.lr)
    ''' Comet Logging '''
    experiment = Experiment(api_key=args.api_key, disabled= not args.do_log
                        ,project_name=args.project_name,workspace=args.workspace)
    experiment.set_name(args.namestr)

    ''' Train Loop '''
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=8, pin_memory=True, collate_fn=collate_fn)
    # train_compositional_reddit_classifier(args,modelD,G,sensitive_nodes,\
            # u_to_idx,train_fairness_set,test_fairness_set,experiment,all_masks,filter_set=filter_set)
    with experiment.train():
        for epoch in tqdm(range(1, args.num_epochs + 1)):
            train_fair_reddit(train_loader,all_hash,epoch,args,modelD,optimizerD,\
                    fairD_set, optimizer_fairD_set, filter_set, train_masks, experiment)

            if epoch % args.valid_freq == 0:
                test_reddit_nce(test_set,epoch,all_hash,\
                        args,modelD,experiment,filter_set,subsample=1)

                if args.use_attr:
                    for i, fairD in enumerate(fairD_set):
                        if filter_set is not None:
                            test_sensitive_sr(args,test_fairness_set,modelD,fairD,\
                                    experiment,epoch,[filter_set[i]])
                        else:
                            test_sensitive_sr(args,test_fairness_set,modelD,fairD,\
                                    experiment,epoch,filter_set)

    constant = len(fairD_set) - fairD_set.count(None)
    if constant != 0 or args.test_new_disc:
        if args.test_new_disc:
            args.use_attr = True
        ''' Training Fresh Discriminators'''
        args.freeze_encoder = True
        freeze_model(modelD)
        with experiment.test():
            ''' Train Classifier '''
            if args.use_attr:
                if args.sample_mask and args.held_out_comp:
                    ''' Compositional Held Out Test '''
                    train_compositional_reddit_classifier(args,modelD,G,sensitive_nodes,\
                            u_to_idx,train_fairness_set,test_fairness_set,experiment,test_masks,filter_set=filter_set)
                elif args.sample_mask and not args.held_out_comp:
                    ''' Compositional All '''
                    train_compositional_reddit_classifier(args,modelD,G,sensitive_nodes,\
                            u_to_idx,train_fairness_set,test_fairness_set,experiment,all_masks,filter_set=filter_set)
                else:
                    ''' Non Compositional '''
                    for sens_node in sensitive_nodes:
                        train_reddit_classifier(args,modelD,G,sens_node[0],u_to_idx,\
                                train_fairness_set,test_fairness_set,\
                                experiment=experiment,filter_set=filter_set)
        experiment.end()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main(parse_args())
