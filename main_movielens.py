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
from transD_movielens import *
import joblib
from collections import Counter, OrderedDict
import ipdb
sys.path.append('../')
import gc
from model import *

ftensor = torch.FloatTensor
ltensor = torch.LongTensor

v2np = lambda v: v.data.cpu().numpy()
USE_SPARSE_EMB = True

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
    parser.add_argument('--gamma', type=int, default=1, help='Tradeoff for Adversarial Penalty')
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
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--use_attr', type=bool, default=False, help='Initialize all Attribute')
    parser.add_argument('--use_occ_attr', type=bool, default=False, help='Use Only Occ Attribute')
    parser.add_argument('--use_gender_attr', type=bool, default=False, help='Use Only Gender Attribute')
    parser.add_argument('--use_age_attr', type=bool, default=False, help='Use Only Age Attribute')
    parser.add_argument('--use_random_attr', type=bool, default=False, help='Use a Random Attribute')
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

    ''' Offset Movie ID's by # users because in TransD they share the same
    embedding Layer '''

    args.train_ratings['movie_id'] += len(args.users)
    args.test_ratings['movie_id'] += len(args.users)

    if args.use_random_attr:
        rand_attr = np.random.choice(2, len(args.users))
        args.users['rand'] = rand_attr
    args.num_ent = len(args.users) + len(args.movies)
    args.num_users = len(args.users)
    args.num_movies = len(args.movies)
    args.num_rel = 5
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.outname_base = os.path.join(args.save_dir,args.namestr+'_MovieLens_results')
    args.saved_path = os.path.join(args.save_dir,args.namestr+'_MovieLens_resultsD_final.pts')
    args.gender_filter_saved_path = args.outname_base + 'GenderFilter.pts'
    args.occupation_filter_saved_path = args.outname_base + 'OccupationFilter.pts'
    args.age_filter_saved_path = args.outname_base + 'AgeFilter.pts'
    args.random_filter_saved_path = args.outname_base + 'RandomFilter.pts'

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ##############################################################
    return args

def main(args):
    train_set = KBDataset(args.train_ratings, args.prefetch_to_gpu)
    test_set = KBDataset(args.test_ratings, args.prefetch_to_gpu)
    if args.prefetch_to_gpu:
        train_hash = set([r.tobytes() for r in train_set.dataset.cpu().numpy()])
    else:
        train_hash = set([r.tobytes() for r in train_set.dataset])

    all_hash = train_hash.copy()
    all_hash.update(set([r.tobytes() for r in test_set.dataset]))

    ''' Comet Logging '''
    experiment = Experiment(api_key="Ht9lkWvTm58fRo9ccgpabq5zV",
                        project_name="graph-fairness", workspace="joeybose")
    experiment.set_name(args.namestr)
    modelD = TransD(args.num_ent, args.num_rel, args.embed_dim, args.p)

    ''' Initialize Everything to None '''
    fairD_gender, fairD_occupation, fairD_age, fairD_random = None,None,None,None
    optimizer_fairD_gender, optimizer_fairD_occupation, \
            optimizer_fairD_age, optimizer_fairD_random = None,None,None,None
    gender_filter, occupation_filter, age_filter = None, None, None
    if args.use_attr:
        attr_data = [args.users,args.movies]
        ''' Initialize Discriminators '''
        fairD_gender = DemParDisc(args.embed_dim,attr_data,use_cross_entropy=args.use_cross_entropy)
        fairD_occupation = DemParDisc(args.embed_dim,attr_data,\
                attribute='occupation',use_cross_entropy=args.use_cross_entropy)
        fairD_age = DemParDisc(args.embed_dim,attr_data,\
                attribute='age',use_cross_entropy=args.use_cross_entropy)
        fairD_random = DemParDisc(args.embed_dim,attr_data,\
                attribute='random',use_cross_entropy=args.use_cross_entropy)

        ''' Initialize Optimizers '''
        if args.sample_mask:
            gender_filter = AttributeFilter(args.embed_dim,attribute='gender')
            occupation_filter = AttributeFilter(args.embed_dim,attribute='occupation')
            age_filter = AttributeFilter(args.embed_dim,attribute='age')
            if not args.use_trained_filters:
                ''' Optimize the Filters oth. it's Pretrained '''
                optimizer_fairD_gender = optimizer(list(fairD_gender.parameters()) + \
                        list(gender_filter.parameters()),'adam', args.lr)
                optimizer_fairD_occupation = optimizer(list(fairD_occupation.parameters()) + \
                        list(occupation_filter.parameters()),'adam',args.lr)
                optimizer_fairD_age = optimizer(list(fairD_age.parameters()) + \
                        list(age_filter.parameters()),'adam', args.lr)
        else:
            optimizer_fairD_gender = optimizer(fairD_gender.parameters(),'adam', args.lr)
            optimizer_fairD_occupation = optimizer(fairD_occupation.parameters(),'adam',args.lr)
            optimizer_fairD_age = optimizer(fairD_age.parameters(),'adam', args.lr)

    elif args.use_occ_attr:
        attr_data = [args.users,args.movies]
        fairD_occupation = DemParDisc(args.embed_dim,attr_data,\
                attribute='occupation',use_cross_entropy=args.use_cross_entropy)
        optimizer_fairD_occupation = optimizer(fairD_occupation.parameters(),'adam',args.lr)
    elif args.use_gender_attr:
        attr_data = [args.users,args.movies]
        fairD_gender = DemParDisc(args.embed_dim,attr_data,use_cross_entropy=args.use_cross_entropy)
        optimizer_fairD_gender = optimizer(fairD_gender.parameters(),'adam', args.lr)
    elif args.use_age_attr:
        attr_data = [args.users,args.movies]
        fairD_age = DemParDisc(args.embed_dim,attr_data,\
                attribute='age',use_cross_entropy=args.use_cross_entropy)
        optimizer_fairD_age = optimizer(fairD_age.parameters(),'adam', args.lr)
    elif args.use_random_attr:
        attr_data = [args.users,args.movies]
        fairD_random = DemParDisc(args.embed_dim,attr_data,\
                attribute='random',use_cross_entropy=args.use_cross_entropy)
        optimizer_fairD_random = optimizer(fairD_random.parameters(),'adam', args.lr)
        fairD_random.cuda()

    if args.load_transD:
        modelD.load(args.saved_path)

    if args.load_filters:
        gender_filter.load(args.gender_filter_saved_path)
        occupation_filter.load(args.occupation_filter_saved_path)
        age_filter.load(args.age_filter_saved_path)

    ''' Create Sets '''
    fairD_set = [fairD_gender,fairD_occupation,fairD_age,fairD_random]
    filter_set = [gender_filter,occupation_filter,age_filter,None]
    optimizer_fairD_set = [optimizer_fairD_gender, optimizer_fairD_occupation,\
            optimizer_fairD_age,optimizer_fairD_random]

    ''' Initialize CUDA if Available '''
    if args.use_cuda:
        modelD.cuda()
        for fairD,filter_ in zip(fairD_set,filter_set):
            if fairD is not None:
                fairD.cuda()
            if filter_ is not None:
                filter_.cuda()

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
                if args.do_log:
                    experiment.log_current_epoch(epoch)
                train(train_loader,epoch,args,train_hash,modelD,optimizerD,\
                        fairD_set,optimizer_fairD_set,filter_set,experiment)
                gc.collect()
                if args.decay_lr:
                    if args.decay_lr == 'ReduceLROnPlateau':
                        schedulerD.step(monitor['D_loss_epoch_avg'])
                    else:
                        schedulerD.step()

                if epoch % args.valid_freq == 0:
                    with torch.no_grad():
                        l_ranks,r_ranks,avg_mr,avg_mrr,avg_h10,avg_h5 = test(test_set, args, all_hash,\
                                modelD,subsample=10)

                    if args.use_attr:
                        test_fairness(test_set,args,modelD,experiment,\
                                fairD_gender, attribute='gender',epoch=epoch)
                        test_fairness(test_set,args,modelD,experiment,\
                                fairD_occupation,attribute='occupation', epoch=epoch)
                        test_fairness(test_set,args,modelD,experiment,\
                                fairD_age,attribute='age', epoch=epoch)
                    elif args.use_gender_attr:
                        test_fairness(test_set,args,modelD,experiment,\
                                fairD_gender, attribute='gender',epoch=epoch)
                    elif args.use_occ_attr:
                        test_fairness(test_set,args,modelD,experiment,\
                                fairD_occupation,attribute='occupation', epoch=epoch)
                    elif args.use_age_attr:
                        test_fairness(test_set,args,modelD,experiment,\
                                fairD_age,attribute='age', epoch=epoch)
                    elif args.use_random_attr:
                        test_fairness(test_set,args,modelD,experiment,\
                                fairD_random,attribute='random', epoch=epoch)

                    joblib.dump({'l_ranks':l_ranks, 'r_ranks':r_ranks},args.outname_base+\
                            'epoch{}_validation_ranks.pkl'.format(epoch), compress=9)

                    if args.do_log: # Tensorboard logging
                        experiment.log_metric("Mean Rank",float(avg_mr),step=epoch)
                        experiment.log_metric("Mean Reciprocal Rank",\
                                float(avg_mrr),step=epoch)
                        experiment.log_metric("Hit @10",float(avg_h10),step=epoch)
                        experiment.log_metric("Hit @5",float(avg_h5),step=epoch)

                if epoch % (args.valid_freq * 5) == 0:
                    l_ranks,r_ranks,avg_mr,avg_mrr,avg_h10,avg_h5 = test(test_set,args, all_hash,\
                            modelD)

        l_ranks,r_ranks,avg_mr,avg_mrr,avg_h10,avg_h5 = test(test_set,args, all_hash, modelD)
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
        if args.use_attr or args.use_random_attr:
            fairD_random.save(args.outname_base+'RandomFairD_final.pts')

        if args.sample_mask:
            gender_filter.save(args.outname_base+'GenderFilter.pts')
            occupation_filter.save(args.outname_base+'OccupationFilter.pts')
            age_filter.save(args.outname_base+'AgeFilter.pts')

    if args.test_new_disc:
        ''' Testing with fresh discriminators '''
        args.freeze_transD = True
        if args.sample_mask:
            args.use_trained_filters = True
            ''' Test With Filters '''
            retrain_disc(args,train_loader,train_hash,test_set,modelD,\
                    optimizerD,experiment,gender_filter,occupation_filter=None,\
                    age_filter=None,attribute='gender')
            retrain_disc(args,train_loader,train_hash,test_set,modelD,\
                    optimizerD,experiment,occupation_filter=occupation_filter,\
                    gender_filter=None,age_filter=None,attribute='occupation')
            retrain_disc(args,train_loader,train_hash,test_set,modelD,\
                    optimizerD,experiment,age_filter=age_filter,gender_filter=None,\
                    occupation_filter=None,attribute='age')

        args.use_trained_filters = False

        '''Test Without Filters '''
        if args.use_attr or args.use_gender_attr:
            retrain_disc(args,train_loader,train_hash,test_set,modelD,\
                    optimizerD,experiment,gender_filter=None,\
                    occupation_filter=None,age_filter=None,attribute='gender')
        if args.use_attr or args.use_occ_attr:
            retrain_disc(args,train_loader,train_hash,test_set,modelD,\
                    optimizerD,experiment,gender_filter=None,\
                    occupation_filter=None,age_filter=None,attribute='occupation')
        if args.use_attr or args.use_age_attr:
            retrain_disc(args,train_loader,train_hash,test_set,modelD,\
                    optimizerD,experiment,gender_filter=None,\
                    occupation_filter=None,age_filter=None,attribute='age')
        if args.use_attr or args.use_random_attr:
            retrain_disc(args,train_loader,train_hash,test_set,modelD,\
                    optimizerD,experiment,gender_filter=None,\
                    occupation_filter=None,age_filter=None,attribute='random')
        experiment.end()

if __name__ == '__main__':
    main(parse_args())
