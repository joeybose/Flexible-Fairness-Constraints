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
from utils import *
from preprocess_movie_lens import *
from transD_movielens import *
import joblib
from collections import Counter, OrderedDict
import ipdb
sys.path.append('../')
import gc
from model import *

# ftensor = torch.FloatTensor
ltensor = torch.LongTensor

v2np = lambda v: v.data.cpu().numpy()
USE_SPARSE_EMB = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--show_tqdm', type=bool, default=False, help='')
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
    parser.add_argument('--D_steps', type=int, default=10, help='Number of D steps')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs (default: 500)')
    parser.add_argument('--num_classifier_epochs', type=int, default=100, help='Number of training epochs (default: 500)')
    parser.add_argument('--batch_size', type=int, default=8192, help='Batch size (default: 512)')
    parser.add_argument('--dropout_p', type=float, default=0.2, help='Batch size (default: 512)')
    parser.add_argument('--gamma', type=int, default=1, help='Tradeoff for Adversarial Penalty')
    parser.add_argument('--valid_freq', type=int, default=99, help='Validate frequency in epochs (default: 50)')
    parser.add_argument('--print_freq', type=int, default=5, help='Print frequency in epochs (default: 5)')
    parser.add_argument('--embed_dim', type=int, default=20, help='Embedding dimension (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.001)')
    parser.add_argument('--margin', type=float, default=3, help='Loss margin (default: 1)')
    parser.add_argument('--p', type=int, default=1, help='P value for p-norm (default: 1)')
    parser.add_argument('--prefetch_to_gpu', type=int, default=0, help="")
    parser.add_argument('--full_loss_penalty', type=int, default=0, help="")
    parser.add_argument('--filter_false_negs', type=int, default=1, help="filter out sampled false negatives")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--use_1M', type=bool, default=False, help='Use 1M dataset')
    parser.add_argument('--report_bias', type=bool, default=True, help='Report dataset bias')
    parser.add_argument('--use_attr', type=bool, default=False, help='Initialize all Attribute')
    parser.add_argument('--use_occ_attr', type=bool, default=False, help='Use Only Occ Attribute')
    parser.add_argument('--use_gender_attr', type=bool, default=False, help='Use Only Gender Attribute')
    parser.add_argument('--use_age_attr', type=bool, default=False, help='Use Only Age Attribute')
    parser.add_argument('--use_random_attr', type=bool, default=False, help='Use a Random Attribute')
    parser.add_argument('--use_gcmc', type=bool, default=False, help='Use a GCMC')
    parser.add_argument('--dont_train', action='store_true', help='Dont Do Train Loop')
    parser.add_argument('--debug', action='store_true', help='Stop before Train Loop')
    parser.add_argument('--sample_mask', type=bool, default=False, help='Sample a binary mask for discriminators to use')
    parser.add_argument('--use_trained_filters', type=bool, default=False, help='Sample a binary mask for discriminators to use')
    parser.add_argument('--optim_mode', type=str, default='adam', help='optimizer')
    parser.add_argument('--fairD_optim_mode', type=str, default='adam_hyp2',help='optimizer for Fairness Discriminator')
    parser.add_argument('--namestr', type=str, default='', help='additional info in output filename to help identify experiments')

    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.use_cuda else "cpu")
    if not args.use_1M:
        args.train_ratings,args.test_ratings,args.users,args.movies = make_dataset(True)
    else:
        args.train_ratings,args.test_ratings,args.users,args.movies = make_dataset_1M(True)

    ''' Offset Movie ID's by # users because in TransD they share the same
    embedding Layer '''

    args.train_ratings['movie_id'] += int(np.max(args.users['user_id']))
    args.test_ratings['movie_id'] += int(np.max(args.users['user_id']))

    if args.use_random_attr:
        rand_attr = np.random.choice(2, len(args.users))
        args.users['rand'] = rand_attr
    args.num_users = int(np.max(args.users['user_id'])) + 1
    args.num_movies = int(np.max(args.movies['movie_id'])) + 1
    args.num_ent = args.num_users + args.num_movies
    args.num_rel = 5
    users = np.asarray(list(set(args.users['user_id'])))
    np.random.shuffle(users)
    if args.use_1M:
        cutoff_constant = 0.9
    else:
        cutoff_constant = 0.8
    train_cutoff_row = int(np.round(len(users)*cutoff_constant))
    args.cutoff_row = train_cutoff_row
    args.users_train = users[:train_cutoff_row]
    args.users_test = users[train_cutoff_row:]
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
    train_fairness_set = NodeClassification(args.users_train, args.prefetch_to_gpu)
    test_fairness_set = NodeClassification(args.users_test, args.prefetch_to_gpu)
    if args.prefetch_to_gpu:
        train_hash = set([r.tobytes() for r in train_set.dataset.cpu().numpy()])
    else:
        train_hash = set([r.tobytes() for r in train_set.dataset])

    all_hash = train_hash.copy()
    all_hash.update(set([r.tobytes() for r in test_set.dataset]))

    ''' Comet Logging '''
    experiment = Experiment(api_key="Ht9lkWvTm58fRo9ccgpabq5zV", disabled= not args.do_log
                        ,project_name="graph-invariance-icml", workspace="joeybose")
    experiment.set_name(args.namestr)
    if not args.use_gcmc:
        # modelD = TransD(args.num_ent, args.num_rel, args.embed_dim,\
                # args.p).to(args.device)
        modelD = TransE(args.num_ent, args.num_rel, args.embed_dim,\
                args.p).to(args.device)
    else:
        decoder = SharedBilinearDecoder(args.num_rel,2,args.embed_dim).to(args.device)
        modelD = SimpleGCMC(decoder,args.embed_dim,args.num_ent,args.p).to(args.device)

    ''' Initialize Everything to None '''
    fairD_gender, fairD_occupation, fairD_age, fairD_random = None,None,None,None
    optimizer_fairD_gender, optimizer_fairD_occupation, \
            optimizer_fairD_age, optimizer_fairD_random = None,None,None,None
    gender_filter, occupation_filter, age_filter = None, None, None
    if args.use_attr:
        attr_data = [args.users,args.movies]
        ''' Initialize Discriminators '''
        fairD_gender = GenderDiscriminator(args.use_1M,args.embed_dim,attr_data,\
                'gender',use_cross_entropy=args.use_cross_entropy).to(args.device)
        fairD_occupation = OccupationDiscriminator(args.use_1M,args.embed_dim,attr_data,\
                attribute='occupation',use_cross_entropy=args.use_cross_entropy)
        fairD_age = AgeDiscriminator(args.use_1M,args.embed_dim,attr_data,\
                attribute='age',use_cross_entropy=args.use_cross_entropy)

        ''' Initialize Optimizers '''
        if args.sample_mask:
            gender_filter = AttributeFilter(args.embed_dim,attribute='gender').to(args.device)
            occupation_filter = AttributeFilter(args.embed_dim,attribute='occupation').to(args.device)
            age_filter = AttributeFilter(args.embed_dim,attribute='age').to(args.device)
            optimizer_fairD_gender = optimizer(fairD_gender.parameters(),'adam', args.lr)
            optimizer_fairD_occupation = optimizer(fairD_occupation.parameters(),'adam',args.lr)
            optimizer_fairD_age = optimizer(fairD_age.parameters(),'adam', args.lr)

    elif args.use_occ_attr:
        attr_data = [args.users,args.movies]
        fairD_occupation = OccupationDiscriminator(args.use_1M,args.embed_dim,attr_data,\
                attribute='occupation',use_cross_entropy=args.use_cross_entropy)
        optimizer_fairD_occupation = optimizer(fairD_occupation.parameters(),'adam',args.lr)
    elif args.use_gender_attr:
        attr_data = [args.users,args.movies]
        fairD_gender = GenderDiscriminator(args.use_1M,args.embed_dim,attr_data,\
                'gender',use_cross_entropy=args.use_cross_entropy)
        optimizer_fairD_gender = optimizer(fairD_gender.parameters(),'adam', args.lr)
    elif args.use_age_attr:
        attr_data = [args.users,args.movies]
        fairD_age = AgeDiscriminator(args.use_1M,args.embed_dim,attr_data,\
                attribute='age',use_cross_entropy=args.use_cross_entropy)
        optimizer_fairD_age = optimizer(fairD_age.parameters(),'adam', args.lr)
    elif args.use_random_attr:
        attr_data = [args.users,args.movies]
        fairD_random = RandomDiscriminator(args.use_1M,args.embed_dim,attr_data,\
                'random',use_cross_entropy=args.use_cross_entropy).to(args.device)
        # fairD_random = DemParDisc(args.use_1M,args.embed_dim,attr_data,\
                # attribute='random',use_cross_entropy=args.use_cross_entropy)
        optimizer_fairD_random = optimizer(fairD_random.parameters(),'adam', args.lr)

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
        for fairD,filter_ in zip(fairD_set,filter_set):
            if fairD is not None:
                fairD.to(args.device)
            if filter_ is not None:
                filter_.to(args.device)

    if args.use_gcmc:
        if args.sample_mask and not args.use_trained_filters:
            optimizerD = optimizer(list(modelD.parameters()) + \
                    list(gender_filter.parameters()) + \
                    list(occupation_filter.parameters()) + \
                    list(age_filter.parameters()), 'adam', args.lr)
            # optimizer_fairD_gender = optimizer(list(fairD_gender.parameters()) + \
                    # list(gender_filter.parameters()),'adam', args.lr)
        else:
            optimizerD = optimizer(modelD.parameters(), 'adam', args.lr)
    else:
        optimizerD = optimizer(modelD.parameters(), 'adam_sparse', args.lr)

    _cst_inds = torch.LongTensor(np.arange(args.num_ent, \
            dtype=np.int64)[:,None]).to(args.device).repeat(1, args.batch_size//2)
    _cst_s = torch.LongTensor(np.arange(args.batch_size//2)).to(args.device)
    _cst_s_nb = torch.LongTensor(np.arange(args.batch_size//2,args.batch_size)).to(args.device)
    _cst_nb = torch.LongTensor(np.arange(args.batch_size)).to(args.device)

    if args.prefetch_to_gpu:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                  num_workers=0, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                  num_workers=4, pin_memory=True, collate_fn=collate_fn)

    if args.freeze_transD:
        freeze_model(modelD)

    if args.debug:
        attr_data = [args.users,args.movies]
        ipdb.set_trace()

    ''' Joint Training '''
    if not args.dont_train:
        with experiment.train():
            for epoch in tqdm(range(1, args.num_epochs + 1)):

                if epoch % args.valid_freq == 0 or epoch == 1:
                    with torch.no_grad():
                        if args.use_gcmc:
                            rmse,test_loss = test_gcmc(test_set,args,modelD,filter_set)
                        else:
                            # l_ranks,r_ranks,avg_mr,avg_mrr,avg_h10,avg_h5 = test(test_set, args, all_hash,\
                                    # modelD,subsample=20)
                            test_nce(test_set,args,modelD,epoch,experiment)

                    if args.use_attr:
                        test_gender(args,test_fairness_set,modelD,fairD_gender,experiment,epoch,filter_set)
                        test_occupation(args,test_fairness_set,modelD,fairD_occupation,experiment,epoch,filter_set)
                        test_age(args,test_fairness_set,modelD,fairD_age,experiment,epoch,filter_set)
                    elif args.use_gender_attr:
                        test_gender(args,test_fairness_set,modelD,fairD_gender,experiment,epoch,filter_set)
                    elif args.use_occ_attr:
                        test_occupation(args,test_fairness_set,modelD,fairD_occupation,experiment,epoch,filter_set)
                    elif args.use_age_attr:
                        test_age(args,test_fairness_set,modelD,fairD_age,experiment,epoch,filter_set)
                    elif args.use_random_attr:
                        test_random(args,test_fairness_set,modelD,fairD_random,experiment,epoch,filter_set)
                        # test_fairness(test_fairness_set,args,modelD,experiment,\
                                # fairD_random,attribute='random',\
                                # epoch=epoch)

                    if args.do_log: # Tensorboard logging
                        if args.use_gcmc:
                            experiment.log_metric("RMSE",float(rmse),step=epoch)
                            experiment.log_metric("Test Loss",float(rmse),step=epoch)
                        # else:
                            # experiment.log_metric("Mean Rank",float(avg_mr),step=epoch)
                            # experiment.log_metric("Mean Reciprocal Rank",\
                                    # float(avg_mrr),step=epoch)
                            # experiment.log_metric("Hit @10",float(avg_h10),step=epoch)
                            # experiment.log_metric("Hit @5",float(avg_h5),step=epoch)

                train(train_loader,epoch,args,train_hash,modelD,optimizerD,\
                        fairD_set,optimizer_fairD_set,filter_set,experiment)
                gc.collect()

                if epoch % (args.valid_freq * 5) == 0:
                    if args.use_gcmc:
                        rmse = test_gcmc(test_set, args, modelD)
                    else:
                        test_nce(test_set,args,modelD,epoch,experiment)
                        # l_ranks,r_ranks,avg_mr,avg_mrr,avg_h10,avg_h5 = test(test_set,args, all_hash,\
                                # modelD,subsample=20)
        # if not args.use_gcmc:
            # l_ranks,r_ranks,avg_mr,avg_mrr,avg_h10,avg_h5 = test(test_set,args, all_hash, modelD)
            # joblib.dump({'l_ranks':l_ranks, 'r_ranks':r_ranks}, args.outname_base+'test_ranks.pkl', compress=9)

        modelD.save(args.outname_base+'D_final.pts')
        if args.use_attr or args.use_gender_attr:
            fairD_gender.save(args.outname_base+'GenderFairD_final.pts')
        if args.use_attr or args.use_occ_attr:
            fairD_occupation.save(args.outname_base+'OccupationFairD_final.pts')
        if args.use_attr or args.use_age_attr:
            fairD_age.save(args.outname_base+'AgeFairD_final.pts')
        if args.use_random_attr:
            fairD_random.save(args.outname_base+'RandomFairD_final.pts')

        if args.sample_mask:
            gender_filter.save(args.outname_base+'GenderFilter.pts')
            occupation_filter.save(args.outname_base+'OccupationFilter.pts')
            age_filter.save(args.outname_base+'AgeFilter.pts')

    constant = len(fairD_set) - fairD_set.count(None)
    if constant != 0 or args.test_new_disc:
        if args.test_new_disc:
            args.use_attr = True
        ''' Training Fresh Discriminators'''
        args.freeze_transD = True
        attr_data = [args.users,args.movies]
        if args.use_random_attr:
            new_fairD_random = DemParDisc(args.use_1M,args.embed_dim,attr_data,\
                    attribute='random',use_cross_entropy=args.use_cross_entropy).to(args.device)
            new_optimizer_fairD_random = optimizer(new_fairD_random.parameters(),'adam', args.lr)

        freeze_model(modelD)
        with experiment.test():
            ''' Train Classifier '''
            if args.use_gender_attr or args.use_attr:
                train_gender(args,modelD,train_fairness_set,test_fairness_set,\
                        attr_data,experiment,filter_set)
            if args.use_occ_attr or args.use_attr:
                train_occupation(args,modelD,train_fairness_set,test_fairness_set,\
                        attr_data,experiment,filter_set)
            if args.use_age_attr or args.use_attr:
                train_age(args,modelD,train_fairness_set,test_fairness_set,\
                        attr_data,experiment,filter_set)
            if args.use_random_attr:
                train_random(args,modelD,train_fairness_set,test_fairness_set,\
                        attr_data,experiment,filter_set)
                # train_fairness_classifier(train_fairness_set,args,modelD,experiment,new_fairD_random,\
                        # new_optimizer_fairD_random,epoch,filter_=None,retrain=False)

        if args.report_bias:
            gender_bias = calc_attribute_bias('Train',args,modelD,experiment,\
                    'gender',epoch,[gender_filter])
            occ_bias = calc_attribute_bias('Train',args,modelD,experiment,\
                    'occupation',epoch,[occupation_filter])
            age_bias = calc_attribute_bias('Train',args,modelD,experiment,\
                    'age',epoch,[age_filter])
            gender_bias = calc_attribute_bias('Test',args,modelD,experiment,\
                    'gender',epoch,[gender_filter])
            occ_bias = calc_attribute_bias('Test',args,modelD,experiment,\
                    'occupation',epoch,[occupation_filter])
            age_bias = calc_attribute_bias('Test',args,modelD,experiment,\
                    'age',epoch,[age_filter])
        experiment.end()

if __name__ == '__main__':
    main(parse_args())
