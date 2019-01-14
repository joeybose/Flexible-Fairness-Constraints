import argparse
import pickle
import json
import logging
import sys, os
import ipdb
import subprocess
from tqdm import tqdm
import concurrent.futures
import pandas as pd
import networkx as nx
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filedir',type=str,default='./reddit_data/Reddit_split_2017-11/split_csv/',\
            help='Reddit dump')
    parser.add_argument('--k_core', type=int, default=10, help="K-core for Graph")
    parser.add_argument('--fileprefix', type=str,\
            default='Split_RC_2017-11*', help='Split Prefix Reddit dump')
    parser.add_argument('--save_dir_prefix', type=str,\
            default='split_csv/', help="output path")
    parser.add_argument('--save_master', type=str,\
            default='master_G_graph.pkl', help="output path")
    parser.add_argument('--save_master_k_core', type=str,\
            default='master_G_k_core_graph.pkl', help="output path")
    args = parser.parse_args()
    return args

def process_csv(filename):
    reddit_dump = pd.read_csv(filename,encoding='utf-8')
    users = reddit_dump['users']
    subreddit = reddit_dump['subreddit']
    total_entries = len(users)
    G = nx.Graph()
    data_itr = tqdm(enumerate(zip(users,subreddit)))
    file_prefix = filename.split('/')[-1]
    file_prefix = file_prefix.replace('.csv','')
    save_path = "./reddit_data/Reddit_split_2017-11/split_csv/" + \
            file_prefix + '_graph.pkl'
    for idx, (user, subreddit) in data_itr:
        user = "U_" + user
        subreddit = "SR_" + subreddit
        G.add_edge(user, subreddit)
    nx.write_gpickle(G,save_path)
    return G

def main(args):
    save_path_base = "./reddit_data/Reddit_split_2017-11/split_csv/"
    save_path_master = save_path_base + args.save_master
    ipdb.set_trace()
    if  os.path.isfile(save_path_master):
        master_G = nx.read_gpickle(save_path_master)
    else:
        file_paths = args.filedir + args.fileprefix
        split_files = glob.glob(file_paths)
        G_list = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for file_, sub_G in tqdm(zip(split_files,executor.map(process_csv,split_files)),total=len(split_files)):
                print("Split file %s" %(file_))
                G_list.append(sub_G)
        save_path_base = "./reddit_data/Reddit_split_2017-11/split_csv/"
        master_G = nx.compose_all(G_list)
        nx.write_gpickle(master_G,save_path_master)
    print("Created Master Graph")
    master_G_k_core = nx.k_core(master_G, k=args.k_core)
    print("K-core of Master Graph")
    save_path_k_core = save_path_base + str(args.k_core) + \
            '_' + args.save_master_k_core
    nx.write_gpickle(master_G_k_core,save_path_k_core)

if __name__ == '__main__':
    main(parse_args())
