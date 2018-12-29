import argparse
import pickle
import json
import logging
import sys, os
import subprocess
from tqdm import tqdm
import pandas as pd
import glob
import concurrent.futures

def process_file(filename):
    col_names =  ['users', 'subreddit']
    reddit_df  = pd.DataFrame(columns = col_names)
    row_idx = 0
    file_prefix = filename.split('/')[-1]
    save_path = "./reddit_data/Reddit_split_2017-11/split_csv/" + \
            file_prefix + '.csv'
    for line in open(filename, 'r'):
        loaded_json = json.loads(line)
        user = loaded_json["author"]
        subreddit = loaded_json["subreddit"]
        reddit_df.loc[row_idx] = [user,subreddit]
        row_idx +=1
    reddit_df.to_csv(save_path)
    print("Finished Processing %s" %(file_prefix))
    return file_prefix

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filedir', type=str, default='./reddit_data/Reddit_split_2017-11/',\
            help='Reddit dump')
    parser.add_argument('--fileprefix', type=str,\
            default='Split_RC_2017-11*', help='Split Prefix Reddit dump')
    parser.add_argument('--save_dir_prefix', type=str,\
            default='split_csv/', help="output path")
    args = parser.parse_args()
    args.save_path = args.filedir + args.save_dir_prefix
    return args

def main(args):
    file_paths = args.filedir + args.fileprefix
    split_files = glob.glob(file_paths)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for file_, saved_file in tqdm(zip(split_files,executor.map(process_file,split_files)),total=len(split_files)):
            print("Split file %s was saved as %s" %(file_,saved_file))

if __name__ == '__main__':
    main(parse_args())
