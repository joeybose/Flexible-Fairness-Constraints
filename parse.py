"""
Parse WordNet and FB15k datasets
"""

import os
import json
import pickle
import argparse
import ipdb

if 'data' not in os.listdir('./'):
    os.mkdir('./data')

def parse_line(line):
    lhs, rel, rhs = line.strip('\n').split('\t')
    return lhs, rel, rhs

def parse_file(lines):
    parsed = []
    for line in lines:
        lhs, rel, rhs = parse_line(line)
        parsed += [[lhs, rel, rhs]]
    return parsed

def get_idx_dicts(data):
    ent_set, rel_set = set(), set()
    for lhs, rel, rhs in data:
        ent_set.add(lhs)
        rel_set.add(rel)
        ent_set.add(rhs)
    ent_list = sorted(list(ent_set))
    rel_list = sorted(list(rel_set))

    ent_to_idx, rel_to_idx = {}, {}
    for i, ent in enumerate(ent_list):
        ent_to_idx[ent] = i
    for j, rel in enumerate(rel_list):
        rel_to_idx[rel] = j
    return ent_to_idx, rel_to_idx

def transform_data(data, ent_to_idx, rel_to_idx):
    dataset = []
    for lhs, rel, rhs in data:
        dataset += [[ent_to_idx[lhs], rel_to_idx[rel], ent_to_idx[rhs]]]
    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="Choose to parse WN or FB15k")
    args = parser.parse_args()
    if args.dataset == 'WN':
        path = './wordnet-mlj12/wordnet-mlj12-%s.txt'
    elif args.dataset == 'FB15k':
        path = './fb15k/%s.txt'
    else:
        raise Exception("Argument 'dataset' can only be WN or FB15k.")

    train_file = open(path % 'train', 'r').readlines()
    valid_file = open(path % 'valid', 'r').readlines()
    test_file = open(path % 'test', 'r').readlines()
    train_data = parse_file(train_file)
    valid_data = parse_file(valid_file)
    test_data = parse_file(test_file)

    ent_to_idx, rel_to_idx = get_idx_dicts(train_data + valid_data + test_data)

    train_set = transform_data(train_data, ent_to_idx, rel_to_idx)
    valid_set = transform_data(valid_data, ent_to_idx, rel_to_idx)
    test_set = transform_data(test_data, ent_to_idx, rel_to_idx)

    pickle.dump(train_set, open('./data/%s-train.pkl' % args.dataset, 'wb'), protocol=-1)
    pickle.dump(valid_set, open('./data/%s-valid.pkl' % args.dataset, 'wb'), protocol=-1)
    pickle.dump(test_set, open('./data/%s-test.pkl' % args.dataset, 'wb'), protocol=-1)

    json.dump(ent_to_idx, open('./data/%s-ent_to_idx.json' % args.dataset, 'w'))
    json.dump(rel_to_idx, open('./data/%s-rel_to_idx.json' % args.dataset, 'w'))

    print("Dataset: %s" % args.dataset)
    print("# entities: %s; # relations: %s" % (len(ent_to_idx),
                                               len(rel_to_idx)))
    print("train set size: %s; valid set size: %s; test set size: %s" % (len(train_set),
                                                                         len(valid_set),
                                                                         len(test_set)))

if __name__ == '__main__':
    main()
