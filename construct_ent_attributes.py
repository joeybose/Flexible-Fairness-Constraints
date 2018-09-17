import os
import json
import pickle
import argparse
import numpy as np
from collections import Counter
import ipdb

if 'data' not in os.listdir('./'):
    os.mkdir('./data')

def parse_line(line):
    lhs, attr = line.strip('\n').split('\t')
    return lhs, attr

def parse_file(lines):
    parsed = []
    for line in lines:
        lhs, attr = parse_line(line)
        parsed += [[lhs, attr]]
    return parsed

def get_idx_dicts(data):
    ent_set, attr_set = set(), set()
    for ent, attr in data:
        ent_set.add(ent)
        attr_set.add(attr)
    ent_list = sorted(list(ent_set))
    attr_list = sorted(list(attr_set))

    ent_to_idx, attr_to_idx = {}, {}
    for i, ent in enumerate(ent_list):
        ent_to_idx[ent] = i
    for j, attr in enumerate(attr_list):
        attr_to_idx[attr] = j
    return ent_to_idx, attr_to_idx

def count_attributes(data, attr_to_idx):
    dataset = []
    for ent, attr in data:
        dataset += [attr_to_idx[attr]]
    counts = Counter(dataset)
    return counts

def reindex_attributes(count_list):
    reindex_attr_idx = {}
    for i, attr_tup in enumerate(count_list):
        attr_idx, count = attr_tup[0], attr_tup[1]
        reindex_attr_idx[attr_idx] = i
    return reindex_attr_idx

def transform_data(data, ent_to_idx, attr_to_idx, \
        reindex_attr_idx, attribute_mat):
    dataset = []
    for ent, attr in data:
        attr_idx = attr_to_idx[attr]
        try:
            reidx = reindex_attr_idx[attr_idx]
            attribute_mat[ent_to_idx[ent]][reidx] = 1
        except:
            pass
    return attribute_mat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="Choose to parse WN or FB15k")
    args = parser.parse_args()
    if args.dataset == 'WN':
        path = './wordnet-mlj12/wordnet-mlj12-%s.txt'
    elif args.dataset == 'FB15k':
        path = './knowledge_graph/datasets/Freebase/FB15k_Entity_Types/FB15k_Entity_Type_%s.txt'
    else:
        raise Exception("Argument 'dataset' can only be WN or FB15k.")

    train_file = open(path % 'train', 'r').readlines()
    valid_file = open(path % 'valid', 'r').readlines()
    test_file = open(path % 'test', 'r').readlines()
    train_data = parse_file(train_file)
    valid_data = parse_file(valid_file)
    test_data = parse_file(test_file)
    ent_to_idx, attr_to_idx = get_idx_dicts(train_data + valid_data + test_data)

    ''' Count attributes '''
    train_attr_count = count_attributes(train_data, attr_to_idx)

    ''' Reindex Attribute Dictionary with 100 Most Common '''
    train_reindex_attr_idx = reindex_attributes(train_attr_count.most_common(50))

    ipdb.set_trace()
    attribute_mat = np.zeros((len(ent_to_idx),50))
    attribute_mat = transform_data(train_data, ent_to_idx, attr_to_idx,\
            train_reindex_attr_idx, attribute_mat)

    pickle.dump(train_set, open('./data/Attributes_%s-train.pkl' % args.dataset, 'wb'), protocol=-1)

    json.dump(ent_to_idx, open('./data/Attributes_%s-ent_to_idx.json' % args.dataset, 'w'))
    json.dump(attr_to_idx, open('./data/Attributes_%s-attr_to_idx.json' % args.dataset, 'w'))

    print("Dataset: %s" % args.dataset)
    print("# entities: %s; # Attributes: %s" % (len(ent_to_idx),
                                               len(attr_to_idx)))
    print("train set size: %s; valid set size: %s; test set size: %s" % (len(train_set),
                                                                         len(valid_set),
                                                                         len(test_set)))

if __name__ == '__main__':
    main()
