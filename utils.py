import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import ipdb

class RedditDataset(Dataset):
    def __init__(self,edges,u_to_idx,sr_to_idx,prefetch_to_gpu=False):
        self.dataset = edges
        self.u_to_idx = u_to_idx
        self.sr_to_idx = sr_to_idx
        self.prefetch_to_gpu = prefetch_to_gpu

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ''' Always return [User, SR] '''
        edge = self.dataset[idx]
        if edge[0].split('_')[0] == 'U':
            user = torch.LongTensor([self.u_to_idx[edge[0]]])
            sr = torch.LongTensor([self.sr_to_idx[edge[1]]])
        else:
            user = torch.LongTensor([self.u_to_idx[edge[1]]])
            sr = torch.LongTensor([self.sr_to_idx[edge[0]]])
        datum = torch.cat((user,sr),0)
        return datum

    def shuffle(self):
        data = self.dataset
        np.random.shuffle(data)

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

class NodeClassification(Dataset):
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
        # np.random.shuffle(data)
        data = np.ascontiguousarray(data)
        self.dataset = ltensor(data)

        if self.prefetch_to_gpu:
            self.dataset = self.dataset.cuda().contiguous()

class PredBias(Dataset):
    def __init__(self,use_1M,movies,users,attribute,prefetch_to_gpu=False):
        self.prefetch_to_gpu = prefetch_to_gpu
        self.dataset = np.ascontiguousarray(movies)
        self.users = users
        self.groups = defaultdict(list)
        if attribute == 'gender':
            users_sex = self.users['sex']
            self.num_groups = 2
            [self.groups[val].append(ind) for ind,val in enumerate(users_sex)]
        elif attribute == 'occupation':
            users_occupation = self.users['occupation']
            if use_1M:
                [self.groups[val].append(ind) for ind,val in \
                        enumerate(users_occupation)]
                self.num_groups = 21
            else:
                users_occupation_list = sorted(set(users_occupation))
                occ_to_idx = {}
                for i, occ in enumerate(users_occupation_list):
                    occ_to_idx[occ] = i
                users_occupation = [occ_to_idx[occ] for occ in users_occupation]
        elif attribute == 'random':
            users_random = self.users['rand']
            self.num_groups = 2
            [self.groups[val].append(ind) for ind,val in enumerate(users_random)]
        else:
            users_age = self.users['age'].values
            users_age_list = sorted(set(users_age))
            if not use_1M:
                bins = np.linspace(5, 75, num=15, endpoint=True)
                inds = np.digitize(users_age, bins) - 1
                self.users_sensitive = np.ascontiguousarray(inds)
            else:
                reindex = {1:0, 18:1, 25:2, 35:3, 45:4, 50:5, 56:6}
                self.num_groups = 7
                inds = [reindex.get(n, n) for n in users_age]
                [self.groups[val].append(ind) for ind,val in enumerate(inds)]

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

def reddit_check_edges(edges):
    print("Printing Bad Edges")
    for edge in edges:
        if edge[0].split('_')[0] == edge[1].split('_')[1]:
            print(edge)

def reddit_mappings(nodes):
    users, subreddits = [], []
    for ent in nodes:
        if ent.split('_')[0] == 'U':
            users.append(ent)
        else:
            subreddits.append(ent)

    user_to_idx, sr_to_idx = {},{}
    for i, ent in enumerate(users):
        user_to_idx[ent] = i

    for j, sr in enumerate(subreddits):
        sr_to_idx[sr] = j
    return user_to_idx, sr_to_idx

def compute_rank(enrgs, target, mask_observed=None):
    enrg = enrgs[target]
    if mask_observed is not None:
        mask_observed[target] = 0
        enrgs = enrgs + 100*mask_observed

    return (enrgs < enrg).sum() + 1


def create_or_append(d, k, v, v2np=None):
    if v2np is None:
        if k in d:
            d[k].append(v)
        else:
            d[k] = [v]
    else:
        if k in d:
            d[k].append(v2np(v))
        else:
            d[k] = [v2np(v)]

def to_multi_gpu(model):
    cuda_stat = torch.cuda.is_available()
    if cuda_stat:
        model = torch.nn.DataParallel(model,\
                device_ids=range(torch.cuda.device_count())).cuda()
    return model
