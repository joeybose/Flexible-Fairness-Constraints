import joblib
import numpy as np
import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ipdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./results/', help="output path")
    parser.add_argument('--dataset', type=str, default='FB15k', help='Knowledge base version (default: FB15k)')
    args = parser.parse_args()
    outname_base = os.path.join(args.save_dir, 'FairD_[15]_{}'.format(args.dataset))
    epochs = np.arange(20,1000,20)

    mrr_list = []
    h10_list = []
    h5_list = []
    mean_rank = []

    for epoch in epochs:
        fn = outname_base+'epoch{}_validation_ranks.pkl'.format(epoch)
        f_ = joblib.load(fn)
        l_ranks = f_['l_ranks']
        r_ranks = f_['r_ranks']

        ''' Calculate Stats '''
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

        mean_rank.append(avg_mr)
        mrr_list.append(avg_mrr)
        h10_list.append(avg_h10)
        h5_list.append(avg_h5)

    ''' Plots '''
    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
    ax1.plot(epochs, mean_rank, label='Mean Rank')
    ax2.plot(epochs, mrr_list, label='MRR')
    ax2.plot(epochs, h5_list, label='H5')
    ax2.plot(epochs, h10_list, label='H10')
    ax1.set_xlabel('Epochs')
    f.legend()
    f.savefig('FairD_[15]_TransD_FB15k_NCE_plot.jpg')

