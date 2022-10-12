import pickle
import random
import torch
import numpy as np
from random import shuffle

import os

root_sampled_data_path = "/root/sampled/data/path"
root_origin_data_path = "/root/origin/data/path"


# func of SSL_Domain_Dataset
def get_2_ch_idx(ch_num):
    random.seed()
    select = random.randint(0, 1)  # 2 optional ways with 50%
    if select == 0:
        return random.randint(0, ch_num - 1), random.randint(0, ch_num - 1), 0
    elif select == 1:
        res = random.randint(0, ch_num - 1)
        return res, res, 1


def get_2_diff_ch_idx(ch_num):
    while True:
        ch_1, ch_2 = random.randint(0, ch_num - 1), random.randint(0, ch_num - 1)
        if ch_1 != ch_2:
            return ch_1, ch_2


def swap_ctxt(swp_seg_1, swp_seg_2, seg_length, half_context_num):
    swp = random.randint(0, 1)
    if swp == 1:
        l_or_r = random.randint(0, 1)
        if l_or_r == 0:
            l_ctxt_1 = swp_seg_1[: seg_length * half_context_num]
            swp_seg_1[: seg_length * half_context_num] = swp_seg_2[: seg_length * half_context_num]
            swp_seg_2[: seg_length * half_context_num] = l_ctxt_1
        else:
            r_ctxt_1 = swp_seg_1[seg_length * (half_context_num + 1):]
            swp_seg_1[seg_length * (half_context_num + 1):] = swp_seg_2[seg_length * (half_context_num + 1):]
            swp_seg_2[seg_length * (half_context_num + 1):] = r_ctxt_1

    return swp_seg_1, swp_seg_2, swp


def get_data(patients, shfl=True):
    pats_x, pats_y, pats_si, pats_ch_num = [], [], [], []
    pairs = []
    for pat_idx, patient in enumerate(patients):
        pats_x.append(pickle.load(open(os.path.join(root_sampled_data_path, f'x_{patient}.pkl'), 'rb')))
        pats_y.append(pickle.load(open(os.path.join(root_sampled_data_path, f'y_{patient}.pkl'), 'rb')))
        pats_si.append(pickle.load(open(os.path.join(root_sampled_data_path, f'sampled_indices_aug_{patient}.pkl'), 'rb')))
        pats_ch_num.append(len(pats_si[-1]))

        cur_ch_idx = list(range(pats_ch_num[-1]))
        cur_pairs = [(pat_idx, ch_idx) for ch_idx in cur_ch_idx]
        pairs += cur_pairs

    num_pairs = len(pairs)
    if shfl is True:
        shuffle(pairs)
    num_pats = len(patients)
    train_pairs, valid_pairs = pairs[: (num_pats - 1) * num_pairs // num_pats], pairs[num_pairs // num_pats:]
    return pats_x, pats_y, pats_si, train_pairs, valid_pairs


class SSLEarlyStopping(object):
    def __init__(self, path, patience=3, delta=0, low_better=False, metric='loss', exp_setting_id=''):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.metric = metric
        self.low_better = low_better
        self.path = path
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.init_value = np.Inf if self.low_better else -np.inf
        self.value_best = self.init_value
        self.delta = delta
        self.exp_setting_id = exp_setting_id

    def __call__(self, value, ssl_model):
        score = -value if self.low_better else value
        if value is np.nan:
            score = self.init_value
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(value, ssl_model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(value, ssl_model)
            self.counter = 0

    def save_checkpoint(self, value, ssl_model):
        print(
            f'Validation metric {"decreased" if self.low_better else "increased"} ({self.value_best:.6f} --> {value :.6f}).  Saving model ...')
        torch.save(ssl_model.state_dict(), self.path)
        self.value_best = value
