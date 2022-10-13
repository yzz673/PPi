# coding: UTF-8
import math
import os
import pickle
import time

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn

from dataset import MainDataset  # sorted_raw_metric
from config.config import Config
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, fbeta_score

root_sampled_data_path = "/root/sampled/data/path"
root_origin_data_path = "/root/origin/data/path"
root_power_path = "/root/power/path"


class Metrics:
    ...



class MulticlassMetrics:
    ...



class MainEarlyStopping(object):
    """Early stops the training if f2 doesn't improve after a given patience."""

    def __init__(self, exp_id, finetune=True, patience=7, delta=0, low_better=False, metric='loss', exp_setting_id=''):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.metric = metric
        self.low_better = low_better
        self.exp_id = exp_id
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.init_value = np.Inf if self.low_better else -np.inf
        self.value_best = self.init_value
        self.delta = delta
        self.exp_setting_id = exp_setting_id
        self.finetune = finetune

    def __call__(self, value, ssl_fe, power_fe, ssl_power_net):
        score = -value if self.low_better else value
        if value is np.nan:
            score = self.init_value
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(value, ssl_fe, power_fe, ssl_power_net)
        elif score < self.best_score + self.delta or math.isnan(score):
            self.counter += 1
            print(f'Validation metric = {score}. EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(value, ssl_fe, power_fe, ssl_power_net)
            self.counter = 0

    def save_checkpoint(self, value, ssl_fe, power_fe, main_model):
        print(f'Validation metric {"decreased" if self.low_better else "increased"} ({self.value_best:.6f} --> {value :.6f}).  Saving model ...')
        torch.save(ssl_fe.state_dict(), '/path/to/saved/ssl/fe')
        torch.save(power_fe.state_dict(), '/path/to/saved/power/fe')
        torch.save(main_model.state_dict(), '/path/to/saved/main/model')
        self.value_best = value


def get_background(config: Config, ssl_fe, patients, sampled=True, tqdm_dis=False, cal_new_bg=True):
    bgs = []
    for patient in patients:
        start = time.time()

        pat_bg_path = f'/path/to/patient/background'
        if os.path.exists(pat_bg_path) and cal_new_bg is False:
            # load bg
            pat_bg = torch.load(pat_bg_path, map_location=config.gpu_map_loc)
        else:
            # cal bg
            if sampled:
                train_x = pickle.load(open(os.path.join(root_sampled_data_path, f'x_{patient}.pkl'), 'rb'))
                train_y = pickle.load(open(os.path.join(root_sampled_data_path, f'y_{patient}.pkl'), 'rb'))
                sampled_indices = pickle.load(open(os.path.join(root_sampled_data_path, f'sampled_indices_aug_{patient}.pkl'), 'rb'))
                channel_num = len(sampled_indices)
            else:
                train_x = np.load(os.path.join(root_origin_data_path, f'{patient}_data.npy'))
                train_y = np.load(os.path.join(root_origin_data_path, f'{patient}_label.npy'))
                channel_num = len(train_x)
            pat_bg = torch.zeros(channel_num, config.d_model, dtype=torch.float32).to(config.device)
            for ch_idx in tqdm(range(channel_num), desc=f'background of {patient}', disable=tqdm_dis):
                subject = config.pat2subject[patient]
                power_dir = os.path.join(root_power_path, '/path/to/channel/background')
                dataset = MainDataset(raw_x=train_x[ch_idx],
                                      raw_y=train_y[ch_idx],
                                      ch_idx=ch_idx,
                                      power_dir=power_dir,
                                      step=config.step,
                                      subject=subject,
                                      sampled_indices=sampled_indices[ch_idx] if sampled else None,
                                      config=config,
                                      br=config.br,
                                      )
                # dataset_len = len(dataset)
                data_iter = DataLoader(dataset, shuffle=False, batch_size=config.bg_batch_size, drop_last=False, num_workers=0)
                # get background
                seg_num = 0
                with torch.no_grad():
                    for ctc_x, _, tar_y in data_iter:
                        ctc_x = ctc_x.float()
                        _, _, ssl_feat = ssl_fe(ctc_x)
                        ssl_feat = ssl_feat.float()
                        pat_bg[ch_idx] += torch.sum(ssl_feat, dim=0)
                        seg_num += ssl_feat.shape[0]
                pat_bg[ch_idx] /= seg_num
            # save bg
            torch.save(pat_bg, pat_bg_path)

        bgs.append(pat_bg)
        print(f"Getting background of {'un' if not sampled else ''}sampled {patient} spend %d s." % (time.time() - start))

    return bgs


def get_brain_region(brain_dict, ch_idx):
    for key, val in brain_dict.items():
        if ch_idx in val:
            return key


def pred_label_transfer(pred, label):
    for i in range(len(pred)):
        pred[i] = np.where(pred[i] % 2 == 0, 0, 1)
        label[i] = np.where(label[i] % 2 == 0, 0, 1)

    return pred, label

