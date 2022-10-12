# coding: UTF-8
import os

import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader

import utils
from config import paths

from config.config import Config
from dataset import MainDataset


def evaluate(config: Config, ssl_fe, power_fe, ssl_power_net, loss_func, eval_bgs, eval_patients):
    path = paths.Paths(config)
    root_sampled_data_path = path.root_sampled_data_path
    root_power_path = path.root_power_path

    tot_y = []
    tot_pred = []
    tot_loss = 0
    with torch.no_grad():
        ssl_fe.eval()
        power_fe.eval()
        ssl_power_net.eval()

        for pat_idx, patient in enumerate(eval_patients):
            eval_x = pickle.load(open(os.path.join(root_sampled_data_path, f'x_{patient}.pkl'), 'rb'))
            eval_y = pickle.load(open(os.path.join(root_sampled_data_path, f'y_{patient}.pkl'), 'rb'))
            sampled_indices = pickle.load(open(os.path.join(root_sampled_data_path, f'sampled_indices_aug_{patient}.pkl'), 'rb'))

            ch_num = len(sampled_indices)

            for ch_idx in range(ch_num):
                suffix = 'single_ch/sampled/' + eval_patients[pat_idx] + f'/{ch_idx}.npy'
                power_dir = os.path.join(root_power_path, suffix)
                subject = config.pat2subject[patient]
                dataset = MainDataset(raw_x=eval_x[ch_idx],
                                      raw_y=eval_y[ch_idx],
                                      ch_idx=ch_idx,
                                      power_dir=power_dir,
                                      subject=subject,
                                      step=config.step,
                                      config=config,
                                      sub_bg=config.use_bg,
                                      br=config.br,
                                      )

                data_iter = DataLoader(dataset, shuffle=False, batch_size=config.test_batch_size, drop_last=False, num_workers=0)
                bg_sum = torch.zeros(config.d_model).to(config.device)
                bg_num = 1
                for ctc_x, ctc_power, tar_y in data_iter:
                    ctc_x = ctc_x.float()
                    _, _, ssl_feat = ssl_fe(ctc_x)
                    ssl_feat_sub_bg = ssl_feat - bg_sum / bg_num
                    ssl_feat_sub_bg = ssl_feat_sub_bg.float().to(config.device)

                    ctc_power = torch.swapaxes(ctc_power, axis0=1, axis1=2).float()
                    power_feat = power_fe(ctc_power).to(config.device)

                    pred = ssl_power_net(ssl_feat_sub_bg, power_feat)

                    bat_loss = loss_func(pred, tar_y.to(config.device))

                    pred_y = torch.argmax(pred, dim=1).long().cpu().numpy().tolist()
                    tar_y = tar_y.view(-1).cpu().numpy().tolist()
                    pred_y, tar_y = utils.pred_label_transfer(pred_y, tar_y)
                    normal_idx = np.where(pred_y == 0)
                    bg_sum += torch.sum(ssl_feat[normal_idx], dim=1)
                    bg_num += normal_idx.shape[0]
                    tot_pred += pred_y
                    tot_loss += bat_loss.cpu().item()

        metr = utils.Metrics(tot_pred, tot_y)
        return tot_loss, metr
