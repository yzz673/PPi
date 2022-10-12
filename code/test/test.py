# coding: UTF-8

import os
import pickle
import torch
import numpy as np

from torch.utils.data import DataLoader

from ..config.config import Config
from ..dataset import MainDataset
from ...ssl.utils import pred_label_transfer


def test(config: Config, ssl_fe, power_fe, ssl_power_net, tar_bgs):
    target_patients = config.tar_patients
    root_origin_data_path = '/path/to/origin/data'
    root_power_path = '/path/to/root/power'
    with torch.no_grad():
        ssl_fe.eval()
        power_fe.eval()
        ssl_power_net.eval()

        for pat_idx, patient in enumerate(target_patients):
            # Save predict label
            result_dir = '/path/to/result/dir'

            test_x = np.load(os.path.join(root_origin_data_path, f'{patient}_data.npy'))
            test_y = np.load(os.path.join(root_origin_data_path, f'{patient}_label.npy'))

            ch_num = len(test_x)
            for ch_idx in range(ch_num):
                cur_ch_y, cur_ch_pred, cur_ch_pred_prob = [], [], []
                suffix = 'single_ch/unsampled/' + target_patients[pat_idx] + f'/{ch_idx}.npy'
                power_dir = os.path.join(root_power_path, suffix)
                subject = config.pat2subject[patient]
                dataset = MainDataset(raw_x=test_x[ch_idx],
                                      raw_y=test_y[ch_idx],
                                      ch_idx=ch_idx,
                                      power_dir=power_dir,
                                      subject=subject,
                                      step=config.step,
                                      config=config,
                                      sub_bg=config.use_bg,
                                      br=config.br,
                                      )
                data_iter = DataLoader(dataset, shuffle=False, batch_size=config.train_batch_size, drop_last=False, num_workers=0)
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

                    pred_y = torch.argmax(pred, dim=1).long().cpu().numpy().tolist()
                    tar_y = tar_y.view(-1).cpu().numpy().tolist()
                    pred_y, tar_y = pred_label_transfer(pred_y, tar_y)
                    normal_idx = np.where(pred_y == 0)
                    bg_sum += torch.sum(ssl_feat[normal_idx], dim=1)
                    bg_num += normal_idx.shape[0]

                    cur_ch_y += tar_y
                    cur_ch_pred += pred_y
                    cur_ch_pred_prob += pred[:, 1].cpu().numpy().tolist()

                cur_ch_y, cur_ch_pred, cur_ch_pred_prob = np.array(cur_ch_y), np.array(cur_ch_pred), np.array(cur_ch_pred_prob)

                pickle.dump(cur_ch_pred[cur_ch_pred != -1].tolist(), open(os.path.join(result_dir, f'pred_ch{ch_idx}.pkl'), 'wb'))
                pickle.dump(cur_ch_y[cur_ch_y != -1].tolist(), open(os.path.join(result_dir, f'label_ch{ch_idx}.pkl'), 'wb'))
                pickle.dump(cur_ch_pred_prob[cur_ch_pred_prob != -1].tolist(), open(os.path.join(result_dir, f'prob_ch{ch_idx}.pkl'), 'wb'))

                print(f"predict label of channel {ch_idx} of {patient} has been saved.")
