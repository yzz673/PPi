import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from ssl_dataset import SSLDataset
from ssl_config import SSLConfig
from ..utils import Metrics

root_sampled_data_path = "/root/sampled/data/path"
root_origin_data_path = "/root/origin/data/path"


def ssl_test(config: SSLConfig, model):
    model.eval()
    tot_judge_y, tot_judge_pred = [], []
    tot_swp_y, tot_swp_pred = [], []
    test_patients = config.tar_patients

    with torch.no_grad():
        for patient in test_patients:
            test_x = pickle.load(open(os.path.join(root_sampled_data_path, f'x_{patient}.pkl'), 'rb'))
            sampled_indices = pickle.load(open(os.path.join(root_sampled_data_path, f'sampled_indices_aug_{patient}.pkl'), 'rb'))

            dataset = SSLDataset(raw_x_list=test_x,
                                 half_context_num=config.half_context_num,
                                 segment_length=config.segment_length,
                                 step=config.step,
                                 device=config.device,
                                 sampled_indices=sampled_indices)
            data_iter = DataLoader(dataset, shuffle=False, batch_size=config.train_batch_size, drop_last=True, num_workers=0)

            for batch_idx, (ctc_x1, ctc_x2, y, swp_seg_1, swp_seg_2, swp) in enumerate(data_iter):
                ctc_x1, ctc_x2, swp_seg_1, swp_seg_2 = ctc_x1.float(), ctc_x2.float(), swp_seg_1.float(), swp_seg_2.float()
                model = model.float()
                _, _, _, _, judge_pred, swp_pred = model(ctc_x1, ctc_x2, swp_seg_1, swp_seg_2)

                judge_pred_y = torch.argmax(judge_pred, dim=1)
                swp_pred_y = torch.argmax(swp_pred, dim=1)

                tot_judge_y += y.view(-1).cpu().numpy().tolist()
                tot_swp_y += swp.view(-1).cpu().numpy().tolist()
                tot_judge_pred += judge_pred_y.long().cpu().numpy().tolist()
                tot_swp_pred += swp_pred_y.long().cpu().numpy().tolist()

    judge_metric = Metrics(np.array(tot_judge_pred), np.array(tot_judge_y))
    swp_metric = Metrics(np.array(tot_swp_pred), np.array(tot_swp_y))
    print("\tOn test judge task:", judge_metric.get_metrics(one_line=True), judge_metric.get_confusion())
    print("\tOn test swap task:", swp_metric.get_metrics(one_line=True), swp_metric.get_confusion())
