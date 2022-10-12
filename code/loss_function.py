import os
import pickle

import numpy as np
import torch

from config.config import Config
from utils import get_brain_region

root_sampled_data_path = '/path/to/sampled/data'
root_brain_region_path = '/path/to/brain/region'


def loss_weight_cal(config: Config, patients):
    fre = np.zeros(2*(90+1))
    num_points = 0
    for patient in patients:
        pat_y = pickle.load(open(os.path.join(root_sampled_data_path, f'y_{patient}.pkl'), 'rb'))

        subject = config.pat2subject[patient]
        suffix = subject + '/brain_dict.pkl'
        dir = os.path.join(root_brain_region_path, suffix)
        with open(dir, 'rb') as f:
            brain_dict = pickle.load(f)

        for ch_idx, ch_y in enumerate(pat_y):
            brain_region = get_brain_region(brain_dict, ch_idx)
            fre[2*brain_region] += np.sum(ch_y == 0)
            fre[2*brain_region+1] += np.sum(ch_y == 1)

            num_points += len(ch_y)

    fre = torch.tensor(fre / num_points, dtype=torch.float32)

    alpha = 0.01
    use_log = False
    weight = torch.log(1 / (fre+alpha)) if use_log else 1 / (fre + alpha)

    return weight.to(config.device)




