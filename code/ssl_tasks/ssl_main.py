import os
import argparse
import random

import numpy as np
import torch

from ..config.exp_settings import ExpSettings
from ssl_config import SSLConfig
from ssl_test import ssl_test
from ssl_train import ssl_train
from ..config.model_settings import model_setting

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SSL')
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--only_test", type=bool, default=False)
    args = parser.parse_args()

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    random.seed(1)
    torch.backends.cudnn.benchmark = True
    exp_id = args.exp_id
    exp_set = ExpSettings()

    exp_ids = ['some exp ids']

    for exp_id in exp_ids:
        num_pats = 7

        ssl_config = SSLConfig()
        ssl_config.exp_id = exp_id
        ssl_config.device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
        ssl_config.src_patients = exp_set.exp_ids[exp_id]['source']['filenames']
        ssl_config.val_patients = exp_set.exp_ids[exp_id]['valid']['filenames']
        ssl_config.tar_patients = exp_set.exp_ids[exp_id]['target']['filenames']
        print(str(ssl_config))

        model = model_setting(ssl_config)

        ssl_model_path = '/path/to/ssl/model'

        if args.only_test is False:
            ssl_train(ssl_config, model, ssl_model_path)

        model.load_state_dict(torch.load(ssl_model_path))

        ssl_test(ssl_config, model)