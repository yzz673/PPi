import argparse
import time

import numpy as np
import torch
import random

from config.config import Config
from config.exp_settings import ExpSettings
from ssl_tasks.ssl_config import SSLConfig
from test.test import test
from config.model_settings import model_setting

from train import train
from model import MainModel, PowerFE

from utils import get_background


def run(ssl_config: SSLConfig, config: Config):

    # --------- model preparation ---------
    ssl_model = model_setting(ssl_config)
    ssl_model_path = f'/path/to/saved/ssl/model'
    ssl_model.load_state_dict(torch.load(ssl_model_path, map_location=config.gpu_map_loc))

    ssl_fe = ssl_model.FE.to(config.device).float()
    ssl_fe.device = config.device
    del ssl_model
    power_fe = PowerFE(seg_num=2 * config.power_half_context_num + 1, band_num=8, feat_dim=config.d_model).to(config.device).float()

    main_model = MainModel(config=config, pre_batch_norm=True).to(config.device).float()

    # --------- train ---------
    # calculate src_bgs and val_bgs
    src_bgs = get_background(config, ssl_fe, config.src_patients, sampled=True, tqdm_dis=config.tqdm_dis, cal_new_bg=True) if config.use_bg else None
    val_bgs = get_background(config, ssl_fe, config.val_patients, sampled=True, tqdm_dis=config.tqdm_dis, cal_new_bg=True) if config.use_bg else None

    if args.only_test is False:
        train(exp_id, config, ssl_fe, power_fe, main_model, src_bgs, val_bgs)

    # --------- load model ---------
    print("\n>>> Loading trained models to get performance metrics. <<<\n")
    ssl_fe.load_state_dict(torch.load('/path/to/saved/ssl/fe', map_location=config.gpu_map_loc))
    power_fe.load_state_dict(torch.load('/path/to/saved/power/fe', map_location=config.gpu_map_loc))
    main_model.load_state_dict(torch.load('/path/to/saved/main/model', map_location=config.gpu_map_loc))

    # --------- get performance metrics ---------
    # calculate tar_bgs
    tar_bgs = get_background(config, ssl_fe, config.tar_patients, sampled=False, tqdm_dis=config.tqdm_dis, cal_new_bg=True) if config.use_bg else None
    test(config, ssl_fe, power_fe, main_model, tar_bgs)
    print("predict result saved. ")



if __name__ == '__main__':
    print('This progress began at: ' + time.asctime(time.localtime(time.time())))
    torch.set_default_tensor_type(torch.DoubleTensor)

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=7)
    parser.add_argument("--exp_id", type=str, default='an exp setting id')
    parser.add_argument("--tqdm_dis", type=bool, default=False)
    parser.add_argument("--only_test", type=bool, default=False)
    args = parser.parse_args()

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    random.seed(1)
    torch.backends.cudnn.benchmark = True

    exp_ids = ['some exp settings']
    ssl_config = SSLConfig()

    config = Config()
    config.num_pats = 7
    exp_id = args.exp_id
    exp_set = ExpSettings()

    for exp_id in exp_ids:
        config.device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
        config.exp_id = exp_id
        config.tqdm_dis = args.tqdm_dis
        config.gpu_map_loc = lambda storage, loc: storage.cuda(args.gpu_id)
        config.src_patients = exp_set.exp_ids[exp_id]['source']['filenames']
        config.val_patients = exp_set.exp_ids[exp_id]['valid']['filenames']
        config.tar_patients = exp_set.exp_ids[exp_id]['target']['filenames']
        config.train_batch_size = 8192
        config.test_batch_size = 8192
        print(str(config))

        run(ssl_config, config)


    print('This progress finished at: ' + time.asctime(time.localtime(time.time())))
