import numpy as np
import pickle
import os
from tqdm import tqdm
from ..config.config import Config
from ..config.exp_settings import ExpSettings
from ..utils import Metrics
import argparse

import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--exp_id", type=str, default='an exp id')
args = parser.parse_args()

exp_id = args.exp_id
num_pats = 7

exp_set = ExpSettings()
patients = exp_set.exp_ids[exp_id]['target']['filenames']

config = Config()

for p_idx, patient in enumerate(patients):
    result_dir = '/path/to/result/dir'

    n_channel = len(os.listdir(result_dir)) // 3

    p_label_ch, p_pred_ch = [], []    # channel level
    p_label_p, p_pred_p = None, None  # patient level
    print(f"begin to calculate metric of {patient}...")
    for ch_idx in tqdm(range(n_channel), disable=True):
        ch_pred = pickle.load(open(os.path.join(result_dir, f'pred_ch{ch_idx}.pkl'), 'rb'))
        ch_label = pickle.load(open(os.path.join(result_dir, f'label_ch{ch_idx}.pkl'), 'rb'))

        ch_pred = np.array(ch_pred)
        ch_label = np.array(ch_label)

        ch_pred = np.where(ch_pred % 2 == 0, 0, 1)
        ch_label = np.where(ch_label % 2 == 0, 0, 1)

        p_label_ch += ch_label.tolist()
        p_pred_ch += ch_pred.tolist()

        ch_metr = Metrics(ch_pred, ch_label)
        print(f"\tMetrics of channel {ch_idx} of exp{exp_id}: " + ch_metr.get_metrics(one_line=True),ch_metr.get_confusion())

    p_pred_ch, p_label_ch = np.array(p_pred_ch), np.array(p_label_ch)
    pos = np.sum(p_label_ch)
    neg = p_label_ch.shape[0] - pos

    print(f'patient {patient}: pos={pos}, neg={neg}')

    metr_ch = Metrics(p_pred_ch, p_label_ch)
    res_ch = metr_ch.get_metrics(one_line=False)
    print(f"Channel level performance metrics of {patient}:")
    print(res_ch)
    print(metr_ch.get_confusion())

