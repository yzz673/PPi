import os

from torch.utils.data import Dataset
import numpy as np
import torch
import pickle

from power.utils import get_power

root_brain_region_path = '/path/to/brain/region'


class MainDataset(Dataset):
    def __init__(self, raw_x, raw_y, power_dir, subject, ch_idx, step, config,
                 br=True, sub_bg=True, sampled_indices=None, initial_length=11059200):
        self.device = config.device
        self.step = step
        self.sub_bg = sub_bg
        self.ch_idx = ch_idx
        self.subject = subject
        if br is True:
            self.br_idx = self.get_brain_region()
        self.br = br

        self.x = raw_x
        self.x = self.x[: initial_length] if sampled_indices is not None else self.x
        self.y = raw_y
        self.y = self.y[: initial_length] if sampled_indices is not None else self.y

        self.ssl_half_context_num = config.ssl_half_context_num
        self.power_half_context_num = config.power_half_context_num
        self.segment_length = config.segment_length
        self.data_half_context_len = self.ssl_half_context_num * self.segment_length
        self.power_half_context_len = self.power_half_context_num * self.segment_length
        self.sampled_indices = sampled_indices

        self.datasetlen = len(self.x) // self.segment_length if self.sampled_indices is None else \
            len(self.sampled_indices) // self.segment_length

        self.seg_num = len(self.x) // self.segment_length
        self.x = self.x[: self.seg_num * self.segment_length]
        self.y = self.y[: self.seg_num * self.segment_length]

        if os.path.exists(power_dir):
            self.power = np.load(power_dir)
        else:
            self.power = get_power(self.x, self.segment_length)
            np.save(power_dir, self.power)

        if self.sub_bg is True:
            self.power = self.power - np.mean(self.power, axis=0)

        self.data_pad_len = self.data_half_context_len + self.segment_length
        data_padding = np.zeros([self.data_pad_len])
        self.x = np.concatenate((data_padding, self.x, data_padding), axis=0)

        self.power_pad_num = self.power_half_context_num + 1
        power_padding = np.zeros([self.power_pad_num, 8])
        self.power = np.concatenate((power_padding, self.power, power_padding), axis=0)

    def __getitem__(self, seg_idx):
        # seg_idx : [0, datasetlen), which is totally the same as ctc_idx
        point_idx = seg_idx * self.segment_length

        if self.sampled_indices is not None:
            point_idx = self.sampled_indices[point_idx]
            seg_idx = point_idx // self.segment_length

        data_point_idx = point_idx + self.data_pad_len  # data padding at beginning
        power_seg_idx = seg_idx + self.power_pad_num  # power padding at beginning
        label_point_idx = point_idx  # no label padding

        # y
        tar_y = self.y[label_point_idx: label_point_idx + self.segment_length]
        tar_y = 1 if np.sum(tar_y) > 0 else 0
        tar_y = torch.tensor(tar_y + (self.br_idx * 2 if self.br is True else 0)).long()

        # # context-target-context
        ctc_x = self.x[data_point_idx - self.data_half_context_len: data_point_idx + self.data_half_context_len + self.segment_length][::self.step]
        ctc_power = self.power[power_seg_idx - self.power_half_context_num: power_seg_idx + self.power_half_context_num + 1]

        assert ctc_x.shape[0] == (self.data_half_context_len * 2 + self.segment_length) // self.step, \
            f'This would result in misaligned batch, see at seg_idx {seg_idx}.'

        return torch.from_numpy(ctc_x).to(self.device), \
               torch.from_numpy(ctc_power).to(self.device), \
               tar_y.to(self.device)

    def __len__(self):
        return self.datasetlen

    def get_brain_region(self):
        brain_dict = self.get_brain_dict()
        for br_idx, ch_list in brain_dict.items():
            if self.ch_idx in ch_list:
                return br_idx

    def get_brain_dict(self):
        suffix = self.subject + '/brain_dict.pkl'
        dir = os.path.join(root_brain_region_path, suffix)
        with open(dir, 'rb') as f:
            brain_dict = pickle.load(f)

        # print(f'brain dict of patient {subject}:\n', brain_dict)
        return brain_dict
