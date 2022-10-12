from torch.utils.data import Dataset
import numpy as np
import torch

from utils import get_2_ch_idx, get_2_diff_ch_idx, swap_ctxt


class SSLDataset(Dataset):
    def __init__(self, raw_x_list, half_context_num, segment_length, step, device='cpu', sampled_indices=None, initial_length=11059200):
        self.device = device
        self.step = step

        self.ch_num = len(raw_x_list)
        self.xlist = raw_x_list
        for ch_idx in range(self.ch_num):
            self.xlist[ch_idx] = self.xlist[ch_idx][: initial_length] if sampled_indices is not None else \
                self.xlist[ch_idx]

        self.half_context_num = half_context_num
        self.segment_length = segment_length
        self.half_context_len = self.half_context_num * self.segment_length
        self.sampled_indices = sampled_indices

        x_min_len = np.min([len(self.xlist[i]) for i in range(len(self.xlist))])
        si_min_len = np.min([len(self.sampled_indices[i]) for i in range(len(self.sampled_indices))])
        self.datasetlen = x_min_len // self.segment_length if self.sampled_indices is None else \
            si_min_len // self.segment_length

        self.seg_num = []
        self.pad_len = self.half_context_len + self.segment_length
        padding = np.zeros([self.pad_len])

        for ch_idx in range(self.ch_num):
            self.seg_num.append(len(self.xlist[ch_idx]) // self.segment_length)
            self.xlist[ch_idx] = self.xlist[ch_idx][: self.seg_num[ch_idx] * self.segment_length]
            self.xlist[ch_idx] = np.concatenate((padding, self.xlist[ch_idx], padding), axis=0)

    def __getitem__(self, index):
        ch1, ch2, y = get_2_ch_idx(self.ch_num)
        swp_ch_1, swp_ch_2 = get_2_diff_ch_idx(self.ch_num)

        index1 = index * self.segment_length
        index2 = index * self.segment_length
        swp_idx_1 = index * self.segment_length
        swp_idx_2 = index * self.segment_length
        if self.sampled_indices is not None:
            index1 = self.sampled_indices[ch1][index1]
            index2 = self.sampled_indices[ch2][index2]
            swp_idx_1 = self.sampled_indices[swp_ch_1][swp_idx_1]
            swp_idx_2 = self.sampled_indices[swp_ch_2][swp_idx_2]
        else:
            index1 = index1
            index2 = index2
            swp_idx_1 = swp_idx_1
            swp_idx_2 = swp_idx_2

        index1 += self.pad_len
        index2 += self.pad_len
        swp_idx_1 += self.pad_len
        swp_idx_2 += self.pad_len

        ctc_x1 = self.xlist[ch1][index1 - self.half_context_len: index1 + self.half_context_len + self.segment_length]
        ctc_x2 = self.xlist[ch2][index2 - self.half_context_len: index2 + self.half_context_len + self.segment_length]

        swp_seg_1 = self.xlist[swp_ch_1][
                    swp_idx_1 - self.half_context_len: swp_idx_1 + self.half_context_len + self.segment_length]
        swp_seg_2 = self.xlist[swp_ch_2][
                    swp_idx_2 - self.half_context_len: swp_idx_2 + self.half_context_len + self.segment_length]

        swp_seg_1, swp_seg_2, swp = swap_ctxt(swp_seg_1, swp_seg_2, self.segment_length, self.half_context_num)
        return torch.from_numpy(ctc_x1).to(self.device)[:: self.step], \
               torch.from_numpy(ctc_x2).to(self.device)[:: self.step], \
               torch.tensor(y, dtype=torch.long).to(self.device), \
               torch.from_numpy(swp_seg_1).to(self.device)[:: self.step], \
               torch.from_numpy(swp_seg_2).to(self.device)[:: self.step], \
               torch.tensor(swp, dtype=torch.long).to(self.device)

    def __len__(self):
        return self.datasetlen
