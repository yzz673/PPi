from torch import nn
import torch

from torch.utils.data import DataLoader
from ssl_config import SSLConfig

from ssl_dataset import SSLDataset

root_sampled_data_path = "/root/sampled/data/path"
root_origin_data_path = "/root/origin/data/path"


def ssl_evaluate(config: SSLConfig, model, valid_pairs, pats_x, pats_si):
    model.eval()
    parts_num = len(valid_pairs) // 5

    ce_loss_func = nn.CrossEntropyLoss()
    mse_loss_func = nn.MSELoss()

    epo_loss = 0
    epo_judge_loss, epo_rec_loss, epo_swp_loss = 0, 0, 0

    with torch.no_grad():
        for part_idx in range(parts_num):
            part_pair = valid_pairs[part_idx * 5: (part_idx + 1) * 5]

            valid_x = [pats_x[pat_idx][ch_idx] for pat_idx, ch_idx in part_pair]
            valid_si = [pats_si[pat_idx][ch_idx] for pat_idx, ch_idx in part_pair]

            dataset = SSLDataset(raw_x_list=valid_x,
                                 half_context_num=config.half_context_num,
                                 segment_length=config.segment_length,
                                 step=config.step,
                                 device=config.device,
                                 sampled_indices=valid_si,
                                 )
            data_iter = DataLoader(dataset, shuffle=False, batch_size=config.test_batch_size, drop_last=True, num_workers=0)
            for batch_idx, (ctc_x1, ctc_x2, y, swp_seg_1, swp_seg_2, swp) in enumerate(data_iter):
                ctc_x1, ctc_x2, swp_seg_1, swp_seg_2 = ctc_x1.float(), ctc_x2.float(), swp_seg_1.float(), swp_seg_2.float()
                model = model.float()
                src_x1, rec_x1, src_x2, rec_x2, judge_pred, swp_pred = model(ctc_x1, ctc_x2, swp_seg_1, swp_seg_2)

                rec_loss = mse_loss_func(src_x1, rec_x1) + mse_loss_func(src_x2, rec_x2)
                judge_loss = ce_loss_func(judge_pred, y)
                swp_loss = ce_loss_func(swp_pred, swp)
                bat_loss = rec_loss + judge_loss + swp_loss

                epo_loss += bat_loss
                epo_rec_loss += rec_loss
                epo_judge_loss += judge_loss
                epo_swp_loss += swp_loss

    print("On valid: \nepo_loss = %.4f. epo_rec_loss = %.4f. epo_judge_loss = %.4f. epo_swp_loss = %.4f."
          % (epo_loss, epo_rec_loss, epo_judge_loss, epo_swp_loss))
    return epo_loss
