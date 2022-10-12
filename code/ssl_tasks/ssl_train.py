import time
from random import shuffle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import SSLEarlyStopping
from ssl_dataset import SSLDataset
from ssl_config import SSLConfig
from ssl_eval import ssl_evaluate
from utils import get_data


def ssl_train(config: SSLConfig, model, ssl_model_path):
    patients = config.src_patients + config.val_patients
    early_stop = SSLEarlyStopping(path=ssl_model_path, patience=3, metric='loss', low_better=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    ce_loss_func = nn.CrossEntropyLoss()
    mse_loss_func = nn.MSELoss()

    parts_num = (len(patients) - 1) * 4  # leave the data for validation
    pats_x, _, pats_si, train_pairs, valid_pairs = get_data(patients)

    for epo_idx in range(config.num_epochs):
        start = time.time()
        model.train()
        epo_loss = 0
        epo_judge_loss, epo_rec_loss, epo_swp_loss = 0, 0, 0

        if config.shuffle_data is True:
            shuffle(train_pairs)
        for part_idx in range(parts_num):
            part_pair = train_pairs[part_idx * 5: (part_idx + 1) * 5]

            part_x = [pats_x[pat_idx][ch_idx] for pat_idx, ch_idx in part_pair]
            part_si = [pats_si[pat_idx][ch_idx] for pat_idx, ch_idx in part_pair]

            dataset = SSLDataset(raw_x_list=part_x,
                                 half_context_num=config.half_context_num,
                                 segment_length=config.segment_length,
                                 step=config.step,
                                 device=config.device,
                                 sampled_indices=part_si,
                                 )
            data_iter = DataLoader(dataset, shuffle=False, batch_size=config.train_batch_size, drop_last=True, num_workers=0)

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

                optimizer.zero_grad()
                bat_loss.backward()
                optimizer.step()

        print("Train epoch %2d: \nepo_loss = %.4f. epo_rec_loss = %.4f. epo_judge_loss = %.4f. epo_swp_loss = %.4f."
              % (epo_idx, epo_loss, epo_rec_loss, epo_judge_loss, epo_swp_loss))

        eval_loss = ssl_evaluate(config, model, valid_pairs, pats_x, pats_si)
        early_stop(eval_loss, model)
        if early_stop.early_stop: break
        print(f"This epoch spent %d s.\n" % (time.time() - start))

    torch.save(model.state_dict(), ssl_model_path)
    print(f">>> model state saved to {ssl_model_path}.")
