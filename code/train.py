# coding: UTF-8
import os

import time
from random import shuffle

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import MainDataset
from evaluate import evaluate
from utils import Metrics, get_background, pred_label_transfer, MainEarlyStopping

from config.config import Config
from loss_function import loss_weight_cal
from ssl_tasks.utils import get_data


def train(exp_id, config: Config, ssl_fe, power_fe, main_model, src_bgs, val_bgs):

    train_loss_weight = loss_weight_cal(config, config.src_patients)
    valid_loss_weight = loss_weight_cal(config, config.val_patients)

    train_loss_func = torch.nn.CrossEntropyLoss(weight=train_loss_weight)
    valid_loss_func = torch.nn.CrossEntropyLoss(weight=valid_loss_weight)

    root_power_path = '/path/to/saved/power'

    metric = config.metric
    low_better = True if metric == 'loss' else False
    early_stopping = MainEarlyStopping(exp_id=exp_id, finetune=config.finetune, patience=2, metric=metric, low_better=low_better)

    ssl_fe_params = list(ssl_fe.parameters())
    power_fe_params = list(power_fe.parameters())
    main_model_params = list(main_model.parameters())

    if config.finetune is True:
        optimizer = torch.optim.Adam(
            [{'params': ssl_fe_params, 'lr': config.ssl_learning_rate},
             {'params': power_fe_params, 'lr': config.power_learning_rate},
             {'params': main_model_params, 'lr': config.learning_rate}],
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    else:
        optimizer = torch.optim.Adam(
            [{'params': power_fe_params, 'lr': config.power_learning_rate},
             {'params': main_model_params, 'lr': config.learning_rate}],
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    pats_x, pats_y, pats_si, pairs1, pairs2 = get_data(config.src_patients, shfl=config.shuffle_data)
    pairs = pairs1 + pairs2

    for epoch in range(config.num_epochs):
        start = time.time()

        ssl_fe.train()
        power_fe.train()
        main_model.train()

        print(f"epoch {epoch} begins")
        tot_y = []
        tot_pred = []
        epo_loss = 0
        if config.shuffle_data is True:
            shuffle(pairs)

        for pat_idx, ch_idx in pairs:
            patient = config.src_patients[pat_idx]
            suffix = 'single_ch/sampled/' + patient + f'/{ch_idx}.npy'
            power_dir = os.path.join(root_power_path, suffix)
            subject = config.pat2subject[patient]
            dataset = MainDataset(raw_x=pats_x[pat_idx][ch_idx],
                                  raw_y=pats_y[pat_idx][ch_idx],
                                  ch_idx=ch_idx,
                                  power_dir=power_dir,
                                  subject=subject,
                                  step=config.step,
                                  sampled_indices=pats_si[pat_idx][ch_idx],
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

                pred = main_model(ssl_feat_sub_bg, power_feat)

                bat_loss = train_loss_func(pred, tar_y.to(config.device))

                optimizer.zero_grad()
                bat_loss.backward()
                optimizer.step()

                pred_y = torch.argmax(pred, dim=1).long().cpu().numpy().tolist()
                tar_y = tar_y.view(-1).cpu().numpy().tolist()
                pred_y, tar_y = pred_label_transfer(pred_y, tar_y)
                normal_idx = np.where(pred_y == 0)
                bg_sum += torch.sum(ssl_feat[normal_idx], dim=1)
                bg_num += normal_idx.shape[0]
                tot_y += tar_y
                tot_pred += pred_y
                epo_loss += bat_loss

        train_metr = Metrics(tot_pred, tot_y)
        print("Train epoch %2d: loss = %.4f." % (epoch, epo_loss))
        print("On train:", train_metr.get_metrics(one_line=True), train_metr.get_confusion())

        if epoch >= 0:
            valid_loss, valid_metr = evaluate(config, ssl_fe, power_fe, main_model, valid_loss_func, val_bgs, config.val_patients)
            print("On valid:", valid_metr.get_metrics(one_line=True), valid_metr.get_confusion())
            stop_value = valid_loss if metric == 'loss' else valid_metr.f_doub
            early_stopping(stop_value, ssl_fe, power_fe, main_model)
            if early_stopping.early_stop: break

        print(f"epoch {epoch} spent %d s.\n" % (time.time() - start))
