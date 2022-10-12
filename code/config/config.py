# coding: UTF-8
import torch


class Config(object):

    """配置参数"""
    def __init__(self):
        self.src_patients, self.val_patients, self.tar_patients = None, None, None
        self.exp_id = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.gpu_map_loc = lambda storage, loc: storage.cuda(0)
        self.tqdm_dis = True
        self.update_bg = True
        self.update_power = False
        self.sub_power_bg = True
        self.num_epochs = 50
        self.bg_batch_size = 512
        self.train_batch_size = 512
        self.test_batch_size = 512
        self.learning_rate = 1e-4
        self.ssl_learning_rate = 1e-5
        self.power_learning_rate = 1e-3
        self.finetune = True
        self.ssl_half_context_num = 3
        self.power_half_context_num = 7
        self.attention_num = 5
        self.segment_length = 300
        self.step = 1
        self.fe_name = 'ssl'
        self.metric = 'loss'
        self.d_model = 64
        self.classes = 2*(1+90)
        self.br = True
        self.shuffle_data = True
        self.use_bg = True
        self.neg_loss_weight = 0.05
        self.pat2subject = {'some mapping from patients file to subject'}

    def __str__(self):
        out = ''
        out += '-' * 15 + 'Config' + '-' * 15 + '\n'
        out += 'exp_id               ' + str(self.exp_id) + '\n'
        out += 'device               ' + str(self.device) + '\n'
        out += 'tqdm_dis             ' + str(self.tqdm_dis) + '\n'
        out += 'update_bg            ' + str(self.update_bg) + '\n'
        out += 'update_power         ' + str(self.update_power) + '\n'
        out += 'sub_power_bg         ' + str(self.sub_power_bg) + '\n'
        out += 'bg_batch_size        ' + str(self.bg_batch_size) + '\n'
        out += 'train_batch_size     ' + str(self.train_batch_size) + '\n'
        out += 'test_batch_size      ' + str(self.test_batch_size) + '\n'
        out += 'learning_rate        ' + str(self.learning_rate) + '\n'
        out += 'ssl_learning_rate    ' + str(self.ssl_learning_rate) + '\n'
        out += 'power_learning_rate  ' + str(self.power_learning_rate) + '\n'
        out += 'finetune             ' + str(self.finetune) + '\n'
        out += 'half_context_num     ' + str(self.ssl_half_context_num) + '\n'
        out += 'power_half_ctc_num   ' + str(self.power_half_context_num) + '\n'
        out += 'attention_num        ' + str(self.attention_num) + '\n'
        out += 'segment_length       ' + str(self.segment_length) + '\n'
        out += 'step                 ' + str(self.step) + '\n'
        out += 'fe_name              ' + str(self.fe_name) + '\n'
        out += 'metric               ' + str(self.metric) + '\n'
        out += 'd_model              ' + str(self.d_model) + '\n'
        out += 'classes              ' + str(self.classes) + '\n'
        out += 'br                   ' + str(self.br) + '\n'
        out += 'shuffle_data         ' + str(self.shuffle_data) + '\n'
        out += 'use_bg               ' + str(self.use_bg) + '\n'
        out += 'neg_loss_weight      ' + str(self.neg_loss_weight) + '\n'

        out += '\n'
        out += 'src_patients             ' + str(self.src_patients) + '\n'
        out += 'val_patients             ' + str(self.val_patients) + '\n'
        out += 'tar_patients             ' + str(self.tar_patients) + '\n'
        return out

