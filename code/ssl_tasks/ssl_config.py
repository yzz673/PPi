import torch


class SSLConfig:
    def __init__(self):
        self.exp_id = None
        self.src_patients, self.val_patients, self.tar_patients = None, None, None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = 50
        self.train_batch_size = 128
        self.test_batch_size = 128
        self.learning_rate = 1e-4
        # SSL model arch
        self.half_context_num = 3
        self.segment_length = 300
        self.step = 1
        self.dropout = 0.2
        self.d_model = 64
        self.n_head = 4
        self.n_enc_layers = 2
        self.shuffle_data = True

    def __str__(self):
        out = ''
        out += '-' * 15 + 'Config' + '-' * 15 + '\n'
        out += 'exp_id               ' + str(self.exp_id) + '\n'
        out += 'device               ' + str(self.device) + '\n'
        out += 'train_batch_size     ' + str(self.train_batch_size) + '\n'
        out += 'test_batch_size      ' + str(self.test_batch_size) + '\n'
        out += 'learning_rate        ' + str(self.learning_rate) + '\n'
        out += 'half_context_num     ' + str(self.half_context_num) + '\n'
        out += 'segment_length       ' + str(self.segment_length) + '\n'
        out += 'step                 ' + str(self.step) + '\n'
        out += 'dropout              ' + str(self.dropout) + '\n'
        out += 'd_model              ' + str(self.d_model) + '\n'
        out += 'n_head               ' + str(self.n_head) + '\n'
        out += 'n_enc_layers         ' + str(self.n_enc_layers) + '\n'
        out += 'shuffle_data         ' + str(self.shuffle_data) + '\n'
        out += '\n'
        out += 'src_patients         ' + str(self.src_patients) + '\n'
        out += 'val_patients         ' + str(self.val_patients) + '\n'
        out += 'tar_patients         ' + str(self.tar_patients) + '\n'
        return out
