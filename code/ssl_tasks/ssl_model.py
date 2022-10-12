import torch
import torch.nn as nn


def t2v(tau, f, w, b, w0, b0):
    v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], 1)


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.w, self.b, self.w0, self.b0)


class Time2Vec(nn.Module):
    def __init__(self, activation, hidden_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, hidden_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        # x = x.unsqueeze(1)
        bat_size, seg_len = x.shape
        x = x.reshape(-1, 1)
        x = self.l1(x)

        x = x.reshape(bat_size, seg_len, -1)
        # x = self.fc1(x)
        return x


# input: [batch size, 1, seg_len * seg_num]
# output: [batch size, seg_len * seg_num, hidden_dim]
class CNNTime2Vec(nn.Module):
    def __init__(self, hidden_dim):
        super(CNNTime2Vec, self).__init__()
        self.cnn_time2vec = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=hidden_dim // 4, kernel_size=7, padding='same', stride=1),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=hidden_dim // 4, out_channels=hidden_dim // 2, kernel_size=7, padding='same', stride=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=hidden_dim // 2, out_channels=hidden_dim, kernel_size=7, padding='same', stride=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):  # x: [batch size, seg_len * seg_num]
        x = x.unsqueeze(dim=1)
        x = self.cnn_time2vec(x)
        x = torch.swapaxes(x, axis0=1, axis1=2)

        return x


# input: [batch size, seg_len * seg_num, input size]
# output: [batch size, seg_len * seg_num]
class CNNVec2Time(nn.Module):
    def __init__(self, input_size):
        super(CNNVec2Time, self).__init__()
        self.cnn_vec2time = nn.Sequential(
            nn.Conv1d(input_size, input_size // 2, kernel_size=7, padding='same', stride=1),
            nn.BatchNorm1d(input_size // 2),
            nn.ReLU(inplace=True),

            nn.Conv1d(input_size // 2, input_size // 4, kernel_size=7, padding='same', stride=1),
            nn.BatchNorm1d(input_size // 4),
            nn.ReLU(inplace=True),

            nn.Conv1d(input_size // 4, 1, kernel_size=7, padding='same', stride=1),
        )

    def forward(self, x):
        x = torch.swapaxes(x, axis0=1, axis1=2)  # x: [bat size, input size, seg_len * seg_num]
        x = self.cnn_vec2time(x)
        bat_size = x.shape[0]
        x = x.reshape(bat_size, -1)

        return x


# LSTM FE & Decoder

class LSTMFE(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, device):
        super(LSTMFE, self).__init__()
        self.num_layers = num_layers
        self.h_out = hidden_dim
        self.device = device

        # self.point2vec = Time2Vec(activation='sin', hidden_dim=input_size)
        self.point2vec = CNNTime2Vec(hidden_dim=input_size)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2, device=device)

    def forward(self, x):
        bat_size = x.shape[0]
        point_emb = self.point2vec(x)
        h0, c0 = torch.randn(self.num_layers, bat_size, self.h_out).float().to(self.device), \
                 torch.randn(self.num_layers, bat_size, self.h_out).float().to(self.device)
        enc_output, (_1, _2) = self.lstm(point_emb, (h0, c0))
        enc_output, _1, _2 = enc_output.to(self.device), _1.to(self.device), _2.to(self.device)
        seg_emb = torch.mean(enc_output, dim=1)  # mean pooling
        # seg_emb = torch.max(enc_output, dim=1).values  # max pooling
        return x, enc_output, seg_emb


class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, device, whole_seg_len=300 * 13):
        super(LSTMDecoder, self).__init__()
        self.num_layers = num_layers
        self.h_out = hidden_dim
        self.device = device

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.vec2point = CNNVec2Time(input_size=hidden_dim)


    def forward(self, x):
        bat_size = x.shape[0]
        h0, c0 = torch.randn(self.num_layers, bat_size, self.h_out), torch.randn(self.num_layers, bat_size, self.h_out)
        enc_output, (_, _) = self.lstm(x, (h0.to(self.device), c0.to(self.device)))
        rec = self.vec2point(enc_output)
        return rec



# SSL model & its components

class ChannelDiscriminator(nn.Module):
    def __init__(self, indim):
        super(ChannelDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=indim, out_features=indim // 2),
            nn.BatchNorm1d(indim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=indim // 2, out_features=2),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.discriminator(x)
        x = self.softmax(x)
        return x


class ContextSwapDiscriminator(nn.Module):
    def __init__(self, indim):
        super(ContextSwapDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=indim, out_features=indim // 2),
            nn.BatchNorm1d(indim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=indim // 2, out_features=2),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.discriminator(x)
        x = self.softmax(x)
        return x


class SSLModel(nn.Module):
    def __init__(self, FE, decoder, judgeDiscriminator: ChannelDiscriminator, ctxtSwpDiscriminator: ContextSwapDiscriminator):
        super(SSLModel, self).__init__()
        self.FE = FE
        self.decoder = decoder
        self.judgeDiscriminator = judgeDiscriminator
        self.ctxtSwpDiscriminator = ctxtSwpDiscriminator

    def forward(self, x1, x2, swp_x1, swp_x2):
        src_x1, point_h1, seg_h1 = self.FE(x1)
        src_x2, point_h2, seg_h2 = self.FE(x2)
        _, _, seg_swp_h1 = self.FE(swp_x1)
        _, _, seg_swp_h2 = self.FE(swp_x2)

        rec_x1 = self.decoder(point_h1)
        rec_x2 = self.decoder(point_h2)

        gts = torch.abs(seg_h1 - seg_h2)
        judge_pred = self.judgeDiscriminator(gts)

        gts = torch.cat([seg_swp_h1, seg_swp_h2], dim=1)
        swp_pred = self.ctxtSwpDiscriminator(gts)
        return src_x1, rec_x1, src_x2, rec_x2, judge_pred, swp_pred

