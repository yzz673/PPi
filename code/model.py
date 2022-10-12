import torch
import torch.nn as nn
from config.config import Config


class Classifier(nn.Module):
    def __init__(self, indim, outdim, pre_batch_norm=True):
        super(Classifier, self).__init__()
        self.pre_batch_norm = pre_batch_norm
        self.preprocess = nn.BatchNorm1d(indim, track_running_stats=False)
        self.hidden_dim = (indim+outdim)*2//3
        self.detector = nn.Sequential(
            nn.Linear(in_features=indim, out_features=self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim, track_running_stats=False),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.hidden_dim, out_features=outdim),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if self.pre_batch_norm:
            x = self.preprocess(x)
        x = self.detector(x)
        x = self.softmax(x)
        return x


class MainModel(nn.Module):
    def __init__(self, config: Config, pre_batch_norm=True):
        super(MainModel, self).__init__()
        feat_dim = config.d_model
        self.attention = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=2, batch_first=True)
        out_dim = config.classes if config.br is True else 2
        self.classifier = Classifier(indim=feat_dim, outdim=out_dim, pre_batch_norm=pre_batch_norm)

    def forward(self, ssl_feat, power_feat):
        ssl_feat = torch.unsqueeze(ssl_feat, dim=1)
        power_feat = torch.unsqueeze(power_feat, dim=1)

        feat = torch.concat([ssl_feat, power_feat], dim=1)
        attn_output, _ = self.attention(feat, feat, feat)

        out_feat = torch.sum(attn_output, dim=1)

        pred = self.classifier(out_feat)
        return pred


# input shape: [batch size, band num(8), seg num]
# output shape: [batch size, feat dim]
class PowerFE(nn.Module):
    def __init__(self, seg_num, band_num, feat_dim, device=None):
        super(PowerFE, self).__init__()
        self.seg_num = seg_num
        self.band_num = band_num
        self.feat_dim = feat_dim
        self.device = device

        self.power_fe = nn.Sequential(
            nn.Conv1d(in_channels=self.band_num, out_channels=self.band_num * 2, kernel_size=3, padding='same', stride=1),
            nn.BatchNorm1d(self.band_num * 2, track_running_stats=False),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=self.band_num * 2, out_channels=self.band_num * 4, kernel_size=3, padding='same', stride=1),
            nn.BatchNorm1d(self.band_num * 4, track_running_stats=False),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.seg_num * self.band_num * 4, out_features=self.feat_dim),
        )

    def forward(self, x):  # x: [batch size, band num, seg num]
        x = self.power_fe(x)
        batch_size, _, _ = x.shape
        x = x.reshape(batch_size, -1)
        x = self.fc(x)

        return x


