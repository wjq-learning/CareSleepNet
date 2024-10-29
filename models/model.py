import torch
import torch.nn as nn
from models.attention import CMEncoder, UniEncoder
from models.config import Config, UniConfig
import copy
import torch.nn.functional as F
from models.transformer import TransformerEncoder, CMTransformerEncoder


class Transpose(nn.Module):
    def __init__(self):
        super(Transpose, self).__init__()
    def forward(self, x):
        return x.transpose(1, 2)

class MSDconv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups):
        super(MSDconv, self).__init__()
        self.downsample = nn.Conv1d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=1,
            stride=stride,
            bias=False,
            groups=groups
        )
        self.bn0 = nn.BatchNorm1d(out_planes)
        self.dconv1 = nn.Conv1d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            dilation=1,
            groups=groups
        )
        self.bn1 = nn.BatchNorm1d(out_planes)
        self.dconv2 = nn.Conv1d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding*2,
            bias=False,
            dilation=2,
            groups=groups
        )
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.dconv3 = nn.Conv1d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding*4,
            bias=False,
            dilation=4,
            groups=groups
        )
        self.bn3 = nn.BatchNorm1d(out_planes)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(out_planes, eps=1e-6)
        # self.fc = nn.Linear(out_planes * 4, out_planes)
        # self.conv = nn.Conv1d()
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        down = self.bn0(self.downsample(x))
        x1 = F.gelu(self.bn1(self.dconv1(x)))
        x2 = F.gelu(self.bn2(self.dconv2(x)))
        x3 = F.gelu(self.bn3(self.dconv3(x)))
        # print(down.shape, x1.shape, x2.shape, x3.shape)
        out = down + x1 + x2 + x3
        out = self.dropout(out)
        out = self.layer_norm(out.transpose(1, 2)).transpose(1, 2)

        return out

# class MSDconv(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
#         super(MSDconv, self).__init__()
#         self.downsample = nn.Conv1d(
#             in_channels=in_planes,
#             out_channels=out_planes,
#             kernel_size=1,
#             stride=stride,
#             bias=False,
#         )
#         self.bn0 = nn.BatchNorm1d(out_planes)
#         self.dconv1 = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=in_planes,
#                 out_channels=in_planes,
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 padding=padding,
#                 bias=False,
#                 dilation=1,
#                 groups=in_planes,
#             ),
#             nn.Conv1d(
#                 in_channels=in_planes,
#                 out_channels=out_planes,
#                 kernel_size=1,
#                 bias=False,
#             ),
#         )
#         self.bn1 = nn.BatchNorm1d(out_planes)
#         self.dconv2 = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=in_planes,
#                 out_channels=in_planes,
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 padding=padding * 2,
#                 bias=False,
#                 dilation=2,
#                 groups=in_planes,
#             ),
#             nn.Conv1d(
#                 in_channels=in_planes,
#                 out_channels=out_planes,
#                 kernel_size=1,
#                 bias=False,
#             ),
#         )
#         self.bn2 = nn.BatchNorm1d(out_planes)
#         self.dconv3 = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=in_planes,
#                 out_channels=in_planes,
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 padding=padding * 4,
#                 bias=False,
#                 dilation=4,
#                 groups=in_planes,
#             ),
#             nn.Conv1d(
#                 in_channels=in_planes,
#                 out_channels=out_planes,
#                 kernel_size=1,
#                 bias=False,
#             ),
#         )
#         self.bn3 = nn.BatchNorm1d(out_planes)
#         self.dropout = nn.Dropout(0.1)
#         self.layer_norm = nn.LayerNorm(out_planes, eps=1e-6)
#         # self.fc = nn.Linear(out_planes * 4, out_planes)
#         # self.conv = nn.Conv1d()
#         self.apply(self.init_weights)
#
#     def init_weights(self, module):
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             module.weight.data.normal_(mean=0.0, std=0.02)
#         elif isinstance(module, LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#         if isinstance(module, nn.Linear) and module.bias is not None:
#             module.bias.data.zero_()
#
#     def forward(self, x):
#         down = self.bn0(self.downsample(x))
#         x1 = F.gelu(self.bn1(self.dconv1(x)))
#         x2 = F.gelu(self.bn2(self.dconv2(x)))
#         x3 = F.gelu(self.bn3(self.dconv3(x)))
#         # print(down.shape, x1.shape, x2.shape, x3.shape)
#         out = down + x1 + x2 + x3
#         out = self.dropout(out)
#         out = self.layer_norm(out.transpose(1, 2)).transpose(1, 2)
#
#         return out

class EpochEncoder(nn.Module):
    def __init__(self, params, in_plane):
        super(EpochEncoder, self).__init__()
        self.encoder = nn.Sequential(
            MSDconv(in_plane, 64, kernel_size=49, stride=12,  padding=24, groups=1),
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),

            # nn.Dropout(params.dropout),

            MSDconv(64, 128, kernel_size=9, stride=1,  padding=4, groups=1),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),

            Transpose(),
            # nn.TransformerEncoderLayer(128, 8, 512, batch_first=True),
            UniEncoder(UniConfig(hidden_size=128, intermediate_size=512, )),
            Transpose(),

            MSDconv(128, 256, kernel_size=9, stride=1, padding=4, groups=1),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),

            Transpose(),
            # nn.TransformerEncoderLayer(256, 8, 1024, batch_first=True),
            UniEncoder(UniConfig(hidden_size=256, intermediate_size=1024)),
            Transpose(),

            MSDconv(256, 512, kernel_size=9, stride=1, padding=4, groups=1),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),

            Transpose(),
            # nn.TransformerEncoderLayer(512, 8, 2048, batch_first=True),
            UniEncoder(UniConfig(hidden_size=512, intermediate_size=2048)),
            Transpose(),
        )

        # self.encoder = nn.Sequential(
        #     nn.Conv1d(in_plane, 64, kernel_size=49, stride=12, padding=24, bias=False),
        #     nn.BatchNorm1d(64),
        #     nn.GELU(),
        #     # nn.MaxPool1d(kernel_size=9, stride=2, padding=4),
        #     # nn.Dropout(params.dropout),
        #
        #     nn.Conv1d(64, 128, kernel_size=9, stride=1, padding=4, bias=False),
        #     nn.BatchNorm1d(128),
        #     nn.GELU(),
        #     nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
        #
        #     Transpose(),
        #     # nn.TransformerEncoderLayer(128, 8, 512, batch_first=True),
        #     UniEncoder(UniConfig(hidden_size=128, intermediate_size=512, )),
        #     Transpose(),
        #
        #     nn.Conv1d(128, 256, kernel_size=9, stride=1, padding=4, bias=False),
        #     nn.BatchNorm1d(256),
        #     nn.GELU(),
        #     nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
        #     #
        #     Transpose(),
        #     # nn.TransformerEncoderLayer(256, 8, 1024, batch_first=True),
        #     UniEncoder(UniConfig(hidden_size=256, intermediate_size=1024)),
        #     Transpose(),
        #
        #     nn.Conv1d(256, 512, kernel_size=9, stride=1, padding=4, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.GELU(),
        #     nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
        #
        #     Transpose(),
        #     # nn.TransformerEncoderLayer(512, 8, 2048, batch_first=True),
        #     UniEncoder(UniConfig(hidden_size=512, intermediate_size=2048)),
        #     Transpose(),
        # )

        # self.encoder = nn.Sequential(
        #     nn.Conv1d(in_plane, 128, kernel_size=49, stride=6, bias=False, padding=24),
        #     nn.BatchNorm1d(128),
        #     nn.GELU(),
        #     nn.MaxPool1d(kernel_size=9, stride=2, padding=4),
        #
        #     nn.Dropout(params.dropout),
        #
        #     nn.Conv1d(128, 256, kernel_size=9, stride=1, bias=False, padding=4),
        #     nn.BatchNorm1d(256),
        #     nn.GELU(),
        #     nn.MaxPool1d(kernel_size=9, stride=2, padding=4),
        #
        #     Transpose(),
        #     UniEncoder(UniConfig(hidden_size=256, intermediate_size=1024, )),
        #     Transpose(),
        #
        #     nn.Conv1d(256, 512, kernel_size=9, stride=1, bias=False, padding=4),
        #     nn.BatchNorm1d(512),
        #     nn.GELU(),
        #     nn.MaxPool1d(kernel_size=9, stride=2, padding=4),
        #
        #     Transpose(),
        #     UniEncoder(UniConfig(hidden_size=512, intermediate_size=2048)),
        #     Transpose(),
        #
        # )


        self.avg = nn.AdaptiveAvgPool1d(1)
        # self.layer_norm = LayerNorm(512)

    def forward(self, x: torch.tensor):
        x = self.encoder(x)
        # print(x.shape)
        x = self.avg(x).squeeze()
        return x


class PositionEncoding(nn.Module):
    def __init__(self, params):
        super(PositionEncoding, self).__init__()
        self.params = params
        self.position_embeddings = nn.Embedding(100, 512)

        self.layer_norm = LayerNorm(512)
        self.dropout = nn.Dropout(self.params.dropout)

    def forward(self, x):
        bz = x.shape[0]
        seq_length = x.shape[1]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(bz, seq_length)
        # print(position_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embedding = x + position_embeddings * 0.1
        embedding = self.layer_norm(embedding)
        embedding = self.dropout(embedding)
        return embedding


class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.params = params
        self.epoch_encoder_eeg = EpochEncoder(self.params, 6)
        self.epoch_encoder_eog = EpochEncoder(self.params, 2)
        # self.position_encoding = PositionEncoding(self.params)
        # self.dropout = nn.Dropout(params.dropout)
        # self.modality_fusion = PointWiseAtt(params, 1024, 512)
        self.eog2eeg_encoder = CMTransformerEncoder(
            seq_length=20,
            num_layers=1,
            num_heads=8,
            hidden_dim=512,
            mlp_dim=512,
            dropout=self.params.dropout,
            attention_dropout=self.params.dropout,
        )
        self.eeg2eog_encoder = CMTransformerEncoder(
            seq_length=20,
            num_layers=1,
            num_heads=8,
            hidden_dim=512,
            mlp_dim=512,
            dropout=self.params.dropout,
            attention_dropout=self.params.dropout,
        )
        self.sequence_encoder = TransformerEncoder(
            seq_length=20,
            num_layers=1,
            num_heads=8,
            hidden_dim=512,
            mlp_dim=512,
            dropout=self.params.dropout,
            attention_dropout=self.params.dropout,
        )
        # self.layer_norm = LayerNorm(512, eps=1e-6)
        self.classifier = nn.Linear(512, self.params.num_of_classes)

    def forward(self, x):
        bz = x.shape[0]

        x_eeg = x[:, :, :6, :]
        x_eeg = x_eeg.view(bz * 20, 6, -1)
        x_eeg = self.epoch_encoder_eeg(x_eeg)
        # print(x_eeg.shape)
        # x_eeg = self.avg(x_eeg)
        # print(x_eeg.shape)
        x_eeg = x_eeg.view(bz, 20, -1)

        x_eog = x[:, :, 6:, :]
        x_eog = x_eog.view(bz * 20, 2, -1)
        x_eog = self.epoch_encoder_eog(x_eog)
        # x_eog = self.avg(x_eog)
        x_eog = x_eog.view(bz, 20, -1)
        x_eeg_ = self.eog2eeg_encoder(x_eeg, x_eog)
        x_eog_ = self.eeg2eog_encoder(x_eog, x_eeg)
        x = x_eeg_ + x_eog_
        #
        # x = self.layer_norm(x_eeg + x_eog)
        # x = torch.cat((x_eeg, x_eog), dim=1)
        # x = self.position_encoding(x)
        # x = self.dropout(x)
        # x = self.modality_fusion(x_eeg, x_eog)

        x = self.sequence_encoder(x)
        # x = x[:, :20, :] + x[:, 20:, :]

        return self.classifier(x)


class PointWiseAtt(nn.Module):
    def __init__(self, params, in_planes, out_planes):
        super(PointWiseAtt, self).__init__()
        self.fc1 = nn.Linear(in_planes, out_planes)
        self.dropout = nn.Dropout(params.dropout)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_planes, out_planes)
        self.sigmod = nn.Sigmoid()
        self.att = None
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.act(x)
        x = self.fc2(x)
        att = self.sigmod(x)
        self.att = att
        out = att * x1 + (1 - att) * x2
        return out


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
