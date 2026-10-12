"""Architectures trained in nirm3_week7-8_colab_training.ipynb, for loading their weights locally.

The Colab notebook keeps its own copy of these classes so it can run standalone; parameter names here
must stay identical to it, otherwise the saved state dicts will not load.
"""
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, kernels=(8, 5, 3)):
        super().__init__()
        layers, c = [], c_in
        for i, k in enumerate(kernels):
            layers += [nn.Conv1d(c, c_out, k, padding='same', bias=False), nn.BatchNorm1d(c_out)]
            if i < len(kernels) - 1:
                layers.append(nn.ReLU())
            c = c_out
        self.body = nn.Sequential(*layers)
        self.skip = nn.Sequential(nn.Conv1d(c_in, c_out, 1, bias=False), nn.BatchNorm1d(c_out)) \
            if c_in != c_out else nn.BatchNorm1d(c_out)

    def forward(self, x):
        return torch.relu(self.body(x) + self.skip(x))


class ResNet1d(nn.Module):
    def __init__(self, widths=(64, 128, 128), n_leads=12, n_classes=5):
        super().__init__()
        blocks, c = [], n_leads
        for w in widths:
            blocks.append(ResBlock(c, w))
            c = w
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(2 * c, n_classes))

    def forward(self, x):
        h = self.features(x)
        return self.head(torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1))


def conv_block(c_in, c_out, k=7, pool=True):
    layers = [nn.Conv1d(c_in, c_out, k, padding='same', bias=False), nn.BatchNorm1d(c_out), nn.ReLU(),
              nn.Conv1d(c_out, c_out, k, padding='same', bias=False), nn.BatchNorm1d(c_out), nn.ReLU()]
    return nn.Sequential(*layers, nn.MaxPool1d(2)) if pool else nn.Sequential(*layers)


class PlainCNN(nn.Module):
    def __init__(self, widths=(32, 64, 128, 128), n_leads=12, n_classes=5):
        super().__init__()
        blocks, c = [], n_leads
        for w in widths:
            blocks.append(conv_block(c, w))
            c = w
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(2 * c, n_classes))

    def forward(self, x):
        h = self.features(x)
        return self.head(torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1))


class AttentionPool(nn.Module):
    """Weighted mean over time; the last weights are kept in self.weights for attention maps."""

    def __init__(self, d):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))
        self.weights = None

    def forward(self, h):  # h: (batch, time, d)
        w = torch.softmax(self.score(h).squeeze(-1), dim=1)
        self.weights = w.detach()
        return (w.unsqueeze(-1) * h).sum(dim=1)


def conv_stem(width, n_leads=12):
    return nn.Sequential(conv_block(n_leads, width // 2), conv_block(width // 2, width))  # time / 4


class CNNLSTM(nn.Module):
    def __init__(self, width=128, hidden=128, n_classes=5):
        super().__init__()
        self.stem = conv_stem(width)
        self.lstm = nn.LSTM(width, hidden, batch_first=True, bidirectional=True)
        self.pool = AttentionPool(2 * hidden)
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(2 * hidden, n_classes))

    def forward(self, x):
        h, _ = self.lstm(self.stem(x).transpose(1, 2))
        return self.head(self.pool(h))


class CNNTransformer(nn.Module):
    def __init__(self, width=128, n_layers=3, n_heads=4, max_tokens=256, n_classes=5):
        super().__init__()
        self.stem = conv_stem(width)
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, width))
        layer = nn.TransformerEncoderLayer(width, n_heads, dim_feedforward=2 * width, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.pool = AttentionPool(width)
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(width, n_classes))

    def forward(self, x):
        h = self.stem(x).transpose(1, 2)
        h = self.encoder(h + self.pos[:, :h.shape[1]])
        return self.head(self.pool(h))


MODELS = {'resnet1d': ResNet1d, 'cnn': PlainCNN, 'cnn_lstm': CNNLSTM, 'cnn_transformer': CNNTransformer}


def load_model(info, weights_path):
    """Rebuild a model from a results/*.json record and load its weights on CPU in eval mode."""
    model = MODELS[info['config']['model']](info['config']['widths'])
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    return model.eval()
