import torch
import torch.nn.functional as F


class ResidualLayer(torch.nn.Module):
    def __init__(self, nb_features, hidden_size=None, dropout=0):
        super().__init__()
        self.nb_features = nb_features

        if hidden_size is not None:
            self.hidden_size = hidden_size
        else:
            self.hidden_size = nb_features

        self.dropout = None
        if dropout != 0:
            self.dropout = torch.nn.Dropout(dropout)

        self.linear1 = torch.nn.Linear(self.nb_features, self.hidden_size)
        self.bn = torch.nn.BatchNorm1d(self.nb_features)

        # self.linear2 = torch.nn.Linear(self.hidden_size, self.nb_features)

    def _init_weights(self):
        torch.nn.init.normal(self.linear1.weight.data, std=1e-6)
        self.linear1.bias.data.zero_()

        # torch.nn.init.normal(self.linear2.weight.data, std=1e-6)
        # self.linear2.bias.data.zero_()

    def forward(self, inputs):
        transformed = self.linear1(inputs)
        transformed = self.bn(transformed)
        transformed = F.elu(transformed)

        if self.dropout is not None:
            transformed = self.dropout(transformed)

        # transformed = self.linear2(transformed)

        outputs = inputs + transformed
        return outputs


class HighwayLayer(torch.nn.Module):
    def __init__(self, nb_features):
        super().__init__()

        self.T = torch.nn.Linear(nb_features, nb_features)
        self.H = torch.nn.Linear(nb_features, nb_features)

        self._init_weights()

    def _init_weights(self):
        # torch.nn.init.xavier_normal(self.H.weight.data)
        self.H.bias.data.zero_()
        torch.nn.init.normal(self.H.weight.data, std=0.001)

        # torch.nn.init.xavier_normal(self.T.weight.data)
        self.T.bias.data.zero_()
        torch.nn.init.normal(self.T.weight.data, std=0.001)

    def forward(self, inputs):
        h = F.elu(self.H(inputs))
        t = F.sigmoid(self.T(inputs))

        outputs = h * t + (1 - t) * inputs

        return outputs
