import torch

from common_utils.torch.layers.residual import HighwayLayer


def init_weights(modules):
    if isinstance(modules, torch.nn.Module):
        modules = modules.modules()

    for m in modules:
        if isinstance(m, torch.nn.Sequential):
            init_weights(m_inner for m_inner in m)

        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal(m.weight.data)
            m.bias.data.zero_()

        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_normal(m.weight.data)
            m.bias.data.zero_()

        if isinstance(m, torch.nn.Conv1d):
            torch.nn.init.xavier_normal(m.weight.data)
            m.bias.data.zero_()
