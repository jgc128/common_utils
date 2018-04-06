import torch


class Squeeze(torch.nn.Module):
    def __init__(self, dim=-1):
        super().__init__()

        self.dim = -1

    def forward(self, inputs):
        outputs = inputs.squeeze(dim=self.dim)
        return outputs
