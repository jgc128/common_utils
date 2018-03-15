import torch


class SegmentationLoss(torch.nn.Module):
    def __init__(self):
        super(SegmentationLoss, self).__init__()

        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, outputs, masks):
        if outputs.size(1) == 1:
            outputs = outputs.squeeze(1)

        loss_bce = self.bce_loss(outputs, masks)

        return loss_bce


class SequenceReconstructionLoss(torch.nn.Module):
    def __init__(self, ignore_index=-100):
        super(SequenceReconstructionLoss, self).__init__()

        self.xent_loss = torch.nn.CrossEntropyLoss(size_average=True, ignore_index=ignore_index)

    def _calc_sent_xent(self, outputs, targets):
        if len(outputs.shape) > 2:
            targets = targets.view(-1)
            outputs = outputs.view(targets.size(0), -1)

        xent = self.xent_loss(outputs, targets)

        return xent

    def forward(self, outputs, targets):
        loss = self._calc_sent_xent(outputs, targets)

        return loss


class BinaryCrossEntropyWithLogits(torch.nn.BCEWithLogitsLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs, targets):
        targets = targets.float()
        return super().forward(inputs, targets)
