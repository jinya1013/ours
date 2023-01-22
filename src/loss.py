import torch
import torch.nn.functional as F
from torch import nn


class BinaryCrossEntropyEasy(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyEasy, self).__init__()

    def forward(self, outputs, targets):

        probs = torch.sigmoid(
            outputs[:, 0] + outputs[:, 1] + outputs[:, 8] + outputs[:, 9]
        )

        loss = F.binary_cross_entropy(probs, targets)
        return loss


if __name__ == "__main__":
    x = torch.ones(1000, 10)
    y = torch.zeros(1000)
    metrics = BinaryCrossEntropyEasy()
    metrics(x, y)
