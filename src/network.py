import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union, List
import torchvision  # type: ignore


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Module:
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Module:
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, bias=False
    )


class AllCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        channels1: int = 96,
        channels2: int = 192,
        num_classes: int = 10,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.num_classes = 1 if num_classes == 2 else num_classes

        self.conv1: nn.Module = nn.Sequential(
            conv3x3(in_channels, channels1),
            nn.BatchNorm2d(channels1),
            nn.ReLU(inplace=True),
            conv3x3(channels1, channels1),
            nn.BatchNorm2d(channels1),
            nn.ReLU(inplace=True),
            conv3x3(channels1, channels2, stride=2),
            nn.BatchNorm2d(channels2),
            nn.ReLU(inplace=True),
        )

        self.conv2: nn.Module = nn.Sequential(
            conv3x3(channels2, channels2),
            nn.BatchNorm2d(channels2),
            nn.ReLU(inplace=True),
            conv3x3(channels2, channels2),
            nn.BatchNorm2d(channels2),
            nn.ReLU(inplace=True),
            conv3x3(channels2, channels2, stride=2),
            nn.BatchNorm2d(channels2),
            nn.ReLU(inplace=True),
        )

        self.conv3: nn.Module = nn.Sequential(
            conv3x3(channels2, channels2),
            nn.BatchNorm2d(channels2),
            nn.ReLU(inplace=True),
            conv1x1(channels2, channels2),
            nn.BatchNorm2d(channels2),
            nn.ReLU(inplace=True),
            conv1x1(channels2, self.num_classes),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.squeeze()
        if self.num_classes == 1:
            return torch.sigmoid(out)
        return out


"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_classes = 1 if num_classes == 2 else num_classes

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, self.num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if self.num_classes == 1:
            return torch.sigmoid(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


if __name__ == "__main__":
    model = AllCNN(3)
    # model = ResNet18()
    N = 20
    W = 32
    H = 32
    C = 3
    x: Tensor = torch.ones((N, C, W, H))
    y = model(x)
    print(model, y.shape)
    assert y.shape == (N, 10)
