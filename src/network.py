import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union, List
import torchvision #type: ignore


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
        stride: int = 1,
    ) -> None:
        super().__init__()

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
            conv1x1(channels2, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = torch.mean(out, dim=[2, 3])  # global average pooling
        out = F.softmax(out, dim=1)
        return out


class BasicBlock(nn.Module):
    expansion: int = 1  # 出力のチャネル数を入力のチャネル数の何倍に拡大するか

    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1: nn.Module = conv3x3(in_channels, channels, stride)
        self.bn1: nn.Module = nn.BatchNorm2d(channels)
        self.relu: nn.Module = nn.ReLU(inplace=True)
        self.conv2: nn.Module = conv3x3(channels, channels)
        self.bn2: nn.Module = nn.BatchNorm2d(channels)

        # 入力と出力のチャネル数が異なる場合, x をダウンサンプリングする
        if in_channels != channels * self.expansion:
            self.shortcut: nn.Module = nn.Sequential(
                conv1x1(in_channels, channels * self.expansion, stride),
                nn.BatchNorm2d(channels * self.expansion),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)

        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    expansion: int = 4

    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        print(in_channels, channels)
        super().__init__()
        self.conv1: nn.Module = conv1x1(in_channels, channels)
        self.bn1: nn.Module = nn.BatchNorm2d(channels)
        self.conv2: nn.Module = conv3x3(channels, channels, stride)
        self.bn2: nn.Module = nn.BatchNorm2d(channels)
        self.conv3: nn.Module = conv1x1(in_channels, channels * self.expansion)
        self.bn3: nn.Module = nn.BatchNorm2d(channels * self.expansion)
        self.relu: nn.Module = nn.ReLU(inplace=True)

        # 入力と出力のチャネル数が異なる場合, x をダウンサンプリングする
        if in_channels != channels * self.expansion:
            self.shortcut: nn.Module = nn.Sequential(
                conv1x1(in_channels, channels * self.expansion, stride),
                nn.BatchNorm2d(channels * self.expansion),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(x)

        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Union[BasicBlock, BottleneckBlock],
        layers: List[int],
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        self.in_channels: int = 64
        self.conv1: nn.Module = nn.Conv2d(
            3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1: nn.Module = nn.BatchNorm2d(self.in_channels)
        self.relu: nn.Module = nn.ReLU(inplace=True)
        self.maxpool: nn.Module = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1: nn.Module = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2: nn.Module = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3: nn.Module = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4: nn.Module = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool: nn.Module = nn.AdaptiveAvgPool2d((1, 1))
        self.fc: nn.Module = nn.Linear(512 * block.expansion, num_classes)

        # 重みの初期化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: Union[BasicBlock, BottleneckBlock],
        channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Module:
        layers: List[nn.Module] = []

        # 最初の Residual Block
        layers.append(block(self.in_channels, channels, stride))
        self.in_channels = channels * block.expansion
        # 残りの Residual Block
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


if __name__ == "__main__":
    # model = AllCNN(3)
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
    N = 20
    W = 32
    H = 32
    C = 3
    x: Tensor = torch.ones((N, C, W, H))
    y = model(x)
    print(model)
