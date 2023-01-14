import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Union
import wandb


import torch
from torch import Tensor, nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import torchinfo

device: Union[int, str] = 0 if torch.cuda.is_available() else "cpu"


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Module:
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Module:
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, bias=False
    )


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


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.savefig("sample.png")


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


def model_pipeline(hyperparameters):

    # wandbを開始する設定(configには辞書型で渡すこと)
    with wandb.init(project="pytorch-demo", config=hyperparameters):

        # wandb.configを通して, wandbのHPとアクセス
        config = wandb.config

        # model, dataloader, criterion, optimizerを作成
        model, train_loader, test_loader, criterion, optimizer = make(config)
        print(model)

        # train用のUtile
        run(model, train_loader, test_loader, criterion, optimizer, config)

    return model


def make(config):

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    # Dataset, Dataloaderを作成
    train_dataset = torchvision.datasets.CIFAR10(
        root=".data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2
    )

    # Modelを作成
    model = ResNet(BasicBlock, config.blocks).to(device)

    # torchinfo.summary(model, input_size=(config.batch_size, 3, 32, 32))

    # lossとoptimizerを設定
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)

    return model, train_loader, test_loader, criterion, optimizer


def run(model, train_loader, test_loader, criterion, optimizer, config):

    # 任意 : log_freqステップの学習ごとにパラメータの勾配とモデルのパラメータを記録
    wandb.watch(model, criterion, log="all", log_freq=10)

    for epoch in tqdm(range(config.epochs)):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config)
        test_loss = inference(model, test_loader, criterion)
        take_log(train_loss, test_loss, epoch)


def train_epoch(model, train_loader, criterion, optimizer, config):
    train_loss = 0
    model.train()
    for step, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss / len(train_loader)
    return train_loss


def inference(model, test_loader, criterion):
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
        test_loss = test_loss / len(test_loader)

    # ONNXフォーマットでモデルを保存する
    torch.onnx.export(model, images, "model.onnx")
    wandb.save("model.onnx")
    return test_loss


def take_log(train_loss, test_loss, epoch):
    wandb.log({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss})
    print(f"train_loss: {train_loss:.5f}, test_loss : {test_loss:.5f}")


if __name__ == "__main__":
    # wandb.login()

    print(torch.cuda.is_available())

    r = model_pipeline(
        {"batch_size": 32, "learning_rate": 0.009, "blocks": [2, 2, 2, 2], "epochs": 10}
    )
