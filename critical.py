import numpy as np

from PIL import Image

import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Union
import wandb

import torch
from torch import Tensor, nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

PROJECT_NAME = "critical"
device: Union[int, str] = 0 if torch.cuda.is_available() else "cpu"


def imshow(img: Image, name: str = "sample.png") -> None:
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.savefig(name)


def model_pipeline(hyperparameters):

    # wandbを開始する設定(configには辞書型で渡すこと)
    with wandb.init(project=PROJECT_NAME, config=hyperparameters):

        # wandb.configを通して, wandbのHPとアクセス
        config = wandb.config

        # model, dataloader, criterion, optimizerを作成
        model, train_loader, test_loader, criterion, optimizer, scheduler = make(config)
        print(model)

        # train用のUtile
        run(model, train_loader, test_loader, criterion, optimizer, scheduler, config)

    return model


def make(config):

    transform = transforms.ToTensor()
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
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
    model = model.to(device)

    # torchinfo.summary(model, input_size=(config.batch_size, 3, 32, 32))

    # lossとoptimizerを設定
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, config.lr_decay)
    return model, train_loader, test_loader, criterion, optimizer, scheduler


def run(model, train_loader, test_loader, criterion, optimizer, scheduler, config):

    # 任意 : log_freqステップの学習ごとにパラメータの勾配とモデルのパラメータを記録
    wandb.watch(model, criterion, log="all", log_freq=10)

    for epoch in tqdm(range(config.epochs)):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config)
        test_loss = test(model, test_loader, criterion)
        take_log(train_loss, test_loss, epoch)
        scheduler.step()


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


def test(model, test_loader, criterion):
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
        {
            "batch_size": 32,
            "learning_rate": 0.009,
            "epochs": 10,
            "weight_decay": 0.0005,
            "lr_decay": 0.97,
        }
    )
