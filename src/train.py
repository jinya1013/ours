import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision  # type: ignore
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor  # type: ignore
import wandb
from tqdm import tqdm
from typing import Union, Dict, Tuple, Any

from PIL import Image

import argparse
from pprint import pprint

from network import AllCNN, ResNet18
from utils import imshow
from dataset import Cifar10

device: Union[int, str] = 0 if torch.cuda.is_available() else "cpu"


def model_pipeline(hyperparameters):
    print("hyperparameters: ", hyperparameters)

    # wandbを開始する設定(configには辞書型で渡すこと)
    with wandb.init(project="critical", config=hyperparameters):

        # wandb.configを通して, wandbのHPとアクセス
        config = wandb.config

        # model, dataloader, criterion, optimizerを作成
        (
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            scheduler,
        ) = make(config)
        print(model)

        # train用のUtile
        run(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            scheduler,
            config,
        )

    return model


def make(
    config,
) -> Tuple[
    nn.Module,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    nn.CrossEntropyLoss,
    optim.SGD,
    optim.lr_scheduler.ExponentialLR,
]:

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_train_down_sample = transforms.Compose(
        [transforms.Resize((8, 8)), transforms.Resize((32, 32)), transform_train]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Dataset, Dataloaderを作成
    if config.aug:
        train_dataset = Cifar10(
            root=".data",
            train=True,
            download=True,
            transform=[transform_train, transform_train_down_sample],
        )
    else:
        train_dataset = Cifar10(
            root=".data",
            train=True,
            download=True,
            transform=[transform_train, transform_train_down_sample],
        )

    test_dataset = Cifar10(
        root="./data", train=False, download=True, transform=transform_test
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2
    )

    # Modelを作成
    if config.model == "AllCNN":
        model = AllCNN()
    else:
        model = ResNet18()
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
        train_loss, train_accuracy = train_epoch(
            model, train_loader, criterion, optimizer, config
        )
        test_loss, test_accuracy = test(model, test_loader, criterion)
        take_log(
            {
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "epoch": epoch,
            }
        )
        scheduler.step()


def train_epoch(model, train_loader, criterion, optimizer, config, down_sampled=False):
    train_loss = 0
    train_accuracy = 0
    num_data = 0
    model.train()
    for images, images_sub, labels in train_loader:
        if down_sampled:
            images, labels = images_sub.to(device), labels.to(device)
        else:
            images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        _, preds = torch.max(outputs, dim=1)
        loss = criterion(outputs, labels)
        train_accuracy += torch.sum(preds == labels).item()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        num_data += len(labels)
    train_loss = train_loss / len(train_loader)
    train_accuracy = train_accuracy / num_data
    return train_loss, train_accuracy


def test(model, test_loader, criterion):
    test_loss = 0
    accuracy = 0
    num_data = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            accuracy += torch.sum(preds == labels).item()
            num_data += len(labels)
        test_loss = test_loss / len(test_loader)
        test_accuracy = accuracy / num_data

    # ONNXフォーマットでモデルを保存する
    torch.onnx.export(model, images, "model.onnx")
    wandb.save("model.onnx")
    return test_loss, test_accuracy


def take_log(info_dict):
    wandb.log(info_dict)
    pprint(info_dict)


def main():
    parser = argparse.ArgumentParser("Experiments for Critical Learning Period")

    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.05)
    parser.add_argument("-e", "--epochs", type=int, default=140)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--lr_decay", type=float, default=0.97)
    parser.add_argument("-m", "--model", type=str, default="ResNet18")
    parser.add_argument("--aug", action="store_true")

    args = parser.parse_args()

    r = model_pipeline(
        {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay,
            "lr_decay": args.lr_decay,
            "model": args.model,
            "aug": args.aug,
        }
    )


if __name__ == "__main__":
    print(torch.cuda.is_available())

    main()
