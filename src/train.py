import torch
from torch import nn
from torch import optim
import torchvision  # type: ignore
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor  # type: ignore
from torchmetrics.classification import MulticlassAccuracy
import wandb
from tqdm import tqdm
from typing import Union

import argparse

from network import AllCNN

device: Union[int, str] = 0 if torch.cuda.is_available() else "cpu"


def model_pipeline(hyperparameters):
    print("hyperparameters: ", hyperparameters)

    # wandbを開始する設定(configには辞書型で渡すこと)
    with wandb.init(project="pytorch-demo", config=hyperparameters):

        # wandb.configを通して, wandbのHPとアクセス
        config = wandb.config

        # model, dataloader, criterion, optimizerを作成
        (
            model,
            train_loader,
            test_loader,
            criterion,
            metrics,
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
            metrics,
            optimizer,
            scheduler,
            config,
        )

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
    if config.model == "AllCNN":
        model = AllCNN()
    else:
        model = torchvision.models.resnet18()
        model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
    model = model.to(device)

    # torchinfo.summary(model, input_size=(config.batch_size, 3, 32, 32))

    # lossとoptimizerを設定
    criterion = nn.CrossEntropyLoss()
    metrics = MulticlassAccuracy(num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, config.lr_decay)

    return model, train_loader, test_loader, criterion, metrics, optimizer, scheduler


def run(
    model, train_loader, test_loader, criterion, metrics, optimizer, scheduler, config
):

    # 任意 : log_freqステップの学習ごとにパラメータの勾配とモデルのパラメータを記録
    wandb.watch(model, criterion, log="all", log_freq=10)

    for epoch in tqdm(range(config.epochs)):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config)
        test_loss, test_accuracy = test(model, test_loader, criterion, metrics)
        take_log(train_loss, test_loss, test_accuracy, epoch)
        scheduler.step()


def train_epoch(model, train_loader, criterion, optimizer, config, down_sampled=False):
    train_loss = 0
    model.train()
    for step, (images, labels) in enumerate(train_loader):
        if down_sampled:
            images_pil = to_pil_image(images)
            images_down_sampled = images_pil.resize(8, 8)
            images = to_tensor(images_down_sampled)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss / len(train_loader)
    return train_loss


def test(model, test_loader, criterion, metrics):
    test_loss = 0
    accuracy = 0
    num_data = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            accuracy += metrics(outputs, labels) * len(labels)
            num_data += len(labels)
        test_loss = test_loss / len(test_loader)
        test_accuracy = accuracy / num_data

    # ONNXフォーマットでモデルを保存する
    torch.onnx.export(model, images, "model.onnx")
    wandb.save("model.onnx")
    return test_loss, test_accuracy


def take_log(train_loss, test_loss, test_accuracy, epoch):
    wandb.log(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }
    )
    print(
        f"train_loss: {train_loss:.5f}, test_loss : {test_loss:.5f}, test_accuracy : {test_accuracy:.5f}"
    )


def main():
    parser = argparse.ArgumentParser("Experiments for Critical Learning Period")

    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.05)
    parser.add_argument("-e", "--epochs", type=int, default=140)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--lr_decay", type=float, default=0.97)
    parser.add_argument("--model", type=str, default="ResNet18")

    args = parser.parse_args()

    r = model_pipeline(
        {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay,
            "lr_decay": args.lr_decay,
            "model": args.model,
        }
    )


if __name__ == "__main__":
    print(torch.cuda.is_available())

    main()
