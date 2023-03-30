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

import argparse
from pprint import pprint

from models import AllCNN, ResNet18
from metrics import imshow, compute_fisher
from data import CIFAR10, DogAircraftCIFAR10, DogDeerCIFAR10

from sam.sam import SAM

device: Union[int, str] = 0 if torch.cuda.is_available() else "cpu"


def model_pipeline(hyperparameters):
    print("hyperparameters: ", hyperparameters)

    # wandbを開始する設定(configには辞書型で渡すこと)
    with wandb.init(
        project=hyperparameters["project"],
        name=f"critical_period:{hyperparameters['deficit_start']}_{hyperparameters['deficit_end']}",
        config=hyperparameters,
    ):

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
):
    transform_train = {
        "original": transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
        "subsample": transforms.Compose(
            [
                transforms.Resize((8, 8)),
                transforms.Resize((32, 32)),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    }

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Dataset, Dataloaderを作成
    if config.num_classes == 2:
        train_dataset = DogAircraftCIFAR10(
            root=".data",
            train=True,
            download=True,
            transform=transform_train,
        )
        test_dataset = DogAircraftCIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
    else:
        train_dataset = CIFAR10(
            root=".data",
            train=True,
            download=True,
            transform=transform_train,
            deficit=config.deficit,
        )
        test_dataset = CIFAR10(
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
        model = AllCNN(num_classes=config.num_classes)
    else:
        model = ResNet18(num_classes=config.num_classes)
    model = model.to(device)

    # torchinfo.summary(model, input_size=(config.batch_size, 3, 32, 32))

    # lossとoptimizerを設定
    criterion = nn.BCELoss() if config.num_classes == 2 else nn.CrossEntropyLoss()
    if config.optimizer == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    elif config.optimizer == "SAM":
        base_optimizer = optim.SGD
        optimizer = SAM(
            model.parameters(),
            base_optimizer,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, config.lr_decay)

    return model, train_loader, test_loader, criterion, optimizer, scheduler


def run(model, train_loader, test_loader, criterion, optimizer, scheduler, config):

    # 任意 : log_freqステップの学習ごとにパラメータの勾配とモデルのパラメータを記録
    wandb.watch(model, criterion, log="all", log_freq=10)

    train(model, train_loader, test_loader, criterion, optimizer, scheduler, config)


def train(model, train_loader, test_loader, criterion, optimizer, scheduler, config):
    for epoch in tqdm(range(0, config.deficit_start)):
        (
            train_loss,
            train_accuracy,
        ) = train_epoch(model, train_loader, criterion, optimizer, down_sampled=False)
        test_loss, test_accuracy = test(model, test_loader, criterion)
        take_log(
            {
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "test_accuracy_decrerase": 0.9161 - test_accuracy,
                "epoch": epoch,
            }
        )
        scheduler.step()
    for epoch in tqdm(range(config.deficit_start, config.deficit_end)):
        (
            train_loss,
            train_accuracy,
        ) = train_epoch(model, train_loader, criterion, optimizer, down_sampled=True)
        test_loss, test_accuracy = test(model, test_loader, criterion)
        take_log(
            {
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                # "test_accuracy_decrerase": 0.9161 - test_accuracy,
                "epoch": epoch,
            }
        )
        scheduler.step()
    for epoch in tqdm(range(config.deficit_end, config.epochs)):
        (
            train_loss,
            train_accuracy,
        ) = train_epoch(model, train_loader, criterion, optimizer, down_sampled=False)
        test_loss, test_accuracy = test(model, test_loader, criterion)
        take_log(
            {
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                # "test_accuracy_decrerase": 0.9161 - test_accuracy,
                "epoch": epoch,
            }
        )
        scheduler.step()


def train_epoch(model, train_loader, criterion, optimizer, down_sampled=False):
    train_loss = 0
    train_accuracy = 0
    num_data = 0
    model.train()
    for images, images_sub, labels in train_loader:
        images, images_sub, labels = (
            images.to(device),
            images_sub.to(device),
            labels.to(device),
        )
        optimizer.zero_grad()
        if down_sampled:
            outputs = model(images_sub)
        else:
            outputs = model(images)
        if outputs.ndim > 1:
            _, preds = torch.max(outputs, dim=1)
        else:
            preds = torch.where(outputs > 0.5, 1.0, 0.0)
        train_accuracy += torch.sum(preds == labels).item()
        loss = criterion(outputs, labels)
        loss.backward()

        if optimizer.__class__.__name__ == "SAM":
            # first forward-backward pass
            optimizer.first_step(zero_grad=True)

            # second forward-backward pass
            if down_sampled:
                criterion(
                    model(images_sub), labels
                ).backward()  # make sure to do a full forward pass
            else:
                criterion(
                    model(images), labels
                ).backward()  # make sure to do a full forward pass
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.step()

        train_loss += loss.item()
        num_data += len(labels)
    train_loss = train_loss / len(train_loader)
    train_accuracy = train_accuracy / num_data
    return (
        train_loss,
        train_accuracy,
    )


def test(model, test_loader, criterion):
    test_loss = 0
    accuracy = 0
    num_data = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if outputs.ndim > 1:
                _, preds = torch.max(outputs, dim=1)
            else:
                preds = torch.where(outputs > 0.5, 1.0, 0.0)
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
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.05)
    parser.add_argument("-e", "--epochs", type=int, default=160)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--lr_decay", type=float, default=0.97)
    parser.add_argument("-m", "--model", type=str, default="ResNet18")
    parser.add_argument("--deficit_start", type=int, default=0)
    parser.add_argument("--deficit_end", type=int, default=0)
    parser.add_argument("--deficit", type=str, default="subsample")
    parser.add_argument("--aug", action="store_true")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--project", type=str, default="critical")

    args = parser.parse_args()

    r = model_pipeline(
        {
            "batch_size": args.batch_size,
            "optimizer": args.optimizer,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay,
            "lr_decay": args.lr_decay,
            "model": args.model,
            "aug": args.aug,
            "deficit_start": args.deficit_start,
            "deficit_end": args.deficit_end,
            "deficit": args.deficit,
            "num_classes": args.num_classes,
            "project": args.project,
        }
    )


if __name__ == "__main__":
    print(torch.cuda.is_available())
    main()
