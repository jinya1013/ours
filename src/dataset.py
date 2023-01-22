import torch
import torchvision  # type: ignore
from torchvision import transforms
import numpy as np
from utils import imshow


class CIFAR10(torch.utils.data.Dataset):
    def __init__(
        self,
        root="./data",
        transform=None,
        train=True,
        download=True,
        deficit="subsample",
    ):
        self.dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=None
        )
        self.transform = transform
        self.train = train
        self.deficit = deficit

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        if self.train:
            if self.deficit == "subsample":
                img_org = self.transform["original"](img)
                img_sub = self.transform["subsample"](img)
                return img_org, img_sub, label
            elif self.deficit == "noise":
                img_org = self.transform["original"](img)
                img_noise = torch.randn_like(img_org)
                return img_org, img_noise, label
        else:
            img = self.transform(img)
            return img, label


class DogAircraftCIFAR10(torch.utils.data.Dataset):
    def __init__(
        self,
        root="./data",
        train=True,
        download=True,
        transform=None,
        deficit="subsample",
    ):
        self.transform = transform
        trainset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=None
        )
        self.subset = torch.utils.data.Subset(
            trainset, [i for i in range(len(trainset)) if trainset[i][1] in [0, 5]]
        )
        self.targets = [
            np.float32(0.0) if self.subset[i][1] == 0 else np.float32(1.0)
            for i in range(len(self.subset))
        ]
        self.train = train

    def __getitem__(self, index):
        img, _ = self.subset[index]
        target = self.targets[index]
        if self.train:
            if self.deficit == "subsample":
                img_org = self.transform["original"](img)
                img_sub = self.transform["subsample"](img)
                return img_org, img_sub, target
            elif self.deficit == "noise":
                img_org = self.transform["original"](img)
                img_noise = torch.randn_like(img_org)
                return img_org, img_noise, target
        else:
            img = self.transform(img)
            return img, target

    def __len__(self):
        return len(self.subset)


class DogDeerCIFAR10(torch.utils.data.Dataset):
    def __init__(
        self,
        root="./data",
        train=True,
        download=True,
        transform=None,
        deficit="subsample",
    ):
        self.transform = transform
        trainset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=None
        )
        self.subset = torch.utils.data.Subset(
            trainset, [i for i in range(len(trainset)) if trainset[i][1] in [4, 5]]
        )
        self.targets = [
            np.float32(0.0) if self.subset[i][1] == 4 else np.float32(1.0)
            for i in range(len(self.subset))
        ]
        self.train = train

    def __getitem__(self, index):
        img, _ = self.subset[index]
        target = self.targets[index]
        if self.train:
            if self.deficit == "subsample":
                img_org = self.transform["original"](img)
                img_sub = self.transform["subsample"](img)
                return img_org, img_sub, target
            elif self.deficit == "noise":
                img_org = self.transform["original"](img)
                img_noise = torch.randn_like(img_org)
                return img_org, img_noise, target
        else:
            img = self.transform(img)
            return img, target

    def __len__(self):
        return len(self.subset)


if __name__ == "__main__":
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
    d = CIFAR10(".data", transform=transform_train, deficit="noise")
    imshow(d[0][0], "img_org.png")
    imshow(d[0][1], "img_noise.png")
