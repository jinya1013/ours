import torch
import torchvision  # type: ignore


class Cifar10(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, train=True, download=True):
        self.dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=None
        )
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        if self.train:
            img_org = self.transform[0](img)
            img_sub = self.transform[1](img)

            return img_org, img_sub, label
        else:
            img = self.transform(img)
            return img, label
