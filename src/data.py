import torchvision  # type: ignore
import torch
import numpy as np
import copy

mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def load_dataset(dataset_name):
    transforms_train = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ]
    )

    transforms_test = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)]
    )

    if dataset_name == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            root="./", train=True, transform=transforms_train, download=True
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="./", train=False, transform=transforms_test, download=True
        )
    else:
        train_dataset = torchvision.datasets.CIFAR100(
            root="./", train=True, transform=transforms_train, download=True
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root="./", train=False, transform=transforms_test, download=True
        )

    return train_dataset, test_dataset


from torch.utils.data import WeightedRandomSampler


def get_loaders(train_dataset, test_dataset, batch_size):
    targets = torch.Tensor([y for _, y in train_dataset])
    num_classes = len(torch.unique(targets))

    # Compute class weights
    class_counts = {i: 0 for i in range(10)}
    print("num_classes", num_classes)
    print("targets", targets)
    for target in targets:
        class_counts[int(target)] += 1

    class_weights = {
        class_: 1.0 / count for class_, count in class_counts.items() if count > 0
    }

    print(class_weights)

    # Assign sample weights based on their respective class weights
    sample_weights = [class_weights[int(target)] for target in targets]

    # Create a WeightedRandomSampler instance using the sample weights
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(targets), replacement=True
    )

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


def task_construction(task_labels, dataset_name, order=None):
    train_dataset, test_dataset = load_dataset(dataset_name)

    train_dataset.targets = torch.tensor(train_dataset.targets)
    test_dataset.targets = torch.tensor(test_dataset.targets)

    if order is not None:
        train_targets = -1 * torch.tensor(
            np.ones(len(train_dataset.targets)), dtype=torch.long
        )
        test_targets = -1 * torch.tensor(
            np.ones(len(test_dataset.targets)), dtype=torch.long
        )
        for i, label in enumerate(order):
            train_targets[train_dataset.targets == label] = i
            test_targets[test_dataset.targets == label] = i

        train_dataset.targets = train_targets.clone()
        test_dataset.targets = test_targets.clone()

    train_dataset = split_dataset_by_labels(train_dataset, task_labels)

    test_task_labels = []
    for i in range(len(task_labels)):
        test_task_labels.append(task_labels[i][:])
        for j in range(i):
            test_task_labels[i].extend(task_labels[j])

    test_dataset = split_dataset_by_labels(test_dataset, test_task_labels)
    return train_dataset, test_dataset


def split_dataset_by_labels(dataset, task_labels):
    datasets = []
    for labels in task_labels:
        idx = np.in1d(dataset.targets, labels)
        splited_dataset = copy.deepcopy(dataset)
        splited_dataset.targets = torch.tensor(splited_dataset.targets)[idx]
        splited_dataset.data = splited_dataset.data[idx]
        datasets.append(splited_dataset)
    return datasets


def create_labels(num_classes, num_tasks, num_classes_first_task):
    classes_first_task = [list(range(0, num_classes_first_task))]
    classes_remained_task = [[i] for i in range(num_classes_first_task, num_classes)]
    labels = classes_first_task + classes_remained_task
    return labels


def create_class_task_map(num_classes_first_task, order):
    class_task_map = {}

    for i, class_label in enumerate(order):
        if i < num_classes_first_task:
            task_number = 0
        else:
            task_number = i - num_classes_first_task + 1

        class_task_map[class_label] = task_number

    return class_task_map
