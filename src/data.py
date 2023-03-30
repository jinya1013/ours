import torchvision

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


def get_loaders(train_dataset, test_dataset, batch_size):
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
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
    test_dataset = split_dataset_by_labels(test_dataset, task_labels)
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


from torch.utils.data import Dataset, DataLoader, Subset


def divide_dataset(dataset, num_classes_first_task, order=None):
    # Get the total number of classes in the dataset
    num_classes_total = len(set(dataset.targets))

    # Ensure the number of classes for the first task is valid
    if num_classes_first_task < 2 or num_classes_first_task >= num_classes_total:
        raise ValueError(
            "num_classes_first_task must be greater than 1 and less than the total number of classes"
        )

    # Find the indices for each class
    class_indices = {label: [] for label in set(dataset.targets)}
    for idx, label in enumerate(dataset.targets):
        class_indices[label].append(idx)

    # Divide the dataset into tasks
    divided_dataset = []
    if order is not None:
        first_task_classes = order[:num_classes_first_task]
        remaining_classes = order[num_classes_first_task:num_classes_total]
    else:
        first_task_classes = list(range(num_classes_first_task))
        remaining_classes = list(range(num_classes_first_task, num_classes_total))

    # Create the first task dataset
    first_task_indices = []
    for label in first_task_classes:
        first_task_indices.extend(class_indices[label])
    divided_dataset.append(Subset(dataset, first_task_indices))

    # Create the remaining tasks datasets
    for label in remaining_classes:
        divided_dataset.append(Subset(dataset, class_indices[label]))

    return divided_dataset


def create_labels(num_classes, num_tasks, num_classes_per_task):
    """
    Creates a label matrix for a multi-task classification problem.

    Args:
        num_classes (int): The total number of classes.
        num_tasks (int): The number of tasks.
        num_classes_per_task (int): The number of classes per task.

    Returns:
        np.ndarray: A label matrix of shape (num_tasks, num_classes_per_task),
                    where each row represents the class labels for a single task.
    """
    # Initialize an array with all class indices
    tasks_order = np.arange(num_classes)

    # Reshape the array to create num_tasks sets of num_classes_per_task class indices
    labels = tasks_order.reshape((num_tasks, num_classes_per_task))

    return labels


def set_task(model, task_id):
    model.task_id = task_id
    for layer in range(len(model.num_blocks)):
        for block in range(model.num_blocks[layer]):
            Block = list(model.children())[layer + 2][block]
            Block.task_id = task_id


if __name__ == "__main__":
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor

    # Load CIFAR-10 dataset
    cifar10_dataset = CIFAR10(
        root="./data", train=True, download=True, transform=ToTensor()
    )

    # Divide the dataset into tasks
    num_classes_first_task = 3
    divided_dataset = divide_dataset(cifar10_dataset, num_classes_first_task)

    # Check the output
    for i, task_dataset in enumerate(divided_dataset):
        print(f"Task {i + 1}: {len(task_dataset)} samples")
