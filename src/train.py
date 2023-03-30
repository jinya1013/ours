import numpy as np
import copy
import random
import datetime
import os
import argparse
from torch.utils.data import Dataset, Subset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics.pairwise import euclidean_distances

from pruning import iterative_pruning
from models import init_model
from data import set_task
from experiment import rewrite_parameters


def accuracy(model, data_loader, prototype_vectors, task_class_map, device):
    """
    Computes the accuracy of the model on the provided data_loader.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        data_loader (DataLoader): The DataLoader providing the data to evaluate the model on.
        device (str or torch.device): The device to perform the evaluation on.

    Returns:
        float: The accuracy of the model on the provided data_loader.
    """
    correct_preds = 0
    n = 0

    for X, y_true in data_loader:
        X = X.to(device)
        y_true = y_true.to(device)
        y_preds = predict(model, X, prototype_vectors, task_class_map)
        n += y_true.size(0)
        correct_preds += (y_preds == y_true).float().sum()

    return (correct_preds / n).item()


def create_task_class_map(num_classes, num_classes_first_task, order=None):
    if order is None:
        order = list(range(num_classes))

    if len(order) != num_classes:
        raise ValueError("The length of 'order' should be equal to 'num_classes'")

    task_class_map = {}
    task_class_map[0] = order[:num_classes_first_task]

    for t in range(1, num_classes - num_classes_first_task + 1):
        task_class_map[t] = [order[num_classes_first_task + t - 1]]

    return task_class_map


# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", type=str, default="cifar100", help="dataset to use"
    )
    parser.add_argument(
        "--path_data", type=str, default="./", help="path to save/load dataset"
    )
    parser.add_argument(
        "--download_data", type=bool, default=True, help="download dataset"
    )
    parser.add_argument(
        "--model_name", type=str, default="resnet18", help="network architecture to use"
    )
    parser.add_argument(
        "--path_pretrained_model",
        type=str,
        default="pretrained_model.pth",
        help="path to pretrained parameters",
    )
    parser.add_argument(
        "--path_init_params",
        type=str,
        default="init_params.pth",
        help="path to initialization parameters",
    )
    parser.add_argument(
        "--alpha_conv",
        type=float,
        default=0.9,
        help="fraction of importance to keep in conv layers",
    )
    parser.add_argument("--num_tasks", type=int, default=10, help="number of tasks")
    parser.add_argument(
        "--num_classes", type=int, default=100, help="number of classes"
    )
    parser.add_argument(
        "--num_classes_per_task",
        type=int,
        default=10,
        help="number of classes per task",
    )
    parser.add_argument(
        "--num_iters", type=int, default=3, help="number of pruning iterations"
    )  # 3
    parser.add_argument(
        "--prune_batch_size",
        type=int,
        default=1000,
        help="number of examples for pruning",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="number of examples per training batch",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=20,
        help="number of examples per test batch",
    )
    parser.add_argument(
        "--train_epochs", type=int, default=70, help="number training epochs"
    )  # 70
    parser.add_argument(
        "--retrain_epochs",
        type=int,
        default=50,
        help="number of retraining epochs after pruning",
    )  # 30
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer")
    parser.add_argument(
        "--lr_decay_type",
        type=str,
        default="multistep",
        help="learning rate decay type",
    )
    parser.add_argument("--lr", type=float, default=1e-2, help="initial learning rate")
    parser.add_argument(
        "--decay_epochs_train",
        nargs="+",
        type=int,
        default=[20, 40, 60],
        help="epochs for multistep decay",
    )  # [20, 40, 60]
    parser.add_argument(
        "--decay_epochs_retrain",
        nargs="+",
        type=int,
        default=[15, 25, 40],
        help="epochs for multistep decay",
    )  # [15, 25, 40]
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.2,
        help="multiplicative factor of learning rate decay",
    )  # 0.1
    parser.add_argument(
        "--wd", type=float, default=5e-4, help="weight decay during retraining"
    )
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument(
        "--order_name",
        type=str,
        default="default",
        help="name of class ordering. Options: defult, seed1993, seed1605",
    )
    parser.add_argument(
        "--task_select_method", type=str, default="max", help="task selection method"
    )
    parser.add_argument(
        "--train",
        type=bool,
        default=True,
        help="train the model; if False inference mode only",
    )
    parser.add_argument(
        "--num_class",
        type=int,
        default=10,
        help="the number of classes",
    )
    parser.add_argument(
        "--num_class_first_task",
        type=int,
        default=5,
        help="the number of classes for the first task; should be larger than 1",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path_results = (
        f"./results/{args.dataset_name}-{args.model_name}/{args.num_tasks}_tasks/"
    )
    if not os.path.isdir(path_results):
        os.makedirs(path_results)

    print("STARTED")
    print(args)

    # Load dataset and apply necessary transformations
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Define the base model
    model = init_model(args, device)

    num_tasks = args.num_tasks

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.decay_epochs_train, gamma=args.gamma
    )

    order_name = args.order_name
    # Task class mapping
    task_class_map = create_task_class_map(
        args.num_classes, args.num_classes_first_task
    )

    prototype_vectors = {}

    # loop for tasks
    for task_id, task_classes in task_class_map.items():
        path_to_save = (
            path_results
            + "{}_task{}_{}classes_{}_{}_it{}_order_{}.pth".format(
                args.model_name,
                task_id + 1,
                args.num_classes_per_task,
                args.optimizer,
                args.alpha_conv,
                args.num_iters,
                order_name,
            )
        )
        set_task(net, task_id)

        # Construct dataset for each task
        train_dataset = construct_dataset("train_dataset", task_classes, transform)
        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, num_workers=1
        )
        valid_dataset = construct_dataset("valid_dataset", task_classes, transform)
        valid_loader = DataLoader(
            valid_dataset, batch_size=32, shuffle=False, num_workers=1
        )

        net.set_trainable_masks(task_id)

        # Train the model on the current task
        net = train_task(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            shceduler=scheduler,
            train_loader=train_loader,
            valid_loader=valid_loader,
            prototype_vectors=prototype_vectors,
            task_class_map=task_class_map,
            epochs=args.train_epochs,
            task_id=task_id,
            device=device,
            file_name="model.pth",
            print_every=1,
        )

        # # Prune the network
        # random.seed(args.seed)
        # np.random.seed(args.seed)
        # prune_idx = np.random.permutation(train_dataset[task_id].data.shape[0])[
        #     : args.prune_batch_size
        # ]

        # x_prune = torch.FloatTensor(train_dataset[task_id].data[prune_idx]).to(device)
        # x_prune = x_prune.permute(0, 3, 1, 2)
        # x_prune = torchvision.transforms.Normalize(mean, std)(x_prune.float() / 255)

        # net = iterative_pruning(
        #     args=args,
        #     net=net,
        #     train_loader=train_loader,
        #     test_loader=test_loader,
        #     x_prune=x_prune,
        #     task_id=task_id,
        #     device=device,
        #     path_to_save=path_to_save,
        # )

        # Compute and store the prototype vector for the current task
        prototype_vector = compute_prototype_vector(model, train_loader, device)
        prototype_vectors.update(prototype_vector)

        # # 精度を計算
        # acc = accuracy(net, test_loader, device)
        # print("Accuracy: ", np.round(100 * acc, 2))

    # # Inference
    # input_data = torch.randn(1, 3, 224, 224).to(device)  # Dummy input
    # prediction = predict(model, input_data, prototype_vectors, masks)
    # print("Prediction:", prediction)


if __name__ == "__main__":
    main()


# Custom Dataset class for task construction
class TaskDataset(Dataset):
    def __init__(self, dataset, task_classes, transform=None):
        self.dataset = dataset
        self.task_classes = task_classes
        self.transform = transform

        # Filter the dataset for the current task's classes
        self.indices = []
        for idx in range(len(self.dataset)):
            target = self.dataset[idx][1]
            if target in self.task_classes:
                self.indices.append(idx)

    def __getitem__(self, index):
        image, label = self.dataset[self.indices[index]]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.indices)


# Dataset construction for each task
def construct_dataset(dataset_name, task_classes, transform):
    if dataset_name == "CIFAR10":
        dataset = datasets.CIFAR10(root="./data", train=True, download=True)
    elif dataset_name == "CIFAR100":
        dataset = datasets.CIFAR100(root="./data", train=True, download=True)
    # Add other datasets as needed

    task_dataset = TaskDataset(dataset, task_classes, transform)
    return task_dataset


def compute_prototype_vector(model, dataloader, device):
    model.eval()
    class_sum = {}  # Dictionary to store sum of feature vectors for each class
    class_count = {}  # Dictionary to store count of samples for each class

    # Iterate over the dataset to compute sum of feature vectors for each class
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            features = model.feature_map(inputs)

            for feature, label in zip(features, labels):
                if label.item() not in class_sum:
                    class_sum[label.item()] = feature
                    class_count[label.item()] = 1
                else:
                    class_sum[label.item()] += feature
                    class_count[label.item()] += 1

    # Compute prototype vectors by averaging the feature vectors
    prototype_vectors = {}
    for label in class_sum.keys():
        prototype_vectors[label] = class_sum[label] / class_count[label]

    return prototype_vectors


# Train each epoch
def train_epoch(train_loader, model, criterion, optimizer, old_params, device):
    model.train()
    running_loss = 0

    for X, y_true in train_loader:
        optimizer.zero_grad()

        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass
        y_hat, feature = model(X)

        loss = criterion(y_true, y_hat, feature)

        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            rewrite_parameters(model, old_params, device)

    epoch_loss = running_loss / len(train_loader.dataset)

    return model, optimizer, epoch_loss


def validate(valid_loader, model, criterion, device):
    model.eval()
    running_loss = 0

    for X, y_true in valid_loader:

        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass
        y_hat, feature = model(X)

        loss = criterion(y_true, y_hat, feature)

        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)

    return model, epoch_loss


# Train task
def train_task(
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    valid_loader,
    prototype_vectors,
    task_class_map,
    epochs,
    task_id,
    device,
    file_name="model.pth",
    print_every=1,
):
    best_loss = 1e10
    best_acc = 0
    train_losses = []
    valid_losses = []

    old_params = copy.deepcopy(model.named_parameters)

    for epoch in range(epochs):
        # training
        model, optimizer, train_loss = train_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            old_params,
            device,
            task_id=task_id,
        )
        # validation
        with torch.no_grad():
            model, valid_loss = validate(
                valid_loader, model, criterion, task_id, device
            )
            valid_losses.append(valid_loss)
            scheduler.step()

        # compute prototype veector for inference
        prototype_vector = compute_prototype_vector(model, train_loader, device)
        prototype_vectors.update(prototype_vector)

        train_acc = accuracy(
            model, train_loader, prototype_vectors, task_class_map, device=device
        )
        valid_acc = accuracy(
            model, valid_loader, prototype_vectors, task_class_map, device=device
        )

        if valid_acc > best_acc:
            torch.save(model.state_dict(), file_name)
            best_acc = valid_acc

        if epoch % print_every == (print_every - 1):
            print(
                f"{datetime.now().time().replace(microsecond=0)} --- "
                f"Epoch: {epoch}\t"
                f"Train loss: {train_loss:.4f}\t"
                f"Valid loss: {valid_loss:.4f}\t"
                f"Train accuracy: {100 * train_acc:.2f}\t"
                f"Valid accuracy: {100 * valid_acc:.2f}"
            )

    return model, (train_losses, valid_losses)


def predict(model, inputs, prototype_vectors, task_class_map):
    n = len(inputs)
    model.eval()
    min_distance = float("inf") * np.ones(n)
    min_distance_class = -1 * np.ones(n)
    with torch.no_grad():
        for task_id in task_class_map.keys():
            # Specify task id
            model.task_id = task_id
            # Compute the feature extraction for the input
            features = model.feature(inputs).cpu()

            # Prepare prototype veectors in the task
            idx_class_map, prototype_vectors_in_task = tuple(
                zip(*[(c, prototype_vectors[c]) for c in task_class_map(task_id)])
            )

            # Compare the distance to the prototype vectors
            distances = euclidean_distances(features, prototype_vectors_in_task)

            # Identify the nearest prototype vector
            min_idx = np.argmin(distances, axis=1)

            min_distance = np.where(
                distances[min_idx] < min_distance, distances[min_idx], min_distance
            )

            min_distance_class = np.where(
                distances[min_idx] < min_distance,
                idx_class_map[min_idx],
                min_distance_class,
            )

    return min_distance_class
