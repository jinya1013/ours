import torch
import random
import numpy as np
import pickle
import argparse

import torchvision  # type: ignore
import torch
import os

from data import (
    mean,
    std,
    create_labels,
    task_construction,
    create_class_task_map,
    get_loaders,
)

from models import init_model, set_task

from train import train, accuracy, update_prototype_vector
from pruning import iterative_pruning
from eval import backward_transfer


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
        "--num_classes_first_task",
        type=int,
        default=10,
        help="number of classes in the first task",
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

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    TRAIN = args.train
    EVAL = True

    path_results = (
        f"./results/{args.dataset_name}-{args.model_name}/{args.num_tasks}_tasks/"
    )
    if not os.path.isdir(path_results):
        os.makedirs(path_results)

    print("STARTED")
    print(args)

    orders = {
        "default": [i for i in range(args.num_classes)],
        "seed1993": [
            68,
            56,
            78,
            8,
            23,
            84,
            90,
            65,
            74,
            76,
            40,
            89,
            3,
            92,
            55,
            9,
            26,
            80,
            43,
            38,
            58,
            70,
            77,
            1,
            85,
            19,
            17,
            50,
            28,
            53,
            13,
            81,
            45,
            82,
            6,
            59,
            83,
            16,
            15,
            44,
            91,
            41,
            72,
            60,
            79,
            52,
            20,
            10,
            31,
            54,
            37,
            95,
            14,
            71,
            96,
            98,
            97,
            2,
            64,
            66,
            42,
            22,
            35,
            86,
            24,
            34,
            87,
            21,
            99,
            0,
            88,
            27,
            18,
            94,
            11,
            12,
            47,
            25,
            30,
            46,
            62,
            69,
            36,
            61,
            7,
            63,
            75,
            5,
            32,
            4,
            51,
            48,
            73,
            93,
            39,
            67,
            29,
            49,
            57,
            33,
        ],
        "seed1605": [
            47,
            23,
            18,
            97,
            84,
            49,
            98,
            26,
            0,
            14,
            40,
            85,
            29,
            72,
            1,
            48,
            35,
            52,
            3,
            2,
            92,
            7,
            80,
            32,
            6,
            19,
            79,
            58,
            11,
            34,
            89,
            57,
            21,
            37,
            38,
            86,
            73,
            94,
            28,
            67,
            63,
            87,
            51,
            20,
            54,
            33,
            64,
            56,
            31,
            41,
            12,
            46,
            76,
            99,
            61,
            8,
            36,
            75,
            15,
            4,
            10,
            83,
            82,
            78,
            96,
            27,
            30,
            93,
            74,
            66,
            90,
            70,
            81,
            69,
            5,
            65,
            13,
            25,
            88,
            17,
            71,
            60,
            44,
            68,
            95,
            59,
            45,
            53,
            50,
            43,
            55,
            22,
            24,
            9,
            39,
            62,
            16,
            77,
            91,
            42,
        ],
        "seed2022": [
            79,
            76,
            83,
            5,
            35,
            57,
            22,
            96,
            67,
            58,
            93,
            3,
            69,
            60,
            39,
            17,
            54,
            44,
            61,
            94,
            32,
            84,
            70,
            20,
            50,
            81,
            47,
            51,
            4,
            97,
            30,
            10,
            1,
            25,
            65,
            7,
            26,
            31,
            82,
            6,
            9,
            28,
            62,
            63,
            89,
            34,
            95,
            66,
            8,
            40,
            90,
            59,
            36,
            0,
            68,
            77,
            46,
            43,
            78,
            73,
            21,
            74,
            85,
            29,
            71,
            64,
            91,
            42,
            52,
            13,
            80,
            98,
            12,
            56,
            37,
            23,
            15,
            2,
            87,
            99,
            14,
            72,
            38,
            86,
            48,
            75,
            19,
            11,
            27,
            33,
            41,
            53,
            16,
            24,
            18,
            88,
            55,
            49,
            45,
            92,
        ],
        "seed9999": [
            83,
            13,
            57,
            5,
            31,
            44,
            50,
            55,
            89,
            74,
            73,
            1,
            35,
            15,
            23,
            34,
            7,
            65,
            93,
            49,
            0,
            4,
            66,
            40,
            97,
            17,
            18,
            52,
            98,
            2,
            80,
            72,
            82,
            85,
            77,
            78,
            26,
            42,
            91,
            81,
            6,
            10,
            62,
            87,
            53,
            27,
            60,
            84,
            67,
            29,
            96,
            58,
            63,
            94,
            21,
            48,
            95,
            64,
            75,
            70,
            32,
            88,
            71,
            56,
            61,
            99,
            11,
            3,
            41,
            30,
            19,
            9,
            43,
            39,
            14,
            45,
            68,
            28,
            8,
            22,
            12,
            20,
            59,
            36,
            38,
            79,
            90,
            51,
            69,
            46,
            33,
            76,
            16,
            24,
            25,
            37,
            92,
            54,
            47,
            86,
        ],
    }

    order_name = args.order_name
    tasks_order = np.array(orders[order_name])

    if TRAIN:
        task_labels = create_labels(
            args.num_classes, args.num_tasks, args.num_classes_first_task
        )
        train_dataset, test_dataset = task_construction(
            task_labels, args.dataset_name, tasks_order
        )

        net = init_model(args, device)

        net.class_task_map = create_class_task_map(
            args.num_classes_first_task, tasks_order
        )

        # torch.save(net.state_dict(), args.path_init_params)

        num_tasks = args.num_tasks

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        prototype_vectors = {}

        start_task = 0
        if start_task > 0:
            path_to_save = (
                path_results
                + "{}_task{}_{}classes_{}_{}_it{}_order_{}.pth".format(
                    args.model_name,
                    args.model_name,
                    start_task,
                    args.num_classes_first_task,
                    args.optimizer,
                    args.alpha_conv,
                    args.num_iters,
                    order_name,
                )
            )
            net.load_state_dict(torch.load(path_to_save, map_location=device))

            net._load_masks(
                file_name=path_results
                + "{}_task{}_masks_{}classes_{}_{}_it{}_order_{}.pth".format(
                    args.model_name,
                    args.model_name,
                    start_task,
                    args.num_classes_first_task,
                    args.optimizer,
                    args.alpha_conv,
                    args.num_iters,
                    order_name,
                ),
                num_tasks=start_task,
            )

            if start_task < num_tasks:
                net._add_mask(task_id=start_task)

        for task_id in range(start_task, num_tasks):
            path_to_save = (
                path_results
                + "{}_task{}_{}classes_{}_{}_it{}_order_{}.pth".format(
                    args.model_name,
                    task_id + 1,
                    args.num_classes_first_task,
                    args.optimizer,
                    args.alpha_conv,
                    args.num_iters,
                    order_name,
                )
            )
            set_task(net, task_id)
            train_loader, test_loader = get_loaders(
                train_dataset[task_id], test_dataset[task_id], args.batch_size
            )
            # class_counts = {}
            # for _, labels in train_loader:
            #     for label in labels:
            #         label = label.item()
            #         if label not in class_counts:
            #             class_counts[label] = 0
            #         class_counts[label] += 1
            #     print("各クラスのデータ数:")
            #     for label, count in class_counts.items():
            #         print(f"クラス {label}: {count} 個")

            net.set_trainable_masks(task_id)

            net = train(
                args=args,
                model=net,
                train_loader=train_loader,
                test_loader=test_loader,
                prototype_vectors=prototype_vectors,
                device=device,
                task_id=task_id,
            )

            # prototype update
            net.eval()
            set_task(net, task_id)
            with torch.no_grad():
                prototype_vectors = update_prototype_vector(
                    train_loader, prototype_vectors, net, device=device
                )

            acc = accuracy(net, test_loader, prototype_vectors, device)
            print("Accuracy: ", np.round(100 * acc, 2))

            data_iterator = iter(test_loader)
            x, y = next(data_iterator)
            x = x.to(device)
            print("y: ", y[0])

            for class_, prototype_vector in prototype_vectors.items():
                # print("After training before pruning\n")
                # print("pv", class_, prototype_vector)

                set_task(net, net.class_task_map[class_])

                feature = net.features(x)
                # print("fv", y[0], feature[0])

            with open("prototype.pickle", "wb") as f:
                pickle.dump(prototype_vectors, f)

            random.seed(args.seed)
            np.random.seed(args.seed)
            prune_idx = np.random.permutation(train_dataset[task_id].data.shape[0])[
                : args.prune_batch_size
            ]

            x_prune = torch.FloatTensor(train_dataset[task_id].data[prune_idx]).to(
                device
            )
            x_prune = x_prune.permute(0, 3, 1, 2)
            x_prune = torchvision.transforms.Normalize(mean, std)(x_prune.float() / 255)

            set_task(net, task_id)
            net = iterative_pruning(
                args=args,
                net=net,
                train_loader=train_loader,
                test_loader=test_loader,
                prototype_vectors=prototype_vectors,
                x_prune=x_prune,
                task_id=task_id,
                device=device,
                path_to_save=path_to_save,
            )

            net.set_masks_intersection()
            net.set_masks_union()

            torch.save(net.state_dict(), path_to_save)
            net._save_masks(
                path_results
                + "{}_task{}_masks_{}classes_{}_{}_it{}_order_{}.pth".format(
                    args.model_name,
                    task_id + 1,
                    args.num_classes_first_task,
                    args.optimizer,
                    args.alpha_conv,
                    args.num_iters,
                    order_name,
                )
            )

            if task_id < num_tasks - 1:
                net._add_mask(task_id=task_id + 1)
                print(
                    "-------------------TASK {}------------------------------".format(
                        task_id + 1
                    )
                )

    if EVAL:
        NUM_LEARNED = args.num_tasks
        if not TRAIN:
            path_to_save = (
                path_results
                + "{}_task{}_{}classes_{}_{}_it{}_order_{}.pth".format(
                    args.dataset_name,
                    args.model_name,
                    args.model_name,
                    args.num_tasks,
                    args.num_classes_first_task,
                    args.optimizer,
                    args.alpha_conv,
                    args.num_iters,
                    order_name,
                )
            )
            net.load_state_dict(torch.load(path_to_save, map_location=device))

            net._load_masks(
                file_name=path_results
                + "{}_task{}_masks_{}classes_{}_{}_it{}_order_{}.pth".format(
                    args.dataset_name,
                    args.model_name,
                    args.model_name,
                    args.num_tasks,
                    args.num_classes_first_task,
                    args.optimizer,
                    args.alpha_conv,
                    args.num_iters,
                    order_name,
                ),
                num_tasks=args.num_tasks,
            )

        accs1 = []

        avg_inc_acc1 = []

        net.eval()

        for task_id in range(NUM_LEARNED):
            set_task(net, task_id)
            train_loader, test_loader = get_loaders(
                train_dataset[task_id], test_dataset[task_id], args.batch_size
            )

            accs1.append(
                np.round(100 * accuracy(net, test_loader, prototype_vectors, device), 2)
            )

            print("Task {} accuracy with task_id: ".format(task_id + 1), accs1[task_id])

            avg_inc_acc1.append(np.array(accs1).mean())

        print("Upper-bound Top-1: ", torch.FloatTensor(avg_inc_acc1))

        NUM_RUNS = 5
        shuffle = True

        net.eval()

        batch_size = [args.test_batch_size]
        method = args.task_select_method

        for bs in batch_size:
            total_accs = []
            task_acc_matrices = np.zeros((net.num_tasks, net.num_tasks))
            total_acc_matrices = np.zeros((net.num_tasks, net.num_tasks))
            print("BATCH SIZE: ", bs)
            for i in range(NUM_RUNS):
                print("RUN ", i + 1)
                total_acc, task_acc_matrix, total_acc_matrix = eval(
                    args,
                    net,
                    train_dataset,
                    test_dataset,
                    prototype_vectors,
                    path_to_save,
                    device,
                    method=method,
                    batch_size=bs,
                    max_num_learned=NUM_LEARNED,
                    shuffle=shuffle,
                )
                total_accs.append(torch.stack(total_acc))
                task_acc_matrices += task_acc_matrix
                total_acc_matrices += total_acc_matrix

            task_acc_matrices /= NUM_RUNS
            total_acc_matrices /= NUM_RUNS

            print("####")
            print("INCREMENTAL ACCURACY: ", torch.stack(total_accs, dim=0).mean(dim=0))
            print("BWT: ", backward_transfer(total_acc_matrices, net.num_tasks))

            print(
                "####################################################################"
            )

            with open(
                path_results
                + "{}_{}_{}tasks_{}classes_{}_{}_avg_acc_{}_bs{}_order_{}.pickle".format(
                    args.dataset_name,
                    args.model_name,
                    args.dataset_name,
                    args.model_name,
                    args.num_tasks,
                    args.num_classes_first_task,
                    args.optimizer,
                    args.alpha_conv,
                    method,
                    bs,
                    order_name,
                ),
                "wb",
            ) as handle:
                pickle.dump(
                    total_acc_matrices, handle, protocol=pickle.HIGHEST_PROTOCOL
                )

            with open(
                path_results
                + "{}_{}_{}tasks_{}classes_{}_{}_task_select_{}_bs{}_order_{}.pickle".format(
                    args.dataset_name,
                    args.model_name,
                    args.dataset_name,
                    args.model_name,
                    args.num_tasks,
                    args.num_classes_first_task,
                    args.optimizer,
                    args.alpha_conv,
                    method,
                    bs,
                    order_name,
                ),
                "wb",
            ) as handle:
                pickle.dump(task_acc_matrices, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
