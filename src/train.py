import copy
from datetime import datetime

import torch
import torch.nn.functional as F

from models import set_task


# def prototypical_loss(input, target, n_support):
#     """
#     Compute the prototypical loss for few-shot learning.
#     Args:
#     - input: the model output for a batch of samples, tensor of shape (batch_size, feature_dim)
#     - target: ground truth for the batch of samples, tensor of shape (batch_size,)
#     - n_support: number of support samples for each class
#     Returns:
#     - loss_val: the computed loss value, scalar
#     - acc_val: the computed accuracy, scalar
#     """

#     # Move tensors to CPU
#     input_cpu = input.to("cpu")
#     target_cpu = target.to("cpu")

#     # Find the unique classes and the number of classes in the target
#     classes = torch.unique(target_cpu)
#     n_classes = len(classes)

#     # Find the support indices for each class
#     def supp_idxs(c):
#         return target_cpu.eq(c).nonzero(as_tuple=True)[0][:n_support]

#     support_idxs = list(map(supp_idxs, classes))

#     # Compute the class prototypes (the mean of the support samples for each class)
#     prototypes = torch.stack([input_cpu[idx_list].mean(dim=0) for idx_list in support_idxs])

#     # Get the query indices and samples
#     query_indices = torch.cat(
#         [target_cpu.eq(c).nonzero(as_tuple=True)[0][n_support:] for c in classes]
#     )
#     query_samples = input_cpu[query_indices]

#     # Calculate the Euclidean distances between the query samples and the prototypes
#     dists = torch.cdist(query_samples, prototypes)

#     # Calculate the log probabilities
#     log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, -1, n_query)

#     # Prepare the target indices tensor for loss and accuracy computation
#     target_inds = (
#         torch.arange(n_classes)
#         .view(n_classes, 1, 1)
#         .long()
#         .expand(n_classes, n_query, 1)
#     )

#     # Compute the loss
#     loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

#     # Compute the accuracy
#     _, y_hat = log_p_y.max(dim=2)
#     acc_val = y_hat.eq(target_inds.squeeze(dim=2)).float().mean()

#     return loss_val, acc_val

import torch
import torch.nn.functional as F


def prototypical_loss(input, target, n_support, prototype_vectors, device):
    # target_cpu = target.to("cpu")
    # input_cpu = input.to("cpu")

    target_cpu = target
    input_cpu = input

    def get_support_indices(c):
        return target_cpu.eq(c).nonzero(as_tuple=True)[0][:n_support].to(device)

    classes = torch.unique(target_cpu).to(device)
    # n_classes = len(classes)

    # n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    # Use the prototype_vectors if available
    if len(prototype_vectors) > 0:
        stored_prototypes = torch.stack(
            [prototype_vectors[c] for c in range(10) if c in prototype_vectors]
        ).to(device)
    else:
        stored_prototypes = torch.empty(0, input_cpu.size(1)).to(device)

    # Calculate prototypes for new classes
    support_indices = list(map(get_support_indices, classes))
    new_prototypes = torch.stack(
        [input_cpu[idx_list].mean(dim=0) for idx_list in support_indices]
    ).to(device)

    # Combine stored and new prototypes
    prototypes = torch.cat((stored_prototypes, new_prototypes), dim=0)

    query_indices = torch.cat(
        [target_cpu.eq(c).nonzero(as_tuple=True)[0][n_support:] for c in classes]
    )
    # print(query_indices, input_cpu.shape[0], target_cpu.shape[0])
    query_samples = input_cpu[query_indices]
    dists = torch.cdist(query_samples, prototypes)

    # log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    # target_indices = (
    #     torch.arange(0, n_classes)
    #     .view(n_classes, 1, 1)
    #     .expand(n_classes, n_query, 1)
    #     .long()
    # )

    # loss_value = -log_p_y.gather(2, target_indices).squeeze().view(-1).mean()

    log_p_y = F.log_softmax(-dists, dim=1)  # (num_query, num_prototype)

    # print("log_p_y", log_p_y.shape, "gt", target_cpu[query_indices].shape)

    loss_value = F.nll_loss(log_p_y, target_cpu[query_indices])

    if torch.isnan(loss_value):
        print("log_p_y", log_p_y)
        print("target", target_cpu[query_indices])
        raise ValueError(f"loss_value is nan")

    return loss_value


def accuracy(model, test_loader, prototype_vectors, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for X, y_true in test_loader:
            X, y_true = X.to(device), y_true.to(device)
            y_preds = model.predict(X, prototype_vectors)
            # print("ypred", y_preds, "ytrue", y_true)

            total += y_true.size(0)
            correct += (y_preds == y_true).float().sum()

    return (correct / total).item()


def validate(valid_loader, prototype_vectors, model, criterion, task_id, device):
    model.eval()
    running_loss = 0

    for X, y_true in valid_loader:
        # X_prototype = None

        # if len(prototype_vectors) > 0:
        #     keys, values = zip(*prototype_vectors.items())
        #     X_prototype = torch.stack(values)
        #     y_prototype = torch.tensor(keys)

        #     y_true = torch.cat((y_true, y_prototype), dim=0)

        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass
        features = model.features(X)

        y, c = torch.unique(y_true, return_counts=True)
        # print("Y:", y, "C:", c)
        n_support = int(torch.min(c) / 2)

        loss = criterion(features, y_true, n_support, prototype_vectors, device)

        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)

    return model, epoch_loss


def rewrite_parameters(net, old_params, device):
    for (name, param), (old_name, old_param) in zip(
        net.named_parameters(), old_params()
    ):
        if name == "conv1.weight":
            param.data = old_param.data * (1 - net.trainable_mask[0]).to(
                device
            ) + param.data * net.trainable_mask[0].to(device)
        elif "linear" in name:
            if "weight" in name:
                param.data = old_param.data * (1 - net.trainable_mask[-1][0]).to(
                    device
                ) + param.data * net.trainable_mask[-1][0].to(device)
            else:
                param.data = old_param.data * (1 - net.trainable_mask[-1][1]).to(
                    device
                ) + param.data * net.trainable_mask[-1][1].to(device)
        else:
            for layer_num in range(len(net.num_blocks)):
                for block_num in range(net.num_blocks[layer_num]):
                    if name == "layer{}.{}.conv1.weight".format(
                        layer_num + 1, block_num
                    ):
                        param.data = old_param.data * (
                            1 - net.trainable_mask[1][layer_num][block_num][0]
                        ).to(device) + param.data * net.trainable_mask[1][layer_num][
                            block_num
                        ][
                            0
                        ].to(
                            device
                        )
                    elif name == "layer{}.{}.conv2.weight".format(
                        layer_num + 1, block_num
                    ):
                        param.data = old_param.data * (
                            1 - net.trainable_mask[1][layer_num][block_num][1]
                        ).to(device) + param.data * net.trainable_mask[1][layer_num][
                            block_num
                        ][
                            1
                        ].to(
                            device
                        )
                    elif name == "layer{}.{}.shortcut.0.weight".format(
                        layer_num + 1, block_num
                    ):
                        param.data = old_param.data * (
                            1 - net.trainable_mask[1][layer_num][block_num][-1]
                        ).to(device) + param.data * net.trainable_mask[1][layer_num][
                            block_num
                        ][
                            -1
                        ].to(
                            device
                        )

    for (name, param), (old_name, old_param) in zip(
        net.named_parameters(), old_params()
    ):
        for task_id in range(0, net.task_id):
            if "bns.{}".format(task_id) in name:
                param.data = 1 * old_param.data


def freeze_bn(net):
    for name, param in net.named_parameters():
        if "bns.{}".format(net.task_id) in name:
            param.requires_grad = False
            net.bn1.bns[net.task_id].track_running_stats = False
            print(name)

    return net


def train_resnet(
    train_loader,
    prototype_vectors,
    model,
    criterion,
    optimizer,
    old_params,
    device,
    task_id,
):
    model.train()
    running_loss = 0

    for X, y_true in train_loader:
        optimizer.zero_grad()

        # X_prototype = None

        # if len(prototype_vectors) > 0:
        #     keys, values = zip(*prototype_vectors.items())
        #     X_prototype = torch.stack(values)
        #     y_prototype = torch.tensor(keys)

        #     y_true = torch.cat((y_true, y_prototype), dim=0)

        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass
        features = model.features(X)

        y, c = torch.unique(y_true, return_counts=True)
        # print("Y:", y, "C:", c)
        n_support = int(torch.min(c) / 2)

        loss = criterion(features, y_true, n_support, prototype_vectors, device)

        # print(loss)

        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()

        with torch.no_grad():
            for name, param in list(model.named_parameters()):
                if name == "conv1":
                    param.grad.data = param.grad.data * (model.trainable_mask[0]).to(
                        device
                    )
                elif "linear" in name:
                    pass
                    # if "weight" in name:
                    #     param.grad.data = param.grad.data * (
                    #         model.trainable_mask[-1][0]
                    #     ).to(device)
                    # else:
                    #     param.grad.data = param.grad.data * (
                    #         model.trainable_mask[-1][1]
                    #     ).to(device)
                else:
                    for layer in range(len(model.num_blocks)):
                        for block in range(model.num_blocks[layer]):
                            if name == "layer{}.{}.conv1.weight".format(
                                layer + 1, block
                            ):
                                param.grad.data = param.grad.data * (
                                    model.trainable_mask[1][layer][block][0]
                                ).to(device)
                            elif name == "layer{}.{}.conv2.weight".format(
                                layer + 1, block
                            ):
                                param.grad.data = param.grad.data * (
                                    model.trainable_mask[1][layer][block][1]
                                ).to(device)
                            elif name == "layer{}.{}.shortcut.0.weight".format(
                                layer + 1, block
                            ):
                                param.data = param.data * (
                                    model.layers_masks[layer][block][-1]
                                ).to(device)

        optimizer.step()

        with torch.no_grad():
            rewrite_parameters(model, old_params, device)

    epoch_loss = running_loss / len(train_loader.dataset)

    return model, optimizer, epoch_loss, prototype_vectors


def update_prototype_vector(train_loader, prototype_vectors, model, device):
    model.eval()
    class_sum = {}  # Dictionary to store sum of feature vectors for each class
    class_count = {}  # Dictionary to store count of samples for each class

    # Iterate over the dataset to compute sum of feature vectors for each class
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        features = model.features(inputs)

        for feature, label in zip(features, labels):
            if label.item() not in class_sum:
                class_sum[label.item()] = feature
                class_count[label.item()] = 1
            else:
                class_sum[label.item()] += feature
                class_count[label.item()] += 1

    # Compute prototype vectors by averaging the feature vectors
    for label in sorted(class_sum.keys()):
        prototype_vector = class_sum[label] / class_count[label]
        prototype_vectors.update({label: prototype_vector})

    return prototype_vectors


def training_loop(
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    valid_loader,
    prototype_vectors,
    epochs,
    task_id,
    model_name,
    device,
    file_name="model.pth",
    print_every=1,
):
    best_loss = 1e10
    best_acc = 0
    train_losses = []
    valid_losses = []

    if "lenet" in model_name:
        train = train_lenet
    elif "vgg" in model_name:
        train = train_vgg
    else:
        train = train_resnet

    old_params = copy.deepcopy(model.named_parameters)
    # Train model
    print("TRAINING...")
    for epoch in range(0, epochs):
        # training
        set_task(model, task_id)
        model, optimizer, train_loss, prototype_vectors = train(
            train_loader,
            prototype_vectors,
            model,
            criterion,
            optimizer,
            old_params,
            device,
            task_id=task_id,
        )
        train_losses.append(train_loss)

        # prototype update
        set_task(model, task_id)
        with torch.no_grad():
            prototype_vectors_tmp = update_prototype_vector(
                train_loader, prototype_vectors, model, device=device
            )

        # validation
        set_task(model, task_id)
        with torch.no_grad():
            model, valid_loss = validate(
                valid_loader, prototype_vectors_tmp, model, criterion, task_id, device
            )
            valid_losses.append(valid_loss)
            scheduler.step()

        train_acc = accuracy(model, train_loader, prototype_vectors_tmp, device=device)
        valid_acc = accuracy(model, valid_loader, prototype_vectors_tmp, device=device)

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


def train(args, model, train_loader, test_loader, prototype_vectors, device, task_id=0):
    loss = prototypical_loss
    # loss = (
    #     lambda y_hat, y_true, feature: (torch.nn.CrossEntropyLoss())(y_hat, y_true)
    #     + 1.0 * torch.square(torch.norm(feature, p=2, dim=1) - 1.0).sum()
    # )
    # loss = torch.nn.BCEWithLogitsLoss()

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.wd
        )
    # elif args.optimizer == 'radam':
    #     optimizer = RAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.decay_epochs_retrain, gamma=args.gamma
        )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.decay_epochs_train, gamma=args.gamma
    )
    net, _ = training_loop(
        model=model,
        criterion=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=test_loader,
        prototype_vectors=prototype_vectors,
        epochs=args.train_epochs,
        task_id=task_id,
        model_name=args.model_name,
        device=device,
        file_name=args.path_pretrained_model,
    )

    net.load_state_dict(torch.load(args.path_pretrained_model, map_location=device))

    return net
