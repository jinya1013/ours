import torch
import torch.nn as nn
import torch.optim as optim
import copy
from datetime import datetime

from metrics import accuracy


def validate(valid_loader, model, criterion, task_id, device):
    model.eval()
    running_loss = 0

    for X, y_true in valid_loader:

        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass
        y_hat = model(X)
        offset_a = task_id * model.num_classes_per_task
        offset_b = (task_id + 1) * model.num_classes_per_task

        loss = criterion(y_hat[:, offset_a:offset_b], y_true - offset_a)

        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)

    return model, epoch_loss


def features_out(model, x, device):
    i = 0
    bs = 32
    out = []

    while i + bs < len(x):
        out.append(model.features(x[i : (i + bs)].to(device)).cpu().detach())

        i += bs

    if i < len(x) and i + bs >= len(x):
        out.append(model.features(x[i:].to(device)).cpu().detach())

    out = torch.cat(out)

    return out


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
    train_loader, model, criterion, optimizer, old_params, device, task_id
):
    model.train()
    running_loss = 0

    for X, y_true in train_loader:
        optimizer.zero_grad()

        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass
        y_hat = model(X)
        offset_a = task_id * model.num_classes_per_task
        offset_b = (task_id + 1) * model.num_classes_per_task

        loss = criterion(y_hat[:, offset_a:offset_b], y_true - offset_a)

        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        """
        with torch.no_grad():
            for name, param in list(model.named_parameters()):
                if (name == 'conv1'):
                    param.grad.data = param.grad.data*(model.trainable_mask[0]).to(device)
                elif ('linear' in name):
                    if ('weight' in name):
                        param.grad.data = param.grad.data*(model.trainable_mask[-1][0]).to(device)
                    else:
                        param.grad.data = param.grad.data*(model.trainable_mask[-1][1]).to(device)
                else:
                    for layer in range(len(model.num_blocks)):
                        for block in range(model.num_blocks[layer]):
                            if (name == 'layer{}.{}.conv1.weight'.format(layer + 1, block)):
                                param.grad.data = param.grad.data*(model.trainable_mask[1][layer][block][0]).to(device)
                            elif (name == 'layer{}.{}.conv2.weight'.format(layer + 1, block)):
                                param.grad.data = param.grad.data*(model.trainable_mask[1][layer][block][1]).to(device)
                            elif (name == 'layer{}.{}.shortcut.0.weight'.format(layer+1, block)):
                                param.data = param.data*(model.layers_masks[layer][block][-1]).to(device)     
        """

        optimizer.step()

        with torch.no_grad():
            rewrite_parameters(model, old_params, device)

    epoch_loss = running_loss / len(train_loader.dataset)

    return model, optimizer, epoch_loss


def training_loop(
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    valid_loader,
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

    # if "lenet" in model_name:
    #     train = train_lenet
    # elif "vgg" in model_name:
    #     train = train_vgg
    # else:
    #     train = train_resnet
    train = train_resnet

    old_params = copy.deepcopy(model.named_parameters)
    # Train model
    print("TRAINING...")
    for epoch in range(0, epochs):
        # training

        model, optimizer, train_loss = train(
            train_loader,
            model,
            criterion,
            optimizer,
            old_params,
            device,
            task_id=task_id,
        )
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(
                valid_loader, model, criterion, task_id, device
            )
            valid_losses.append(valid_loss)
            scheduler.step()

        train_acc = accuracy(model, train_loader, device=device)
        valid_acc = accuracy(model, valid_loader, device=device)

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


def train(args, model, train_loader, test_loader, device, task_id=0):

    loss = torch.nn.CrossEntropyLoss()
    # loss = torch.nn.BCEWithLogitsLoss()

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.wd
        )
    # elif args.optimizer == "radam":
    #     optimizer = RAdam(
    #         model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd
    #     )
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
        epochs=args.train_epochs,
        task_id=task_id,
        model_name=args.model_name,
        device=device,
        file_name=args.path_pretrained_model,
    )

    net.load_state_dict(torch.load(args.path_pretrained_model, map_location=device))

    return net
