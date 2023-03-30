import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import copy


class NonAffineBN(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBN, self).__init__(dim, affine=False)


class AffineBN(nn.BatchNorm2d):
    def __init__(self, dim):
        super(AffineBN, self).__init__(dim, affine=True)


class NonAffineNoStatsBN(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineNoStatsBN, self).__init__(
            dim, affine=False, track_running_stats=False
        )


class MultitaskBN(nn.Module):
    def __init__(self, dim, num_tasks, affine=True):
        super(MultitaskBN, self).__init__()
        if affine:
            self.bns = nn.ModuleList([AffineBN(dim) for _ in range(num_tasks)])
        else:
            self.bns = nn.ModuleList([NonAffineBN(dim) for _ in range(num_tasks)])

    def forward(self, x, task_id):
        return self.bns[task_id](x)


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


def _masks_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.ones_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, args, device, stride=1, option="B"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = MultitaskBN(planes, args.num_tasks)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = MultitaskBN(planes, args.num_tasks)

        self.task_id = 0

        self.shortcut = nn.Sequential()
        self.planes = planes
        self.device = device
        self.in_planes = in_planes
        self.stride = stride

        self.block_masks = self._make_masks(
            in_planes, planes, stride
        )  # masks for current task

        self.tasks_masks = []  # mask archive

        self.planes = planes
        self.in_planes = in_planes
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, self.planes // 4, self.planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    MultitaskBN(self.expansion * planes, args.num_tasks),
                )

    def add_mask(self):
        """
        Append a deep copy of the current block_masks to the tasks_masks list.
        """
        self.tasks_masks.append(copy.deepcopy(self.block_masks))

    def _make_masks(self, in_planes, planes, stride):
        """Create and return a list of masks depending on the provided parameters.

        The masks have shapes corresponding to the weight tensors of the
        convolutional layers in the block.

        Args:
            in_planes (int): The number of input channels.
            planes (int): The number of output channels for the first and second
                convolutional layers.
            stride (int): The stride value for the first convolutional layer.

        Returns:
            List[torch.Tensor]: A list of masks with specific shapes depending
                on the provided parameters.
        """
        if stride != 1 or in_planes != self.expansion * planes:
            mask = [
                torch.ones(planes, in_planes, 3, 3),
                torch.ones(planes, planes, 3, 3),
                torch.ones(self.expansion * planes, in_planes, 1, 1),
            ]
        else:
            mask = [
                torch.ones(planes, in_planes, 3, 3),
                torch.ones(planes, planes, 3, 3),
                torch.ones(planes),
            ]

        return mask

    def forward(self, x):
        """Compute the output of the BasicBlock using input tensor x.

        Applies the convolutional layers and the shortcut connection, and
        performs element-wise addition between the output of the second
        batch normalization layer and the shortcut connection.

        Args:
            x (torch.Tensor): The input tensor with shape (batch_size, in_planes, height, width).

        Returns:
            torch.Tensor: The output tensor with shape (batch_size, planes, output_height, output_width).
        """
        active_conv = self.conv1.weight * self.tasks_masks[self.task_id][0].to(
            self.device
        )
        out = F.conv2d(
            x,
            weight=active_conv,
            bias=None,
            stride=self.conv1.stride,
            padding=self.conv1.padding,
            groups=self.conv1.groups,
        )
        out = F.relu(self.bn1(out, self.task_id))

        active_conv = self.conv2.weight * self.tasks_masks[self.task_id][1].to(
            self.device
        )
        out = F.conv2d(
            out,
            weight=active_conv,
            bias=None,
            stride=self.conv2.stride,
            padding=self.conv2.padding,
            groups=self.conv2.groups,
        )
        out = self.bn2(out, self.task_id)

        if self.stride != 1 or self.in_planes != self.planes:
            shortcut = list(self.shortcut.children())[0]
            active_shortcut = shortcut.weight * self.tasks_masks[self.task_id][-1].to(
                self.device
            )

            bn = list(self.shortcut.children())[1]
            shortcut = F.conv2d(
                x,
                weight=active_shortcut,
                bias=None,
                stride=shortcut.stride,
                padding=shortcut.padding,
                groups=shortcut.groups,
            )
            shortcut = bn(shortcut, self.task_id)
            out += shortcut
        else:
            shortcut = self.shortcut(x)
            out += shortcut * (
                self.tasks_masks[self.task_id][-1]
                .reshape((-1, 1, 1))
                .expand((shortcut.size(-3), shortcut.size(-2), shortcut.size(-1)))
                .to(self.device)
            )

        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, args, device, task_id=0):
        super(ResNet, self).__init__()

        # _outputs = [64, 128, 256, 512]
        _outputs = [21, 42, 85, 170]
        self.in_planes = _outputs[0]

        self.num_blocks = num_blocks

        self.num_classes = args.num_classes
        self.num_classes_per_task = args.num_classes_per_task
        self.num_tasks = 0
        self.args = args

        self.device = device

        self.task_id = task_id

        self.conv1 = nn.Conv2d(
            3, _outputs[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv1_masks = torch.ones(_outputs[0], 3, 3, 3)
        self.bn1 = MultitaskBN(_outputs[0], args.num_tasks)

        self.layer1, self.layer1_masks = self._make_layer(
            block, _outputs[0], num_blocks[0], stride=1
        )
        self.layer2, self.layer2_masks = self._make_layer(
            block, _outputs[1], num_blocks[1], stride=2
        )
        self.layer3, self.layer3_masks = self._make_layer(
            block, _outputs[2], num_blocks[2], stride=2
        )
        self.layer4, self.layer4_masks = self._make_layer(
            block, _outputs[3], num_blocks[3], stride=2
        )

        self.layers_masks = [
            self.layer1_masks,
            self.layer2_masks,
            self.layer3_masks,
            self.layer4_masks,
        ]

        self.linear = nn.Linear(_outputs[3] * block.expansion, self.num_classes)
        self.linear_masks = [
            torch.ones(self.num_classes, _outputs[3] * block.expansion),
            torch.ones(self.num_classes),
        ]

        self.apply(_weights_init)

        self.tasks_masks = []

        self._add_mask(task_id=0)

        self.trainable_mask = copy.deepcopy(self.tasks_masks[0])
        self.masks_union = copy.deepcopy(self.tasks_masks[0])
        self.masks_intersection = copy.deepcopy(self.tasks_masks[0])

    def _add_mask(self, task_id):
        """Add a mask to the network for a specific task.

        This function creates a new mask for the given task ID and appends
        it to the tasks_masks list. It also updates the tasks_masks list
        for the current task ID by copying masks from the individual blocks
        of the network.

        Args:
            task_id (int): The ID of the task for which to add a mask.

        """
        self.num_tasks += 1
        network_mask = [
            copy.deepcopy(self.conv1_masks),
            copy.deepcopy(self.layers_masks),
            copy.deepcopy(self.linear_masks),
        ]

        self.tasks_masks.append(copy.deepcopy(network_mask))

        for layer in range(len(network_mask[1])):
            for block in range(len(network_mask[1][layer])):
                Block = list(list(self.children())[layer + 2])[block]
                Block.add_mask()
                for conv in range(2):
                    self.tasks_masks[task_id][1][layer][block][
                        conv
                    ] = Block.tasks_masks[task_id][conv]

                self.tasks_masks[task_id][1][layer][block][-1] = Block.tasks_masks[
                    task_id
                ][-1]

        index = self.num_classes_per_task * task_id
        # Check if the current task's class range doesn't exceed the total number of classes
        if index + self.num_classes_per_task < self.num_classes - 1:
            # Set the masks for classes beyond the current task's range to 0
            self.tasks_masks[-1][-1][-1][(index + self.num_classes_per_task) :] = 0
            self.tasks_masks[-1][-1][-2][(index + self.num_classes_per_task) :, :] = 0
        # Check if it is not the first task
        if task_id > 0:
            # Set the masks for classes before the current task's range to 0
            self.tasks_masks[-1][-1][-1][:index] = 0
            self.tasks_masks[-1][-1][-2][:index, :] = 0

    def set_masks_union(self, num_learned=-1):
        """
        Set the union of masks for a given number of learned tasks.

        This function computes the union of masks for the specified number
        of tasks (or all tasks if num_learned is not specified) and stores
        the result in the masks_union attribute.

        Args:
            num_learned (int, optional): The number of learned tasks to consider for the union. Defaults to -1.
        """
        self.masks_union = copy.deepcopy(self.tasks_masks[0])

        # If num_learned is not specified, set it to the total number of tasks
        if num_learned < 0:
            num_learned = self.num_tasks

        # Iterate through the learned tasks and compute the union of masks
        for id in range(1, num_learned):
            # Compute the union of conv1_masks
            self.masks_union[0] = 1 * torch.logical_or(
                self.masks_union[0], self.tasks_masks[id][0]
            )

            for layer in range(len(self.masks_union[1])):
                for block in range(0, len(self.masks_union[1][layer])):
                    # Compute the union of conv_masks in each layer and block
                    for conv in range(2):
                        self.masks_union[1][layer][block][conv] = 1 * torch.logical_or(
                            self.masks_union[1][layer][block][conv],
                            self.tasks_masks[id][1][layer][block][conv],
                        )

                    # Compute the union of downsample_masks in each layer and block
                    self.masks_union[1][layer][block][-1] = 1 * torch.logical_or(
                        self.masks_union[1][layer][block][-1],
                        self.tasks_masks[id][1][layer][block][-1],
                    )

            # Compute the union of linear_masks
            self.masks_union[-1][0] = 1 * torch.logical_or(
                self.masks_union[-1][0], self.tasks_masks[id][-1][0]
            )
            self.masks_union[-1][1] = 1 * torch.logical_or(
                self.masks_union[-1][1], self.tasks_masks[id][-1][1]
            )

    def set_masks_intersection(self):
        """
        Set the intersection of masks for all learned tasks.

        This function computes the intersection of masks for all learned tasks
        and stores the result in the masks_intersection attribute.
        """
        self.masks_intersection = copy.deepcopy(self.tasks_masks[0])

        # Iterate through the learned tasks and compute the intersection of masks
        for id in range(1, self.num_tasks):
            # Compute the intersection of conv1_masks
            self.masks_intersection[0] = 1 * torch.logical_and(
                self.masks_intersection[0], self.tasks_masks[id][0]
            )

            for layer in range(len(self.masks_intersection[1])):
                for block in range(0, len(self.masks_intersection[1][layer])):
                    # Compute the intersection of conv_masks in each layer and block
                    for conv in range(2):
                        self.masks_intersection[1][layer][block][
                            conv
                        ] = 1 * torch.logical_and(
                            self.masks_intersection[1][layer][block][conv],
                            self.tasks_masks[id][1][layer][block][conv],
                        )

                    # Compute the intersection of downsample_masks in each layer and block
                    self.masks_intersection[1][layer][block][
                        -1
                    ] = 1 * torch.logical_and(
                        self.masks_intersection[1][layer][block][-1],
                        self.tasks_masks[id][1][layer][block][-1],
                    )

            # Compute the intersection of linear_masks
            self.masks_intersection[-1][0] = 1 * torch.logical_and(
                self.masks_intersection[-1][0], self.tasks_masks[id][-1][0]
            )
            self.masks_intersection[-1][1] = 1 * torch.logical_and(
                self.masks_intersection[-1][1], self.tasks_masks[id][-1][1]
            )

    def set_trainable_masks(self, task_id):
        """
        Set the trainable masks for a specific task.

        This function computes the trainable masks for the given task ID
        and stores the result in the trainable_mask attribute. For the first
        task (task_id == 0), the trainable masks are equal to the task's masks.
        For subsequent tasks, the trainable masks are computed by subtracting
        the union of previous tasks' masks from the current task's masks.

        Args:
            task_id (int): The ID of the task for which to set trainable masks.
        """
        if task_id > 0:
            # Compute the trainable conv1_masks
            self.trainable_mask[0] = copy.deepcopy(
                1 * ((self.tasks_masks[task_id][0] - self.masks_union[0]) > 0)
            )

            # Iterate through layers and blocks to compute the trainable masks
            for layer in range(len(self.trainable_mask[1])):  # layer x block x 0/1
                for block in range(len(self.trainable_mask[1][layer])):
                    Block = list(list(self.children())[layer + 2])[block]

                    # Compute the trainable conv_masks for each layer and block
                    for conv in range(2):
                        self.trainable_mask[1][layer][block][conv] = copy.deepcopy(
                            1
                            * (
                                (
                                    self.tasks_masks[task_id][1][layer][block][conv]
                                    - self.masks_union[1][layer][block][conv]
                                )
                                > 0
                            )
                        )

                    # Compute the trainable downsample_masks if the block has a downsample layer
                    if Block.stride != 1 or Block.in_planes != Block.planes:
                        self.trainable_mask[1][layer][block][-1] = copy.deepcopy(
                            1
                            * (
                                (
                                    self.tasks_masks[task_id][1][layer][block][-1]
                                    - self.masks_union[1][layer][block][-1]
                                )
                                > 0
                            )
                        )

            # Compute the trainable linear_masks
            self.trainable_mask[-1][0] = copy.deepcopy(
                1 * ((self.tasks_masks[task_id][-1][0] - self.masks_union[-1][0]) > 0)
            )
            self.trainable_mask[-1][1] = copy.deepcopy(
                1 * ((self.tasks_masks[task_id][-1][1] - self.masks_union[-1][1]) > 0)
            )
        else:
            # For the first task, the trainable masks are equal to the task's masks
            self.trainable_mask = copy.deepcopy(self.tasks_masks[0])

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        Create a layer with a sequence of blocks and their corresponding masks.

        This function constructs a layer with the specified number of blocks, planes,
        and initial stride. It also creates a list of masks for each block in the layer.
        The layer is returned as an nn.Sequential object, and the list of masks is returned
        as a separate list.

        Args:
            block (nn.Module): The type of block to use in the layer (e.g., BasicBlock).
            planes (int): The number of output channels for each block in the layer.
            num_blocks (int): The number of blocks to include in the layer.
            stride (int): The stride for the first block in the layer.

        Returns:
            nn.Sequential: The layer containing the sequence of blocks.
            list: The list of masks corresponding to each block in the layer.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        layers_masks = []

        # Iterate through the blocks, create each block and its corresponding mask
        for stride in strides:
            basicblock = block(self.in_planes, planes, self.args, self.device, stride)
            layers.append(basicblock)
            layers_masks.append(basicblock.block_masks)

            # Update the number of input channels for the next block
            self.in_planes = planes * block.expansion

        # Return the layer as an nn.Sequential object and the list of masks
        return nn.Sequential(*layers), layers_masks

    def features(self, x):
        """
        Compute the features of the input tensor x using the network's convolutional layers.

        This function processes the input tensor x through the network's convolutional layers,
        applying the task-specific masks as needed. The final feature tensor is reshaped into
        a 2D tensor and returned.

        Args:
            x (torch.Tensor): The input tensor with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The output feature tensor with shape (batch_size, num_features).
        """
        # Apply the task-specific mask to the first convolution layer
        active_conv = self.conv1.weight * self.tasks_masks[self.task_id][0].to(
            self.device
        )

        # Process the input tensor through the first masked convolution layer
        out = F.conv2d(
            x,
            weight=active_conv,
            bias=None,
            stride=self.conv1.stride,
            padding=self.conv1.padding,
        )

        # Apply the ReLU activation function and the batch normalization layer
        out = F.relu(self.bn1(out, self.task_id))

        # Process the tensor through the remaining convolutional layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Apply average pooling to the output tensor
        out = F.avg_pool2d(out, out.size()[3])

        # Reshape the output tensor into a 2D tensor (batch_size, num_features)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        """
        Forward pass of the input tensor x through the network.

        This function processes the input tensor x through the network's layers,
        applying the task-specific masks as needed. The final output tensor is returned.

        Args:
            x (torch.Tensor): The input tensor with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The output tensor with shape (batch_size, num_classes).
        """
        # Apply the task-specific mask to the first convolution layer
        active_conv = self.conv1.weight * self.tasks_masks[self.task_id][0].to(
            self.device
        )

        # Process the input tensor through the first masked convolution layer
        out = F.conv2d(
            x,
            weight=active_conv,
            bias=None,
            stride=self.conv1.stride,
            padding=self.conv1.padding,
        )

        # Apply the ReLU activation function and the batch normalization layer
        out = F.relu(self.bn1(out, self.task_id))

        # Process the tensor through the remaining convolutional layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Apply average pooling to the output tensor
        out = F.avg_pool2d(out, out.size()[3])

        # Reshape the output tensor into a 2D tensor (batch_size, num_features)
        feature = out.view(out.size(0), -1)

        # Apply the task-specific mask to the linear layer
        active_weight = self.linear.weight * self.tasks_masks[self.task_id][-1][0].to(
            self.device
        )
        active_bias = self.linear.bias * self.tasks_masks[self.task_id][-1][1].to(
            self.device
        )

        # Process the feature tensor through the masked linear layer
        out = F.linear(feature, weight=active_weight, bias=active_bias)

        return out, feature

    def _save_masks(self, file_name="net_masks.pt"):
        """
        Save the masks of all tasks to a file.

        This function saves the masks for each task in the network into a dictionary
        and then stores that dictionary in a file with the given file name. The masks
        are stored with keys representing their layer, block, convolution number, and
        task ID.

        Args:
            file_name (str, optional): The name of the file to save the masks. Defaults to "net_masks.pt".
        """
        masks_database = {}

        for task_id in range(self.num_tasks):
            # Save the mask for the first convolution layer
            masks_database["conv1.mask.task{}".format(task_id)] = self.tasks_masks[
                task_id
            ][0]

            for layer in range(len(self.num_blocks)):
                for block in range(self.num_blocks[layer]):
                    for conv_num in range(2):
                        # Save the mask for each convolution within a block
                        name = "layer{}.{}.conv{}.mask.task{}".format(
                            layer + 1, block, conv_num + 1, task_id
                        )
                        masks_database[name] = self.tasks_masks[task_id][1][layer][
                            block
                        ][conv_num]

                    # Save the mask for the shortcut in each block
                    name = "layer{}.{}.shortcut.mask.task{}".format(
                        layer + 1, block, task_id
                    )
                    masks_database[name] = self.tasks_masks[task_id][1][layer][block][
                        -1
                    ]

            # Save the masks for the linear layer's weight and bias
            masks_database[
                "linear.weight.mask.task{}".format(task_id)
            ] = self.tasks_masks[task_id][-1][0]
            masks_database[
                "linear.bias.mask.task{}".format(task_id)
            ] = self.tasks_masks[task_id][-1][1]

        # Save the masks database to the specified file
        torch.save(masks_database, file_name)

    def _load_masks(self, file_name="net_masks.pt", num_tasks=1):
        """
        Load the masks of all tasks from a file.

        This function loads the masks for each task in the network from a file with
        the given file name. The masks are loaded into the tasks_masks attribute
        and additional masks are added if the number of tasks specified is greater
        than the number of tasks currently in the network.

        Args:
            file_name (str, optional): The name of the file to load the masks from. Defaults to "net_masks.pt".
            num_tasks (int, optional): The number of tasks to load masks for. Defaults to 1.
        """
        masks_database = torch.load(file_name)

        for task_id in range(num_tasks):
            # Load the mask for the first convolution layer
            self.tasks_masks[task_id][0] = masks_database[
                "conv1.mask.task{}".format(task_id)
            ]

            for layer in range(len(self.num_blocks)):
                for block in range(self.num_blocks[layer]):
                    Block = list(list(self.children())[layer + 2])[block]
                    for conv in range(2):
                        # Load the mask for each convolution within a block
                        name = "layer{}.{}.conv{}.mask.task{}".format(
                            layer + 1, block, conv + 1, task_id
                        )
                        Block.tasks_masks[task_id][conv] = masks_database[name]
                        self.tasks_masks[task_id][1][layer][block][
                            conv
                        ] = Block.tasks_masks[task_id][conv]

                    # Load the mask for the shortcut in each block
                    name = "layer{}.{}.shortcut.mask.task{}".format(
                        layer + 1, block, task_id
                    )
                    Block.tasks_masks[task_id][-1] = masks_database[name]
                    self.tasks_masks[task_id][1][layer][block][-1] = Block.tasks_masks[
                        task_id
                    ][-1]

            # Load the masks for the linear layer's weight and bias
            self.tasks_masks[task_id][-1][0] = masks_database[
                "linear.weight.mask.task{}".format(task_id)
            ]
            self.tasks_masks[task_id][-1][1] = masks_database[
                "linear.bias.mask.task{}".format(task_id)
            ]

            # Add mask for the next task if needed
            if task_id + 1 < num_tasks:
                self._add_mask(task_id + 1)

        # Set the masks union and masks intersection after loading all masks
        self.set_masks_union()
        self.set_masks_intersection()


def resnet18(args, device):
    return ResNet(BasicBlock, [2, 2, 2, 2], args, device)


def resnet20(num_classes, num_classes_per_task, device):
    return ResNet(BasicBlock, [3, 3, 3], num_classes)


def resnet32(num_classes, num_classes_per_task, device):
    return ResNet(BasicBlock, [5, 5, 5], num_classes, num_classes_per_task, device)


def resnet44(num_classes, num_classes_per_task, device):
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56(num_classes):
    return ResNet(BasicBlock, [9, 9, 9], num_classes)


def resnet110(num_classes):
    return ResNet(BasicBlock, [18, 18, 18])


def init_model(args, device):
    """
    Initialize a ResNet-18 model with the given arguments and device.

    This function creates a ResNet-18 model with the specified arguments and
    moves the model to the target device. It returns the initialized model.

    Args:
        args: A namespace or dictionary containing the arguments for the ResNet-18 model.
        device (torch.device): The device to move the model to (e.g., 'cpu' or 'cuda').

    Returns:
        model (nn.Module): The initialized ResNet-18 model.
    """
    model = resnet18(args, device)
    model = model.to(device)
    return model


def resnet_total_params(model):
    """
    Calculate the total number of non-zero parameters in a ResNet model.

    This function computes the total number of non-zero parameters in the
    given ResNet model by iterating through its state dictionary and counting
    the number of elements in each parameter tensor that are not zero.

    Args:
        model (nn.Module): The ResNet model to calculate the total number of non-zero parameters for.

    Returns:
        total_number (int): The total number of non-zero parameters in the model.
    """
    total_number = 0
    # Iterate through the model's state dictionary
    for param_name in list(model.state_dict()):
        # Get the parameter tensor
        param = model.state_dict()[param_name]
        # Count non-zero elements and add to the total number
        total_number += torch.numel(param[param != 0])

    return total_number


def resnet_total_params_mask(model, task_id=0):
    """
    Calculate the total number of non-zero parameters in a ResNet model using masks for a given task.

    This function computes the total number of non-zero parameters in the
    given ResNet model for a specific task using the associated masks. It
    iterates through the model's named parameters and counts the number of
    non-zero elements in each parameter tensor using the masks.

    Args:
        model (nn.Module): The ResNet model to calculate the total number of non-zero parameters for.
        task_id (int, optional): The task ID for which to calculate the total number of non-zero parameters. Defaults to 0.

    Returns:
        total (int): The total number of non-zero parameters in the model for the given task.
        total_number_conv (int): The total number of non-zero parameters in the convolutional layers for the given task.
        total_number_fc (int): The total number of non-zero parameters in the fully connected layer for the given task.
    """
    total_number_conv = 0
    total_number_fc = 0

    # Iterate through the model's named parameters
    for name, param in list(model.named_parameters()):
        if name == "conv1":
            total_number_conv += model.tasks_masks[task_id][0].sum()
        elif "linear" in name:
            if "weight" in name:
                total_number_fc += model.tasks_masks[task_id][-1][0].sum()
            else:
                total_number_fc += model.tasks_masks[task_id][-1][1].sum()
        else:
            for layer in range(len(model.num_blocks)):
                for block in range(model.num_blocks[layer]):
                    if name == "layer{}.{}.conv1.weight".format(layer + 1, block):
                        total_number_conv += model.tasks_masks[task_id][1][layer][
                            block
                        ][0].sum()
                    elif name == "layer{}.{}.conv2.weight".format(layer + 1, block):
                        total_number_conv += model.tasks_masks[task_id][1][layer][
                            block
                        ][1].sum()
                    elif name == "layer{}.{}.shortcut.0.weight".format(
                        layer + 1, block
                    ):
                        total_number_conv += model.tasks_masks[task_id][1][layer][
                            block
                        ][-1].sum()

    total = total_number_conv + total_number_fc

    return total.item(), total_number_conv.item(), total_number_fc.item()


def resnet_get_architecture(model):
    """
    Gets the architecture of a ResNet model in terms of the number of
    active filters in each convolutional and fully-connected layer.

    Args:
        model (torch.nn.Module): A ResNet model.

    Returns:
        arch: A list of lists, where the first list contains the number of active
        filters in each convolutional layer and the second list contains the
        number of active units in the fully-connected layer.

    """
    arch = []
    convs = []
    fc = []

    # Get the number of active filters in the first convolutional layer
    convs.append(torch.sum(model.conv1_masks.sum(dim=(1, 2, 3)) > 0).item())

    # Loop over each residual block in the model
    for block, num_block in enumerate(model.num_blocks):
        block_masks = []
        # Loop over each convolutional layer in the block
        for i in range(num_block):
            block_conv_masks = []
            # Loop over the two convolutional layers in each block
            for conv_num in range(2):
                # Get the number of active filters in the current convolutional layer
                block_conv_masks.append(
                    torch.sum(
                        model.layers_masks[block][i][conv_num].sum(dim=(1, 2, 3)) > 0
                    ).item()
                )

            block_masks.append(block_conv_masks)

        convs.append(block_masks)

    arch.append(convs)

    # Get the number of active units in the fully-connected layer
    fc.append(torch.sum(model.linear_masks[0].sum(dim=0) > 0).item())

    arch.append(fc)

    return arch


def resnet_compute_flops(model):
    arch_conv, arch_fc = resnet_get_architecture(model)
    flops = 0
    k = 3
    h = 32
    w = 32
    in_channels = 3

    flops += (2 * h * w * (in_channels * k**2) - 1) * arch_conv[0]

    in_channels = arch_conv[0]

    for block, num_block in enumerate(model.num_blocks):
        for i in range(num_block):
            for conv_num in range(2):
                if conv_num == 1:
                    out_channels = torch.sum(
                        torch.logical_or(
                            model.layers_masks[block][i][1].sum(dim=(1, 2, 3)),
                            model.layers_masks[block][i][-1],
                        )
                        > 0
                    ).item()
                else:
                    out_channels = arch_conv[1 + block][i][conv_num]

                flops += (2 * h * w * (in_channels * k**2) - 1) * out_channels

            in_channels = out_channels

            flops += h * w * model.layers_masks[block][i][-1].sum().item()

        h /= 2
        w /= 2

    flops += (2 * arch_fc[0] - 1) * 10

    return flops
