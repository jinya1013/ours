import torch
import numpy as np


def accuracy(model, data_loader, device):
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

    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:
            X = X.to(device)
            y_true = y_true.to(device)
            y_preds = model(X)

            n += y_true.size(0)
            correct_preds += (y_preds.argmax(dim=1) == y_true).float().sum()

    return (correct_preds / n).item()


def backward_transfer(acc_matrix: np.ndarray, tasks_learned: int) -> list:
    """Computes the backward transfer of a continual learning model.

    The backward transfer measures how much the learning of a new task affects the
    performance of previously learned tasks. It is defined as the average loss in
    accuracy on the previous tasks, caused by learning the new task.

    Args:
        acc_matrix (np.ndarray): A square numpy array of shape (tasks_learned, tasks_learned),
            where each element (i, j) is the accuracy of the model on task i after training
            on task j. The diagonal elements represent the accuracies on each task when trained
            only on that task.
        tasks_learned (int): The number of tasks learned by the model.

    Returns:
        list: A list of backward transfer values, one for each task learned after the first.
            The values are expressed as percentages, i.e., floats between 0 and 100, and are
            rounded to two decimal places.
    """

    bwt = np.zeros(tasks_learned)

    for t in range(1, tasks_learned):
        for task_id in range(t):
            # Compute the backward transfer for the current task and the current reference task
            bwt[t] += 100 * (acc_matrix[task_id, t] - acc_matrix[task_id, task_id]) / t

    # Convert the backward transfer values to a list and round them to two decimal places
    return list(-np.round(bwt, 2))
