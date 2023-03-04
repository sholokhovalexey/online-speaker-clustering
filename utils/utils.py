import torch


def one_hot(labels, neg_label=None):
    # (n_samples,) -> (n_samples, n_classes)
    classes, y = torch.unique(labels, return_inverse=True)
    n_classes = len(classes)
    I = torch.eye(n_classes).to(labels.device)
    labels_onehot = I[y]  # one-hot encoding
    # put neg_label to the last column
    if neg_label is not None:
        idx = (classes == neg_label).nonzero(as_tuple=True)[0]
        # swap columns
        labels_onehot[:, [idx, -1]] = labels_onehot[:, [-1, idx]]
    return labels_onehot


def one_hot_batch(labels_batch, neg_label=None):
    # (batch, n_samples) -> (batch, n_samples, n_classes)
    labels_onehot = []
    for labels in labels_batch:
        labels_onehot += [one_hot(labels, neg_label).unsqueeze(0)]
    return torch.cat(
        labels_onehot
    )  # all sets in the batch must have the same number of classes

