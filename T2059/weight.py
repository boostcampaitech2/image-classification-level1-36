import numpy as np
import torch


# defualt weight function
def my_weight(dataset, device, num_classed=18):
    labels = np.array([0] * num_classed)
    for i in dataset.dataset.targets:
        labels[i] += 1
    weight = [1 - (i / sum(labels)) for i in labels]
    weight = torch.FloatTensor(weight).to(device)
    return weight


# https://medium.com/gumgum-tech/handling-class-imbalance-by-introducing-sample-weighting-in-the-loss-function-3bdebd8203b4 참고
def ins_weight(dataset, device, num_classes=18):
    labels = np.array([0] * 18)
    for i in dataset.dataset.targets:
        labels[i] += 1

    weight = 1 / labels
    weight = weight / np.sum(weight) * num_classes
    weight = torch.FloatTensor(weight).to(device)
    return weight
