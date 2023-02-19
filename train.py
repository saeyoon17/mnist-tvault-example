# Basic routines for trainning DL model
# Source: https://nextjournal.com/gkoehler/pytorch-mnist
# Import necessary files
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP

from module import Net

# seeding
seed = 2023
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = False
np.random.seed(seed)

# configurations
batch_size = 32
learning_rate = 1e-3
log_interval = 10

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "/MNIST/", train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    ),
    batch_size=batch_size,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "/MNIST/", train=False, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    ),
    batch_size=batch_size,
    shuffle=True,
)


def train(model, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch, batch_idx * len(data), len(train_loader.dataset), 100.0 * batch_idx / len(train_loader), loss.item()))


def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print("\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    # Model
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    train(model)
    test(model)
