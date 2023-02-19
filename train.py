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
import argparse

# seeding
seed = 2023
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = False
np.random.seed(seed)

# configurations
batch_size = 128
learning_rate = 1e-3
log_interval = 10
device = "cuda"


def train(model, train_epoch, train_loader):
    model.train()
    loss_acc = 0
    for epoch in range(train_epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss_acc += loss.item() / batch_size
            loss.backward()
            optimizer.step()
        if epoch % log_interval == 0:
            print(f"Train Epoch: {epoch} \tLoss: {loss_acc / len(train_loader)}")


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print("\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)))


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epoch", type=int, default=90)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--gpu_ids", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--world_size", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=0)
    return parser


def init_for_distributed(args):
    # 2. init_process_group
    dist.init_process_group(
        backend="nccl",
    )

    # 1. setting for distributed training
    torch.cuda.set_device(args.local_rank)
    if args.local_rank is not None:
        print("Use GPU: {} for training".format(args.local_rank))

    print(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MNIST arg parser", parents=[get_args_parser()])
    args = parser.parse_args()

    # DDP
    init_for_distributed(args)

    # Model
    train_dataset = torchvision.datasets.MNIST(
        "/MNIST/", train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        sampler=train_sampler,
    )

    test_dataset = torchvision.datasets.MNIST(
        "/MNIST/", train=False, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        sampler=test_sampler,
    )

    model = Net().to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=args.gpu_ids)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    train(model, 20, train_loader)
    test(model, test_loader)
