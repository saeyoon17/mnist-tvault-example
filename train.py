import os
import sys
import ast
import git
import glob
import json
import difflib
import astunparse
from collections import defaultdict


class TorchVault:
    def __init__(self, log_dir="./model_log", model_dir="./"):
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.use_astunparse = True if sys.version_info.minor < 9 else False

    """
    From model directory, retrieves every class and function definition from .py files
    """

    def get_defs(self, model_dir):
        function_defs = defaultdict(lambda: "")
        class_defs = defaultdict(lambda: "")
        for filename in glob.iglob(model_dir + "**/*.py", recursive=True):
            with open(filename, "r") as f:
                file_ast = ast.parse(f.read())
            for stmt in file_ast.body:
                if type(stmt) == ast.ClassDef:
                    class_defs[filename + ":" + stmt.name] = stmt
                elif type(stmt) == ast.FunctionDef:
                    function_defs[filename + ":" + stmt.name] = stmt
        return class_defs, function_defs

    """
    From class definitions, retrieve function names that are not class methods from __init__.
    """

    def match_external_funcs(self, class_defs):
        target_funcs = []
        for class_def in class_defs.values():
            # for each body in class definitions,
            for body in class_def.body:
                try:
                    # if the function is __init__,
                    if body.name == "__init__":
                        init_body = body
                        # for each stmt in init_body,
                        for stmt in init_body.body:
                            # external if satisfies following condition
                            if (
                                type(stmt) == ast.Assign
                                and type(stmt.value) == ast.Call
                                and type(stmt.value.func) == ast.Name
                            ):
                                # this is the function we need to track
                                function_name = stmt.value.func.id
                                target_funcs.append(function_name)
                # parsing errors will happen by default
                except:
                    pass
        return list(set(target_funcs))

    """
    Provide logging for pytorch model.
    1. Retrives target modules from pytorch model representation.
    2. Get class definition of target modules.
    3. Get external function definition of those used in target model.
    """

    def analyze_model(self, model):
        os.makedirs(self.log_dir, exist_ok=True)

        model = model.__str__()
        target_modules = set()

        # retrieve target modules
        for line in model.split("\n"):
            if "(" in line:
                if line == line.strip():
                    # model classname
                    target_module = line.split("(")[0]
                else:
                    # submodules
                    target_module = line.split("(")[1].split(" ")[-1]
                target_modules.add(target_module)

        # retrieve class / function definitions
        class_defs, function_defs = self.get_defs(self.model_dir)

        # get target module defs.
        filter_class_defs = defaultdict(lambda: "")
        for k, v in class_defs.items():
            if k.split(":")[-1] in target_modules:
                filter_class_defs[k] = v

        # find functions that we only want to track
        target_funcs = self.match_external_funcs(filter_class_defs)

        # unparse
        filter_target_class = defaultdict(lambda: "")
        for k, v in class_defs.items():
            if k.split(":")[-1] in target_modules:
                if self.use_astunparse:
                    filter_target_class[k] = astunparse.unparse(v)
                else:
                    filter_target_class[k] = ast.unparse(v)

        filter_target_funcs = defaultdict(lambda: "")
        for k, v in function_defs.items():
            if k.split(":")[-1] in target_funcs:
                if self.use_astunparse:
                    filter_target_funcs[k] = astunparse.unparse(v)
                else:
                    filter_target_funcs[k] = ast.unparse(v)

        # get git hash
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        short_sha = sha[:7]
        model_log = dict()
        model_log["model"] = model.__str__()
        model_log["src"] = dict(filter_target_class)
        model_log["external_func"] = dict(filter_target_funcs)
        model_json = json.dumps(model_log, indent=4)

        with open(f"{self.log_dir}/model_{short_sha}", "w") as f:
            f.write(model_json)

    def diff(self, sha1, sha2):
        with open(f"{self.log_dir}/model_{sha1}", "r") as f1:
            prev_model = json.load(f1)
        with open(f"{self.log_dir}/model_{sha2}", "r") as f2:
            cur_model = json.load(f2)
        diff_dict = dict()

        # 1. get model diff using string
        model_diff = [e for e in difflib.ndiff(prev_model["model"], cur_model["model"])]
        import ipdb

        ipdb.set_trace()
        filter_model_diff = [l for l in model_diff if not l.startswith("? ")]
        model_diff = "".join(filter_model_diff)
        diff_dict["model"] = model_diff

        # 2. Check module definition between modules
        src_diff = dict()
        for p_module, p_source in prev_model["src"].items():
            # if module still exists in current model
            if p_module in cur_model["src"].keys():
                class_diff = [
                    e
                    for e in difflib.ndiff(
                        p_source.split("\n"), cur_model["src"][p_module].split("\n")
                    )
                ]  # generator requires this wrapping
                changes = [l for l in class_diff if l.startswith("+ ") or l.startswith("- ")]
                filter_class_diff = [l for l in class_diff if not l.startswith("? ")]
                if len(changes) > 0:
                    src_diff[p_module] = "\n".join(filter_class_diff)
            else:
                src_diff[p_module] = "module removed"
        for c_module, c_source in cur_model["src"].items():
            if c_module not in prev_model["src"].keys():
                src_diff[c_module] = "module added"
        diff_dict["src"] = src_diff

        # 3. Check external function diff
        func_diff = dict()
        for p_func, p_source in prev_model["external_func"].items():
            if p_func in cur_model["external_func"].keys():
                func_diff = [
                    e
                    for e in difflib.ndiff(
                        p_source.split("\n"), cur_model["external_func"][p_func].split("\n")
                    )
                ]  # generator requires this wrapping
                changes = [l for l in func_diff if l.startswith("+ ") or l.startswith("- ")]
                filter_func_diff = [l for l in func_diff if not l.startswith("? ")]
                if len(changes) > 0:
                    func_diff[p_func] = "\n".join(filter_func_diff)
            else:
                func_diff[p_func] = "function removed"
        for c_func, c_source in cur_model["external_func"].items():
            if c_func not in prev_model["external_func"].keys():
                func_diff[c_func] = "function added"
        diff_dict["func"] = func_diff

        diff_json = json.dumps(diff_dict, indent=4)
        # Writing to sample.json
        with open(f"{self.log_dir}/diff_{sha1}_{sha2}", "w") as f:
            f.write(diff_json)


# Basic routines for trainning DL model
# Source: https://nextjournal.com/gkoehler/pytorch-mnist
# Import necessary files
import os
import git
import torch
import pickle
import torch.distributed as dist
import torch.optim as optim
import torchvision
import numpy as np
from utils import analyze_model, get_model_diff
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP

from module import Net, resnet18
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


def train(model, train_epoch, train_loader, local_rank, criterion):
    model.train()
    loss_acc = 0
    for epoch in range(train_epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(local_rank)
            target = target.to(local_rank)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss_acc += loss.item() / batch_size
            loss.backward()
            optimizer.step()
        if epoch % log_interval == 0:
            print(f"Train Epoch: {epoch} \tLoss: {loss_acc / len(train_loader)}")


def test(model, test_loader, local_rank, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(local_rank)
            target = target.to(local_rank)
            output = model(data)
            test_loss += criterion(output, target).item()  # size avg?
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epoch", type=int, default=90)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--gpu_ids", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--local_rank", type=int, default=0)
    # ?? debug for python3.9 trial
    # why passed on using local-rank ..?
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--sha1", type=str, default="")
    parser.add_argument("--sha2", type=str, default="")
    return parser


def init_for_distributed(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl", init_method="env://")
    if args.local_rank is not None:
        args.local_rank = local_rank
        print("Use GPU: {} for training".format(args.local_rank))
        torch.cuda.set_device(args.local_rank)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MNIST arg parser", parents=[get_args_parser()])
    args = parser.parse_args()

    # DDP
    init_for_distributed(args)

    # Model
    train_dataset = torchvision.datasets.MNIST(
        "/MNIST/",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
    )

    test_dataset = torchvision.datasets.MNIST(
        "/MNIST/",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
    )
    model = resnet18(10)
    if args.sha1 != "":
        print(f"get model diff between commit {args.sha1} and {args.sha2}")
        get_model_diff(args.sha1, args.sha2)
    else:
        print("log current model")

        # tvault debugging session
        tvault = TorchVault("./logs", "./")
        tvault.diff("c93198d", "b86c619")

        # import tvault

        # tvault.log(model, "./logs", "./")
        # class_log, function_log = analyze_model(model, "./")

        # # get git hash
        # repo = git.Repo(search_parent_directories=True)
        # sha = repo.head.object.hexsha
        # with open(f"logs/model_str_{sha}.txt", "w") as f:
        #     f.write(model.__str__())
        # with open(f"logs/class_def_{sha}.pkl", "wb") as f:
        #     pickle.dump(dict(class_log), f)
        # with open(f"logs/func_def_{sha}.pkl", "wb") as f:
        #     pickle.dump(dict(function_log), f)

    # import ipdb

    # ipdb.set_trace()
    # model = model.to(args.local_rank)
    # model = DDP(model, device_ids=[args.local_rank])
    # criterion = torch.nn.NLLLoss()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # train(model, 20, train_loader, args.local_rank, criterion)
    # if args.local_rank == 0:
    #     test(model, test_loader, args.local_rank, criterion)
