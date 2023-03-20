import os
import ast


def analyze_model(model):
    model = model.__str__()
    target_modules = []
    for line in model.split("\n"):
        if "(" in line:
            if line == line.strip():
                # model classname
                target_module = line.split("(")[0]
            else:
                # submodules
                target_module = line.split("(")[1].split(" ")[-1]
            target_modules.append(target_module)
    print(target_modules)
