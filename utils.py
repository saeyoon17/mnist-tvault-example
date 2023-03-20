import os
import ast
import glob
from collections import defaultdict


def get_class_defs(model_dir):
    # root_dir needs a trailing slash (i.e. /root/dir/)
    class_defs = defaultdict(lambda: "")
    for filename in glob.iglob(model_dir + "**/*.py", recursive=True):
        with open(filename, "r") as f:
            file_ast = ast.parse(f.read())
        for stmt in file_ast.body:
            if type(stmt) == ast.ClassDef:
                class_defs[filename + ":" + stmt.name] = stmt
    return class_defs


def analyze_model(model, model_dir, torch_dir=None):
    model = model.__str__()
    target_modules = set()
    for line in model.split("\n"):
        if "(" in line:
            if line == line.strip():
                # model classname
                target_module = line.split("(")[0]
            else:
                # submodules
                target_module = line.split("(")[1].split(" ")[-1]
            target_modules.add(target_module)
    print(target_modules)
    print("\n\n")
    class_defs = get_class_defs(model_dir)
    import ipdb

    ipdb.set_trace()
