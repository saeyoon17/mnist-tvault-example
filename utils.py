import os
import ast
import glob


def get_class_defs(model_dir):
    # root_dir needs a trailing slash (i.e. /root/dir/)
    class_defs = []
    for filename in glob.iglob(model_dir + "**/*.py", recursive=True):
        with open(filename, "r") as f:
            file_ast = ast.parse(f)
        import ipdb

        ipdb.set_trace()


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
    get_class_defs(model_dir)
