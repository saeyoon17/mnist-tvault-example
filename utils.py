import os
import ast
import glob
from collections import defaultdict


def get_defs(model_dir):
    # root_dir needs a trailing slash (i.e. /root/dir/)
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


def match_external_funcs(class_defs):
    target_funcs = []
    for class_def in class_defs:
        # for each body in class definitions,
        for body in class_def.body:
            try:
                # if the function is __init__,
                if body.name == "__init__":
                    init_body = body
                    # for each stmt in init_body,
                    for stmt in init_body:
                        # if the statement is assign, and its value is function call, and is external
                        if (
                            type(stmt) == ast.Assign
                            and type(stmt.value) == ast.Call
                            and type(stmt.value.func) == ast.Name
                        ):
                            # this is the function we need to track
                            function_name = stmt.value.func.id
                            target_funcs.append(function_name)
                        else:
                            # not into other types
                            pass
            # parsing errors will happen by default
            except:
                pass
    return target_funcs


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
    class_defs, function_defs = get_defs(model_dir)
    target_funcs = match_external_funcs(class_defs)
    import ipdb

    ipdb.set_trace()
