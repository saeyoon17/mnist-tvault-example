import os
import ast
import glob
import pickle
import difflib
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
    for class_def in class_defs.values():
        # for each body in class definitions,
        # import ipdb

        # ipdb.set_trace()
        for body in class_def.body:
            try:
                # if the function is __init__,
                if body.name == "__init__":
                    init_body = body
                    # for each stmt in init_body,
                    for stmt in init_body.body:
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
    return list(set(target_funcs))


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
    class_defs, function_defs = get_defs(model_dir)
    target_funcs = match_external_funcs(class_defs)

    filter_target_class = defaultdict(lambda: "")
    filter_target_funcs = defaultdict(lambda: "")
    for k, v in class_defs.items():
        if k.split(":")[-1] in target_modules:
            filter_target_class[k] = ast.unparse(v)
    for k, v in function_defs.items():
        if k.split(":")[-1] in target_funcs:
            filter_target_funcs[k] = ast.unparse(v)
    return filter_target_class, filter_target_funcs


def get_model_diff(sha1, sha2):
    with open(f"logs/model_str_{sha1}.txt", "r") as f:
        prev_model = f.readlines()
    with open(f"logs/class_def_{sha1}.pkl", "rb") as f:
        prev_class_def = pickle.load(f)
    with open(f"logs/func_def_{sha1}.pkl", "rb") as f:
        prev_func_def = pickle.load(f)

    with open(f"logs/model_str_{sha2}.txt", "r") as f:
        cur_model = f.readlines()
    with open(f"logs/class_def_{sha2}.pkl", "rb") as f:
        cur_class_def = pickle.load(f)
    with open(f"logs/func_def_{sha2}.pkl", "rb") as f:
        cur_func_def = pickle.load(f)

    # 1. get model diff using string
    print("===== MACRO MODEL DIFF =====")
    model_diff = [e for e in difflib.ndiff(prev_model, cur_model)]
    filter_model_diff = [l for l in model_diff if not l.startswith("? ")]
    print("".join(filter_model_diff))

    # 2. Check module definition between modules
    for p_module, p_source in prev_class_def.items():
        if p_module in cur_class_def.keys():
            class_diff = [
                e for e in difflib.ndiff(p_source.split("\n"), cur_class_def[p_module].split("\n"))
            ]  # generator requires this wrapping
            changes = [l for l in class_diff if l.startswith("+ ") or l.startswith("- ")]
            filter_class_diff = [l for l in class_diff if not l.startswith("? ")]
            if len(changes) > 0:
                print(f"===== CHANGE IN MODULE: {p_module} =====")
                print("\n".join(filter_class_diff))
        else:
            print(f"===== MODULE REMOVED: {p_module} =====")
    for c_module, c_source in cur_class_def.items():
        if c_module not in prev_class_def.keys():
            print(f"===== MODULE ADDED: {p_module} =====")
            print(c_source)

    # 3. Check external function diff
    for p_func, p_source in prev_func_def.items():
        if p_func in cur_func_def.keys():
            func_diff = [
                e for e in difflib.ndiff(p_source.split("\n"), cur_func_def[p_func].split("\n"))
            ]  # generator requires this wrapping
            changes = [l for l in func_diff if l.startswith("+ ") or l.startswith("- ")]
            filter_func_diff = [l for l in func_diff if not l.startswith("? ")]
            if len(changes) > 0:
                print(f"===== CHANGE IN FUNCTION: {p_func} =====")
                print("\n".join(filter_func_diff))
        else:
            print(f"===== FUNCTION REMOVED: {p_func} =====")
    for c_func, c_source in cur_func_def.items():
        if c_func not in prev_func_def.keys():
            print(f"===== FUNCTION ADDED: {p_func} =====")
            print(c_source)
