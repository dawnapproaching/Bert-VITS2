"""Initialize server package."""

import pkgutil
import importlib


import importlib
import pkgutil

def import_submodules(package, recursive=True):
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        try:
            results[name] = importlib.import_module(name)
        except Exception as e:
            # 如果想要看到导入错误的具体信息，可以取消注释下面的行
            # print(f"Failed to import {name}: {e}")
            pass
        if recursive and is_pkg:
            results.update(import_submodules(name))
    return results

import_submodules(__name__)