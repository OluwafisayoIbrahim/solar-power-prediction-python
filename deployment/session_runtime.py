"""Coherent, immutable upload-workflow modules for Streamlit hot reruns.

Do not reload shared modules in place: another session may still be training.
Each local source snapshot receives its own import namespace and dependencies.
"""
import builtins
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import sys
from threading import RLock
from types import ModuleType, SimpleNamespace

MODULES = ('feature_mapping', 'model_registry', 'batch_prediction', 'training', 'mapping_ui')
_LOCK = RLock()


@lru_cache(maxsize=4)
def _load_bundle(sources, directory):
    digest = sha256(repr(sources).encode()).hexdigest()
    namespace = '_solar_session_' + digest
    modules = {}
    package = ModuleType(namespace)
    package.__path__ = [directory]
    sys.modules[namespace] = package
    original_import = builtins.__import__
    source_by_name = dict(sources)

    def local_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in source_by_name:
            return load(name)
        return original_import(name, globals, locals, fromlist, level)

    def load(name):
        if name in modules:
            return modules[name]
        qualified = namespace + '.' + name
        module = ModuleType(qualified)
        module.__file__ = str(Path(directory) / (name + '.py'))
        module.__package__ = namespace
        module.__builtins__ = dict(vars(builtins), __import__=local_import)
        # Dataclasses/type resolution require registration before execution.
        modules[name] = module
        sys.modules[qualified] = module
        try:
            exec(compile(source_by_name[name], module.__file__, 'exec'), module.__dict__)
        except BaseException:
            modules.pop(name, None)
            sys.modules.pop(qualified, None)
            raise
        setattr(package, name, module)
        return module

    try:
        for name in MODULES:
            load(name)
    except BaseException:
        for name in modules:
            sys.modules.pop(namespace + '.' + name, None)
        sys.modules.pop(namespace, None)
        raise
    return SimpleNamespace(**modules, version=digest)


def load_session_runtime():
    directory = Path(__file__).resolve().parent
    sources = tuple((name, (directory / (name + '.py')).read_bytes()) for name in MODULES)
    with _LOCK:
        return _load_bundle(sources, str(directory))
