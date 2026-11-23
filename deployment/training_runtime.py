"""Load matching upload-training code across Streamlit hot reruns.

Streamlit reruns page scripts without necessarily invalidating imported modules.
Keep each source version independent so existing runs retain their own globals.
"""
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from types import ModuleType


@lru_cache(maxsize=2)
def _load_source(source, filename):
    module = ModuleType("_solar_upload_training_" + sha256(source).hexdigest())
    module.__file__ = filename
    # Compile the trusted local source directly: timestamp-based bytecode caches
    # can also lag a rapidly edited file of the same size.
    exec(compile(source, filename, "exec"), module.__dict__)
    return module


def load_training_backend():
    """Compatibility entrypoint; all upload dependencies now share one snapshot."""
    from session_runtime import load_session_runtime
    return load_session_runtime().training
