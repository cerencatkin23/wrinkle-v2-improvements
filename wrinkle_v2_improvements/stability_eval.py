import importlib.util
import os
import sys

_src_path = os.path.join(os.path.dirname(__file__), "..", "src", "wrinkle_v2_improvements", "stability_eval.py")
_src_path = os.path.abspath(_src_path)
_spec = importlib.util.spec_from_file_location("wrinkle_v2_improvements_stability_impl", _src_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

globals().update({k: getattr(_mod, k) for k in dir(_mod) if not k.startswith("_")})

if __name__ == "__main__":
    _mod.main()
