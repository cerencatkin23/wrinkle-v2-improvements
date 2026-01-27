import os
import sys

src_root = os.path.join(os.path.dirname(__file__), "..", "src")
if src_root not in sys.path:
    sys.path.insert(0, src_root)

__all__ = ["stability_eval"]
