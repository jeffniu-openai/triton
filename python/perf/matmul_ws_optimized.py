from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_ws_optimized_example():
    module_name = "triton_examples_gluon_ws_optimized"
    if module_name in sys.modules:
        return sys.modules[module_name]

    path = Path(__file__).resolve().parent.parent / "examples" / "gluon" / "05-matmul-ws-optimized.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_example = _load_ws_optimized_example()

KernelConfig = _example.KernelConfig
get_operand_layout = _example.get_operand_layout
make_operand_descriptor = _example.make_operand_descriptor
ws_matmul_kernel_optimized = _example.ws_matmul_kernel_optimized
matmul = _example.matmul

__all__ = [
    "KernelConfig",
    "get_operand_layout",
    "make_operand_descriptor",
    "ws_matmul_kernel_optimized",
    "matmul",
]
