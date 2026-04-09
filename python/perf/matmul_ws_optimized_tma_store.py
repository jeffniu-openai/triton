from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_ws_optimized_example():
    module_name = "triton_examples_gluon_ws_optimized"
    if module_name in sys.modules:
        return sys.modules[module_name]

    path = Path(__file__).resolve().parent.parent / "examples" / "gluon" / "05-moe-bmm1-fused-gather.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_example = _load_ws_optimized_example()

KernelConfig = _example.KernelConfig
make_ragged_packed_output_descriptor = _example.make_ragged_packed_output_descriptor
ws_matmul_kernel_optimized_tma_store = _example.ws_matmul_kernel_optimized_tma_store
matmul = _example.matmul_tma_store

__all__ = [
    "KernelConfig",
    "make_ragged_packed_output_descriptor",
    "ws_matmul_kernel_optimized_tma_store",
    "matmul",
]
