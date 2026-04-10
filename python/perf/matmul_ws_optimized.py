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

BLOCK_SCHEDULE_ROW_MAJOR = _example.BLOCK_SCHEDULE_ROW_MAJOR
BLOCK_SCHEDULE_N_MAJOR = _example.BLOCK_SCHEDULE_N_MAJOR
BLOCK_SCHEDULE_GROUPED_M_4 = _example.BLOCK_SCHEDULE_GROUPED_M_4
BLOCK_SCHEDULE_GROUPED_M_8 = _example.BLOCK_SCHEDULE_GROUPED_M_8
BLOCK_SCHEDULE_GROUPED_M_16 = _example.BLOCK_SCHEDULE_GROUPED_M_16
BLOCK_SCHEDULE_GROUPED_N_4 = _example.BLOCK_SCHEDULE_GROUPED_N_4
BLOCK_SCHEDULE_GROUPED_N_8 = _example.BLOCK_SCHEDULE_GROUPED_N_8
BLOCK_SCHEDULE_GROUPED_N_16 = _example.BLOCK_SCHEDULE_GROUPED_N_16
BLOCK_SCHEDULE_SNAKE_M_4 = _example.BLOCK_SCHEDULE_SNAKE_M_4
BLOCK_SCHEDULE_SNAKE_M_8 = _example.BLOCK_SCHEDULE_SNAKE_M_8
BLOCK_SCHEDULE_SNAKE_M_16 = _example.BLOCK_SCHEDULE_SNAKE_M_16
BLOCK_SCHEDULE_SNAKE_N_4 = _example.BLOCK_SCHEDULE_SNAKE_N_4
BLOCK_SCHEDULE_SNAKE_N_8 = _example.BLOCK_SCHEDULE_SNAKE_N_8
BLOCK_SCHEDULE_SNAKE_N_16 = _example.BLOCK_SCHEDULE_SNAKE_N_16
BLOCK_SCHEDULE_ROW_MAJOR_SWIZZLE_2 = _example.BLOCK_SCHEDULE_ROW_MAJOR_SWIZZLE_2
BLOCK_SCHEDULE_ROW_MAJOR_SWIZZLE_4 = _example.BLOCK_SCHEDULE_ROW_MAJOR_SWIZZLE_4
BLOCK_SCHEDULE_N_MAJOR_SWIZZLE_2 = _example.BLOCK_SCHEDULE_N_MAJOR_SWIZZLE_2
BLOCK_SCHEDULE_N_MAJOR_SWIZZLE_4 = _example.BLOCK_SCHEDULE_N_MAJOR_SWIZZLE_4
BLOCK_SCHEDULE_BAND_N_20_ROW_MAJOR = _example.BLOCK_SCHEDULE_BAND_N_20_ROW_MAJOR
BLOCK_SCHEDULE_STRATEGIES = _example.BLOCK_SCHEDULE_STRATEGIES
BLOCK_SCHEDULE_STRATEGY_NAMES = _example.BLOCK_SCHEDULE_STRATEGY_NAMES
KernelConfig = _example.KernelConfig
get_operand_layout = _example.get_operand_layout
make_operand_descriptor = _example.make_operand_descriptor
ws_matmul_kernel_optimized = _example.ws_matmul_kernel
matmul = _example.matmul

__all__ = [
    "BLOCK_SCHEDULE_ROW_MAJOR",
    "BLOCK_SCHEDULE_N_MAJOR",
    "BLOCK_SCHEDULE_GROUPED_M_4",
    "BLOCK_SCHEDULE_GROUPED_M_8",
    "BLOCK_SCHEDULE_GROUPED_M_16",
    "BLOCK_SCHEDULE_GROUPED_N_4",
    "BLOCK_SCHEDULE_GROUPED_N_8",
    "BLOCK_SCHEDULE_GROUPED_N_16",
    "BLOCK_SCHEDULE_SNAKE_M_4",
    "BLOCK_SCHEDULE_SNAKE_M_8",
    "BLOCK_SCHEDULE_SNAKE_M_16",
    "BLOCK_SCHEDULE_SNAKE_N_4",
    "BLOCK_SCHEDULE_SNAKE_N_8",
    "BLOCK_SCHEDULE_SNAKE_N_16",
    "BLOCK_SCHEDULE_ROW_MAJOR_SWIZZLE_2",
    "BLOCK_SCHEDULE_ROW_MAJOR_SWIZZLE_4",
    "BLOCK_SCHEDULE_N_MAJOR_SWIZZLE_2",
    "BLOCK_SCHEDULE_N_MAJOR_SWIZZLE_4",
    "BLOCK_SCHEDULE_BAND_N_20_ROW_MAJOR",
    "BLOCK_SCHEDULE_STRATEGIES",
    "BLOCK_SCHEDULE_STRATEGY_NAMES",
    "KernelConfig",
    "get_operand_layout",
    "make_operand_descriptor",
    "ws_matmul_kernel_optimized",
    "matmul",
]
