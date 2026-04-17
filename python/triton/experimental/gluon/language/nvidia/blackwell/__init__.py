from __future__ import annotations
from typing import Tuple, List, TYPE_CHECKING

from dataclasses import dataclass, field
from triton.experimental import gluon
from triton.runtime.jit import constexpr_function
from triton.experimental.gluon.language import _core as ttgl
from triton.experimental.gluon.language._core import builtin, base_type, base_value, _unwrap_if_constexpr
from triton.experimental.gluon.language._layouts import DistributedLinearLayout
from triton.experimental.gluon.language._semantic import _compute_tmem_reg_layout, _finalize_splitn_tmem_reg_layout

from . import tma
from . import clc
from ..hopper import fence_async_shared, mbarrier
from ..ampere import async_copy, mma_v2

from triton._C.libtriton import ir
import triton._C.libtriton.gluon_ir as gluon_ir
if TYPE_CHECKING:
    from triton._C.libtriton.gluon_ir import GluonOpBuilder
    from ..._semantic import GluonSemantic

__all__ = [
    "allocate_tensor_memory",
    "async_copy",
    "clc",
    "fence_async_shared",
    "mbarrier",
    "mma_v2",
    "tensor_memory_descriptor",
    "tensor_memory_descriptor_type",
    "TensorMemoryLayout",
    "TensorMemoryLinearLayout",
    "TensorMemoryScalesLayout",
    "tma",
]


def _count_cta_splits(cga_layout):
    return 2**sum(any(x != 0 for x in basis) for basis in cga_layout)


def _check_tensor_memory_layout_ctas(builder, two_ctas, cga_layout=None):
    options = getattr(builder, "options", None)
    if options is None:
        return
    context_ctas = getattr(options, "num_ctas", None)
    if context_ctas is None:
        return
    if cga_layout:
        layout_ctas = _count_cta_splits(cga_layout)
    elif two_ctas:
        layout_ctas = 2
    else:
        layout_ctas = 1
    if layout_ctas != context_ctas:
        raise ValueError(
            f"Layout has {layout_ctas} CTAs per CGA, but the context requires {context_ctas} CTAs per CGA."
        )


def _target_compute_capability(builder):
    options = getattr(builder, "options", None)
    if options is None or getattr(options, "backend_name", None) != "cuda":
        return None
    arch = getattr(options, "arch", None)
    if isinstance(arch, str) and arch.startswith("sm") and arch[2:].isdigit():
        return int(arch[2:])
    return None


def _format_sm_name(capability):
    suffix = "a" if capability >= 90 else ""
    return f"sm_{capability}{suffix}"


def _strip_zero_reg_bases_from_layout(layout):
    if not hasattr(layout, "reg_bases") or not hasattr(layout, "lane_bases") or not hasattr(layout, "warp_bases"):
        return layout
    reg_bases = [basis for basis in layout.reg_bases if any(value != 0 for value in basis)]
    if len(reg_bases) == len(layout.reg_bases):
        return layout
    return type(layout)(
        reg_bases=reg_bases,
        lane_bases=layout.lane_bases,
        warp_bases=layout.warp_bases,
        block_bases=layout.block_bases,
        shape=layout.shape,
    )


def _is_4x256b_refresh_tmem_layout(layout, element_bitwidth, shape):
    if not isinstance(layout, TensorMemoryLinearLayout):
        return False
    if element_bitwidth != 32:
        return False
    rows = [list(basis) for basis in layout.rows]
    cols = [list(basis) for basis in layout.cols]
    if rows != [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 2]]:
        return False
    if cols != [[1, 0], [2, 0], [0, 4]]:
        return False
    if list(shape) == [4, 8]:
        return list(layout.shape) == [4, 8] and not layout.two_ctas and not layout.block_bases
    if list(shape) == [8, 8]:
        return list(layout.shape) == [8, 8] and layout.two_ctas and layout.block_bases == [[4, 0]]
    return False


def _is_4x256b_refresh_physical_bitcast_tmem_layout(layout, element_bitwidth, shape):
    if not isinstance(layout, TensorMemoryLinearLayout):
        return False
    if element_bitwidth != 8:
        return False
    if list(shape) != [32, 4] or list(layout.shape) != [32, 4]:
        return False
    if layout.two_ctas or layout.block_bases:
        return False
    rows = [list(basis) for basis in layout.rows]
    cols = [list(basis) for basis in layout.cols]
    return rows == [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0]] and cols == [
        [0, 1],
        [0, 2],
        [8, 0],
        [16, 0],
        [4, 0],
    ]


def _is_unsupported_4x256b_refresh_tmem_ldst_layout(layout, element_bitwidth, shape):
    return _is_4x256b_refresh_tmem_layout(
        layout, element_bitwidth, shape
    ) or _is_4x256b_refresh_physical_bitcast_tmem_layout(layout, element_bitwidth, shape)


def _raise_unsupported_4x256b_refresh_tmem_ldst(op_name):
    raise ValueError(
        f"direct TMEM {op_name} is unsupported for the tcgen05.copy.4x256b "
        "refresh-shaped TensorMemoryLinearLayout or its raw physical bitcast. "
        "tcgen05.ld/st packets require TMEM row anchors to be materializable as "
        "warp bases and read whole row footprints, but this refresh image stores "
        "logical row bits in TMEM columns and low logical column bits in sparse "
        "TMEM rows 32/64 without a lane mask. Use tcgen05_copy from shared "
        "memory for this refresh image, or access a directly supported 128-row "
        "physical layout."
    )


def _fold_canonical_single_cta_block_rows(rows, block_bases, shape, two_ctas):
    if two_ctas or not block_bases or len(shape) != 2:
        return rows, block_bases
    folded_rows = [list(basis) for basis in block_bases] + [list(basis) for basis in rows]
    if (
        shape[0] <= 0
        or shape[0] & (shape[0] - 1)
        or len(folded_rows) != shape[0].bit_length() - 1
    ):
        return rows, block_bases
    for idx, basis in enumerate(folded_rows):
        expected = [0] * len(shape)
        expected[0] = 1 << idx
        if list(basis) != expected:
            return rows, block_bases
    return folded_rows, []


@gluon.jit
def _reduce_min_direct(a, b):
    return ttgl.minimum(a, b)


@gluon.jit
def _reduce_max_direct(a, b):
    return ttgl.maximum(a, b)


@gluon.jit
def _reduce_min_propagate_nan(a, b):
    return ttgl.where(a != a, a, ttgl.where(b != b, b, ttgl.minimum(a, b)))


@gluon.jit
def _reduce_max_propagate_nan(a, b):
    return ttgl.where(a != a, a, ttgl.where(b != b, b, ttgl.maximum(a, b)))


@dataclass(frozen=True, eq=True)
class TensorMemoryLayout:
    """
    Describes the layout for tensor memory in Blackwell architecture.

    Args:
        block (Tuple[int, int]): Number of contiguous elements per row / column in a CTA.
        col_stride (int): Number of 32-bit columns to advance between logically
            adjacent columns. Packed layouts use a stride of 1. Unpacked
            layouts use ``32 / bitwidth``.
        cga_layout (Optional[List[List[int]]]): CGA layout bases. Defaults to [].
        two_ctas (bool): Whether the layout is for two-CTA mode. Defaults to False.
    """
    block: Tuple[int, int]
    col_stride: int
    cga_layout: List[List[int]] = field(default_factory=list)
    two_ctas: bool = False

    def __post_init__(self):
        super().__setattr__("block", _unwrap_if_constexpr(self.block))
        super().__setattr__("col_stride", _unwrap_if_constexpr(self.col_stride))
        super().__setattr__("cga_layout", _unwrap_if_constexpr(self.cga_layout))
        super().__setattr__("two_ctas", _unwrap_if_constexpr(self.two_ctas))
        assert len(self.block) == 2
        assert all(len(basis) == 2 for basis in self.cga_layout)
        assert self.col_stride >= 1 and (self.col_stride &
                                         (self.col_stride - 1)) == 0, "tensor memory col_stride must be a power of two"

    def _to_ir(self, builder):
        _check_tensor_memory_layout_ctas(builder, self.two_ctas, self.cga_layout)
        return builder.get_tensor_memory_layout(
            self.block,
            self.col_stride,
            [list(basis) for basis in self.cga_layout],
            self.two_ctas,
        )

    def mangle(self) -> str:
        block_str = f"{self.block[0]}x{self.block[1]}"
        stride_str = f"C{self.col_stride}"
        cga_layout_str = "_".join("~".join(map(str, basis)) for basis in self.cga_layout)
        two_ctas_str = "2CT" if self.two_ctas else ""
        return f"TL{block_str}{stride_str}{cga_layout_str}{two_ctas_str}TL"

    def __hash__(self):
        return hash((tuple(self.block), self.col_stride, tuple(tuple(b) for b in self.cga_layout), self.two_ctas))


@dataclass(frozen=True, eq=True)
class TensorMemoryScalesLayout:
    """
    Describes the layout for tensor memory scales in Blackwell architecture.

    Args:
        cga_layout (Optional[List[List[int]]]): CGA layout bases. Defaults to [].
    """
    cga_layout: List[List[int]] = field(default_factory=list)

    def __post_init__(self):
        super().__setattr__("cga_layout", _unwrap_if_constexpr(self.cga_layout))
        assert all(len(basis) == 2 for basis in self.cga_layout)

    def _to_ir(self, builder):
        return builder.get_tensor_memory_scales_layout([list(basis) for basis in self.cga_layout])

    def mangle(self) -> str:
        cga_layout_str = "_".join("~".join(map(str, basis)) for basis in self.cga_layout)
        return f"TLS{cga_layout_str}TLS"

    def __hash__(self):
        return hash(tuple(tuple(b) for b in self.cga_layout))


@dataclass(frozen=True, eq=True)
class TensorMemoryLinearLayout:
    """
    Canonical tensor-memory layout expressed as a linear mapping from TMEM
    coordinates to logical tensor dimensions.

    Args:
        rows (List[List[int]]): Bases for the TMEM row input dimension.
        cols (List[List[int]]): Bases for the TMEM col input dimension.
        shape (List[int]): Logical tensor shape (`dim0`, `dim1`, ...).
        block_bases (Optional[List[List[int]]]): Bases for optional TMEM block
            input dimension. Defaults to [].
        two_ctas (bool): Whether layout is intended for two-CTA mode.
    """
    rows: List[List[int]]
    cols: List[List[int]]
    shape: List[int]
    block_bases: List[List[int]] = field(default_factory=list)
    two_ctas: bool = False

    def __post_init__(self):
        super().__setattr__("rows", _unwrap_if_constexpr(self.rows))
        super().__setattr__("cols", _unwrap_if_constexpr(self.cols))
        super().__setattr__("shape", _unwrap_if_constexpr(self.shape))
        super().__setattr__("block_bases", _unwrap_if_constexpr(self.block_bases))
        super().__setattr__("two_ctas", _unwrap_if_constexpr(self.two_ctas))
        rank = len(self.shape)
        for in_dim, bases in [("row", self.rows), ("col", self.cols), ("block", self.block_bases)]:
            for basis in bases:
                if len(basis) != rank:
                    raise ValueError(
                        f"Invalid basis rank {len(basis)} for in-dim '{in_dim}'. "
                        f"Expected {rank} entries to match the tensor rank."
                    )
                for idx, value in enumerate(basis):
                    if value < 0 or value >= self.shape[idx]:
                        raise ValueError(
                            f"Invalid basis {value} for in-dim '{in_dim}' and "
                            f"out-dim 'dim{idx}'. Basis must be non-negative and "
                            f"less than the out-dim size."
                        )

    def _to_ir(self, builder):
        _check_tensor_memory_layout_ctas(builder, self.two_ctas)
        rows, block_bases = _fold_canonical_single_cta_block_rows(
            self.rows, self.block_bases, self.shape, self.two_ctas
        )
        return builder.get_tensor_memory_linear_layout(
            rows,
            self.cols,
            block_bases,
            self.shape,
            self.two_ctas,
        )

    def mangle(self):
        return f"TMLL_{self.rows}_{self.cols}_{self.block_bases}_{self.shape}_{self.two_ctas}_TMLL"

    def __hash__(self):
        return hash((
            tuple(map(tuple, self.rows)),
            tuple(map(tuple, self.cols)),
            tuple(map(tuple, self.block_bases)),
            tuple(self.shape),
            self.two_ctas,
        ))


# Backward-compatibility alias for previous private name.
_TensorMemoryLinearLayout = TensorMemoryLinearLayout


def _unwrap_tmem_layout_arg(x):
    if isinstance(x, ttgl.constexpr):
        return _unwrap_tmem_layout_arg(x.value)
    if isinstance(x, list):
        return [_unwrap_tmem_layout_arg(i) for i in x]
    if isinstance(x, tuple):
        return tuple(_unwrap_tmem_layout_arg(i) for i in x)
    return x


class tensor_memory_descriptor_type(base_type):

    def __init__(self, element_ty, shape, layout, alloc_shape, ir_type=None):
        self.element_ty = _unwrap_if_constexpr(element_ty)
        self.shape = _unwrap_if_constexpr(shape)
        self.layout = _unwrap_if_constexpr(layout)
        self.alloc_shape = _unwrap_if_constexpr(alloc_shape)
        self.ir_type = ir_type
        assert isinstance(self.layout, (TensorMemoryLayout, TensorMemoryLinearLayout, TensorMemoryScalesLayout))

    def to_ir(self, builder: GluonOpBuilder) -> None:
        if self.ir_type is not None:
            return self.ir_type
        return builder.get_tensor_mem_desc_ty(
            self.element_ty.to_ir(builder),
            self.shape,
            self.layout._to_ir(builder),
            self.alloc_shape,
        )

    def _unflatten_ir(self, handles: List[ir.Value], cursor: int) -> Tuple[tensor_memory_descriptor, int]:
        value = tensor_memory_descriptor(
            handles[cursor],
            self.element_ty,
            self.shape,
            self.layout,
            self.alloc_shape,
            ir_type=handles[cursor].get_type(),
        )
        return value, cursor + 1

    def _flatten_ir_types(self, builder: GluonOpBuilder, out: List[ir.type]) -> None:
        out.append(self.to_ir(builder))

    def __str__(self) -> str:
        return f"tensor_memory_descriptor<{self.element_ty}, {self.shape}, {self.layout}>"

    def __eq__(self, other) -> bool:
        return (type(self) is type(other) and self.shape == other.shape and self.layout == other.layout
                and self.alloc_shape == other.alloc_shape)

    def __neq__(self, other) -> bool:
        return not (self == other)

    def mangle(self) -> str:
        shape_str = "_".join([str(s) for s in self.shape])
        return f"MD{self.element_ty.mangle()}S{shape_str}SL{self.layout.mangle()}LAS{self.alloc_shape}ASMD"

    @constexpr_function
    def get_reg_layout(self, num_warps=None, instr_variant="auto"):
        """
        Return a DistributedLinearLayout compatible with TMEM load/store
        instructions for this descriptor type.

        Args:
            num_warps (Optional[int]): Number of warps participating in the
                operation. Must be provided when it cannot be inferred by the
                caller.
            instr_variant (str): TMEM instruction variant (e.g. ``"auto"``,
                ``"32x32b"``). ``"auto"`` selects the first supported TMEM
                register layout for the descriptor.
        """
        tmem_ty = _unwrap_tmem_layout_arg(self)
        if not isinstance(tmem_ty, tensor_memory_descriptor_type):
            raise TypeError(f"expected a tensor_memory_descriptor_type but got {type(tmem_ty)!r}")
        num_warps = _unwrap_tmem_layout_arg(num_warps)
        if num_warps is None:
            raise ValueError("num_warps could not be inferred; pass a positive power of two")
        if not isinstance(num_warps, int) or num_warps <= 0 or (num_warps & (num_warps - 1)) != 0:
            raise ValueError(f"num_warps must be a positive power of two, got {num_warps!r}")
        if _is_unsupported_4x256b_refresh_tmem_ldst_layout(
            tmem_ty.layout,
            _unwrap_tmem_layout_arg(tmem_ty.element_ty).primitive_bitwidth,
            _unwrap_tmem_layout_arg(tmem_ty.shape),
        ):
            _raise_unsupported_4x256b_refresh_tmem_ldst("register layout query")

        layout = _compute_tmem_reg_layout(
            _unwrap_tmem_layout_arg(tmem_ty.element_ty),
            _unwrap_tmem_layout_arg(tmem_ty.shape),
            _unwrap_tmem_layout_arg(tmem_ty.alloc_shape),
            _unwrap_tmem_layout_arg(tmem_ty.layout),
            num_warps,
            _unwrap_tmem_layout_arg(instr_variant),
        )
        if isinstance(tmem_ty.layout, TensorMemoryScalesLayout):
            layout = _strip_zero_reg_bases_from_layout(layout)
        return layout


class tensor_memory_descriptor(base_value):
    """
    Represents a tensor memory descriptor handle for Tensor Core Gen5 operations.
    """

    def __init__(self, handle, element_ty, shape, layout, alloc_shape, ir_type=None):
        self.handle = handle
        self.type = tensor_memory_descriptor_type(element_ty, shape, layout, alloc_shape, ir_type=ir_type)

    def _set_name(self, builder: ir.builder, name: str) -> None:
        self.handle.set_loc(builder.create_name_loc(name, self.handle.get_loc()))

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        handles.append(self.handle)

    @property
    def dtype(self):
        return self.type.element_ty

    @property
    def shape(self):
        return self.type.shape

    @property
    def rank(self):
        return len(self.shape)

    @property
    def layout(self):
        return self.type.layout

    def __str__(self) -> str:
        return str(self.type)

    def _require_rank2_tmem_ldst(self, op_name: str) -> None:
        if len(self.shape) != 2:
            raise ValueError(
                f"direct TMEM {op_name} requires a rank-2 descriptor view; "
                "index, slice, or reshape higher-rank TMEM descriptors to a 2D "
                "view before calling get_reg_layout(), load(), or store()."
            )

    @builtin
    def get_reg_layout(self, num_warps=None, instr_variant="auto", _semantic: GluonSemantic = None, _generator=None):
        """
        Return the register layout used to access this tensor memory descriptor.

        Args:
            num_warps (Optional[int]): Number of warps participating in the
                operation. When omitted, infer it from the caller's context.
            instr_variant (str): TMEM instruction variant. Defaults to
                ``"auto"``, which selects the first supported TMEM register
                layout for the descriptor.

        Returns:
            DistributedLayout: A register layout compatible with TMEM
            load/store instructions for this descriptor.
        """
        if num_warps is None:
            num_warps = ttgl.num_warps(_semantic=_semantic, _generator=_generator)
        num_warps = _unwrap_if_constexpr(num_warps)
        requested_variant = _unwrap_if_constexpr(instr_variant)
        self._require_rank2_tmem_ldst(f"{requested_variant} register layout query")
        try:
            layout = gluon_ir.compute_tmem_reg_layout_from_memdesc(
                self.handle, num_warps, requested_variant
            )
        except Exception as e:
            raise ValueError(str(e)) from e
        if layout is not None and requested_variant in ("32x32b_splitn", "16x32bx2"):
            layout = _finalize_splitn_tmem_reg_layout(
                layout,
                self.dtype,
                list(self.shape),
                list(self.type.alloc_shape),
                self.layout,
                num_warps,
                requested_variant,
                isinstance(self.layout, TensorMemoryScalesLayout),
            )
        if isinstance(self.layout, TensorMemoryScalesLayout):
            layout = _strip_zero_reg_bases_from_layout(layout)
        if layout is None:
            reason = gluon_ir.get_tmem_ldst_unsupported_reason_from_memdesc_for_variant(
                self.handle, num_warps, requested_variant
            )
            if reason is None:
                reason = gluon_ir.get_tmem_ldst_unsupported_reason_from_memdesc(self.handle)
            if reason is not None:
                raise ValueError(
                    f"TMEM layout '{instr_variant}' unsupported for descriptor view {self.type}. {reason}"
                )
            raise ValueError(f"TMEM layout '{instr_variant}' unsupported for descriptor view {self.type}")
        return layout

    @builtin
    def load(self, layout=None, _semantic: GluonSemantic = None, _generator=None) -> ttgl.tensor:
        """
        Load a tensor from tensor memory.

        Args:
            layout (Optional[DistributedLayout]): Destination layout of the tensor.
                When omitted, infer the default TMEM register layout for this
                descriptor type and the caller's warp count.

        Returns:
            tensor: A distributed tensor containing the loaded data.
        """
        self._require_rank2_tmem_ldst("load")
        if layout is None:
            num_warps = ttgl.num_warps(_semantic=_semantic, _generator=_generator)
            layout = self.get_reg_layout(
                num_warps=num_warps,
                _semantic=_semantic,
                _generator=_generator,
            )
        layout = _unwrap_if_constexpr(layout)
        ret_ty = ttgl.distributed_type(self.dtype, self.shape, layout)
        builder = _semantic.builder
        handle = builder.create_tmem_load(ret_ty.to_ir(builder), self.handle)
        return ttgl.tensor(handle, ret_ty)

    def _load_red(self, layout, red_op, abs, propagate_nan, _semantic: GluonSemantic, _generator=None):
        #   red_op: MIN/MAX reduction operation
        #   abs (bool): If True, reduce absolute values.
        #   propagate_nan (NONE): If ALL, propagate NaN in specified reduction operation.
        self._require_rank2_tmem_ldst("reduction load")
        abs_flag = _unwrap_if_constexpr(abs)
        propagate_nan = _unwrap_if_constexpr(propagate_nan)
        if layout is None:
            num_warps = ttgl.num_warps(_semantic=_semantic, _generator=_generator)
            raw_layout = _unwrap_if_constexpr(self.layout)
            if self.dtype.primitive_bitwidth == 32 and not isinstance(raw_layout, TensorMemoryScalesLayout):
                layout = gluon_ir.compute_tmem_reduce_reg_layout_from_memdesc(
                    self.handle, num_warps
                )
            if layout is None:
                try:
                    layout = self.get_reg_layout(
                        num_warps=num_warps,
                        instr_variant="32x32b",
                        _semantic=_semantic,
                        _generator=_generator,
                    )
                except Exception as e:
                    raise ValueError(str(e)) from e
        layout = _unwrap_if_constexpr(layout)
        ret_ty = ttgl.distributed_type(self.dtype, self.shape, layout)
        builder = _semantic.builder
        num_warps = builder.options.num_warps

        result, reduced, red_layout = builder.create_tmem_load(ret_ty.to_ir(builder), self.handle, red_op, abs_flag,
                                                               propagate_nan, num_warps)

        red_shape = [self.shape[0]]  # [M] for [M,N] input
        red_ty = ttgl.distributed_type(self.dtype, red_shape, red_layout)

        return (ttgl.tensor(result, ret_ty), ttgl.tensor(reduced, red_ty))

    @builtin
    def load_min(self, layout=None, abs=False, propagate_nan=ir.PROPAGATE_NAN.NONE, _semantic: GluonSemantic = None,
                 _generator=None):
        """
        Load a tensor from tensor memory with MIN reduction along the N-dimension.

        Args:
            layout (Optional[DistributedLayout]): Destination layout of the tensor.
                When omitted, infer the default TMEM register layout for this
                descriptor type and the caller's warp count.
            abs (bool): If True, reduce absolute values. Defaults to False.
            propagate_nan (PROPAGATE_NAN): If ALL, propagate NaN in the reduction operation. Defaults to NONE.

        Returns:
            tuple: A tuple containing (tensor, reduced_tensor) where tensor is the loaded data
                   and reduced_tensor is the result of MIN reduction along the N-dimension of loaded data
        """
        return self._load_red(layout, gluon_ir.TMEM_LOAD_REDUCE_MODIFIER.MIN, abs, propagate_nan, _semantic, _generator)

    @builtin
    def load_max(self, layout=None, abs=False, propagate_nan=ir.PROPAGATE_NAN.NONE, _semantic: GluonSemantic = None,
                 _generator=None):
        """
        Load a tensor from tensor memory with MAX reduction along the N-dimension.

        Args:
            layout (Optional[DistributedLayout]): Destination layout of the tensor.
                When omitted, infer the default TMEM register layout for this
                descriptor type and the caller's warp count.
            abs (bool): If True, reduce absolute values. Defaults to False.
            propagate_nan (PROPAGATE_NAN): If ALL, propagate NaN in the reduction operation. Defaults to NONE.

        Returns:
            tuple: A tuple containing (tensor, reduced_tensor) where tensor is the loaded data
                   and reduced_tensor is the result of MAX reduction along the N-dimension of loaded data.
        """
        return self._load_red(layout, gluon_ir.TMEM_LOAD_REDUCE_MODIFIER.MAX, abs, propagate_nan, _semantic, _generator)

    @builtin
    def store(self, value, pred=True, _semantic: GluonSemantic = None) -> None:
        """
        Store a tensor into tensor memory.

        Args:
            value (tensor): The tensor to store.
            pred (bool): Scalar predicate. Operation is skipped if predicate is False. Defaults to True.
        """
        self._require_rank2_tmem_ldst("store")
        pred = _unwrap_if_constexpr(pred)
        pred = _semantic.to_tensor(pred)
        assert value.shape == self.shape, f"source shape {value.shape} does not match destination shape {self.shape}"
        assert value.dtype == self.dtype, f"source dtype {value.dtype} does not match destination dtype {self.dtype}"
        _semantic.builder.create_tmem_store(self.handle, value.handle, pred.handle)

    @builtin
    def slice(self, start, length, dim=None, _semantic: GluonSemantic = None) -> tensor_memory_descriptor:
        """
        Create a subview of tensor memory by slicing along a given dimension.

        Args:
            start (int): The starting index of the slice.
            length (int): The length of the slice.
            dim (Optional[int]): The dimension to slice. When omitted, preserve
                the historical TMEM shorthand and slice the trailing dimension.

        Returns:
            tensor_memory_descriptor: Descriptor for the sliced subview.
        """
        start = _unwrap_if_constexpr(start)
        length = _unwrap_if_constexpr(length)
        dim = _unwrap_if_constexpr(dim)
        if dim is None:
            dim = self.rank - 1
        return _semantic.memdesc_slice(self, start, length, dim)

    @builtin
    def index(self, index, _semantic: GluonSemantic = None) -> tensor_memory_descriptor:
        """
        Create a subview of tensor memory by indexing the first dimension.

        Args:
            index (tensor): The index tensor for the subview.

        Returns:
            tensor_memory_descriptor: Descriptor for the indexed subview.
        """
        index = _unwrap_if_constexpr(index)
        return _semantic.memdesc_index(self, index)

    @builtin
    def permute(self, order, _semantic: GluonSemantic = None) -> tensor_memory_descriptor:
        """
        Permute the dimensions of the tensor memory descriptor.

        Args:
            order (List[int]): The new ordering of dimensions.

        Returns:
            tensor_memory_descriptor: Descriptor with permuted dimensions.
        """
        order = [_unwrap_if_constexpr(o) for o in order]
        return _semantic.memdesc_trans(self, order)

    @builtin
    def reshape(self, shape, _semantic: GluonSemantic = None) -> tensor_memory_descriptor:
        """
        Reshape the tensor memory descriptor to a new shape and layout.

        Args:
            shape (List[int]): The target shape.

        Returns:
            tensor_memory_descriptor: Descriptor with the new shape and layout.
        """
        shape = [_unwrap_if_constexpr(s) for s in shape]
        return _semantic.memdesc_reshape(self, shape)

    @builtin
    def _reinterpret(self, dtype, shape, layout, _semantic: GluonSemantic = None) -> tensor_memory_descriptor:
        """
        Reinterpret tensor memory descriptor with a new dtype, shape, and layout.

        Args:
            dtype (dtype): The new data type.
            shape (Sequence[int]): The new shape.
            layout (TensorMemoryLayout): The new layout.

        Returns:
            tensor_memory_descriptor: Descriptor with updated type and layout.
        """
        dtype = _unwrap_if_constexpr(dtype)
        shape = [_unwrap_if_constexpr(s) for s in shape]
        layout = _unwrap_if_constexpr(layout)
        return _semantic.memdesc_reinterpret(self, dtype, shape, layout)

    @builtin
    def bitcast(self, dtype, shape, layout=None, _semantic: GluonSemantic = None) -> tensor_memory_descriptor:
        """
        Bitcast a tensor memory descriptor without changing its physical TMEM
        mapping.

        The source descriptor should already identify the physical TMEM region
        to view. The bitcast preserves the total number of bits and derives the
        result layout from the source descriptor's physical mapping unless an
        explicit equivalent tensor memory layout is provided.

        Args:
            dtype (dtype): The new data type.
            shape (Sequence[int]): The new shape.
            layout (Optional[TensorMemoryLayout]): Explicit result layout.

        Returns:
            tensor_memory_descriptor: Descriptor with updated type, shape, and
            tensor-memory layout.
        """
        dtype = _unwrap_if_constexpr(dtype)
        shape = [_unwrap_if_constexpr(s) for s in shape]
        layout = _unwrap_if_constexpr(layout)
        return _semantic.tmem_memdesc_bitcast(self, dtype, shape, layout)


@builtin
def allocate_tensor_memory(element_ty, shape, layout, value=None, _semantic=None):
    """
    Allocate tensor memory.

    Args:
        element_ty (dtype): The element data type.
        shape (Sequence[int]): The descriptor shape.
        layout (TensorMemoryLayout): The layout of the tensor memory.
        value (tensor, optional): Initial tensor to copy. Defaults to None.

    Returns:
        tensor_memory_descriptor: Descriptor for the allocated memory.
    """
    element_ty = _unwrap_if_constexpr(element_ty)
    shape = _unwrap_if_constexpr(shape)
    layout = _unwrap_if_constexpr(layout)
    value = value.handle if value is not None else None

    builder = _semantic.builder
    alloc_shape = shape
    if isinstance(layout, (TensorMemoryLinearLayout, TensorMemoryScalesLayout)):
        alloc_shape = builder.get_tmem_alloc_shape(shape, layout._to_ir(builder))
    ty = tensor_memory_descriptor_type(element_ty, shape, layout, alloc_shape)
    handle = builder.create_tmem_alloc(ty.to_ir(builder), value)
    return tensor_memory_descriptor(handle, element_ty, shape, layout, alloc_shape)


@builtin
def tcgen05_copy(src, dst, _semantic=None):
    """
    Start an asynchronous copy from shared memory to tensor memory.

    Args:
        src (shared_memory_descriptor): Shared memory to copy from.
        dst (tensor_memory_descriptor): Tensor memory to copy to.
    """
    assert isinstance(src, ttgl.shared_memory_descriptor), "source must be a shared memory descriptor"
    assert isinstance(dst, tensor_memory_descriptor), "destination must be a tensor memory descriptor"
    _semantic.builder.create_tmem_copy(src.handle, dst.handle)


@builtin
def tcgen05_mma(a, b, acc, *, use_acc=True, pred=True, multicast=False, mbarriers=None, mbarrier_preds=None,
                _semantic=None):
    """
    Emit an asynchronous 5th generation TensorCore MMA instruction.
    acc = a * b + (acc if use_acc else 0)

    Args:
        a (shared_memory_descriptor or tensor_memory_descriptor): Left hand side operand in shared or tensor memory.
        b (shared_memory_descriptor): Right hand side operand in shared memory.
        acc (tensor_memory_descriptor): Accumulator value in tensor memory (mutated).
        use_acc (bool): Whether to use the initial value of the accumulator. Defaults to True.
        pred (bool): Scalar predicate. Operation is skipped if predicate is False. Defaults to True.
        multicast (bool): Whether tcgen05 commit should multicast across a CTA cluster. Defaults to False.
        mbarriers (Sequence[shared_memory_descriptor], optional): Barriers to signal when the operation is complete. If omitted, the operation still executes asynchronously and must be synchronized later via tcgen05_commit and mbarrier.wait before reading the result. Defaults to None.
        mbarrier_preds (Sequence[bool], optional): Predicates for barriers. Defaults to None.
    """
    capability = _target_compute_capability(_semantic.builder)
    if (capability is not None and capability >= 103 and a.dtype in (ttgl.int8, ttgl.uint8) and
            b.dtype in (ttgl.int8, ttgl.uint8) and acc.dtype == ttgl.int32):
        raise ValueError(
            f"direct tcgen05_mma kind::i8 is not supported on {_format_sm_name(capability)} by current Blackwell "
            "lowering. Use tt.dot so the compiler can select a supported MMA version."
        )

    use_acc = _semantic.to_tensor(use_acc)
    pred = _semantic.to_tensor(pred)

    if mbarriers is None:
        assert mbarrier_preds is None
        mbarriers = []
        mbarrier_preds = []
    else:
        mbarriers = [bar.handle for bar in mbarriers]
        if mbarrier_preds is None:
            true = _semantic.to_tensor(True)
            mbarrier_preds = [true.handle] * len(mbarriers)
        else:
            mbarrier_preds = _semantic._convert_to_ir_values(mbarrier_preds, require_i64=False)

    multicast = _unwrap_if_constexpr(multicast)
    _semantic.builder.create_tcgen05_mma(a.handle, b.handle, acc.handle, use_acc.handle, pred.handle, mbarriers,
                                         mbarrier_preds, acc.layout.two_ctas, multicast)


@builtin
def tcgen05_mma_scaled(a, b, acc, a_scale, b_scale, a_type, b_type, *, use_acc=True, pred=True, mbarriers=None,
                       mbarrier_preds=None, _semantic=None):
    """
    Emit an asynchronous 5th generation TensorCore MMA scaled instruction.
    acc = (a * a_scale) * (b * b_scale) + (acc if use_acc else 0)

    Args:
        a (shared_memory_descriptor): Left hand side operand in shared memory.
        b (shared_memory_descriptor or tensor_memory_descriptor): Right hand side operand in shared or tensor memory.
        acc (tensor_memory_descriptor): Accumulator value in tensor memory (mutated).
        a_scale (tensor): Scale factor for operand A.
        b_scale (tensor): Scale factor for operand B.
        a_type (str): Type of operand A. One of {"e2m1", "e4m3", "e5m2"}.
        b_type (str): Type of operand B. One of {"e2m1", "e4m3", "e5m2"}.
        use_acc (bool): Whether to use the initial value of the accumulator. Defaults to True.
        pred (bool): Scalar predicate. Operation is skipped if predicate is False. Defaults to True.
        mbarriers (Sequence[mbarrier], optional): Barriers to signal when the operation is complete. If omitted, the operation still executes asynchronously and must be synchronized later via tcgen05_commit and mbarrier.wait before reading the result. Defaults to None.
        mbarrier_preds (Sequence[bool], optional): Predicates for barriers. Defaults to None.
    """
    use_acc = _semantic.to_tensor(use_acc)
    pred = _semantic.to_tensor(pred)
    if isinstance(acc.type.layout, TensorMemoryLayout):
        assert acc.type.layout.block[0] != 64, "tcgen05_mma_scaled does not support blockM=64"

    if mbarriers is None:
        assert mbarrier_preds is None
        mbarriers = []
        mbarrier_preds = []
    else:
        mbarriers = [bar.handle for bar in mbarriers]
        if mbarrier_preds is None:
            true = _semantic.to_tensor(True)
            mbarrier_preds = [true.handle] * len(mbarriers)
        else:
            mbarrier_preds = _semantic._convert_to_ir_values(mbarrier_preds, require_i64=False)

    allowed_formats = {"e2m1", "e4m3", "e5m2"}
    assert a_type.value in allowed_formats, f"Unsupported lhs_format: {a_type.value}"
    assert b_type.value in allowed_formats, f"Unsupported rhs_format: {b_type.value}"
    a_type = _semantic._str_to_fp_type(a_type.value)
    b_type = _semantic._str_to_fp_type(b_type.value)
    _semantic.builder.create_tcgen05_mma_scaled(a.handle, b.handle, acc.handle, a_scale.handle, b_scale.handle, a_type,
                                                b_type, use_acc.handle, pred.handle, mbarriers, mbarrier_preds,
                                                acc.layout.two_ctas)


@constexpr_function
def tcgen05_mma_barrier_count(smems, multicast):
    """
    Calculate the number of CTAs that will commit the tcgen05 MMA instruction.

    Args:
        smems (Sequence[shared_memory_descriptor]): Shared memory descriptors used in the tcgen05 instruction.
        multicast (bool): Whether the tcgen05 instruction is multicast.

    Returns:
        int: The number of CTAs that will commit the tcgen05 MMA instruction.
    """
    assert 0 <= len(smems) <= 2, "tcgen05_mma_barrier_count supports 0, 1, or 2 smem descriptors"
    if not smems or not multicast:
        return 1

    def basis_is_zero(basis):
        return all(b == 0 for b in basis)

    def num_broadcast_bits(smem):
        return sum(basis_is_zero(basis) for basis in smem.layout.cga_layout)

    if len(smems) == 1:
        return 2**num_broadcast_bits(smems[0])

    assert len(smems) == 2
    num_broadcast_bits_a = num_broadcast_bits(smems[0])
    num_broadcast_bits_b = num_broadcast_bits(smems[1])
    # Assert that for every basis, at least one of them is non-zero
    # so that the inclusion-exclusion principle below works
    # This can be generalised if needed by substracting below 2**size_intersection
    for i in range(len(smems[0].layout.cga_layout)):
        assert not basis_is_zero(smems[0].layout.cga_layout[i]) or not basis_is_zero(smems[1].layout.cga_layout[i])

    # Inclusion-exclusion
    num_cta_commits = 2**num_broadcast_bits_a + 2**num_broadcast_bits_b - 1
    return num_cta_commits


@builtin
def tcgen05_commit(barrier, pred=True, descs=(), _semantic=None):
    """
    This instruction causes the provided mbarrier to be arrived-on with a count
    of 1 when all async tcgen05 MMA and copy instructions previously issued by
    the thread are complete.

    If `descs` are provided, the commit will be multicast across the CTA cluster
    based on the shared layouts of those descriptors. This should be used when
    the inputs to the tcgen5 MMA come from TMA descriptors using multicast.

    Args:
        barrier (shared_memory_descriptor): The barrier to track completion of tcgen05 MMA and copy instructions.
        pred (bool): Scalar predicate. Operation is skipped if predicate is False. Defaults to True.
        descs (Sequence[shared_memory_descriptor]): Shared memory descriptors for
            the preceding multiplication inputs. Defaults to ().
    """
    pred = _semantic.to_tensor(pred)
    descs = _unwrap_if_constexpr(descs)
    descs = [d.handle for d in descs]
    _semantic.builder.create_tcgen05_commit(barrier.handle, pred.handle, descs)
