import argparse
import csv
import importlib.util
import json
import statistics
from dataclasses import dataclass, replace
from pathlib import Path

import torch


EXAMPLE_PATH = str(Path(__file__).with_name("05-moe-bmm1-fused-gather.py"))


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ex = _load_module("moe_bmm1_base", EXAMPLE_PATH)

gluon = ex.gluon
gl = ex.gl
blackwell = ex.blackwell
tma = ex.tma
mbarrier = ex.mbarrier
triton = ex.triton

PartitionArgs = ex.PartitionArgs
advance = ex.advance
alloc_barrier_ring = ex.alloc_barrier_ring
alloc_empty_ready_barriers = ex.alloc_empty_ready_barriers
load_activations = ex.load_activations
load_weights = ex.load_weights
epilogue_partition = ex.epilogue_partition
epilogue_store_partition = ex.epilogue_store_partition
get_store_layout = ex.get_store_layout
epilogue_direct_store = ex.epilogue_direct_store
epilogue_overlapped_store = ex.epilogue_overlapped_store
mma_partition = ex.mma_partition
unswizzle_mx_scale = ex.unswizzle_mx_scale
float2 = ex.float2

_dynamic_pair_w_reuse_tile_schedule_cache: dict[tuple[int, int, int, int], torch.Tensor] = {}


def _pack_full_tile(pid_m: int, pid_n: int, slice_idx: int) -> int:
    return slice_idx | (pid_m << 16) | (pid_n << 32)


def get_pair_w_reuse_tile_schedule_tensor(
    ragged_metadata,
    block_size: int,
    grid_n: int,
    launch_grid: int,
) -> torch.Tensor:
    key = (id(ragged_metadata), block_size, grid_n, launch_grid)
    cached = _dynamic_pair_w_reuse_tile_schedule_cache.get(key)
    if cached is not None:
        return cached

    block_offs, block_schedule = ex.get_block_schedule_tensors(ragged_metadata, block_size)
    grid_m = int(block_offs[ragged_metadata.n_slices].item())
    num_blocks = grid_m * grid_n
    block_schedule_cpu = block_schedule.cpu().tolist()

    grouped: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    singles: list[tuple[int, int, int]] = []
    for schedule_pid_m in range(grid_m):
        m_entry = int(block_schedule_cpu[schedule_pid_m])
        slice_idx = m_entry & 0xFFFF
        pid_m = m_entry >> 16
        for pid_n in range(grid_n):
            grouped.setdefault((slice_idx, pid_n), []).append((pid_m, pid_n, slice_idx))

    pairs: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    for entries in grouped.values():
        entries.sort()
        i = 0
        while i + 1 < len(entries):
            pairs.append((entries[i], entries[i + 1]))
            i += 2
        if i < len(entries):
            singles.append(entries[i])

    schedule: list[tuple[int, int, int] | None] = [None] * num_blocks
    pair_idx = 0
    column = 0
    while pair_idx < len(pairs) and (column + 1) * launch_grid < num_blocks:
        for program_id in range(launch_grid):
            first = program_id + column * launch_grid
            second = first + launch_grid
            if second >= num_blocks or pair_idx >= len(pairs):
                break
            schedule[first], schedule[second] = pairs[pair_idx]
            pair_idx += 1
        column += 2

    for pair in pairs[pair_idx:]:
        singles.extend(pair)
    single_iter = iter(singles)
    for idx, entry in enumerate(schedule):
        if entry is None:
            schedule[idx] = next(single_iter)

    packed_schedule = [_pack_full_tile(*entry) for entry in schedule if entry is not None]
    assert len(packed_schedule) == num_blocks
    tile_schedule = torch.tensor(packed_schedule, dtype=torch.int64, device=ragged_metadata.slice_sizes.device)
    _dynamic_pair_w_reuse_tile_schedule_cache[key] = tile_schedule
    return tile_schedule


@dataclass(frozen=True)
class LogicalCase:
    batch_size: int
    local_rank: int
    x: torch.Tensor
    w_mx: torch.Tensor
    w_scale_raw: torch.Tensor
    bias: torch.Tensor
    ragged_metadata: object
    gather_indx: torch.Tensor
    fused_activation: object
    x_scale: torch.Tensor
    y_scale: torch.Tensor
    out_shape: tuple[int, int]
    out_dtype: torch.dtype


def fixed_activation():
    return ex.FusedActivation(
        ex.FnSpecs("swiglu", ex.swiglu_fn, ("alpha", "limit"), reduction_n=2),
        (1.10, 1.40),
    )


def make_logical_case(c, batch_size: int, device: str, seed: int, routing: str, local_rank: int | None = None):
    torch.manual_seed(seed)
    if local_rank is None:
        local_rank = int(torch.randint(0, c.num_expert_shards, size=()).item())
    ragged, gather = ex.init_routing_data(
        c,
        batch_size,
        local_rank,
        device,
        uniform_routing=(routing == "uniform"),
    )
    k, n = c.hidden_size, c.intermediate_size
    nlocal = c.num_experts // c.num_expert_shards
    x = ex.alloc_randn((batch_size, k), dtype=torch.float8_e4m3fn, device=device)
    w_src = ex.alloc_randn((nlocal, k, n), dtype=torch.bfloat16, device=device)
    w_mx, w_scale_raw = ex.downcast_to_mxfp(w_src, ex.FP4, axis=1)
    bias = ex.alloc_randn((nlocal, n), dtype=torch.float32, device=device)
    fused = fixed_activation()
    return LogicalCase(
        batch_size=batch_size,
        local_rank=local_rank,
        x=x,
        w_mx=w_mx,
        w_scale_raw=w_scale_raw,
        bias=bias,
        ragged_metadata=ragged,
        gather_indx=gather,
        fused_activation=fused,
        x_scale=torch.full((1,), 1.0, device=device),
        y_scale=torch.full((1,), 4.0, device=device),
        out_shape=(batch_size * c.experts_per_token, n // fused.specs.reduction_n),
        out_dtype=torch.float8_e4m3fn,
    )


def materialize_prepared(logical: LogicalCase, p):
    w_layout = ex.BlackwellMX4ValueShuffledLayout(block_k=p.BLOCK_K, block_n=p.BLOCK_N)
    scale_layout = ex.make_default_matmul_mxfp4_w_scale_layout(mx_axis=1, num_warps=p.NUM_WARPS)
    w = ex.convert_layout(ex.wrap_torch_tensor(logical.w_mx, dtype=ex.FP4), w_layout)
    w_scale = ex.convert_layout(ex.wrap_torch_tensor(logical.w_scale_raw), scale_layout)
    return ex.PreparedCase(
        batch_size=logical.batch_size,
        local_rank=logical.local_rank,
        x=logical.x,
        w=w,
        w_scale=w_scale,
        bias=logical.bias,
        ragged_metadata=logical.ragged_metadata,
        gather_indx=logical.gather_indx,
        fused_activation=logical.fused_activation,
        x_scale=logical.x_scale,
        y_scale=logical.y_scale,
        out_shape=logical.out_shape,
        out_dtype=logical.out_dtype,
    )


def config_dict(p):
    return {
        "bm": p.BLOCK_M,
        "bn": p.BLOCK_N,
        "bk": p.BLOCK_K,
        "ctas": p.NUM_CTAS,
        "num_warps": p.NUM_WARPS,
        "occ": p.OCCUPANCY,
        "x": p.X_NUM_BUFS,
        "w": p.W_NUM_BUFS,
        "acc": p.ACC_NUM_BUFS,
        "sub": p.SWIGLU_SUBTILE_FACTOR,
        "epi": p.EPILOGUE_BUFFER_DEPTH,
        "band": p.BAND_N,
        "maxnreg": p.MAXNREG,
        "fullsched": p.USE_FULL_TILE_SCHEDULE,
        "x_mc": p.X_GATHER_MULTICAST,
        "wscale_mc": p.W_SCALE_MULTICAST,
        "epi_n1": p.FORCE_EPILOGUE_WARPS_N1,
        "direct_store": p.USE_DIRECT_EPILOGUE_STORE,
        "reuse_gather": p.REUSE_GATHER_INDICES,
        "inline_release": p.INLINE_MMA_INPUT_RELEASE,
        "act_warps": p.LOAD_ACTIVATION_WARPS,
        "weight_warps": p.LOAD_WEIGHT_WARPS,
        "mma_warps": p.MMA_WARPS,
        "la": p.LOAD_ACTIVATION_REGS,
        "lw": p.LOAD_WEIGHT_REGS,
        "mma": p.MMA_REGS,
    }


def force_1cta(slice_size: int):
    p = ex._select_occ2_config(slice_size) if slice_size <= 64 else ex._select_occ1_config(slice_size)
    return replace(p, NUM_CTAS=1, BAND_N=ex._select_band_n(slice_size))


def make_candidate(name: str, slice_size: int):
    base = force_1cta(slice_size)
    if name == "1cta":
        return base
    if name == "selected":
        return ex.select_kernel_config(slice_size)
    if name == "m16_bn256_sub1_direct_warps4_x8w5_act1w1m1_regs52_epin1_b22":
        return replace(
            base,
            BLOCK_M=16,
            BLOCK_N=256,
            NUM_CTAS=2,
            NUM_WARPS=4,
            X_NUM_BUFS=8,
            W_NUM_BUFS=5,
            SWIGLU_SUBTILE_FACTOR=1,
            LOAD_ACTIVATION_WARPS=1,
            LOAD_WEIGHT_WARPS=1,
            MMA_WARPS=1,
            LOAD_ACTIVATION_REGS=40,
            LOAD_WEIGHT_REGS=32,
            MMA_REGS=32,
            MAXNREG=52,
            BAND_N=22,
            FORCE_EPILOGUE_WARPS_N1=True,
            USE_DIRECT_EPILOGUE_STORE=True,
        )
    if name == "m16_bn256_sub1_direct_warps4_x10w5_act1w1m1_regs52_epin1_b22":
        return replace(
            base,
            BLOCK_M=16,
            BLOCK_N=256,
            NUM_CTAS=2,
            NUM_WARPS=4,
            X_NUM_BUFS=10,
            W_NUM_BUFS=5,
            SWIGLU_SUBTILE_FACTOR=1,
            LOAD_ACTIVATION_WARPS=1,
            LOAD_WEIGHT_WARPS=1,
            MMA_WARPS=1,
            LOAD_ACTIVATION_REGS=40,
            LOAD_WEIGHT_REGS=32,
            MMA_REGS=32,
            MAXNREG=52,
            BAND_N=22,
            FORCE_EPILOGUE_WARPS_N1=True,
            USE_DIRECT_EPILOGUE_STORE=True,
        )

    prefix = None
    xbuf = None
    wbuf = None
    num_warps = None
    for cur_x in (4, 5, 6):
        for cur_w in (5, 6):
            for cur_warps in (4, 8):
                cur_prefix = f"m32_bn256_sub1_direct_warps{cur_warps}_x{cur_x}w{cur_w}_b"
                if name.startswith(cur_prefix):
                    prefix = cur_prefix
                    xbuf = cur_x
                    wbuf = cur_w
                    num_warps = cur_warps
                    break
            if prefix is not None:
                break
        if prefix is not None:
            break

    suffix = None
    regs = None
    for cur_regs in (48, 52, 56, 60):
        cur_suffix = f"_act2w1m1_regs{cur_regs}_epin1_b32"
        if prefix is not None and name.endswith(cur_suffix):
            suffix = cur_suffix
            regs = cur_regs
            break

    if suffix is None or prefix is None or xbuf is None or wbuf is None or regs is None or num_warps is None:
        raise ValueError(name)

    tokens = name[len(prefix):-len(suffix)].split("_")
    band = int(tokens[0])
    opts = set(tokens[1:])
    if regs == 60:
        load_activation_regs, load_weight_regs, mma_regs = 48, 40, 40
    elif regs in (52, 56):
        load_activation_regs, load_weight_regs, mma_regs = 40, 32, 32
    else:
        load_activation_regs, load_weight_regs, mma_regs = 32, 32, 32
    cfg = dict(
        BLOCK_N=256,
        NUM_CTAS=2,
        NUM_WARPS=num_warps,
        X_NUM_BUFS=xbuf,
        W_NUM_BUFS=wbuf,
        SWIGLU_SUBTILE_FACTOR=1,
        LOAD_ACTIVATION_WARPS=2,
        LOAD_WEIGHT_WARPS=1,
        MMA_WARPS=1,
        LOAD_ACTIVATION_REGS=load_activation_regs,
        LOAD_WEIGHT_REGS=load_weight_regs,
        MMA_REGS=mma_regs,
        MAXNREG=regs,
        BAND_N=band,
        FORCE_EPILOGUE_WARPS_N1=True,
        USE_DIRECT_EPILOGUE_STORE=True,
    )
    if "acc2" in opts:
        cfg["ACC_NUM_BUFS"] = 2
    if "acc3" in opts:
        cfg["ACC_NUM_BUFS"] = 3
    if "acc4" in opts:
        cfg["ACC_NUM_BUFS"] = 4
    if "nomcscale" in opts:
        cfg["W_SCALE_MULTICAST"] = False
    if "nomcx" in opts:
        cfg["X_GATHER_MULTICAST"] = False
    if "nomc" in opts:
        cfg["X_GATHER_MULTICAST"] = False
        cfg["W_SCALE_MULTICAST"] = False
    if "fullsched" in opts:
        cfg["USE_FULL_TILE_SCHEDULE"] = True
    if "reuse" in opts:
        cfg["REUSE_GATHER_INDICES"] = True
    return replace(base, **cfg)


def route_stats(prepared, p):
    ss = prepared.ragged_metadata.slice_sizes
    return {
        "tokens_local": int(ss.sum().item()),
        "active_slices": int((ss > 0).sum().item()),
        "max_slice": int(ss.max().item()),
        "grid_m": int(prepared.ragged_metadata.n_blocks(prepared.ragged_metadata.n_slices,
                                                        prepared.gather_indx.shape[0], p.BLOCK_M)),
    }


def run_split_with_config(prepared, p, out: torch.Tensor):
    pc = ex.make_precision_config(prepared)
    return ex.matmul(
        a=prepared.x,
        b=prepared.w,
        bias=prepared.bias,
        a_ragged_metadata=prepared.ragged_metadata,
        gather_indx=prepared.gather_indx,
        precision_config=pc,
        c=out,
        fused_activation=prepared.fused_activation,
        p=p,
    )


@gluon.jit
def load_inputs_partition(p: PartitionArgs):
    local_cga_layout: gl.constexpr = ((0, 1), ) if p.USE_2CTA else ()
    offs_layout: gl.constexpr = gl.SliceLayout(
        dim=0,
        parent=gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0], cga_layout=local_cga_layout),
    )
    tile_x_bytes: gl.constexpr = p.x_desc.block_type.nbytes * (p.BLOCK_M_PER_CTA if p.USE_2CTA else p.BLOCK_M)
    tile_w_bytes: gl.constexpr = p.w_desc.nbytes_per_cta
    tile_scale_bytes: gl.constexpr = p.scale_desc.nbytes_per_cta
    bytes_per_w_stage: gl.constexpr = tile_w_bytes + tile_scale_bytes
    scale_k_stride: gl.constexpr = p.BLOCK_K // (p.MXFP_BLOCK_SIZE * p.SCALE_SIZE_INNER)

    x_idx = 0
    x_phase = 1
    x_issued = 0
    w_idx = 0
    w_phase = 1
    w_issued = 0

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        pid_m, pid_n, slice_idx, slice_offset = p.apply_block_schedule(block_id)
        off_m = pid_m * p.BLOCK_M
        shape_m = gl.load(p.x_slice_sizes + slice_idx)
        offs_m = off_m + gl.arange(0, p.BLOCK_M, layout=offs_layout)
        mask_m = offs_m < shape_m
        offs_x_m = gl.load(
            p.gather_indx_ptr + slice_offset + offs_m,
            mask=mask_m,
            other=p.x_desc.shape[0],
        )
        scale_idx = slice_idx * p.SCALE_FLAT_N + pid_n * p.SCALE_BLOCK_N_DIV

        for ki in range(p.K_TILES):
            off_k_x = ki * p.BLOCK_K
            off_k_scale = ki * scale_k_stride

            x_empty_bar = p.x_empty_bars.index(x_idx)
            x_ready_bar = p.x_ready_bars.index(x_idx)
            x_buf = p.x_bufs.index(x_idx)
            w_empty_bar = p.w_empty_bars.index(w_idx)
            w_ready_bar = p.w_ready_bars.index(w_idx)
            w_buf = p.w_bufs.index(w_idx)
            scale_buf = p.w_scale_bufs.index(w_idx)

            mbarrier.wait(x_empty_bar, x_phase, pred=x_issued >= p.x_num_bufs)
            mbarrier.wait(w_empty_bar, w_phase, pred=w_issued >= p.w_num_bufs)

            mbarrier.expect(x_ready_bar, tile_x_bytes)
            tma.async_gather(
                p.x_desc,
                offs_x_m,
                off_k_x,
                x_ready_bar,
                x_buf,
                multicast=p.USE_2CTA and p.X_GATHER_MULTICAST,
            )

            mbarrier.expect(w_ready_bar, bytes_per_w_stage)
            tma.async_copy_global_to_shared(p.w_desc, [slice_idx, ki, pid_n, 0, 0], w_ready_bar, w_buf)
            tma.async_copy_global_to_shared(
                p.scale_desc,
                [0, scale_idx, off_k_scale, 0, 0],
                w_ready_bar,
                scale_buf,
                multicast=p.USE_2CTA and p.W_SCALE_MULTICAST,
            )

            x_idx, x_phase = advance(x_idx, x_phase, p.x_num_bufs)
            w_idx, w_phase = advance(w_idx, w_phase, p.w_num_bufs)
            x_issued += 1
            w_issued += 1


@gluon.jit
def load_inputs_fake_nsplit_partition(p: PartitionArgs):
    # Fake 2CTA N-split: each CTA owns a disjoint N shard but needs the full
    # M rows of the activation tile for a regular one-CTA MMA.
    local_cga_layout: gl.constexpr = ((0, 0), ) if p.USE_2CTA else ()
    offs_layout: gl.constexpr = gl.SliceLayout(
        dim=0,
        parent=gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0], cga_layout=local_cga_layout),
    )
    tile_x_bytes: gl.constexpr = p.x_desc.block_type.nbytes * p.BLOCK_M
    tile_w_bytes: gl.constexpr = p.w_desc.nbytes_per_cta
    tile_scale_bytes: gl.constexpr = p.scale_desc.nbytes_per_cta
    bytes_per_w_stage: gl.constexpr = tile_w_bytes + tile_scale_bytes
    scale_k_stride: gl.constexpr = p.BLOCK_K // (p.MXFP_BLOCK_SIZE * p.SCALE_SIZE_INNER)

    x_idx = 0
    x_phase = 1
    x_issued = 0
    w_idx = 0
    w_phase = 1
    w_issued = 0

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        pid_m, pid_n, slice_idx, slice_offset = p.apply_block_schedule(block_id)
        off_m = pid_m * p.BLOCK_M
        shape_m = gl.load(p.x_slice_sizes + slice_idx)
        offs_m = off_m + gl.arange(0, p.BLOCK_M, layout=offs_layout)
        mask_m = offs_m < shape_m
        offs_x_m = gl.load(
            p.gather_indx_ptr + slice_offset + offs_m,
            mask=mask_m,
            other=p.x_desc.shape[0],
        )
        scale_idx = slice_idx * p.SCALE_FLAT_N + pid_n * p.SCALE_BLOCK_N_DIV

        for ki in range(p.K_TILES):
            off_k_x = ki * p.BLOCK_K
            off_k_scale = ki * scale_k_stride

            x_empty_bar = p.x_empty_bars.index(x_idx)
            x_ready_bar = p.x_ready_bars.index(x_idx)
            x_buf = p.x_bufs.index(x_idx)
            w_empty_bar = p.w_empty_bars.index(w_idx)
            w_ready_bar = p.w_ready_bars.index(w_idx)
            w_buf = p.w_bufs.index(w_idx)
            scale_buf = p.w_scale_bufs.index(w_idx)

            mbarrier.wait(x_empty_bar, x_phase, pred=x_issued >= p.x_num_bufs)
            mbarrier.wait(w_empty_bar, w_phase, pred=w_issued >= p.w_num_bufs)

            mbarrier.expect(x_ready_bar, tile_x_bytes)
            tma.async_gather(
                p.x_desc,
                offs_x_m,
                off_k_x,
                x_ready_bar,
                x_buf,
                multicast=False,
            )

            mbarrier.expect(w_ready_bar, bytes_per_w_stage)
            tma.async_copy_global_to_shared(p.w_desc, [slice_idx, ki, pid_n, 0, 0], w_ready_bar, w_buf)
            tma.async_copy_global_to_shared(
                p.scale_desc,
                [0, scale_idx, off_k_scale, 0, 0],
                w_ready_bar,
                scale_buf,
                multicast=False,
            )

            x_idx, x_phase = advance(x_idx, x_phase, p.x_num_bufs)
            w_idx, w_phase = advance(w_idx, w_phase, p.w_num_bufs)
            x_issued += 1
            w_issued += 1


@gluon.jit
def load_inputs_xfirst_partition(p: PartitionArgs):
    local_cga_layout: gl.constexpr = ((0, 1), ) if p.USE_2CTA else ()
    offs_layout: gl.constexpr = gl.SliceLayout(
        dim=0,
        parent=gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0], cga_layout=local_cga_layout),
    )
    tile_x_bytes: gl.constexpr = p.x_desc.block_type.nbytes * (p.BLOCK_M_PER_CTA if p.USE_2CTA else p.BLOCK_M)
    tile_w_bytes: gl.constexpr = p.w_desc.nbytes_per_cta
    tile_scale_bytes: gl.constexpr = p.scale_desc.nbytes_per_cta
    bytes_per_w_stage: gl.constexpr = tile_w_bytes + tile_scale_bytes
    scale_k_stride: gl.constexpr = p.BLOCK_K // (p.MXFP_BLOCK_SIZE * p.SCALE_SIZE_INNER)

    x_idx = 0
    x_phase = 1
    x_issued = 0
    w_idx = 0
    w_phase = 1
    w_issued = 0

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        pid_m, pid_n, slice_idx, slice_offset = p.apply_block_schedule(block_id)
        off_m = pid_m * p.BLOCK_M
        shape_m = gl.load(p.x_slice_sizes + slice_idx)
        offs_m = off_m + gl.arange(0, p.BLOCK_M, layout=offs_layout)
        mask_m = offs_m < shape_m
        offs_x_m = gl.load(
            p.gather_indx_ptr + slice_offset + offs_m,
            mask=mask_m,
            other=p.x_desc.shape[0],
        )
        scale_idx = slice_idx * p.SCALE_FLAT_N + pid_n * p.SCALE_BLOCK_N_DIV

        for ki in range(p.K_TILES):
            off_k_x = ki * p.BLOCK_K
            off_k_scale = ki * scale_k_stride

            x_empty_bar = p.x_empty_bars.index(x_idx)
            x_ready_bar = p.x_ready_bars.index(x_idx)
            x_buf = p.x_bufs.index(x_idx)
            mbarrier.wait(x_empty_bar, x_phase, pred=x_issued >= p.x_num_bufs)
            mbarrier.expect(x_ready_bar, tile_x_bytes)
            tma.async_gather(
                p.x_desc,
                offs_x_m,
                off_k_x,
                x_ready_bar,
                x_buf,
                multicast=p.USE_2CTA and p.X_GATHER_MULTICAST,
            )

            w_empty_bar = p.w_empty_bars.index(w_idx)
            w_ready_bar = p.w_ready_bars.index(w_idx)
            w_buf = p.w_bufs.index(w_idx)
            scale_buf = p.w_scale_bufs.index(w_idx)
            mbarrier.wait(w_empty_bar, w_phase, pred=w_issued >= p.w_num_bufs)
            mbarrier.expect(w_ready_bar, bytes_per_w_stage)
            tma.async_copy_global_to_shared(p.w_desc, [slice_idx, ki, pid_n, 0, 0], w_ready_bar, w_buf)
            tma.async_copy_global_to_shared(
                p.scale_desc,
                [0, scale_idx, off_k_scale, 0, 0],
                w_ready_bar,
                scale_buf,
                multicast=p.USE_2CTA and p.W_SCALE_MULTICAST,
            )

            x_idx, x_phase = advance(x_idx, x_phase, p.x_num_bufs)
            w_idx, w_phase = advance(w_idx, w_phase, p.w_num_bufs)
            x_issued += 1
            w_issued += 1


@gluon.jit
def load_inputs_wfirst_partition(p: PartitionArgs):
    local_cga_layout: gl.constexpr = ((0, 1), ) if p.USE_2CTA else ()
    offs_layout: gl.constexpr = gl.SliceLayout(
        dim=0,
        parent=gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0], cga_layout=local_cga_layout),
    )
    tile_x_bytes: gl.constexpr = p.x_desc.block_type.nbytes * (p.BLOCK_M_PER_CTA if p.USE_2CTA else p.BLOCK_M)
    tile_w_bytes: gl.constexpr = p.w_desc.nbytes_per_cta
    tile_scale_bytes: gl.constexpr = p.scale_desc.nbytes_per_cta
    bytes_per_w_stage: gl.constexpr = tile_w_bytes + tile_scale_bytes
    scale_k_stride: gl.constexpr = p.BLOCK_K // (p.MXFP_BLOCK_SIZE * p.SCALE_SIZE_INNER)

    x_idx = 0
    x_phase = 1
    x_issued = 0
    w_idx = 0
    w_phase = 1
    w_issued = 0

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        pid_m, pid_n, slice_idx, slice_offset = p.apply_block_schedule(block_id)
        off_m = pid_m * p.BLOCK_M
        shape_m = gl.load(p.x_slice_sizes + slice_idx)
        offs_m = off_m + gl.arange(0, p.BLOCK_M, layout=offs_layout)
        mask_m = offs_m < shape_m
        offs_x_m = gl.load(
            p.gather_indx_ptr + slice_offset + offs_m,
            mask=mask_m,
            other=p.x_desc.shape[0],
        )
        scale_idx = slice_idx * p.SCALE_FLAT_N + pid_n * p.SCALE_BLOCK_N_DIV

        for ki in range(p.K_TILES):
            off_k_x = ki * p.BLOCK_K
            off_k_scale = ki * scale_k_stride

            w_empty_bar = p.w_empty_bars.index(w_idx)
            w_ready_bar = p.w_ready_bars.index(w_idx)
            w_buf = p.w_bufs.index(w_idx)
            scale_buf = p.w_scale_bufs.index(w_idx)
            mbarrier.wait(w_empty_bar, w_phase, pred=w_issued >= p.w_num_bufs)
            mbarrier.expect(w_ready_bar, bytes_per_w_stage)
            tma.async_copy_global_to_shared(p.w_desc, [slice_idx, ki, pid_n, 0, 0], w_ready_bar, w_buf)
            tma.async_copy_global_to_shared(
                p.scale_desc,
                [0, scale_idx, off_k_scale, 0, 0],
                w_ready_bar,
                scale_buf,
                multicast=p.USE_2CTA and p.W_SCALE_MULTICAST,
            )

            x_empty_bar = p.x_empty_bars.index(x_idx)
            x_ready_bar = p.x_ready_bars.index(x_idx)
            x_buf = p.x_bufs.index(x_idx)
            mbarrier.wait(x_empty_bar, x_phase, pred=x_issued >= p.x_num_bufs)
            mbarrier.expect(x_ready_bar, tile_x_bytes)
            tma.async_gather(
                p.x_desc,
                offs_x_m,
                off_k_x,
                x_ready_bar,
                x_buf,
                multicast=p.USE_2CTA and p.X_GATHER_MULTICAST,
            )

            x_idx, x_phase = advance(x_idx, x_phase, p.x_num_bufs)
            w_idx, w_phase = advance(w_idx, w_phase, p.w_num_bufs)
            x_issued += 1
            w_issued += 1


@gluon.jit
def load_pair_reuse_inputs_partition(p: PartitionArgs):
    local_cga_layout: gl.constexpr = ((0, 1), ) if p.USE_2CTA else ()
    offs_layout: gl.constexpr = gl.SliceLayout(
        dim=0,
        parent=gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0], cga_layout=local_cga_layout),
    )
    tile_x_bytes: gl.constexpr = p.x_desc.block_type.nbytes * (p.BLOCK_M_PER_CTA if p.USE_2CTA else p.BLOCK_M)
    tile_w_bytes: gl.constexpr = p.w_desc.nbytes_per_cta
    tile_scale_bytes: gl.constexpr = p.scale_desc.nbytes_per_cta
    bytes_per_w_stage: gl.constexpr = tile_w_bytes + tile_scale_bytes
    scale_k_stride: gl.constexpr = p.BLOCK_K // (p.MXFP_BLOCK_SIZE * p.SCALE_SIZE_INNER)

    x_idx = 0
    x_phase = 1
    x_issued = 0
    w_idx = 0
    w_phase = 1
    w_issued = 0

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        pid_m, pid_n, slice_idx, slice_offset = p.apply_block_schedule(block_id)
        prev_block_id = block_id - p.NUM_SMS
        has_prev = block_id >= p.NUM_SMS
        safe_prev_block_id = gl.where(has_prev, prev_block_id, block_id)
        _, prev_pid_n, prev_slice_idx, _ = p.apply_block_schedule(safe_prev_block_id)
        is_second_reuse = has_prev & (prev_pid_n == pid_n) & (prev_slice_idx == slice_idx)

        should_issue = is_second_reuse == False
        if should_issue:
            next_block_id = block_id + p.NUM_SMS
            has_next = next_block_id < p.num_blocks
            safe_next_block_id = gl.where(has_next, next_block_id, block_id)
            next_pid_m, next_pid_n, next_slice_idx, next_slice_offset = p.apply_block_schedule(safe_next_block_id)
            pair_next = has_next & (next_pid_n == pid_n) & (next_slice_idx == slice_idx)

            off_m = pid_m * p.BLOCK_M
            shape_m = gl.load(p.x_slice_sizes + slice_idx)
            offs_m = off_m + gl.arange(0, p.BLOCK_M, layout=offs_layout)
            mask_m = offs_m < shape_m
            offs_x_m = gl.load(
                p.gather_indx_ptr + slice_offset + offs_m,
                mask=mask_m,
                other=p.x_desc.shape[0],
            )

            next_off_m = next_pid_m * p.BLOCK_M
            next_shape_m = gl.load(p.x_slice_sizes + next_slice_idx)
            next_offs_m = next_off_m + gl.arange(0, p.BLOCK_M, layout=offs_layout)
            next_mask_m = next_offs_m < next_shape_m
            next_offs_x_m = gl.load(
                p.gather_indx_ptr + next_slice_offset + next_offs_m,
                mask=next_mask_m,
                other=p.x_desc.shape[0],
            )

            scale_idx = slice_idx * p.SCALE_FLAT_N + pid_n * p.SCALE_BLOCK_N_DIV

            for ki in range(p.K_TILES):
                off_k_x = ki * p.BLOCK_K
                off_k_scale = ki * scale_k_stride

                x_empty_bar = p.x_empty_bars.index(x_idx)
                x_ready_bar = p.x_ready_bars.index(x_idx)
                x_buf = p.x_bufs.index(x_idx)
                w_empty_bar = p.w_empty_bars.index(w_idx)
                w_ready_bar = p.w_ready_bars.index(w_idx)
                w_buf = p.w_bufs.index(w_idx)
                scale_buf = p.w_scale_bufs.index(w_idx)

                mbarrier.wait(x_empty_bar, x_phase, pred=x_issued >= p.x_num_bufs)
                mbarrier.wait(w_empty_bar, w_phase, pred=w_issued >= p.w_num_bufs)

                mbarrier.expect(x_ready_bar, tile_x_bytes)
                tma.async_gather(
                    p.x_desc,
                    offs_x_m,
                    off_k_x,
                    x_ready_bar,
                    x_buf,
                    multicast=p.USE_2CTA and p.X_GATHER_MULTICAST,
                )

                mbarrier.expect(w_ready_bar, bytes_per_w_stage)
                tma.async_copy_global_to_shared(p.w_desc, [slice_idx, ki, pid_n, 0, 0], w_ready_bar, w_buf)
                tma.async_copy_global_to_shared(
                    p.scale_desc,
                    [0, scale_idx, off_k_scale, 0, 0],
                    w_ready_bar,
                    scale_buf,
                    multicast=p.USE_2CTA and p.W_SCALE_MULTICAST,
                )

                next_x_idx, next_x_phase = advance(x_idx, x_phase, p.x_num_bufs)
                if pair_next:
                    next_x_empty_bar = p.x_empty_bars.index(next_x_idx)
                    next_x_ready_bar = p.x_ready_bars.index(next_x_idx)
                    next_x_buf = p.x_bufs.index(next_x_idx)
                    mbarrier.wait(next_x_empty_bar, next_x_phase, pred=(x_issued + 1) >= p.x_num_bufs)
                    mbarrier.expect(next_x_ready_bar, tile_x_bytes)
                    tma.async_gather(
                        p.x_desc,
                        next_offs_x_m,
                        off_k_x,
                        next_x_ready_bar,
                        next_x_buf,
                        multicast=p.USE_2CTA and p.X_GATHER_MULTICAST,
                    )
                    x_idx, x_phase = advance(next_x_idx, next_x_phase, p.x_num_bufs)
                    x_issued += 2
                else:
                    x_idx = next_x_idx
                    x_phase = next_x_phase
                    x_issued += 1

                w_idx, w_phase = advance(w_idx, w_phase, p.w_num_bufs)
                w_issued += 1


@gluon.jit
def load_pair_reuse_activations_partition(p: PartitionArgs):
    local_cga_layout: gl.constexpr = ((0, 1), ) if p.USE_2CTA else ()
    offs_layout: gl.constexpr = gl.SliceLayout(
        dim=0,
        parent=gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0], cga_layout=local_cga_layout),
    )
    tile_x_bytes: gl.constexpr = p.x_desc.block_type.nbytes * (p.BLOCK_M_PER_CTA if p.USE_2CTA else p.BLOCK_M)

    x_idx = 0
    x_phase = 1
    x_issued = 0

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        pid_m, pid_n, slice_idx, slice_offset = p.apply_block_schedule(block_id)
        prev_block_id = block_id - p.NUM_SMS
        has_prev = block_id >= p.NUM_SMS
        safe_prev_block_id = gl.where(has_prev, prev_block_id, block_id)
        _, prev_pid_n, prev_slice_idx, _ = p.apply_block_schedule(safe_prev_block_id)
        is_second_reuse = has_prev & (prev_pid_n == pid_n) & (prev_slice_idx == slice_idx)

        should_issue = is_second_reuse == False
        if should_issue:
            next_block_id = block_id + p.NUM_SMS
            has_next = next_block_id < p.num_blocks
            safe_next_block_id = gl.where(has_next, next_block_id, block_id)
            next_pid_m, next_pid_n, next_slice_idx, next_slice_offset = p.apply_block_schedule(safe_next_block_id)
            pair_next = has_next & (next_pid_n == pid_n) & (next_slice_idx == slice_idx)

            off_m = pid_m * p.BLOCK_M
            shape_m = gl.load(p.x_slice_sizes + slice_idx)
            offs_m = off_m + gl.arange(0, p.BLOCK_M, layout=offs_layout)
            mask_m = offs_m < shape_m
            offs_x_m = gl.load(
                p.gather_indx_ptr + slice_offset + offs_m,
                mask=mask_m,
                other=p.x_desc.shape[0],
            )

            next_off_m = next_pid_m * p.BLOCK_M
            next_shape_m = gl.load(p.x_slice_sizes + next_slice_idx)
            next_offs_m = next_off_m + gl.arange(0, p.BLOCK_M, layout=offs_layout)
            next_mask_m = next_offs_m < next_shape_m
            next_offs_x_m = gl.load(
                p.gather_indx_ptr + next_slice_offset + next_offs_m,
                mask=next_mask_m,
                other=p.x_desc.shape[0],
            )

            for ki in range(p.K_TILES):
                off_k_x = ki * p.BLOCK_K

                x_empty_bar = p.x_empty_bars.index(x_idx)
                x_ready_bar = p.x_ready_bars.index(x_idx)
                x_buf = p.x_bufs.index(x_idx)

                mbarrier.wait(x_empty_bar, x_phase, pred=x_issued >= p.x_num_bufs)
                mbarrier.expect(x_ready_bar, tile_x_bytes)
                tma.async_gather(
                    p.x_desc,
                    offs_x_m,
                    off_k_x,
                    x_ready_bar,
                    x_buf,
                    multicast=p.USE_2CTA and p.X_GATHER_MULTICAST,
                )

                next_x_idx, next_x_phase = advance(x_idx, x_phase, p.x_num_bufs)
                if pair_next:
                    next_x_empty_bar = p.x_empty_bars.index(next_x_idx)
                    next_x_ready_bar = p.x_ready_bars.index(next_x_idx)
                    next_x_buf = p.x_bufs.index(next_x_idx)
                    mbarrier.wait(next_x_empty_bar, next_x_phase, pred=(x_issued + 1) >= p.x_num_bufs)
                    mbarrier.expect(next_x_ready_bar, tile_x_bytes)
                    tma.async_gather(
                        p.x_desc,
                        next_offs_x_m,
                        off_k_x,
                        next_x_ready_bar,
                        next_x_buf,
                        multicast=p.USE_2CTA and p.X_GATHER_MULTICAST,
                    )
                    x_idx, x_phase = advance(next_x_idx, next_x_phase, p.x_num_bufs)
                    x_issued += 2
                else:
                    x_idx = next_x_idx
                    x_phase = next_x_phase
                    x_issued += 1


@gluon.jit
def load_pair_reuse_weights_partition(p: PartitionArgs):
    tile_w_bytes: gl.constexpr = p.w_desc.nbytes_per_cta
    tile_scale_bytes: gl.constexpr = p.scale_desc.nbytes_per_cta
    bytes_per_stage: gl.constexpr = tile_w_bytes + tile_scale_bytes
    scale_k_stride: gl.constexpr = p.BLOCK_K // (p.MXFP_BLOCK_SIZE * p.SCALE_SIZE_INNER)

    idx = 0
    phase = 1
    issued = 0

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        pid_n, slice_idx = p.apply_weight_schedule(block_id)
        prev_block_id = block_id - p.NUM_SMS
        has_prev = block_id >= p.NUM_SMS
        safe_prev_block_id = gl.where(has_prev, prev_block_id, block_id)
        prev_pid_n, prev_slice_idx = p.apply_weight_schedule(safe_prev_block_id)
        is_second_reuse = has_prev & (prev_pid_n == pid_n) & (prev_slice_idx == slice_idx)

        should_issue = is_second_reuse == False
        if should_issue:
            scale_idx = slice_idx * p.SCALE_FLAT_N + pid_n * p.SCALE_BLOCK_N_DIV
            for ki in range(p.K_TILES):
                off_k_scale = ki * scale_k_stride

                w_empty_bar = p.w_empty_bars.index(idx)
                w_ready_bar = p.w_ready_bars.index(idx)
                w_buf = p.w_bufs.index(idx)
                scale_buf = p.w_scale_bufs.index(idx)

                mbarrier.wait(w_empty_bar, phase, pred=issued >= p.w_num_bufs)
                mbarrier.expect(w_ready_bar, bytes_per_stage)
                tma.async_copy_global_to_shared(p.w_desc, [slice_idx, ki, pid_n, 0, 0], w_ready_bar, w_buf)
                tma.async_copy_global_to_shared(
                    p.scale_desc,
                    [0, scale_idx, off_k_scale, 0, 0],
                    w_ready_bar,
                    scale_buf,
                    multicast=p.USE_2CTA and p.W_SCALE_MULTICAST,
                )

                idx, phase = advance(idx, phase, p.w_num_bufs)
                issued += 1


@gluon.jit
def is_full_m_tile(pid_m, shape_m, BLOCK_M: gl.constexpr):
    return (pid_m + 1) * BLOCK_M <= shape_m


@gluon.jit
def load_transition_pair_activations_partition(p: PartitionArgs):
    local_cga_layout: gl.constexpr = ((0, 1), ) if p.USE_2CTA else ()
    offs_layout: gl.constexpr = gl.SliceLayout(
        dim=0,
        parent=gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0], cga_layout=local_cga_layout),
    )
    tile_x_bytes: gl.constexpr = p.x_desc.block_type.nbytes * (p.BLOCK_M_PER_CTA if p.USE_2CTA else p.BLOCK_M)

    x_idx = 0
    x_phase = 1
    x_issued = 0

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        pid_m, pid_n, slice_idx, slice_offset = p.apply_block_schedule(block_id)
        shape_m = gl.load(p.x_slice_sizes + slice_idx)
        cur_full = is_full_m_tile(pid_m, shape_m, p.BLOCK_M)

        prev_block_id = block_id - p.NUM_SMS
        has_prev = block_id >= p.NUM_SMS
        safe_prev_block_id = gl.where(has_prev, prev_block_id, block_id)
        prev_pid_m, prev_pid_n, prev_slice_idx, _ = p.apply_block_schedule(safe_prev_block_id)
        prev_shape_m = gl.load(p.x_slice_sizes + prev_slice_idx)
        prev_full = is_full_m_tile(prev_pid_m, prev_shape_m, p.BLOCK_M)
        is_second_pair = has_prev & (prev_pid_n == pid_n) & (prev_slice_idx == slice_idx) & (prev_full != cur_full)

        should_issue = is_second_pair == False
        if should_issue:
            next_block_id = block_id + p.NUM_SMS
            has_next = next_block_id < p.num_blocks
            safe_next_block_id = gl.where(has_next, next_block_id, block_id)
            next_pid_m, next_pid_n, next_slice_idx, next_slice_offset = p.apply_block_schedule(safe_next_block_id)
            next_shape_m = gl.load(p.x_slice_sizes + next_slice_idx)
            next_full = is_full_m_tile(next_pid_m, next_shape_m, p.BLOCK_M)
            pair_next = has_next & (next_pid_n == pid_n) & (next_slice_idx == slice_idx) & (cur_full != next_full)

            off_m = pid_m * p.BLOCK_M
            offs_m = off_m + gl.arange(0, p.BLOCK_M, layout=offs_layout)
            mask_m = offs_m < shape_m
            offs_x_m = gl.load(
                p.gather_indx_ptr + slice_offset + offs_m,
                mask=mask_m,
                other=p.x_desc.shape[0],
            )

            next_off_m = next_pid_m * p.BLOCK_M
            next_offs_m = next_off_m + gl.arange(0, p.BLOCK_M, layout=offs_layout)
            next_mask_m = next_offs_m < next_shape_m
            next_offs_x_m = gl.load(
                p.gather_indx_ptr + next_slice_offset + next_offs_m,
                mask=next_mask_m,
                other=p.x_desc.shape[0],
            )

            for ki in range(p.K_TILES):
                off_k_x = ki * p.BLOCK_K

                x_empty_bar = p.x_empty_bars.index(x_idx)
                x_ready_bar = p.x_ready_bars.index(x_idx)
                x_buf = p.x_bufs.index(x_idx)

                mbarrier.wait(x_empty_bar, x_phase, pred=x_issued >= p.x_num_bufs)
                mbarrier.expect(x_ready_bar, tile_x_bytes)
                tma.async_gather(
                    p.x_desc,
                    offs_x_m,
                    off_k_x,
                    x_ready_bar,
                    x_buf,
                    multicast=p.USE_2CTA and p.X_GATHER_MULTICAST,
                )

                next_x_idx, next_x_phase = advance(x_idx, x_phase, p.x_num_bufs)
                if pair_next:
                    next_x_empty_bar = p.x_empty_bars.index(next_x_idx)
                    next_x_ready_bar = p.x_ready_bars.index(next_x_idx)
                    next_x_buf = p.x_bufs.index(next_x_idx)
                    mbarrier.wait(next_x_empty_bar, next_x_phase, pred=(x_issued + 1) >= p.x_num_bufs)
                    mbarrier.expect(next_x_ready_bar, tile_x_bytes)
                    tma.async_gather(
                        p.x_desc,
                        next_offs_x_m,
                        off_k_x,
                        next_x_ready_bar,
                        next_x_buf,
                        multicast=p.USE_2CTA and p.X_GATHER_MULTICAST,
                    )
                    x_idx, x_phase = advance(next_x_idx, next_x_phase, p.x_num_bufs)
                    x_issued += 2
                else:
                    x_idx = next_x_idx
                    x_phase = next_x_phase
                    x_issued += 1


@gluon.jit
def load_transition_pair_weights_partition(p: PartitionArgs):
    tile_w_bytes: gl.constexpr = p.w_desc.nbytes_per_cta
    tile_scale_bytes: gl.constexpr = p.scale_desc.nbytes_per_cta
    bytes_per_stage: gl.constexpr = tile_w_bytes + tile_scale_bytes
    scale_k_stride: gl.constexpr = p.BLOCK_K // (p.MXFP_BLOCK_SIZE * p.SCALE_SIZE_INNER)

    idx = 0
    phase = 1
    issued = 0

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        pid_m, pid_n, slice_idx, _ = p.apply_block_schedule(block_id)
        shape_m = gl.load(p.x_slice_sizes + slice_idx)
        cur_full = is_full_m_tile(pid_m, shape_m, p.BLOCK_M)

        prev_block_id = block_id - p.NUM_SMS
        has_prev = block_id >= p.NUM_SMS
        safe_prev_block_id = gl.where(has_prev, prev_block_id, block_id)
        prev_pid_m, _, prev_slice_idx, _ = p.apply_block_schedule(safe_prev_block_id)
        prev_shape_m = gl.load(p.x_slice_sizes + prev_slice_idx)
        prev_full = is_full_m_tile(prev_pid_m, prev_shape_m, p.BLOCK_M)
        is_second_pair = has_prev & (prev_full != cur_full)

        should_issue = is_second_pair == False
        if should_issue:
            next_block_id = block_id + p.NUM_SMS
            has_next = next_block_id < p.num_blocks
            safe_next_block_id = gl.where(has_next, next_block_id, block_id)
            next_pid_m, next_pid_n, next_slice_idx, _ = p.apply_block_schedule(safe_next_block_id)
            next_shape_m = gl.load(p.x_slice_sizes + next_slice_idx)
            next_full = is_full_m_tile(next_pid_m, next_shape_m, p.BLOCK_M)
            pair_next = has_next & (cur_full != next_full)

            scale_idx = slice_idx * p.SCALE_FLAT_N + pid_n * p.SCALE_BLOCK_N_DIV
            next_scale_idx = next_slice_idx * p.SCALE_FLAT_N + next_pid_n * p.SCALE_BLOCK_N_DIV

            for ki in range(p.K_TILES):
                off_k_scale = ki * scale_k_stride

                w_empty_bar = p.w_empty_bars.index(idx)
                w_ready_bar = p.w_ready_bars.index(idx)
                w_buf = p.w_bufs.index(idx)
                scale_buf = p.w_scale_bufs.index(idx)

                mbarrier.wait(w_empty_bar, phase, pred=issued >= p.w_num_bufs)
                mbarrier.expect(w_ready_bar, bytes_per_stage)
                tma.async_copy_global_to_shared(p.w_desc, [slice_idx, ki, pid_n, 0, 0], w_ready_bar, w_buf)
                tma.async_copy_global_to_shared(
                    p.scale_desc,
                    [0, scale_idx, off_k_scale, 0, 0],
                    w_ready_bar,
                    scale_buf,
                    multicast=p.USE_2CTA and p.W_SCALE_MULTICAST,
                )

                next_idx, next_phase = advance(idx, phase, p.w_num_bufs)
                if pair_next:
                    next_w_empty_bar = p.w_empty_bars.index(next_idx)
                    next_w_ready_bar = p.w_ready_bars.index(next_idx)
                    next_w_buf = p.w_bufs.index(next_idx)
                    next_scale_buf = p.w_scale_bufs.index(next_idx)
                    mbarrier.wait(next_w_empty_bar, next_phase, pred=(issued + 1) >= p.w_num_bufs)
                    mbarrier.expect(next_w_ready_bar, bytes_per_stage)
                    tma.async_copy_global_to_shared(
                        p.w_desc,
                        [next_slice_idx, ki, next_pid_n, 0, 0],
                        next_w_ready_bar,
                        next_w_buf,
                    )
                    tma.async_copy_global_to_shared(
                        p.scale_desc,
                        [0, next_scale_idx, off_k_scale, 0, 0],
                        next_w_ready_bar,
                        next_scale_buf,
                        multicast=p.USE_2CTA and p.W_SCALE_MULTICAST,
                    )
                    idx, phase = advance(next_idx, next_phase, p.w_num_bufs)
                    issued += 2
                else:
                    idx = next_idx
                    phase = next_phase
                    issued += 1


@gluon.jit
def apply_bias_and_scale_explicit_tmem_load(
    p: PartitionArgs,
    idx,
    phase,
    pid_n,
    slice_idx,
    split_layout: gl.constexpr,
    bias_layout: gl.constexpr,
    acc_scale,
    LOAD_LAYOUT_VARIANT: gl.constexpr,
):
    off_n = pid_n * p.BLOCK_N
    acc_empty_bar = p.acc_empty_bars.index(idx)
    acc_ready_bar = p.acc_ready_bars.index(idx)
    acc_buf = p.acc_bufs.index(idx)

    offs_bias_n = off_n + gl.arange(0, p.BLOCK_N, layout=bias_layout)
    if p.USE_DIRECT_EPILOGUE_STORE:
        bias = gl.convert_layout(
            gl.expand_dims(gl.load(p.bias_ptr + slice_idx * p.bias_stride + offs_bias_n), axis=0),
            split_layout,
        )
        mbarrier.wait(acc_ready_bar, phase)
        idx, phase = advance(idx, phase, p.acc_num_bufs)
        gl.static_assert(gl.num_warps() == 4, "explicit tmem load layout scratch supports 4 warps")
        gl.static_assert(p.BLOCK_N == 256, "explicit tmem load layout scratch supports BLOCK_N=256")
        gl.static_assert(p.BLOCK_M == 32, "explicit tmem load layout scratch supports BLOCK_M=32")
        raw_block_bases: gl.constexpr = [[128, 0]] if p.USE_2CTA else []
        if LOAD_LAYOUT_VARIANT == 0:
            raw_acc_layout: gl.constexpr = gl.DistributedLinearLayout(
                reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]],
                lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
                warp_bases=[[32, 0], [64, 0]],
                block_bases=raw_block_bases,
                shape=[p.BLOCK_N, p.BLOCK_M],
            )
        elif LOAD_LAYOUT_VARIANT == 1:
            raw_acc_layout: gl.constexpr = gl.DistributedLinearLayout(
                reg_bases=[[0, 1], [8, 0], [0, 8], [0, 16], [16, 0]],
                lane_bases=[[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]],
                warp_bases=[[32, 0], [64, 0]],
                block_bases=raw_block_bases,
                shape=[p.BLOCK_N, p.BLOCK_M],
            )
        elif LOAD_LAYOUT_VARIANT == 2:
            raw_acc_layout: gl.constexpr = gl.DistributedLinearLayout(
                reg_bases=[[8, 0], [0, 4], [0, 8], [0, 16], [16, 0]],
                lane_bases=[[0, 1], [0, 2], [1, 0], [2, 0], [4, 0]],
                warp_bases=[[32, 0], [64, 0]],
                block_bases=raw_block_bases,
                shape=[p.BLOCK_N, p.BLOCK_M],
            )
        else:
            raw_acc_layout: gl.constexpr = gl.DistributedLinearLayout(
                reg_bases=[[0, 2], [0, 4], [0, 8], [0, 16], [16, 0]],
                lane_bases=[[8, 0], [0, 1], [1, 0], [2, 0], [4, 0]],
                warp_bases=[[32, 0], [64, 0]],
                block_bases=raw_block_bases,
                shape=[p.BLOCK_N, p.BLOCK_M],
            )
        acc_regs_raw = acc_buf.load(raw_acc_layout)
        mbarrier.arrive(acc_empty_bar)
        acc_regs = acc_regs_raw.permute((1, 0))
    else:
        mbarrier.wait(acc_ready_bar, phase)
        idx, phase = advance(idx, phase, p.acc_num_bufs)
        bias = gl.convert_layout(
            gl.expand_dims(gl.load(p.bias_ptr + slice_idx * p.bias_stride + offs_bias_n), axis=0),
            split_layout,
        )
        acc_regs = acc_buf.load().permute((1, 0))
        mbarrier.arrive(acc_empty_bar)
    acc = gl.convert_layout(acc_regs, split_layout)
    acc_packed = float2.pack(acc, axis=1)
    bias_packed = float2.pack(bias, axis=1)
    bias_packed = float2.Float2Tensor(gl.convert_layout(bias_packed.value, acc_packed.value.type.layout))
    acc_packed = float2.fma(acc_packed, float2.full_like(acc_packed, acc_scale), bias_packed)
    return idx, phase, acc_packed


@gluon.jit
def epilogue_explicit_tmem_load_partition_impl(p: PartitionArgs, LOAD_LAYOUT_VARIANT: gl.constexpr):
    idx = 0
    phase = 0
    store_idx = 0
    store_phase = 1
    store_issued = 0

    x_scale = 1.0 if p.x_scale_ptr is None else gl.load(p.x_scale_ptr)
    w_scale = 1.0 if p.w_scale_ptr is None else gl.load(p.w_scale_ptr)
    acc_scale = x_scale * w_scale
    out_recip = 1.0 / gl.load(p.out_scale_ptr)

    num_warps: gl.constexpr = gl.num_warps()
    warps_n: gl.constexpr = 1 if p.FORCE_EPILOGUE_WARPS_N1 else (2 if num_warps >= 8 and p.BLOCK_N >= 256 else 1)
    split_cga_layout: gl.constexpr = ((0, 1), ) if p.USE_2CTA else ()
    split_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4],
        [1, 32],
        [num_warps // warps_n, warps_n],
        [1, 0],
        cga_layout=split_cga_layout,
    )
    bias_layout: gl.constexpr = gl.SliceLayout(0, split_layout)
    store_layout: gl.constexpr = get_store_layout(p)

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        pid_m, pid_n, slice_idx, slice_offset = p.apply_block_schedule(block_id)
        if p.USE_DIRECT_EPILOGUE_STORE:
            off_m = pid_m * p.BLOCK_M
            shape_m = gl.load(p.x_slice_sizes + slice_idx)
            out_off_n_packed = pid_n * (p.BLOCK_N // p.REDUCTION_N // 4)
        idx, phase, acc_packed = apply_bias_and_scale_explicit_tmem_load(
            p,
            idx,
            phase,
            pid_n,
            slice_idx,
            split_layout,
            bias_layout,
            acc_scale,
            LOAD_LAYOUT_VARIANT,
        )
        if p.USE_DIRECT_EPILOGUE_STORE:
            epilogue_direct_store(
                p,
                acc_packed,
                out_recip,
                off_m,
                out_off_n_packed,
                shape_m,
                slice_offset,
                store_layout,
            )
        else:
            store_idx, store_phase, store_issued = epilogue_overlapped_store(
                p,
                acc_packed,
                out_recip,
                store_idx,
                store_phase,
                store_issued,
            )


@gluon.jit
def epilogue_explicit_tmem_load_partition(p: PartitionArgs):
    epilogue_explicit_tmem_load_partition_impl(p, 0)


@gluon.jit
def epilogue_explicit_tmem_load_partition_v1(p: PartitionArgs):
    epilogue_explicit_tmem_load_partition_impl(p, 1)


@gluon.jit
def epilogue_explicit_tmem_load_partition_v2(p: PartitionArgs):
    epilogue_explicit_tmem_load_partition_impl(p, 2)


@gluon.jit
def epilogue_explicit_tmem_load_partition_v3(p: PartitionArgs):
    epilogue_explicit_tmem_load_partition_impl(p, 3)


@gluon.jit
def get_natural_store_layout(p: PartitionArgs):
    frag_rows: gl.constexpr = p.BLOCK_M // p.SWIGLU_SUBTILE_FACTOR
    store_rows: gl.constexpr = p.BLOCK_M if p.USE_WIDE_STORE_HANDOFF else frag_rows
    local_cga_layout: gl.constexpr = ((1, 0), ) if p.USE_2CTA else ()
    return gl.BlockedLayout(
        [store_rows // gl.num_warps(), 2],
        [1, 32],
        [gl.num_warps(), 1],
        [1, 0],
        cga_layout=local_cga_layout,
    )


@gluon.jit
def apply_bias_and_scale_natural_acc(
    p: PartitionArgs,
    idx,
    phase,
    pid_n,
    slice_idx,
    split_layout: gl.constexpr,
    bias_layout: gl.constexpr,
    acc_scale,
):
    off_n = pid_n * p.BLOCK_N
    acc_empty_bar = p.acc_empty_bars.index(idx)
    acc_ready_bar = p.acc_ready_bars.index(idx)
    acc_buf = p.acc_bufs.index(idx)

    offs_bias_n = off_n + gl.arange(0, p.BLOCK_N, layout=bias_layout)
    bias = gl.convert_layout(
        gl.expand_dims(gl.load(p.bias_ptr + slice_idx * p.bias_stride + offs_bias_n), axis=0),
        split_layout,
    )
    mbarrier.wait(acc_ready_bar, phase)
    idx, phase = advance(idx, phase, p.acc_num_bufs)
    acc_regs = acc_buf.load()
    mbarrier.arrive(acc_empty_bar)
    acc = gl.convert_layout(acc_regs, split_layout)
    acc_packed = float2.pack(acc, axis=1)
    bias_packed = float2.pack(bias, axis=1)
    bias_packed = float2.Float2Tensor(gl.convert_layout(bias_packed.value, acc_packed.value.type.layout))
    acc_packed = float2.fma(acc_packed, float2.full_like(acc_packed, acc_scale), bias_packed)
    return idx, phase, acc_packed


@gluon.jit
def epilogue_natural_acc_partition(p: PartitionArgs):
    idx = 0
    phase = 0

    x_scale = 1.0 if p.x_scale_ptr is None else gl.load(p.x_scale_ptr)
    w_scale = 1.0 if p.w_scale_ptr is None else gl.load(p.w_scale_ptr)
    acc_scale = x_scale * w_scale
    out_recip = 1.0 / gl.load(p.out_scale_ptr)

    num_warps: gl.constexpr = gl.num_warps()
    warps_n: gl.constexpr = 1 if p.FORCE_EPILOGUE_WARPS_N1 else (2 if num_warps >= 8 and p.BLOCK_N >= 256 else 1)
    split_cga_layout: gl.constexpr = ((1, 0), ) if p.USE_2CTA else ()
    split_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4],
        [1, 32],
        [num_warps // warps_n, warps_n],
        [1, 0],
        cga_layout=split_cga_layout,
    )
    bias_layout: gl.constexpr = gl.SliceLayout(0, split_layout)
    store_layout: gl.constexpr = get_natural_store_layout(p)

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        pid_m, pid_n, slice_idx, slice_offset = p.apply_block_schedule(block_id)
        off_m = pid_m * p.BLOCK_M
        shape_m = gl.load(p.x_slice_sizes + slice_idx)
        out_off_n_packed = pid_n * (p.BLOCK_N // p.REDUCTION_N // 4)
        idx, phase, acc_packed = apply_bias_and_scale_natural_acc(
            p,
            idx,
            phase,
            pid_n,
            slice_idx,
            split_layout,
            bias_layout,
            acc_scale,
        )
        epilogue_direct_store(
            p,
            acc_packed,
            out_recip,
            off_m,
            out_off_n_packed,
            shape_m,
            slice_offset,
            store_layout,
        )


@gluon.jit
def mma_natural_acc_partition(p: PartitionArgs):
    x_idx = 0
    x_phase = 0
    w_idx = 0
    w_phase = 0
    mma_idx = 0
    mma_phase = 1

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        acc_empty_bar = p.acc_empty_bars.index(mma_idx)
        acc_ready_bar = p.acc_ready_bars.index(mma_idx)
        acc_buf = p.acc_bufs.index(mma_idx)
        mbarrier.wait(acc_empty_bar, mma_phase)

        use_acc = False
        for _ in range(p.K_TILES):
            x_ready_bar = p.x_ready_bars.index(x_idx)
            x_empty_bar = p.x_empty_bars.index(x_idx)
            x_buf = p.x_bufs.index(x_idx)
            mbarrier.wait(x_ready_bar, x_phase)

            w_ready_bar = p.w_ready_bars.index(w_idx)
            w_empty_bar = p.w_empty_bars.index(w_idx)
            w_buf = p.w_bufs.index(w_idx)
            scale_buf = p.w_scale_bufs.index(w_idx)
            mbarrier.wait(w_ready_bar, w_phase)

            blackwell.tcgen05_copy(
                unswizzle_mx_scale(scale_buf, p.SCALE_SIZE_OUTER, p.SCALE_SIZE_INNER, p.MXFP_BLOCK_SIZE),
                p.w_scale_tmem,
            )

            blackwell.tcgen05_mma_scaled(
                x_buf,
                w_buf.reshape((p.BLOCK_N, p.BLOCK_K // 2)).permute((1, 0)),
                acc_buf,
                p.x_scale_tmem,
                p.w_scale_tmem,
                a_type="e4m3",
                b_type="e2m1",
                use_acc=use_acc,
                multicast=p.USE_2CTA,
            )
            blackwell.tcgen05_commit(x_empty_bar)
            blackwell.tcgen05_commit(w_empty_bar)

            x_idx, x_phase = advance(x_idx, x_phase, p.x_num_bufs)
            w_idx, w_phase = advance(w_idx, w_phase, p.w_num_bufs)
            use_acc = True

        blackwell.tcgen05_commit(acc_ready_bar)
        mma_idx, mma_phase = advance(mma_idx, mma_phase, p.acc_num_bufs)


@gluon.jit
def mma_multicast_partition(p: PartitionArgs):
    x_idx = 0
    x_phase = 0
    w_idx = 0
    w_phase = 0
    mma_idx = 0
    mma_phase = 1

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        acc_empty_bar = p.acc_empty_bars.index(mma_idx)
        acc_ready_bar = p.acc_ready_bars.index(mma_idx)
        acc_buf = p.acc_bufs.index(mma_idx)
        mbarrier.wait(acc_empty_bar, mma_phase)

        use_acc = False
        for _ in range(p.K_TILES):
            w_ready_bar = p.w_ready_bars.index(w_idx)
            w_empty_bar = p.w_empty_bars.index(w_idx)
            w_buf = p.w_bufs.index(w_idx)
            scale_buf = p.w_scale_bufs.index(w_idx)
            mbarrier.wait(w_ready_bar, w_phase)

            blackwell.tcgen05_copy(
                unswizzle_mx_scale(scale_buf, p.SCALE_SIZE_OUTER, p.SCALE_SIZE_INNER, p.MXFP_BLOCK_SIZE),
                p.w_scale_tmem,
            )

            x_ready_bar = p.x_ready_bars.index(x_idx)
            x_empty_bar = p.x_empty_bars.index(x_idx)
            x_buf = p.x_bufs.index(x_idx)
            mbarrier.wait(x_ready_bar, x_phase)

            blackwell.tcgen05_mma_scaled(
                w_buf.reshape((p.BLOCK_N, p.BLOCK_K // 2)),
                x_buf.permute((1, 0)),
                acc_buf,
                p.w_scale_tmem,
                p.x_scale_tmem,
                a_type="e2m1",
                b_type="e4m3",
                use_acc=use_acc,
                multicast=p.USE_2CTA,
            )
            blackwell.tcgen05_commit(x_empty_bar)
            blackwell.tcgen05_commit(w_empty_bar)

            x_idx, x_phase = advance(x_idx, x_phase, p.x_num_bufs)
            w_idx, w_phase = advance(w_idx, w_phase, p.w_num_bufs)
            use_acc = True

        blackwell.tcgen05_commit(acc_ready_bar)
        mma_idx, mma_phase = advance(mma_idx, mma_phase, p.acc_num_bufs)


@gluon.jit
def mma_multicast_counted_partition(p: PartitionArgs):
    x_idx = 0
    x_phase = 0
    w_idx = 0
    w_phase = 0
    mma_idx = 0
    mma_phase = 1

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        acc_empty_bar = p.acc_empty_bars.index(mma_idx)
        acc_ready_bar = p.acc_ready_bars.index(mma_idx)
        acc_buf = p.acc_bufs.index(mma_idx)
        mbarrier.wait(acc_empty_bar, mma_phase)

        use_acc = False
        for _ in range(p.K_TILES):
            w_ready_bar = p.w_ready_bars.index(w_idx)
            w_empty_bar = p.w_empty_bars.index(w_idx)
            w_buf = p.w_bufs.index(w_idx)
            scale_buf = p.w_scale_bufs.index(w_idx)
            mbarrier.wait(w_ready_bar, w_phase)

            blackwell.tcgen05_copy(
                unswizzle_mx_scale(scale_buf, p.SCALE_SIZE_OUTER, p.SCALE_SIZE_INNER, p.MXFP_BLOCK_SIZE),
                p.w_scale_tmem,
            )

            x_ready_bar = p.x_ready_bars.index(x_idx)
            x_empty_bar = p.x_empty_bars.index(x_idx)
            x_buf = p.x_bufs.index(x_idx)
            mbarrier.wait(x_ready_bar, x_phase)

            blackwell.tcgen05_mma_scaled(
                w_buf.reshape((p.BLOCK_N, p.BLOCK_K // 2)),
                x_buf.permute((1, 0)),
                acc_buf,
                p.w_scale_tmem,
                p.x_scale_tmem,
                a_type="e2m1",
                b_type="e4m3",
                use_acc=use_acc,
                multicast=p.USE_2CTA,
            )
            blackwell.tcgen05_commit(x_empty_bar, descs=[p.w_bufs.index(0), p.x_bufs.index(0)])
            blackwell.tcgen05_commit(w_empty_bar, descs=[p.w_bufs.index(0), p.x_bufs.index(0)])

            x_idx, x_phase = advance(x_idx, x_phase, p.x_num_bufs)
            w_idx, w_phase = advance(w_idx, w_phase, p.w_num_bufs)
            use_acc = True

        blackwell.tcgen05_commit(acc_ready_bar, descs=[p.w_bufs.index(0), p.x_bufs.index(0)])
        mma_idx, mma_phase = advance(mma_idx, mma_phase, p.acc_num_bufs)


@gluon.jit
def mma_pair_reuse_partition(p: PartitionArgs):
    x_idx = 0
    x_phase = 0
    w_idx = 0
    w_phase = 0
    mma_idx = 0
    mma_phase = 1

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        _, pid_n, slice_idx, _ = p.apply_block_schedule(block_id)
        prev_block_id = block_id - p.NUM_SMS
        has_prev = block_id >= p.NUM_SMS
        safe_prev_block_id = gl.where(has_prev, prev_block_id, block_id)
        _, prev_pid_n, prev_slice_idx, _ = p.apply_block_schedule(safe_prev_block_id)
        is_second_reuse = has_prev & (prev_pid_n == pid_n) & (prev_slice_idx == slice_idx)

        should_process = is_second_reuse == False
        if should_process:
            next_block_id = block_id + p.NUM_SMS
            has_next = next_block_id < p.num_blocks
            safe_next_block_id = gl.where(has_next, next_block_id, block_id)
            _, next_pid_n, next_slice_idx, _ = p.apply_block_schedule(safe_next_block_id)
            pair_next = has_next & (next_pid_n == pid_n) & (next_slice_idx == slice_idx)

            acc_empty_bar = p.acc_empty_bars.index(mma_idx)
            acc_ready_bar = p.acc_ready_bars.index(mma_idx)
            acc_buf = p.acc_bufs.index(mma_idx)
            mbarrier.wait(acc_empty_bar, mma_phase)

            next_mma_idx, next_mma_phase = advance(mma_idx, mma_phase, p.acc_num_bufs)
            next_acc_empty_bar = p.acc_empty_bars.index(next_mma_idx)
            next_acc_ready_bar = p.acc_ready_bars.index(next_mma_idx)
            next_acc_buf = p.acc_bufs.index(next_mma_idx)
            mbarrier.wait(next_acc_empty_bar, next_mma_phase, pred=pair_next)

            use_acc = False
            for _ in range(p.K_TILES):
                w_ready_bar = p.w_ready_bars.index(w_idx)
                w_empty_bar = p.w_empty_bars.index(w_idx)
                w_buf = p.w_bufs.index(w_idx)
                scale_buf = p.w_scale_bufs.index(w_idx)
                mbarrier.wait(w_ready_bar, w_phase)

                blackwell.tcgen05_copy(
                    unswizzle_mx_scale(scale_buf, p.SCALE_SIZE_OUTER, p.SCALE_SIZE_INNER, p.MXFP_BLOCK_SIZE),
                    p.w_scale_tmem,
                )

                x_ready_bar = p.x_ready_bars.index(x_idx)
                x_empty_bar = p.x_empty_bars.index(x_idx)
                x_buf = p.x_bufs.index(x_idx)
                mbarrier.wait(x_ready_bar, x_phase)

                blackwell.tcgen05_mma_scaled(
                    w_buf.reshape((p.BLOCK_N, p.BLOCK_K // 2)),
                    x_buf.permute((1, 0)),
                    acc_buf,
                    p.w_scale_tmem,
                    p.x_scale_tmem,
                    a_type="e2m1",
                    b_type="e4m3",
                    use_acc=use_acc,
                )
                blackwell.tcgen05_commit(x_empty_bar)

                next_x_idx, next_x_phase = advance(x_idx, x_phase, p.x_num_bufs)
                if pair_next:
                    next_x_ready_bar = p.x_ready_bars.index(next_x_idx)
                    next_x_empty_bar = p.x_empty_bars.index(next_x_idx)
                    next_x_buf = p.x_bufs.index(next_x_idx)
                    mbarrier.wait(next_x_ready_bar, next_x_phase)
                    blackwell.tcgen05_mma_scaled(
                        w_buf.reshape((p.BLOCK_N, p.BLOCK_K // 2)),
                        next_x_buf.permute((1, 0)),
                        next_acc_buf,
                        p.w_scale_tmem,
                        p.x_scale_tmem,
                        a_type="e2m1",
                        b_type="e4m3",
                        use_acc=use_acc,
                    )
                    blackwell.tcgen05_commit(next_x_empty_bar)
                    x_idx, x_phase = advance(next_x_idx, next_x_phase, p.x_num_bufs)
                else:
                    x_idx = next_x_idx
                    x_phase = next_x_phase

                blackwell.tcgen05_commit(w_empty_bar)
                w_idx, w_phase = advance(w_idx, w_phase, p.w_num_bufs)
                use_acc = True

            blackwell.tcgen05_commit(acc_ready_bar)
            blackwell.tcgen05_commit(next_acc_ready_bar, pred=pair_next)
            if pair_next:
                mma_idx, mma_phase = advance(next_mma_idx, next_mma_phase, p.acc_num_bufs)
            else:
                mma_idx = next_mma_idx
                mma_phase = next_mma_phase


@gluon.jit
def mma_transition_pair_partition(p: PartitionArgs):
    x_idx = 0
    x_phase = 0
    w_idx = 0
    w_phase = 0
    mma_idx = 0
    mma_phase = 1

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        pid_m, _, slice_idx, _ = p.apply_block_schedule(block_id)
        shape_m = gl.load(p.x_slice_sizes + slice_idx)
        cur_full = is_full_m_tile(pid_m, shape_m, p.BLOCK_M)

        prev_block_id = block_id - p.NUM_SMS
        has_prev = block_id >= p.NUM_SMS
        safe_prev_block_id = gl.where(has_prev, prev_block_id, block_id)
        prev_pid_m, _, prev_slice_idx, _ = p.apply_block_schedule(safe_prev_block_id)
        prev_shape_m = gl.load(p.x_slice_sizes + prev_slice_idx)
        prev_full = is_full_m_tile(prev_pid_m, prev_shape_m, p.BLOCK_M)
        is_second_pair = has_prev & (prev_full != cur_full)

        should_process = is_second_pair == False
        if should_process:
            next_block_id = block_id + p.NUM_SMS
            has_next = next_block_id < p.num_blocks
            safe_next_block_id = gl.where(has_next, next_block_id, block_id)
            next_pid_m, _, next_slice_idx, _ = p.apply_block_schedule(safe_next_block_id)
            next_shape_m = gl.load(p.x_slice_sizes + next_slice_idx)
            next_full = is_full_m_tile(next_pid_m, next_shape_m, p.BLOCK_M)
            pair_next = has_next & (cur_full != next_full)

            acc_empty_bar = p.acc_empty_bars.index(mma_idx)
            acc_ready_bar = p.acc_ready_bars.index(mma_idx)
            acc_buf = p.acc_bufs.index(mma_idx)
            mbarrier.wait(acc_empty_bar, mma_phase)

            next_mma_idx, next_mma_phase = advance(mma_idx, mma_phase, p.acc_num_bufs)
            next_acc_empty_bar = p.acc_empty_bars.index(next_mma_idx)
            next_acc_ready_bar = p.acc_ready_bars.index(next_mma_idx)
            next_acc_buf = p.acc_bufs.index(next_mma_idx)
            mbarrier.wait(next_acc_empty_bar, next_mma_phase, pred=pair_next)

            use_acc = False
            for _ in range(p.K_TILES):
                w_ready_bar = p.w_ready_bars.index(w_idx)
                w_empty_bar = p.w_empty_bars.index(w_idx)
                w_buf = p.w_bufs.index(w_idx)
                scale_buf = p.w_scale_bufs.index(w_idx)
                mbarrier.wait(w_ready_bar, w_phase)

                blackwell.tcgen05_copy(
                    unswizzle_mx_scale(scale_buf, p.SCALE_SIZE_OUTER, p.SCALE_SIZE_INNER, p.MXFP_BLOCK_SIZE),
                    p.w_scale_tmem,
                )

                x_ready_bar = p.x_ready_bars.index(x_idx)
                x_empty_bar = p.x_empty_bars.index(x_idx)
                x_buf = p.x_bufs.index(x_idx)
                mbarrier.wait(x_ready_bar, x_phase)

                blackwell.tcgen05_mma_scaled(
                    w_buf.reshape((p.BLOCK_N, p.BLOCK_K // 2)),
                    x_buf.permute((1, 0)),
                    acc_buf,
                    p.w_scale_tmem,
                    p.x_scale_tmem,
                    a_type="e2m1",
                    b_type="e4m3",
                    use_acc=use_acc,
                )
                blackwell.tcgen05_commit(x_empty_bar)
                blackwell.tcgen05_commit(w_empty_bar)

                next_x_idx, next_x_phase = advance(x_idx, x_phase, p.x_num_bufs)
                next_w_idx, next_w_phase = advance(w_idx, w_phase, p.w_num_bufs)
                if pair_next:
                    next_w_ready_bar = p.w_ready_bars.index(next_w_idx)
                    next_w_empty_bar = p.w_empty_bars.index(next_w_idx)
                    next_w_buf = p.w_bufs.index(next_w_idx)
                    next_scale_buf = p.w_scale_bufs.index(next_w_idx)
                    mbarrier.wait(next_w_ready_bar, next_w_phase)
                    blackwell.tcgen05_copy(
                        unswizzle_mx_scale(next_scale_buf, p.SCALE_SIZE_OUTER, p.SCALE_SIZE_INNER,
                                           p.MXFP_BLOCK_SIZE),
                        p.w_scale_tmem,
                    )

                    next_x_ready_bar = p.x_ready_bars.index(next_x_idx)
                    next_x_empty_bar = p.x_empty_bars.index(next_x_idx)
                    next_x_buf = p.x_bufs.index(next_x_idx)
                    mbarrier.wait(next_x_ready_bar, next_x_phase)
                    blackwell.tcgen05_mma_scaled(
                        next_w_buf.reshape((p.BLOCK_N, p.BLOCK_K // 2)),
                        next_x_buf.permute((1, 0)),
                        next_acc_buf,
                        p.w_scale_tmem,
                        p.x_scale_tmem,
                        a_type="e2m1",
                        b_type="e4m3",
                        use_acc=use_acc,
                    )
                    blackwell.tcgen05_commit(next_x_empty_bar)
                    blackwell.tcgen05_commit(next_w_empty_bar)
                    x_idx, x_phase = advance(next_x_idx, next_x_phase, p.x_num_bufs)
                    w_idx, w_phase = advance(next_w_idx, next_w_phase, p.w_num_bufs)
                else:
                    x_idx = next_x_idx
                    x_phase = next_x_phase
                    w_idx = next_w_idx
                    w_phase = next_w_phase

                use_acc = True

            blackwell.tcgen05_commit(acc_ready_bar)
            blackwell.tcgen05_commit(next_acc_ready_bar, pred=pair_next)
            if pair_next:
                mma_idx, mma_phase = advance(next_mma_idx, next_mma_phase, p.acc_num_bufs)
            else:
                mma_idx = next_mma_idx
                mma_phase = next_mma_phase


@gluon.jit
def mma_transition_reuse_pair_partition(p: PartitionArgs):
    x_idx = 0
    x_phase = 0
    w_idx = 0
    w_phase = 0
    mma_idx = 0
    mma_phase = 1

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        pid_m, pid_n, slice_idx, _ = p.apply_block_schedule(block_id)
        shape_m = gl.load(p.x_slice_sizes + slice_idx)
        cur_full = is_full_m_tile(pid_m, shape_m, p.BLOCK_M)

        prev_block_id = block_id - p.NUM_SMS
        has_prev = block_id >= p.NUM_SMS
        safe_prev_block_id = gl.where(has_prev, prev_block_id, block_id)
        prev_pid_m, prev_pid_n, prev_slice_idx, _ = p.apply_block_schedule(safe_prev_block_id)
        prev_shape_m = gl.load(p.x_slice_sizes + prev_slice_idx)
        prev_full = is_full_m_tile(prev_pid_m, prev_shape_m, p.BLOCK_M)
        is_second_pair = has_prev & (prev_pid_n == pid_n) & (prev_slice_idx == slice_idx) & (prev_full != cur_full)

        should_process = is_second_pair == False
        if should_process:
            next_block_id = block_id + p.NUM_SMS
            has_next = next_block_id < p.num_blocks
            safe_next_block_id = gl.where(has_next, next_block_id, block_id)
            next_pid_m, next_pid_n, next_slice_idx, _ = p.apply_block_schedule(safe_next_block_id)
            next_shape_m = gl.load(p.x_slice_sizes + next_slice_idx)
            next_full = is_full_m_tile(next_pid_m, next_shape_m, p.BLOCK_M)
            pair_next = has_next & (next_pid_n == pid_n) & (next_slice_idx == slice_idx) & (cur_full != next_full)

            acc_empty_bar = p.acc_empty_bars.index(mma_idx)
            acc_ready_bar = p.acc_ready_bars.index(mma_idx)
            acc_buf = p.acc_bufs.index(mma_idx)
            mbarrier.wait(acc_empty_bar, mma_phase)

            next_mma_idx, next_mma_phase = advance(mma_idx, mma_phase, p.acc_num_bufs)
            next_acc_empty_bar = p.acc_empty_bars.index(next_mma_idx)
            next_acc_ready_bar = p.acc_ready_bars.index(next_mma_idx)
            next_acc_buf = p.acc_bufs.index(next_mma_idx)
            mbarrier.wait(next_acc_empty_bar, next_mma_phase, pred=pair_next)

            use_acc = False
            for _ in range(p.K_TILES):
                w_ready_bar = p.w_ready_bars.index(w_idx)
                w_empty_bar = p.w_empty_bars.index(w_idx)
                w_buf = p.w_bufs.index(w_idx)
                scale_buf = p.w_scale_bufs.index(w_idx)
                mbarrier.wait(w_ready_bar, w_phase)

                blackwell.tcgen05_copy(
                    unswizzle_mx_scale(scale_buf, p.SCALE_SIZE_OUTER, p.SCALE_SIZE_INNER, p.MXFP_BLOCK_SIZE),
                    p.w_scale_tmem,
                )

                x_ready_bar = p.x_ready_bars.index(x_idx)
                x_empty_bar = p.x_empty_bars.index(x_idx)
                x_buf = p.x_bufs.index(x_idx)
                mbarrier.wait(x_ready_bar, x_phase)

                blackwell.tcgen05_mma_scaled(
                    w_buf.reshape((p.BLOCK_N, p.BLOCK_K // 2)),
                    x_buf.permute((1, 0)),
                    acc_buf,
                    p.w_scale_tmem,
                    p.x_scale_tmem,
                    a_type="e2m1",
                    b_type="e4m3",
                    use_acc=use_acc,
                )
                blackwell.tcgen05_commit(x_empty_bar)

                next_x_idx, next_x_phase = advance(x_idx, x_phase, p.x_num_bufs)
                if pair_next:
                    next_x_ready_bar = p.x_ready_bars.index(next_x_idx)
                    next_x_empty_bar = p.x_empty_bars.index(next_x_idx)
                    next_x_buf = p.x_bufs.index(next_x_idx)
                    mbarrier.wait(next_x_ready_bar, next_x_phase)
                    blackwell.tcgen05_mma_scaled(
                        w_buf.reshape((p.BLOCK_N, p.BLOCK_K // 2)),
                        next_x_buf.permute((1, 0)),
                        next_acc_buf,
                        p.w_scale_tmem,
                        p.x_scale_tmem,
                        a_type="e2m1",
                        b_type="e4m3",
                        use_acc=use_acc,
                    )
                    blackwell.tcgen05_commit(next_x_empty_bar)
                    x_idx, x_phase = advance(next_x_idx, next_x_phase, p.x_num_bufs)
                else:
                    x_idx = next_x_idx
                    x_phase = next_x_phase

                blackwell.tcgen05_commit(w_empty_bar)
                w_idx, w_phase = advance(w_idx, w_phase, p.w_num_bufs)
                use_acc = True

            blackwell.tcgen05_commit(acc_ready_bar)
            blackwell.tcgen05_commit(next_acc_ready_bar, pred=pair_next)
            if pair_next:
                mma_idx, mma_phase = advance(next_mma_idx, next_mma_phase, p.acc_num_bufs)
            else:
                mma_idx = next_mma_idx
                mma_phase = next_mma_phase


@gluon.jit
def mma_load_weights_partition(p: PartitionArgs):
    tile_w_bytes: gl.constexpr = p.w_desc.nbytes_per_cta
    tile_scale_bytes: gl.constexpr = p.scale_desc.nbytes_per_cta
    bytes_per_w_stage: gl.constexpr = tile_w_bytes + tile_scale_bytes
    scale_k_stride: gl.constexpr = p.BLOCK_K // (p.MXFP_BLOCK_SIZE * p.SCALE_SIZE_INNER)

    x_idx = 0
    x_phase = 0
    w_idx = 0
    w_empty_phase = 1
    w_ready_phase = 0
    w_issued = 0
    mma_idx = 0
    mma_phase = 1

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        acc_empty_bar = p.acc_empty_bars.index(mma_idx)
        acc_ready_bar = p.acc_ready_bars.index(mma_idx)
        acc_buf = p.acc_bufs.index(mma_idx)
        mbarrier.wait(acc_empty_bar, mma_phase)

        pid_n, slice_idx = p.apply_weight_schedule(block_id)
        scale_idx = slice_idx * p.SCALE_FLAT_N + pid_n * p.SCALE_BLOCK_N_DIV

        use_acc = False
        for ki in range(p.K_TILES):
            off_k_scale = ki * scale_k_stride
            w_empty_bar = p.w_empty_bars.index(w_idx)
            w_ready_bar = p.w_ready_bars.index(w_idx)
            w_buf = p.w_bufs.index(w_idx)
            scale_buf = p.w_scale_bufs.index(w_idx)

            mbarrier.wait(w_empty_bar, w_empty_phase, pred=w_issued >= p.w_num_bufs)
            mbarrier.expect(w_ready_bar, bytes_per_w_stage)
            tma.async_copy_global_to_shared(p.w_desc, [slice_idx, ki, pid_n, 0, 0], w_ready_bar, w_buf)
            tma.async_copy_global_to_shared(
                p.scale_desc,
                [0, scale_idx, off_k_scale, 0, 0],
                w_ready_bar,
                scale_buf,
                multicast=p.USE_2CTA and p.W_SCALE_MULTICAST,
            )

            x_ready_bar = p.x_ready_bars.index(x_idx)
            x_empty_bar = p.x_empty_bars.index(x_idx)
            x_buf = p.x_bufs.index(x_idx)
            mbarrier.wait(x_ready_bar, x_phase)
            mbarrier.wait(w_ready_bar, w_ready_phase)

            blackwell.tcgen05_copy(
                unswizzle_mx_scale(scale_buf, p.SCALE_SIZE_OUTER, p.SCALE_SIZE_INNER, p.MXFP_BLOCK_SIZE),
                p.w_scale_tmem,
            )
            blackwell.tcgen05_mma_scaled(
                w_buf.reshape((p.BLOCK_N, p.BLOCK_K // 2)),
                x_buf.permute((1, 0)),
                acc_buf,
                p.w_scale_tmem,
                p.x_scale_tmem,
                a_type="e2m1",
                b_type="e4m3",
                use_acc=use_acc,
            )
            blackwell.tcgen05_commit(x_empty_bar)
            blackwell.tcgen05_commit(w_empty_bar)

            next_w_idx, next_w_empty_phase = advance(w_idx, w_empty_phase, p.w_num_bufs)
            _, next_w_ready_phase = advance(w_idx, w_ready_phase, p.w_num_bufs)
            x_idx, x_phase = advance(x_idx, x_phase, p.x_num_bufs)
            w_idx = next_w_idx
            w_empty_phase = next_w_empty_phase
            w_ready_phase = next_w_ready_phase
            w_issued += 1
            use_acc = True

        blackwell.tcgen05_commit(acc_ready_bar)
        mma_idx, mma_phase = advance(mma_idx, mma_phase, p.acc_num_bufs)


@gluon.jit
def mma_load_activations_partition(p: PartitionArgs):
    local_cga_layout: gl.constexpr = ((0, 1), ) if p.USE_2CTA else ()
    offs_layout: gl.constexpr = gl.SliceLayout(
        dim=0,
        parent=gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0], cga_layout=local_cga_layout),
    )
    tile_x_bytes: gl.constexpr = p.x_desc.block_type.nbytes * (p.BLOCK_M_PER_CTA if p.USE_2CTA else p.BLOCK_M)

    x_idx = 0
    x_empty_phase = 1
    x_ready_phase = 0
    x_issued = 0
    w_idx = 0
    w_phase = 0
    mma_idx = 0
    mma_phase = 1

    for block_id in range(gl.program_id(0), p.num_blocks, p.NUM_SMS):
        acc_empty_bar = p.acc_empty_bars.index(mma_idx)
        acc_ready_bar = p.acc_ready_bars.index(mma_idx)
        acc_buf = p.acc_bufs.index(mma_idx)
        mbarrier.wait(acc_empty_bar, mma_phase)

        pid_m, _, slice_idx, slice_offset = p.apply_block_schedule(block_id)
        off_m = pid_m * p.BLOCK_M
        shape_m = gl.load(p.x_slice_sizes + slice_idx)
        offs_m = off_m + gl.arange(0, p.BLOCK_M, layout=offs_layout)
        mask_m = offs_m < shape_m
        offs_x_m = gl.load(
            p.gather_indx_ptr + slice_offset + offs_m,
            mask=mask_m,
            other=p.x_desc.shape[0],
        )

        use_acc = False
        for ki in range(p.K_TILES):
            off_k_x = ki * p.BLOCK_K
            x_empty_bar = p.x_empty_bars.index(x_idx)
            x_ready_bar = p.x_ready_bars.index(x_idx)
            x_buf = p.x_bufs.index(x_idx)

            mbarrier.wait(x_empty_bar, x_empty_phase, pred=x_issued >= p.x_num_bufs)
            mbarrier.expect(x_ready_bar, tile_x_bytes)
            tma.async_gather(
                p.x_desc,
                offs_x_m,
                off_k_x,
                x_ready_bar,
                x_buf,
                multicast=p.USE_2CTA and p.X_GATHER_MULTICAST,
            )

            w_ready_bar = p.w_ready_bars.index(w_idx)
            w_empty_bar = p.w_empty_bars.index(w_idx)
            w_buf = p.w_bufs.index(w_idx)
            scale_buf = p.w_scale_bufs.index(w_idx)
            mbarrier.wait(w_ready_bar, w_phase)
            mbarrier.wait(x_ready_bar, x_ready_phase)

            blackwell.tcgen05_copy(
                unswizzle_mx_scale(scale_buf, p.SCALE_SIZE_OUTER, p.SCALE_SIZE_INNER, p.MXFP_BLOCK_SIZE),
                p.w_scale_tmem,
            )
            blackwell.tcgen05_mma_scaled(
                w_buf.reshape((p.BLOCK_N, p.BLOCK_K // 2)),
                x_buf.permute((1, 0)),
                acc_buf,
                p.w_scale_tmem,
                p.x_scale_tmem,
                a_type="e2m1",
                b_type="e4m3",
                use_acc=use_acc,
            )
            blackwell.tcgen05_commit(x_empty_bar)
            blackwell.tcgen05_commit(w_empty_bar)

            next_x_idx, next_x_empty_phase = advance(x_idx, x_empty_phase, p.x_num_bufs)
            _, next_x_ready_phase = advance(x_idx, x_ready_phase, p.x_num_bufs)
            x_idx = next_x_idx
            x_empty_phase = next_x_empty_phase
            x_ready_phase = next_x_ready_phase
            w_idx, w_phase = advance(w_idx, w_phase, p.w_num_bufs)
            x_issued += 1
            use_acc = True

        blackwell.tcgen05_commit(acc_ready_bar)
        mma_idx, mma_phase = advance(mma_idx, mma_phase, p.acc_num_bufs)


@gluon.jit
def ws_matmul_combined_load_kernel(
    x_desc: tma.tensor_descriptor,
    w_desc: tma.tensor_descriptor,
    scale_desc: tma.tensor_descriptor,
    out_desc: tma.tensor_descriptor,
    out_ptr: gl.tensor,
    bias_ptr: gl.tensor,
    bias_stride: gl.tensor,
    gather_indx_ptr: gl.tensor,
    x_slice_sizes: gl.tensor,
    x_slice_offs: gl.tensor,
    x_block_offs: gl.tensor,
    x_block_schedule: gl.tensor,
    x_tile_schedule: gl.tensor,
    x_scale_ptr: gl.tensor,
    w_scale_ptr: gl.tensor,
    out_scale_ptr: gl.tensor,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    NUM_SLICES: gl.constexpr,
    SWIGLU_ALPHA: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
    REDUCTION_N: gl.constexpr,
    FLEXPOINT_SATURATE_INF: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_SMS: gl.constexpr,
    X_NUM_BUFS: gl.constexpr,
    W_NUM_BUFS: gl.constexpr,
    ACC_NUM_BUFS: gl.constexpr,
    LOAD_ACTIVATION_WARPS: gl.constexpr,
    LOAD_WEIGHT_WARPS: gl.constexpr,
    MMA_WARPS: gl.constexpr,
    STORE_HELPER_WARPS: gl.constexpr,
    LOAD_ACTIVATION_REGS: gl.constexpr,
    LOAD_WEIGHT_REGS: gl.constexpr,
    MMA_REGS: gl.constexpr,
    STORE_HELPER_REGS: gl.constexpr,
    SWIGLU_SUBTILE_FACTOR: gl.constexpr,
    EPILOGUE_BUFFER_DEPTH: gl.constexpr,
    USE_PLANAR_SNAKE: gl.constexpr,
    GRID_MINOR_DIM: gl.constexpr,
    GRID_TILE_WIDTH: gl.constexpr,
    BAND_N: gl.constexpr,
    USE_FULL_TILE_SCHEDULE: gl.constexpr,
    X_GATHER_MULTICAST: gl.constexpr,
    W_SCALE_MULTICAST: gl.constexpr,
    FORCE_EPILOGUE_WARPS_N1: gl.constexpr,
    USE_WIDE_STORE_HANDOFF: gl.constexpr,
    USE_DIRECT_EPILOGUE_STORE: gl.constexpr,
    REUSE_GATHER_INDICES: gl.constexpr,
    INLINE_MMA_INPUT_RELEASE: gl.constexpr,
    SCALE_SIZE_OUTER: gl.constexpr,
    SCALE_SIZE_INNER: gl.constexpr,
    MXFP_BLOCK_SIZE: gl.constexpr,
    STRUCTURAL_MODE: gl.constexpr,
):
    use_2cta: gl.constexpr = gl.num_ctas() > 1
    gl.static_assert(gl.num_ctas() == 1 or gl.num_ctas() == 2, "kernel supports at most 2 CTAs")
    gl.static_assert(USE_DIRECT_EPILOGUE_STORE, "combined-loader prototype only supports direct epilogue")
    gl.static_assert(not REUSE_GATHER_INDICES, "combined-loader prototype does not cache gathered rows")

    grid_m = gl.load(x_block_offs + NUM_SLICES)
    grid_n: gl.constexpr = triton.cdiv(N, BLOCK_N)
    k_tiles: gl.constexpr = triton.cdiv(K, BLOCK_K)
    scale_flat_n: gl.constexpr = N // SCALE_SIZE_OUTER
    scale_block_n_div: gl.constexpr = BLOCK_N // SCALE_SIZE_OUTER
    num_blocks = grid_m * grid_n

    scale_k: gl.constexpr = BLOCK_K // MXFP_BLOCK_SIZE
    natural_acc: gl.constexpr = STRUCTURAL_MODE == 14
    fake_nsplit: gl.constexpr = STRUCTURAL_MODE == 15
    mma_two_ctas: gl.constexpr = use_2cta and not fake_nsplit
    block_m_per_cta: gl.constexpr = BLOCK_M // gl.num_ctas()
    gl.static_assert(
        not (natural_acc and use_2cta),
        "natural 2CTA scratch is classified invalid; use fakens or transposed true-2CTA modes",
    )
    x_scale_layout: gl.constexpr = blackwell.TensorMemoryScalesLayout(
        cga_layout=((1, 0), ) if use_2cta and natural_acc else (
            ((0, 0), ) if fake_nsplit else (((0, 0), ) if use_2cta else ())
        ),
    )
    w_scale_layout: gl.constexpr = blackwell.TensorMemoryScalesLayout(
        cga_layout=((1, 0), ) if use_2cta and natural_acc else (((1, 0), ) if use_2cta else ()),
    )

    x_num_bufs: gl.constexpr = X_NUM_BUFS
    x_bufs = gl.allocate_shared_memory(
        x_desc.dtype,
        [x_num_bufs, BLOCK_M, x_desc.block_type.shape[1]],
        x_desc.layout,
    )

    w_num_bufs: gl.constexpr = W_NUM_BUFS
    w_bufs = gl.allocate_shared_memory(
        w_desc.dtype,
        [w_num_bufs] + w_desc.block_type.shape,
        w_desc.layout,
    )
    w_scale_bufs = gl.allocate_shared_memory(
        scale_desc.dtype,
        [w_num_bufs] + scale_desc.block_type.shape,
        scale_desc.layout,
    )
    mma_barrier_count: gl.constexpr = blackwell.tcgen05_mma_barrier_count(
        [w_bufs.index(0), x_bufs.index(0)],
        multicast=mma_two_ctas,
    )
    counted_mma_bars: gl.constexpr = STRUCTURAL_MODE == 13
    input_empty_count: gl.constexpr = mma_barrier_count if counted_mma_bars else 1
    x_empty_bars = alloc_barrier_ring(x_num_bufs, count=input_empty_count)
    x_ready_bars = alloc_barrier_ring(x_num_bufs, two_ctas=mma_two_ctas)
    w_empty_bars = alloc_barrier_ring(w_num_bufs, count=input_empty_count)
    w_ready_bars = alloc_barrier_ring(w_num_bufs, two_ctas=mma_two_ctas)

    x_scale_tmem = blackwell.allocate_tensor_memory(gl.uint8, [BLOCK_M, scale_k], x_scale_layout)
    w_scale_tmem = blackwell.allocate_tensor_memory(gl.uint8, [BLOCK_N, scale_k], w_scale_layout)

    acc_num_bufs: gl.constexpr = ACC_NUM_BUFS
    if natural_acc:
        acc_layout: gl.constexpr = blackwell.TensorMemoryLayout(
            [128, BLOCK_N],
            col_stride=1,
            cga_layout=((1, 0), ) if use_2cta else (),
            two_ctas=mma_two_ctas,
        )
        acc_tmem = blackwell.allocate_tensor_memory(
            gl.float32,
            [acc_num_bufs, BLOCK_M, BLOCK_N],
            acc_layout,
        )
    else:
        mma_block_col: gl.constexpr = min(128, BLOCK_N // gl.num_ctas())
        acc_layout: gl.constexpr = blackwell.TensorMemoryLayout(
            [mma_block_col, BLOCK_M],
            col_stride=1,
            cga_layout=((1, 0), ) if use_2cta else (),
            two_ctas=mma_two_ctas,
        )
        acc_tmem = blackwell.allocate_tensor_memory(
            gl.float32,
            [acc_num_bufs, BLOCK_N, BLOCK_M],
            acc_layout,
        )
    acc_ready_count: gl.constexpr = mma_barrier_count if counted_mma_bars else 1
    acc_empty_bars = alloc_barrier_ring(acc_num_bufs, two_ctas=mma_two_ctas)
    acc_ready_bars = alloc_barrier_ring(acc_num_bufs, count=acc_ready_count)

    store_bufs = x_bufs
    store_empty_bars = x_empty_bars
    store_ready_bars = x_ready_bars
    x_scale_tmem.store(gl.full((BLOCK_M, scale_k), 127, dtype=gl.uint8, layout=x_scale_tmem.get_reg_layout()))

    p = PartitionArgs(
        x_desc=x_desc,
        w_desc=w_desc,
        scale_desc=scale_desc,
        out_desc=out_desc,
        x_scale_ptr=x_scale_ptr,
        w_scale_ptr=w_scale_ptr,
        out_scale_ptr=out_scale_ptr,
        out_ptr=out_ptr,
        bias_ptr=bias_ptr,
        bias_stride=bias_stride,
        gather_indx_ptr=gather_indx_ptr,
        x_slice_sizes=x_slice_sizes,
        x_slice_offs=x_slice_offs,
        x_block_offs=x_block_offs,
        x_block_schedule=x_block_schedule,
        x_tile_schedule=x_tile_schedule,
        x_bufs=x_bufs,
        x_empty_bars=x_empty_bars,
        x_ready_bars=x_ready_bars,
        x_num_bufs=x_num_bufs,
        w_bufs=w_bufs,
        w_scale_bufs=w_scale_bufs,
        w_empty_bars=w_empty_bars,
        w_ready_bars=w_ready_bars,
        w_num_bufs=w_num_bufs,
        x_scale_tmem=x_scale_tmem,
        w_scale_tmem=w_scale_tmem,
        acc_bufs=acc_tmem,
        acc_empty_bars=acc_empty_bars,
        acc_ready_bars=acc_ready_bars,
        acc_num_bufs=acc_num_bufs,
        store_bufs=store_bufs,
        store_empty_bars=store_empty_bars,
        store_ready_bars=store_ready_bars,
        grid_m=grid_m,
        GRID_N=grid_n,
        K_TILES=k_tiles,
        SCALE_FLAT_N=scale_flat_n,
        SCALE_BLOCK_N_DIV=scale_block_n_div,
        num_blocks=num_blocks,
        NUM_SMS=NUM_SMS,
        USE_2CTA=use_2cta,
        BLOCK_M_PER_CTA=block_m_per_cta,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SCALE_SIZE_OUTER=SCALE_SIZE_OUTER,
        SCALE_SIZE_INNER=SCALE_SIZE_INNER,
        MXFP_BLOCK_SIZE=MXFP_BLOCK_SIZE,
        SWIGLU_ALPHA=SWIGLU_ALPHA,
        SWIGLU_LIMIT=SWIGLU_LIMIT,
        REDUCTION_N=REDUCTION_N,
        FLEXPOINT_SATURATE_INF=FLEXPOINT_SATURATE_INF,
        SWIGLU_SUBTILE_FACTOR=SWIGLU_SUBTILE_FACTOR,
        EPILOGUE_BUFFER_DEPTH=EPILOGUE_BUFFER_DEPTH,
        USE_PLANAR_SNAKE=USE_PLANAR_SNAKE,
        GRID_MINOR_DIM=GRID_MINOR_DIM,
        GRID_TILE_WIDTH=GRID_TILE_WIDTH,
        BAND_N=BAND_N,
        USE_FULL_TILE_SCHEDULE=USE_FULL_TILE_SCHEDULE,
        X_GATHER_MULTICAST=X_GATHER_MULTICAST,
        W_SCALE_MULTICAST=W_SCALE_MULTICAST,
        FORCE_EPILOGUE_WARPS_N1=FORCE_EPILOGUE_WARPS_N1,
        USE_WIDE_STORE_HANDOFF=USE_WIDE_STORE_HANDOFF,
        USE_DIRECT_EPILOGUE_STORE=USE_DIRECT_EPILOGUE_STORE,
        REUSE_GATHER_INDICES=REUSE_GATHER_INDICES,
        INLINE_MMA_INPUT_RELEASE=INLINE_MMA_INPUT_RELEASE,
    )

    if STRUCTURAL_MODE == 0:
        gl.warp_specialize(
            [
                (epilogue_partition, (p, )),
                (load_inputs_partition, (p, )),
                (mma_partition, (p, )),
            ],
            [LOAD_ACTIVATION_WARPS, MMA_WARPS],
            [LOAD_ACTIVATION_REGS, MMA_REGS],
        )
    elif STRUCTURAL_MODE == 3:
        gl.warp_specialize(
            [
                (epilogue_partition, (p, )),
                (load_inputs_xfirst_partition, (p, )),
                (mma_partition, (p, )),
            ],
            [LOAD_ACTIVATION_WARPS, MMA_WARPS],
            [LOAD_ACTIVATION_REGS, MMA_REGS],
        )
    elif STRUCTURAL_MODE == 4:
        gl.warp_specialize(
            [
                (epilogue_partition, (p, )),
                (load_inputs_wfirst_partition, (p, )),
                (mma_partition, (p, )),
            ],
            [LOAD_ACTIVATION_WARPS, MMA_WARPS],
            [LOAD_ACTIVATION_REGS, MMA_REGS],
        )
    elif STRUCTURAL_MODE == 5:
        gl.static_assert(ACC_NUM_BUFS >= 2, "paired W reuse requires two accumulator buffers")
        gl.warp_specialize(
            [
                (epilogue_partition, (p, )),
                (load_pair_reuse_inputs_partition, (p, )),
                (mma_pair_reuse_partition, (p, )),
            ],
            [LOAD_ACTIVATION_WARPS, MMA_WARPS],
            [LOAD_ACTIVATION_REGS, MMA_REGS],
        )
    elif STRUCTURAL_MODE == 6:
        gl.static_assert(ACC_NUM_BUFS >= 2, "paired W reuse requires two accumulator buffers")
        gl.warp_specialize(
            [
                (epilogue_partition, (p, )),
                (load_pair_reuse_activations_partition, (p, )),
                (load_pair_reuse_weights_partition, (p, )),
                (mma_pair_reuse_partition, (p, )),
            ],
            [LOAD_ACTIVATION_WARPS, LOAD_WEIGHT_WARPS, MMA_WARPS],
            [LOAD_ACTIVATION_REGS, LOAD_WEIGHT_REGS, MMA_REGS],
        )
    elif STRUCTURAL_MODE == 7:
        gl.warp_specialize(
            [
                (epilogue_partition, (p, )),
                (load_activations, (p, )),
                (load_weights, (p, )),
                (mma_multicast_partition, (p, )),
            ],
            [LOAD_ACTIVATION_WARPS, LOAD_WEIGHT_WARPS, MMA_WARPS],
            [LOAD_ACTIVATION_REGS, LOAD_WEIGHT_REGS, MMA_REGS],
        )
    elif STRUCTURAL_MODE == 8:
        gl.warp_specialize(
            [
                (epilogue_explicit_tmem_load_partition, (p, )),
                (load_activations, (p, )),
                (load_weights, (p, )),
                (mma_partition, (p, )),
            ],
            [LOAD_ACTIVATION_WARPS, LOAD_WEIGHT_WARPS, MMA_WARPS],
            [LOAD_ACTIVATION_REGS, LOAD_WEIGHT_REGS, MMA_REGS],
        )
    elif STRUCTURAL_MODE == 9:
        gl.warp_specialize(
            [
                (epilogue_explicit_tmem_load_partition_v1, (p, )),
                (load_activations, (p, )),
                (load_weights, (p, )),
                (mma_partition, (p, )),
            ],
            [LOAD_ACTIVATION_WARPS, LOAD_WEIGHT_WARPS, MMA_WARPS],
            [LOAD_ACTIVATION_REGS, LOAD_WEIGHT_REGS, MMA_REGS],
        )
    elif STRUCTURAL_MODE == 10:
        gl.warp_specialize(
            [
                (epilogue_explicit_tmem_load_partition_v2, (p, )),
                (load_activations, (p, )),
                (load_weights, (p, )),
                (mma_partition, (p, )),
            ],
            [LOAD_ACTIVATION_WARPS, LOAD_WEIGHT_WARPS, MMA_WARPS],
            [LOAD_ACTIVATION_REGS, LOAD_WEIGHT_REGS, MMA_REGS],
        )
    elif STRUCTURAL_MODE == 11:
        gl.warp_specialize(
            [
                (epilogue_explicit_tmem_load_partition_v3, (p, )),
                (load_activations, (p, )),
                (load_weights, (p, )),
                (mma_partition, (p, )),
            ],
            [LOAD_ACTIVATION_WARPS, LOAD_WEIGHT_WARPS, MMA_WARPS],
            [LOAD_ACTIVATION_REGS, LOAD_WEIGHT_REGS, MMA_REGS],
        )
    elif STRUCTURAL_MODE == 12:
        gl.warp_specialize(
            [
                (epilogue_partition, (p, )),
                (load_activations, (p, )),
                (load_weights, (p, )),
                (mma_partition, (p, )),
            ],
            [LOAD_ACTIVATION_WARPS, LOAD_WEIGHT_WARPS, MMA_WARPS],
            [LOAD_ACTIVATION_REGS, LOAD_WEIGHT_REGS, MMA_REGS],
        )
    elif STRUCTURAL_MODE == 13:
        gl.warp_specialize(
            [
                (epilogue_partition, (p, )),
                (load_activations, (p, )),
                (load_weights, (p, )),
                (mma_multicast_counted_partition, (p, )),
            ],
            [LOAD_ACTIVATION_WARPS, LOAD_WEIGHT_WARPS, MMA_WARPS],
            [LOAD_ACTIVATION_REGS, LOAD_WEIGHT_REGS, MMA_REGS],
        )
    elif STRUCTURAL_MODE == 14:
        gl.static_assert(USE_DIRECT_EPILOGUE_STORE, "natural accumulator scratch uses direct epilogue")
        gl.warp_specialize(
            [
                (epilogue_natural_acc_partition, (p, )),
                (load_activations, (p, )),
                (load_weights, (p, )),
                (mma_natural_acc_partition, (p, )),
            ],
            [LOAD_ACTIVATION_WARPS, LOAD_WEIGHT_WARPS, MMA_WARPS],
            [LOAD_ACTIVATION_REGS, LOAD_WEIGHT_REGS, MMA_REGS],
        )
    elif STRUCTURAL_MODE == 15:
        gl.static_assert(USE_DIRECT_EPILOGUE_STORE, "fake N-split scratch uses direct epilogue")
        gl.warp_specialize(
            [
                (epilogue_partition, (p, )),
                (load_inputs_fake_nsplit_partition, (p, )),
                (mma_partition, (p, )),
            ],
            [LOAD_ACTIVATION_WARPS, MMA_WARPS],
            [LOAD_ACTIVATION_REGS, MMA_REGS],
        )
    elif STRUCTURAL_MODE == 16:
        gl.static_assert(ACC_NUM_BUFS >= 2, "phasepair needs two accumulator buffers")
        gl.warp_specialize(
            [
                (epilogue_partition, (p, )),
                (load_transition_pair_activations_partition, (p, )),
                (load_pair_reuse_weights_partition, (p, )),
                (mma_transition_reuse_pair_partition, (p, )),
            ],
            [LOAD_ACTIVATION_WARPS, LOAD_WEIGHT_WARPS, MMA_WARPS],
            [LOAD_ACTIVATION_REGS, LOAD_WEIGHT_REGS, MMA_REGS],
        )
    else:
        if STRUCTURAL_MODE == 1:
            gl.warp_specialize(
                [
                    (epilogue_partition, (p, )),
                    (load_activations, (p, )),
                    (mma_load_weights_partition, (p, )),
                ],
                [LOAD_ACTIVATION_WARPS, MMA_WARPS],
                [LOAD_ACTIVATION_REGS, MMA_REGS],
            )
        else:
            gl.static_assert(STRUCTURAL_MODE == 2, "unknown structural mode")
            gl.warp_specialize(
                [
                    (epilogue_partition, (p, )),
                    (load_weights, (p, )),
                    (mma_load_activations_partition, (p, )),
                ],
                [LOAD_WEIGHT_WARPS, MMA_WARPS],
                [LOAD_WEIGHT_REGS, MMA_REGS],
            )


def combined_load_matmul(
    a: torch.Tensor,
    b,
    bias: torch.Tensor,
    a_ragged_metadata,
    gather_indx: torch.Tensor,
    precision_config,
    c: torch.Tensor,
    fused_activation,
    p,
    structural_mode: int,
):
    specs = fused_activation.specs
    assert specs.name == "swiglu"
    reduction_n = specs.reduction_n
    swiglu_alpha, swiglu_limit = fused_activation.fn_args
    b_mx_scales = precision_config.b_mx_scale
    out_dtype = precision_config.out_dtype
    assert out_dtype is not None
    assert c.ndim == 2
    flex_ctx = precision_config.flex_ctx
    assert a.ndim == 2
    _, k = a.shape
    _, _, n = b.shape
    m = gather_indx.shape[0]

    assert isinstance(b, ex.Tensor)
    assert isinstance(b.storage.layout, ex.BlackwellMX4ValueShuffledLayout)
    assert b.storage.layout.block_k == p.BLOCK_K
    assert b.storage.layout.block_n == p.BLOCK_N
    x_block_offs, x_block_schedule = ex.get_block_schedule_tensors(a_ragged_metadata, p.BLOCK_M)
    expected_grid_m = a_ragged_metadata.n_blocks(a_ragged_metadata.n_slices, m, p.BLOCK_M)
    grid_n = triton.cdiv(n, p.BLOCK_N)
    sms = torch.cuda.get_device_properties(bias.device).multi_processor_count
    sms *= p.OCCUPANCY
    launch_grid = max(1, min(max(1, sms // p.NUM_CTAS), expected_grid_m * grid_n))
    if structural_mode in (5, 6):
        x_tile_schedule = get_pair_w_reuse_tile_schedule_tensor(
            a_ragged_metadata,
            p.BLOCK_M,
            grid_n,
            launch_grid,
        )
        use_full_tile_schedule = True
    else:
        x_tile_schedule = (
            ex.get_full_tile_schedule_tensor(
                a_ragged_metadata,
                p.BLOCK_M,
                grid_n,
                use_planar_snake=p.USE_PLANAR_SNAKE,
                grid_minor_dim=p.GRID_MINOR_DIM,
                grid_tile_width=p.GRID_TILE_WIDTH,
                band_n=p.BAND_N,
            )
            if p.USE_FULL_TILE_SCHEDULE else ex.get_dummy_full_tile_schedule_tensor(a_ragged_metadata.slice_sizes.device)
        )
        use_full_tile_schedule = p.USE_FULL_TILE_SCHEDULE
    grid = (launch_grid, )

    fake_nsplit = structural_mode == 15
    acc_cga_layout = ex.get_acc_cga_layout(p.NUM_CTAS)
    x_desc_cga_layout = ((0, 0), ) if fake_nsplit else ex.get_x_desc_cga_layout(acc_cga_layout)
    w_desc_cga_layout = ex.get_w_desc_cga_layout(acc_cga_layout)
    w_scale_desc_cga_layout = ex.get_w_scale_desc_cga_layout(acc_cga_layout)

    x_desc = ex.make_gather_operand_descriptor(
        a,
        (1, p.BLOCK_K),
        (p.BLOCK_M, p.BLOCK_K),
        cga_layout=x_desc_cga_layout,
    )
    w_desc = ex.make_operand_descriptor(b, (1, p.BLOCK_K, p.BLOCK_N), cga_layout=w_desc_cga_layout)
    scale_desc = ex.make_operand_descriptor(
        b_mx_scales,
        (
            1,
            p.BLOCK_N // p.SCALE_SIZE_OUTER,
            p.BLOCK_K // p.MXFP_BLOCK_SIZE // p.SCALE_SIZE_INNER,
            2,
            256,
        ),
        cga_layout=w_scale_desc_cga_layout,
    )
    out_desc = ex.make_output_meta_descriptor(c, (p.BLOCK_M, p.BLOCK_N // reduction_n))

    ws_matmul_combined_load_kernel[grid](
        x_desc=x_desc,
        w_desc=w_desc,
        scale_desc=scale_desc,
        out_desc=out_desc,
        out_ptr=c,
        bias_ptr=bias,
        bias_stride=bias.stride(0),
        gather_indx_ptr=gather_indx,
        x_slice_sizes=a_ragged_metadata.slice_sizes,
        x_slice_offs=a_ragged_metadata.slice_offs,
        x_block_offs=x_block_offs,
        x_block_schedule=x_block_schedule,
        x_tile_schedule=x_tile_schedule,
        x_scale_ptr=flex_ctx.lhs_data.scale,
        w_scale_ptr=flex_ctx.rhs_data.scale,
        out_scale_ptr=flex_ctx.out_data.expected_scale,
        M=m,
        N=n,
        K=k,
        NUM_SLICES=a_ragged_metadata.n_slices,
        SWIGLU_ALPHA=swiglu_alpha,
        SWIGLU_LIMIT=swiglu_limit,
        REDUCTION_N=reduction_n,
        FLEXPOINT_SATURATE_INF=precision_config.flexpoint_saturate_inf,
        BLOCK_M=p.BLOCK_M,
        BLOCK_N=p.BLOCK_N,
        BLOCK_K=p.BLOCK_K,
        NUM_SMS=launch_grid,
        X_NUM_BUFS=p.X_NUM_BUFS,
        W_NUM_BUFS=p.W_NUM_BUFS,
        ACC_NUM_BUFS=p.ACC_NUM_BUFS,
        LOAD_ACTIVATION_WARPS=p.LOAD_ACTIVATION_WARPS,
        LOAD_WEIGHT_WARPS=p.LOAD_WEIGHT_WARPS,
        MMA_WARPS=p.MMA_WARPS,
        STORE_HELPER_WARPS=p.STORE_HELPER_WARPS,
        LOAD_ACTIVATION_REGS=p.LOAD_ACTIVATION_REGS,
        LOAD_WEIGHT_REGS=p.LOAD_WEIGHT_REGS,
        MMA_REGS=p.MMA_REGS,
        STORE_HELPER_REGS=p.STORE_HELPER_REGS,
        SWIGLU_SUBTILE_FACTOR=p.SWIGLU_SUBTILE_FACTOR,
        EPILOGUE_BUFFER_DEPTH=p.EPILOGUE_BUFFER_DEPTH,
        USE_PLANAR_SNAKE=p.USE_PLANAR_SNAKE,
        GRID_MINOR_DIM=p.GRID_MINOR_DIM,
        GRID_TILE_WIDTH=p.GRID_TILE_WIDTH,
        BAND_N=p.BAND_N,
        USE_FULL_TILE_SCHEDULE=use_full_tile_schedule,
        X_GATHER_MULTICAST=p.X_GATHER_MULTICAST,
        W_SCALE_MULTICAST=p.W_SCALE_MULTICAST,
        FORCE_EPILOGUE_WARPS_N1=p.FORCE_EPILOGUE_WARPS_N1,
        USE_WIDE_STORE_HANDOFF=p.USE_WIDE_STORE_HANDOFF,
        USE_DIRECT_EPILOGUE_STORE=p.USE_DIRECT_EPILOGUE_STORE,
        REUSE_GATHER_INDICES=p.REUSE_GATHER_INDICES,
        INLINE_MMA_INPUT_RELEASE=p.INLINE_MMA_INPUT_RELEASE,
        SCALE_SIZE_OUTER=p.SCALE_SIZE_OUTER,
        SCALE_SIZE_INNER=p.SCALE_SIZE_INNER,
        MXFP_BLOCK_SIZE=p.MXFP_BLOCK_SIZE,
        STRUCTURAL_MODE=structural_mode,
        num_warps=p.NUM_WARPS,
        num_ctas=p.NUM_CTAS,
        maxnreg=p.MAXNREG,
    )
    return c


def parse_candidate(name: str, slice_size: int):
    if name.startswith("combined:"):
        spec = name[len("combined:"):]
        mode = "combined"
    elif name.startswith("combinedx:"):
        spec = name[len("combinedx:"):]
        mode = "combinedx"
    elif name.startswith("combinedw:"):
        spec = name[len("combinedw:"):]
        mode = "combinedw"
    elif name.startswith("mmaw:"):
        spec = name[len("mmaw:"):]
        mode = "mmaw"
    elif name.startswith("mmax:"):
        spec = name[len("mmax:"):]
        mode = "mmax"
    elif name.startswith("pair:"):
        spec = name[len("pair:"):]
        mode = "pair"
    elif name.startswith("pairsplit:"):
        spec = name[len("pairsplit:"):]
        mode = "pairsplit"
    elif name.startswith("mmamc:"):
        spec = name[len("mmamc:"):]
        mode = "mmamc"
    elif name.startswith("epiload:"):
        spec = name[len("epiload:"):]
        mode = "epiload"
    elif name.startswith("epiload1:"):
        spec = name[len("epiload1:"):]
        mode = "epiload1"
    elif name.startswith("epiload2:"):
        spec = name[len("epiload2:"):]
        mode = "epiload2"
    elif name.startswith("epiload3:"):
        spec = name[len("epiload3:"):]
        mode = "epiload3"
    elif name.startswith("splitws:"):
        spec = name[len("splitws:"):]
        mode = "splitws"
    elif name.startswith("mmacount:"):
        spec = name[len("mmacount:"):]
        mode = "mmacount"
    elif name.startswith("natural:"):
        spec = name[len("natural:"):]
        mode = "natural"
    elif name.startswith("fakens:"):
        spec = name[len("fakens:"):]
        mode = "fakens"
    elif name.startswith("phasepair:"):
        spec = name[len("phasepair:"):]
        mode = "phasepair"
    elif name.startswith("split:"):
        spec = name[len("split:"):]
        mode = "split"
    else:
        spec = None
        mode = ""
    if spec is not None:
        base_name, _, option_text = spec.partition("@")
        p = make_candidate(base_name, slice_size)
        options = option_text.split(",") if option_text else []
        updates = {}
        explicit_regs = None
        for option in options:
            if option.startswith("x") and option[1:].isdigit():
                updates["X_NUM_BUFS"] = int(option[1:])
            elif option.startswith("w") and option[1:].isdigit():
                updates["W_NUM_BUFS"] = int(option[1:])
            elif option.startswith("acc") and option[3:].isdigit():
                updates["ACC_NUM_BUFS"] = int(option[3:])
            elif option.startswith("bm") and option[2:].isdigit():
                updates["BLOCK_M"] = int(option[2:])
            elif option.startswith("bn") and option[2:].isdigit():
                updates["BLOCK_N"] = int(option[2:])
            elif option.startswith("bk") and option[2:].isdigit():
                updates["BLOCK_K"] = int(option[2:])
            elif option.startswith("ctas") and option[4:].isdigit():
                updates["NUM_CTAS"] = int(option[4:])
            elif option.startswith("sub") and option[3:].isdigit():
                updates["SWIGLU_SUBTILE_FACTOR"] = int(option[3:])
            elif option.startswith("epi") and option[3:].isdigit():
                updates["EPILOGUE_BUFFER_DEPTH"] = int(option[3:])
            elif option.startswith("occ") and option[3:].isdigit():
                updates["OCCUPANCY"] = int(option[3:])
            elif option.startswith("b") and option[1:].isdigit():
                updates["BAND_N"] = int(option[1:])
            elif option.startswith("warps") and option[5:].isdigit():
                updates["NUM_WARPS"] = int(option[5:])
            elif option.startswith("regs") and option[4:].isdigit():
                explicit_regs = int(option[4:])
                updates["MAXNREG"] = explicit_regs
            elif option.startswith("la") and option[2:].isdigit():
                updates["LOAD_ACTIVATION_REGS"] = int(option[2:])
            elif option.startswith("lw") and option[2:].isdigit():
                updates["LOAD_WEIGHT_REGS"] = int(option[2:])
            elif option.startswith("mma") and option[3:].isdigit():
                updates["MMA_REGS"] = int(option[3:])
            elif option == "epin0":
                updates["FORCE_EPILOGUE_WARPS_N1"] = False
            elif option == "epin1":
                updates["FORCE_EPILOGUE_WARPS_N1"] = True
            elif option == "nomcx":
                updates["X_GATHER_MULTICAST"] = False
            elif option == "nomcscale":
                updates["W_SCALE_MULTICAST"] = False
            elif option == "nomc":
                updates["X_GATHER_MULTICAST"] = False
                updates["W_SCALE_MULTICAST"] = False
            elif option == "fullsched":
                updates["USE_FULL_TILE_SCHEDULE"] = True
            elif option == "reuse":
                updates["REUSE_GATHER_INDICES"] = True
            elif option == "inline":
                updates["INLINE_MMA_INPUT_RELEASE"] = True
            elif option == "helper":
                updates["USE_DIRECT_EPILOGUE_STORE"] = False
            elif option == "direct":
                updates["USE_DIRECT_EPILOGUE_STORE"] = True
            elif option == "wide":
                updates["USE_WIDE_STORE_HANDOFF"] = True
            elif option.startswith("sh") and option[2:].isdigit():
                updates["STORE_HELPER_WARPS"] = int(option[2:])
            elif option.startswith("sr") and option[2:].isdigit():
                updates["STORE_HELPER_REGS"] = int(option[2:])
            elif option.startswith("l") and "m" in option:
                load_warps, mma_warps = option[1:].split("m", 1)
                load_warps = int(load_warps)
                mma_warps = int(mma_warps)
                for value, label in ((load_warps, "loader"), (mma_warps, "mma")):
                    if value <= 0 or value & (value - 1):
                        raise ValueError(f"{label} warps must be a positive power of two, got {value}")
                updates["LOAD_ACTIVATION_WARPS"] = load_warps
                updates["MMA_WARPS"] = mma_warps
            elif option.startswith("w") and "m" in option:
                weight_warps, mma_warps = option[1:].split("m", 1)
                weight_warps = int(weight_warps)
                mma_warps = int(mma_warps)
                for value, label in ((weight_warps, "weight"), (mma_warps, "mma")):
                    if value <= 0 or value & (value - 1):
                        raise ValueError(f"{label} warps must be a positive power of two, got {value}")
                updates["LOAD_WEIGHT_WARPS"] = weight_warps
                updates["MMA_WARPS"] = mma_warps
            elif option:
                raise ValueError(f"unknown combined option {option!r}")
        if explicit_regs is not None:
            if explicit_regs == 60:
                updates.setdefault("LOAD_ACTIVATION_REGS", 48)
                updates.setdefault("LOAD_WEIGHT_REGS", 40)
                updates.setdefault("MMA_REGS", 40)
            elif explicit_regs in (52, 56):
                updates.setdefault("LOAD_ACTIVATION_REGS", 40)
                updates.setdefault("LOAD_WEIGHT_REGS", 32)
                updates.setdefault("MMA_REGS", 32)
            else:
                updates.setdefault("LOAD_ACTIVATION_REGS", 32)
                updates.setdefault("LOAD_WEIGHT_REGS", 32)
                updates.setdefault("MMA_REGS", 32)
        if mode in ("pair", "pairsplit", "phasepair"):
            updates.setdefault("ACC_NUM_BUFS", 2)
        updates.setdefault("USE_DIRECT_EPILOGUE_STORE", True)
        updates.setdefault("REUSE_GATHER_INDICES", False)
        updates.setdefault("INLINE_MMA_INPUT_RELEASE", False)
        p = replace(p, **updates)
        return mode, p, base_name
    if name.startswith("split:"):
        base_name = name[len("split:"):]
        return "split", make_candidate(base_name, slice_size), base_name
    return "split", make_candidate(name, slice_size), name


def run_with_candidate(prepared, p, mode: str, out: torch.Tensor):
    pc = ex.make_precision_config(prepared)
    if mode in (
        "combined",
        "combinedx",
        "combinedw",
        "mmaw",
        "mmax",
        "pair",
        "pairsplit",
        "mmamc",
        "epiload",
        "epiload1",
        "epiload2",
        "epiload3",
        "splitws",
        "mmacount",
        "natural",
        "fakens",
        "phasepair",
    ):
        return combined_load_matmul(
            a=prepared.x,
            b=prepared.w,
            bias=prepared.bias,
            a_ragged_metadata=prepared.ragged_metadata,
            gather_indx=prepared.gather_indx,
            precision_config=pc,
            c=out,
            fused_activation=prepared.fused_activation,
            p=p,
            structural_mode={
                "combined": 0,
                "mmaw": 1,
                "mmax": 2,
                "combinedx": 3,
                "combinedw": 4,
                "pair": 5,
                "pairsplit": 6,
                "mmamc": 7,
                "epiload": 8,
                "epiload1": 9,
                "epiload2": 10,
                "epiload3": 11,
                "splitws": 12,
                "mmacount": 13,
                "natural": 14,
                "fakens": 15,
                "phasepair": 16,
            }[mode],
        )
    return run_split_with_config(prepared, p, out)


def validate(prepared, p, mode: str):
    ref, _ = ex.run_provider(prepared, "reference")
    out = ex.make_output_buffer(prepared)
    cand = run_with_candidate(prepared, p, mode, out)
    torch.cuda.synchronize()
    ex.assert_close(
        ref.to(torch.float32),
        cand.to(torch.float32),
        maxtol=0.125,
        rmstol=None,
        description=f"mode={mode}-bs{prepared.batch_size}-{config_dict(p)}",
        verbose=False,
    )


def bench_ms(prepared, p, mode: str, rep: int):
    out = ex.make_output_buffer(prepared)

    def fn():
        return run_with_candidate(prepared, p, mode, out)

    fn()
    torch.cuda.synchronize()
    vals = ex.do_bench_cudagraph(fn, rep=rep, return_mode="all")
    return float(statistics.median(vals)), float(statistics.mean(vals)), [float(v) for v in vals]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", nargs="+", type=int, default=[896])
    ap.add_argument("--routing", choices=["prod", "uniform"], default="uniform")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--local-ranks", nargs="+", type=int, default=[3, 4])
    ap.add_argument("--rep", type=int, default=300)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--out", default="/tmp/moe_bmm1_structural_explore.csv")
    ap.add_argument(
        "--candidates",
        nargs="+",
        default=[
            "1cta",
            "split:m32_bn256_sub1_direct_warps4_x5w6_b21_act2w1m1_regs48_epin1_b32",
            "combined:m32_bn256_sub1_direct_warps4_x5w6_b21_act2w1m1_regs48_epin1_b32@l2m1",
            "combined:m32_bn256_sub1_direct_warps4_x5w6_b20_nomcscale_act2w1m1_regs52_epin1_b32@l2m1",
            "mmaw:m32_bn256_sub1_direct_warps4_x5w6_b21_act2w1m1_regs48_epin1_b32@l2m1",
            "mmax:m32_bn256_sub1_direct_warps4_x5w6_b21_act2w1m1_regs48_epin1_b32@w1m1",
        ],
    )
    ap.add_argument("--candidates-file", help="Read one candidate per non-empty, non-comment line.")
    args = ap.parse_args()
    if args.candidates_file:
        args.candidates = [
            line.strip()
            for line in Path(args.candidates_file).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    c = ex.GPT_OSS_120B_CONFIG
    device = f"cuda:{torch.cuda.current_device()}"
    rows = []
    for seed in args.seeds:
        for local_rank in args.local_ranks:
            for bs in args.batches:
                slice_size = bs * c.experts_per_token // c.num_experts
                logical = make_logical_case(
                    c,
                    bs,
                    device,
                    seed,
                    args.routing,
                    local_rank=local_rank,
                )
                baseline_ms = None
                for name in args.candidates:
                    mode, p, base_name = parse_candidate(name, slice_size)
                    prepared = materialize_prepared(logical, p)
                    stats = route_stats(prepared, p)
                    try:
                        if args.validate:
                            validate(prepared, p, mode)
                        med_ms, mean_ms, vals = bench_ms(prepared, p, mode, args.rep)
                        if name == "1cta":
                            baseline_ms = med_ms
                        speedup = "" if baseline_ms is None else f"{baseline_ms / med_ms:.5f}"
                        flops, nbytes = ex.estimate_benchmark_work(c, prepared)
                        tf = flops * 1e-12 / (med_ms * 1e-3)
                        tb = nbytes * 1e-12 / (med_ms * 1e-3)
                        row = {
                            "routing": args.routing,
                            "seed": seed,
                            "local_rank": logical.local_rank,
                            "batch_size": bs,
                            "slice_size": slice_size,
                            "candidate": name,
                            "mode": mode,
                            "base_candidate": base_name,
                            "median_ms": f"{med_ms:.8f}",
                            "mean_ms": f"{mean_ms:.8f}",
                            "tflops": f"{tf:.4f}",
                            "tbps": f"{tb:.4f}",
                            "speedup_vs_1cta": speedup,
                            "samples": json.dumps(vals),
                            "config": json.dumps(config_dict(p), sort_keys=True),
                            "route_stats": json.dumps(stats, sort_keys=True),
                            "status": "ok",
                            "error": "",
                        }
                    except Exception as exc:
                        row = {
                            "routing": args.routing,
                            "seed": seed,
                            "local_rank": logical.local_rank,
                            "batch_size": bs,
                            "slice_size": slice_size,
                            "candidate": name,
                            "mode": mode,
                            "base_candidate": base_name,
                            "median_ms": "",
                            "mean_ms": "",
                            "tflops": "",
                            "tbps": "",
                            "speedup_vs_1cta": "",
                            "samples": "",
                            "config": json.dumps(config_dict(p), sort_keys=True),
                            "route_stats": json.dumps(stats, sort_keys=True),
                            "status": "error",
                            "error": repr(exc),
                        }
                    rows.append(row)
                    print(",".join(str(row[k]) for k in (
                        "routing", "seed", "local_rank", "batch_size", "candidate", "mode", "status",
                        "median_ms", "speedup_vs_1cta", "tflops", "tbps", "error")), flush=True)

    out = Path(args.out)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
