import copy
import math

from triton.language.core import _aggregate as aggregate

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import allocate_tensor_memory
from triton.experimental.gluon.language.nvidia.blackwell import mbarrier
from triton.experimental.gluon.language.nvidia.blackwell import tensor_memory_descriptor
from triton.experimental.gluon.language.nvidia.blackwell import tma


@aggregate
class BarrierCounter:
    index: gl.tensor
    phase: gl.tensor
    num_barriers: gl.constexpr

    @gluon.must_use_result
    @gluon.jit
    def increment(self):
        if self.num_barriers == 1:
            return BarrierCounter(gl.to_tensor(0), self.phase ^ 1, self.num_barriers)
        next_index = self.index + 1
        rollover = next_index == self.num_barriers
        index = gl.where(rollover, 0, next_index)
        phase = gl.where(rollover, self.phase ^ 1, self.phase)
        return BarrierCounter(index, phase, self.num_barriers)


def Channel(T, alloc_fn):

    @aggregate
    class ChannelType:
        mem: T
        ready_bars: gl.shared_memory_descriptor
        empty_bars: gl.shared_memory_descriptor
        num_buffers: gl.constexpr
        num_consumers: gl.constexpr

        @gluon.jit
        def alloc(
            shape: gl.constexpr,
            dtype: gl.constexpr,
            layout: gl.constexpr,
            num_buffers: gl.constexpr,
            num_consumers: gl.constexpr = 1,
        ):
            mem = alloc_fn(dtype, [num_buffers] + shape, layout)
            ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
            empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
            for i in gl.static_range(num_buffers):
                mbarrier.init(ready_bars.index(i), count=1)
                mbarrier.init(empty_bars.index(i), count=num_consumers)
                mbarrier.arrive(empty_bars.index(i), count=num_consumers)
            return ChannelType(mem, ready_bars, empty_bars, num_buffers, num_consumers)

        @gluon.jit
        def acquire_producer(self, counter):
            index, phase = counter.index, counter.phase
            mem = self.mem.index(index)
            ready_bar = self.ready_bars.index(index)
            empty_bar = self.empty_bars.index(index)
            mbarrier.wait(empty_bar, phase)
            return mem, ready_bar

        @gluon.jit
        def acquire_consumer(self, counter):
            index, phase = counter.index, counter.phase
            mem = self.mem.index(index)
            ready_bar = self.ready_bars.index(index)
            empty_bar = self.empty_bars.index(index)
            mbarrier.wait(ready_bar, phase)
            return mem, empty_bar

        @gluon.jit
        def create_counter(self):
            return BarrierCounter(gl.to_tensor(0), gl.to_tensor(0), self.num_buffers)

        @gluon.jit
        def create_producer(self):
            return Producer(self, self.create_counter())

        @gluon.jit
        def create_consumer(self):
            return Consumer(self, self.create_counter())

        @gluon.jit
        def release(self):
            if isinstance(self.mem, gl.shared_memory_descriptor):
                self.mem._keep_alive()
            for i in gl.static_range(self.num_buffers):
                mbarrier.invalidate(self.ready_bars.index(i))
                mbarrier.invalidate(self.empty_bars.index(i))

    @aggregate
    class Producer:
        channel: ChannelType
        counter: BarrierCounter

        @gluon.jit
        def acquire(self):
            mem, ready_bar = self.channel.acquire_producer(self.counter)
            next_producer = Producer(self.channel, self.counter.increment())
            return mem, ready_bar, next_producer

    @aggregate
    class Consumer:
        channel: ChannelType
        counter: BarrierCounter

        @gluon.jit
        def acquire(self):
            mem, empty_bar = self.channel.acquire_consumer(self.counter)
            next_consumer = Consumer(self.channel, self.counter.increment())
            return mem, empty_bar, next_consumer

    return ChannelType, Producer, Consumer


SharedMemoryChannel, SharedMemoryProducer, SharedMemoryConsumer = Channel(
    gl.shared_memory_descriptor,
    gl.allocate_shared_memory,
)
TensorMemoryChannel, TensorMemoryProducer, TensorMemoryConsumer = Channel(
    tensor_memory_descriptor,
    allocate_tensor_memory,
)


@gluon.jit
def get_desc_channel(desc, num_buffers: gl.constexpr, num_consumers: gl.constexpr = 1):
    shape: gl.constexpr = desc.block_type.shape
    layout: gl.constexpr = desc.layout
    return SharedMemoryChannel.alloc(shape, desc.dtype, layout, num_buffers, num_consumers)


@gluon.jit
def issue_async_tma_load(smem, bar, desc, offset):
    mbarrier.expect(bar, desc.block_type.nbytes)
    tma.async_copy_global_to_shared(desc, [offset, 0], bar, smem)


@gluon.jit
def reshape_last_dim_for_pair_ops(values):
    return values.reshape((values.shape[0], 2, values.shape[1] // 2)).permute((0, 2, 1))


@gluon.constexpr_function
def to_distributed_linear_layout(layout: gl.constexpr):
    if isinstance(layout, gl.DistributedLinearLayout):
        return layout

    assert isinstance(layout, gl.SliceLayout), "split requires a linear or slice-of-linear layout"
    assert isinstance(layout.parent, gl.DistributedLinearLayout), "split requires a slice of a linear layout"
    dim = layout.dim
    parent = layout.parent
    shape = parent.shape[:dim] + parent.shape[dim + 1 :]
    return gl.DistributedLinearLayout(
        [basis[:dim] + basis[dim + 1 :] for basis in parent.reg_bases],
        [basis[:dim] + basis[dim + 1 :] for basis in parent.lane_bases],
        [basis[:dim] + basis[dim + 1 :] for basis in parent.warp_bases],
        [basis[:dim] + basis[dim + 1 :] for basis in parent.block_bases],
        shape,
    )


@gluon.constexpr_function
def swap_basis_to_last_reg(layout: gl.constexpr, target_basis):
    layout = to_distributed_linear_layout(layout)
    last_reg_idx = len(layout.reg_bases) - 1
    reg_last = layout.reg_bases[last_reg_idx]
    if reg_last == target_basis:
        return layout

    ret = copy.deepcopy(layout)
    for basis_list in (ret.reg_bases, ret.lane_bases, ret.warp_bases, ret.block_bases):
        for i, basis in enumerate(basis_list):
            if basis == target_basis:
                basis_list[i], ret.reg_bases[last_reg_idx] = reg_last, target_basis
                return ret
    assert False, f"split requires having a basis {target_basis}. Got\n{layout}"


@gluon.constexpr_function
def get_split_last_dim_layout(layout: gl.constexpr, split_factor: gl.constexpr = 2):
    assert split_factor == 1 or split_factor == 2, "split requires a split factor of 1 or 2"
    layout = to_distributed_linear_layout(layout)
    if split_factor == 1:
        return layout
    return swap_basis_to_last_reg(layout, [0, layout.shape[1] // 2])


@gluon.jit
def split_last_dim_pow2(values, split_factor: gl.constexpr = 2):
    if split_factor == 1:
        return (values,)

    layout: gl.constexpr = values.type.layout
    if isinstance(layout, gl.DistributedLinearLayout) or (
        isinstance(layout, gl.SliceLayout) and isinstance(layout.parent, gl.DistributedLinearLayout)
    ):
        lhs, rhs = reshape_last_dim_for_pair_ops(values).split()
        split_layout: gl.constexpr = get_split_last_dim_layout(lhs.type.layout)
        lhs = gl.convert_layout(lhs, split_layout, assert_trivial=True)
        rhs = gl.convert_layout(rhs, split_layout, assert_trivial=True)
    else:
        lhs, rhs = gl.split(reshape_last_dim_for_pair_ops(values))
    return split_last_dim_pow2(lhs, split_factor // 2) + split_last_dim_pow2(rhs, split_factor // 2)


@gluon.constexpr_function
def get_join_last_dim_layout(layout: gl.constexpr, split_factor: gl.constexpr = 2):
    assert isinstance(layout, gl.DistributedLinearLayout), "join requires a distributed linear layout"
    shape = list(layout.shape)
    regs = [[0, shape[1] * (1 << i)] for i in range(int(math.log2(split_factor)))]
    shape[1] *= split_factor
    return gl.DistributedLinearLayout(
        layout.reg_bases + regs,
        layout.lane_bases,
        layout.warp_bases,
        layout.block_bases,
        shape,
    )


@gluon.jit
def join_last_dim(xs):
    num_values: gl.constexpr = len(xs)
    gl.static_assert(
        num_values == 1
        or num_values == 2
        or num_values == 4
        or num_values == 8
        or num_values == 16,
        "unsupported join factor",
    )
    joined = xs
    for join_level in gl.static_range(4):
        if (1 << join_level) < num_values:
            next_joined = ()
            pair_count: gl.constexpr = num_values // (2 << join_level)
            for pair_idx in gl.static_range(pair_count):
                lhs = joined[2 * pair_idx]
                rhs = joined[2 * pair_idx + 1]
                layout: gl.constexpr = get_join_last_dim_layout(lhs.type.layout)
                values = gl.join(lhs, rhs).permute((0, 2, 1)).reshape([lhs.shape[0], lhs.shape[1] * 2])
                next_joined += (gl.convert_layout(values, layout, assert_trivial=True),)
            joined = next_joined
    return joined[0]


@gluon.jit
def _split_first_dim_in_half(values):
    return gl.split(values.reshape((2, values.shape[0] // 2, values.shape[1])).permute((1, 2, 0)))


@gluon.jit
def split_first_dim_pow2(values, split_factor: gl.constexpr):
    gl.static_assert(
        split_factor == 1
        or split_factor == 2
        or split_factor == 4
        or split_factor == 8
        or split_factor == 16
        or split_factor == 32,
        "unsupported row subtile factor",
    )
    subtiles = (values,)
    for split_level in gl.static_range(5):
        if (1 << split_level) < split_factor:
            next_subtiles = ()
            for subtile_idx in gl.static_range(1 << split_level):
                lhs, rhs = _split_first_dim_in_half(subtiles[subtile_idx])
                next_subtiles += (lhs, rhs)
            subtiles = next_subtiles
    return subtiles
