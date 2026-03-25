// RUN: triton-opt %s --convert-warp-specialize-to-llvm --convert-nv-gpu-to-llvm -allow-unregistered-dialect | FileCheck %s

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32, ttg.tensor_memory_size = 128 : i32, "ttng.two-ctas" = true} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  // CHECK-LABEL: @automatic_tmem_lifecycle
  // CHECK: [[ALLOC_PRED:%.*]] = llvm.icmp "ult" {{.*}} : i32
  // CHECK-COUNT-1: tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32
  // CHECK: [[TMEM_BASE_B32:%.*]] = llvm.load {{.*}} : !llvm.ptr<3> -> i32
  // CHECK: [[TMEM_BASE_PTR:%.*]] = llvm.inttoptr [[TMEM_BASE_B32]] : i32 to !llvm.ptr<6>
  // CHECK-COUNT-1: tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned
  // CHECK-COUNT-1: nvvm.cluster.arrive {aligned}
  // CHECK-COUNT-1: nvvm.cluster.wait {aligned}
  // CHECK-COUNT-1: tcgen05.dealloc.cta_group::2.sync.aligned.b32
  // CHECK: "b,r" [[ALLOC_PRED]], [[TMEM_BASE_PTR]] : (i1, !llvm.ptr<6>) -> !llvm.void
  // CHECK-NOT: tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32
  // CHECK-NOT: nvg.tensor_memory_base
  llvm.func @automatic_tmem_lifecycle() attributes {allocation.offset = 0 : i32, nvvm.kernel = 1 : ui1, nvvm.maxntid = array<i32: 256>} {
    ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      %0 = nvg.tensor_memory_base
      %1 = llvm.ptrtoint %0 : !llvm.ptr<6> to i32
      "use"(%1) : (i32) -> ()
      ttg.warp_return
    } : () -> ()
    llvm.return
  }
}
