// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm=compute-capability=100 -cse | FileCheck %s

#blocked_cf2 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked_cf2_red = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#tmem_cf2 = #ttng.tensor_memory_linear<{row = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]], col = [[0, 1]]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:103", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_memory_cf_same_type_load
  // CHECK: %[[BASE:.+]] = nvg.tensor_memory_base
  // CHECK: %[[BASE_I:.+]] = llvm.ptrtoint %[[BASE]] : !llvm.ptr<6> to i32
  // CHECK: %[[C0:.+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[LHS_ADDR:.+]] = llvm.add %[[BASE_I]], %[[C0]] : i32
  // CHECK: %[[LHS_PTR:.+]] = llvm.inttoptr %[[LHS_ADDR]] : i32 to !llvm.ptr<3>
  // CHECK: %[[LHS_DESC:.+]] = builtin.unrealized_conversion_cast %[[LHS_PTR]] : !llvm.ptr<3> to !ttg.memdesc<128x2xf32
  // CHECK: %[[C64:.+]] = llvm.mlir.constant(64 : i32) : i32
  // CHECK: %[[RHS_ADDR:.+]] = llvm.add %[[BASE_I]], %[[C64]] : i32
  // CHECK: %[[RHS_PTR:.+]] = llvm.inttoptr %[[RHS_ADDR]] : i32 to !llvm.ptr<3>
  // CHECK: %[[RHS_DESC:.+]] = builtin.unrealized_conversion_cast %[[RHS_PTR]] : !llvm.ptr<3> to !ttg.memdesc<128x2xf32
  // CHECK: %[[SELECTED:.+]] = scf.if %arg0 -> (!ttg.memdesc<128x2xf32
  // CHECK: scf.yield %[[RHS_DESC]] : !ttg.memdesc<128x2xf32
  // CHECK: } else {
  // CHECK: scf.yield %[[LHS_DESC]] : !ttg.memdesc<128x2xf32
  // CHECK: }
  // CHECK: %[[SELECTED_PTR:.+]] = builtin.unrealized_conversion_cast %[[SELECTED]] : !ttg.memdesc<128x2xf32{{.*}} to !llvm.ptr<3>
  // CHECK: %[[SELECTED_I:.+]] = llvm.ptrtoint %[[SELECTED_PTR]] : !llvm.ptr<3> to i32
  // CHECK: %[[LOAD_ADDR:.+]] = llvm.add %[[SELECTED_I]], %{{.+}} : i32
  // CHECK: llvm.inline_asm {{.*}}"tcgen05.ld.sync.aligned.32x32b.x2.b32{{.*}}"{{.*}} %[[LOAD_ADDR]] : (i32) ->
  // CHECK: nvvm.tcgen05.wait <load>
  tt.func public @tensor_memory_cf_same_type_load(%choose_rhs: i1) {
    %lhs = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x2xf32, #tmem_cf2, #ttng.tensor_memory, mutable>
    %rhs = ttng.tmem_alloc {tensor_memory_col_offset = 64 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x2xf32, #tmem_cf2, #ttng.tensor_memory, mutable>
    %selected = scf.if %choose_rhs -> (!ttg.memdesc<128x2xf32, #tmem_cf2, #ttng.tensor_memory, mutable>) {
      scf.yield %rhs : !ttg.memdesc<128x2xf32, #tmem_cf2, #ttng.tensor_memory, mutable>
    } else {
      scf.yield %lhs : !ttg.memdesc<128x2xf32, #tmem_cf2, #ttng.tensor_memory, mutable>
    }
    %result = ttng.tmem_load %selected : !ttg.memdesc<128x2xf32, #tmem_cf2, #ttng.tensor_memory, mutable> -> tensor<128x2xf32, #blocked_cf2>
    tt.return
  }
}

// -----

#blocked_cf2 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked_cf2_red = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#tmem_cf2 = #ttng.tensor_memory_linear<{row = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]], col = [[0, 1]]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:103", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_memory_cf_same_type_ld_red
  // CHECK: %[[BASE:.+]] = nvg.tensor_memory_base
  // CHECK: %[[BASE_I:.+]] = llvm.ptrtoint %[[BASE]] : !llvm.ptr<6> to i32
  // CHECK: %[[C0:.+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[LHS_ADDR:.+]] = llvm.add %[[BASE_I]], %[[C0]] : i32
  // CHECK: %[[LHS_PTR:.+]] = llvm.inttoptr %[[LHS_ADDR]] : i32 to !llvm.ptr<3>
  // CHECK: %[[LHS_DESC:.+]] = builtin.unrealized_conversion_cast %[[LHS_PTR]] : !llvm.ptr<3> to !ttg.memdesc<128x2xf32
  // CHECK: %[[C64:.+]] = llvm.mlir.constant(64 : i32) : i32
  // CHECK: %[[RHS_ADDR:.+]] = llvm.add %[[BASE_I]], %[[C64]] : i32
  // CHECK: %[[RHS_PTR:.+]] = llvm.inttoptr %[[RHS_ADDR]] : i32 to !llvm.ptr<3>
  // CHECK: %[[RHS_DESC:.+]] = builtin.unrealized_conversion_cast %[[RHS_PTR]] : !llvm.ptr<3> to !ttg.memdesc<128x2xf32
  // CHECK: %[[SELECTED:.+]] = scf.if %arg0 -> (!ttg.memdesc<128x2xf32
  // CHECK: scf.yield %[[RHS_DESC]] : !ttg.memdesc<128x2xf32
  // CHECK: } else {
  // CHECK: scf.yield %[[LHS_DESC]] : !ttg.memdesc<128x2xf32
  // CHECK: }
  // CHECK: %[[SELECTED_PTR:.+]] = builtin.unrealized_conversion_cast %[[SELECTED]] : !ttg.memdesc<128x2xf32{{.*}} to !llvm.ptr<3>
  // CHECK: %[[SELECTED_I:.+]] = llvm.ptrtoint %[[SELECTED_PTR]] : !llvm.ptr<3> to i32
  // CHECK: %[[LOAD_ADDR:.+]] = llvm.add %[[SELECTED_I]], %{{.+}} : i32
  // CHECK: llvm.inline_asm {{.*}}"tcgen05.ld.red.sync.aligned.32x32b.x2.max.f32{{.*}}"{{.*}} %[[LOAD_ADDR]] : (i32) ->
  // CHECK: nvvm.tcgen05.wait <load>
  tt.func public @tensor_memory_cf_same_type_ld_red(%choose_rhs: i1) {
    %lhs = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x2xf32, #tmem_cf2, #ttng.tensor_memory, mutable>
    %rhs = ttng.tmem_alloc {tensor_memory_col_offset = 64 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x2xf32, #tmem_cf2, #ttng.tensor_memory, mutable>
    %selected = scf.if %choose_rhs -> (!ttg.memdesc<128x2xf32, #tmem_cf2, #ttng.tensor_memory, mutable>) {
      scf.yield %rhs : !ttg.memdesc<128x2xf32, #tmem_cf2, #ttng.tensor_memory, mutable>
    } else {
      scf.yield %lhs : !ttg.memdesc<128x2xf32, #tmem_cf2, #ttng.tensor_memory, mutable>
    }
    %result, %red = ttng.tmem_load %selected {redOp = #ttng.redOp<max>} : !ttg.memdesc<128x2xf32, #tmem_cf2, #ttng.tensor_memory, mutable> -> tensor<128x2xf32, #blocked_cf2>, tensor<128xf32, #blocked_cf2_red>
    tt.return
  }
}


// -----

#blocked_cf2 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem_cf2 = #ttng.tensor_memory_linear<{row = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]], col = [[0, 1]]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:103", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_memory_cf_same_type_store
  // CHECK: %[[BASE:.+]] = nvg.tensor_memory_base
  // CHECK: %[[BASE_I:.+]] = llvm.ptrtoint %[[BASE]] : !llvm.ptr<6> to i32
  // CHECK: %[[C0:.+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[LHS_ADDR:.+]] = llvm.add %[[BASE_I]], %[[C0]] : i32
  // CHECK: %[[LHS_PTR:.+]] = llvm.inttoptr %[[LHS_ADDR]] : i32 to !llvm.ptr<3>
  // CHECK: %[[LHS_DESC:.+]] = builtin.unrealized_conversion_cast %[[LHS_PTR]] : !llvm.ptr<3> to !ttg.memdesc<128x2xf32
  // CHECK: %[[C64:.+]] = llvm.mlir.constant(64 : i32) : i32
  // CHECK: %[[RHS_ADDR:.+]] = llvm.add %[[BASE_I]], %[[C64]] : i32
  // CHECK: %[[RHS_PTR:.+]] = llvm.inttoptr %[[RHS_ADDR]] : i32 to !llvm.ptr<3>
  // CHECK: %[[RHS_DESC:.+]] = builtin.unrealized_conversion_cast %[[RHS_PTR]] : !llvm.ptr<3> to !ttg.memdesc<128x2xf32
  // CHECK: %[[SELECTED:.+]] = scf.if %arg0 -> (!ttg.memdesc<128x2xf32
  // CHECK: scf.yield %[[RHS_DESC]] : !ttg.memdesc<128x2xf32
  // CHECK: } else {
  // CHECK: scf.yield %[[LHS_DESC]] : !ttg.memdesc<128x2xf32
  // CHECK: }
  // CHECK: %[[SELECTED_PTR:.+]] = builtin.unrealized_conversion_cast %[[SELECTED]] : !ttg.memdesc<128x2xf32{{.*}} to !llvm.ptr<3>
  // CHECK: %[[SELECTED_I:.+]] = llvm.ptrtoint %[[SELECTED_PTR]] : !llvm.ptr<3> to i32
  // CHECK: %[[STORE_ADDR:.+]] = llvm.add %[[SELECTED_I]], %{{.+}} : i32
  // CHECK: llvm.inline_asm {{.*}}tcgen05.st.sync.aligned.32x32b.x2.b32{{.*}} %{{.*}}, %[[STORE_ADDR]],
  tt.func public @tensor_memory_cf_same_type_store(%choose_rhs: i1, %value: tensor<128x2xf32, #blocked_cf2>) {
    %lhs = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x2xf32, #tmem_cf2, #ttng.tensor_memory, mutable>
    %rhs = ttng.tmem_alloc {tensor_memory_col_offset = 64 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x2xf32, #tmem_cf2, #ttng.tensor_memory, mutable>
    %true = arith.constant true
    %selected = scf.if %choose_rhs -> (!ttg.memdesc<128x2xf32, #tmem_cf2, #ttng.tensor_memory, mutable>) {
      scf.yield %rhs : !ttg.memdesc<128x2xf32, #tmem_cf2, #ttng.tensor_memory, mutable>
    } else {
      scf.yield %lhs : !ttg.memdesc<128x2xf32, #tmem_cf2, #ttng.tensor_memory, mutable>
    }
    ttng.tmem_store %value, %selected, %true : tensor<128x2xf32, #blocked_cf2> -> !ttg.memdesc<128x2xf32, #tmem_cf2, #ttng.tensor_memory, mutable>
    tt.return
  }
}
