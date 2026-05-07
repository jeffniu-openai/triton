// RUN: triton-opt %s -split-input-file --gluon-infer-coalesced-encodings | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @infer_efficient(%in_ptr : !tt.ptr<f32>, %out_ptr : !tt.ptr<f32>) {
    // CHECK: [[BLOCKED:#.+]] = #ttg.blocked
    // CHECK: %[[IN_PTRS:.+]] = gluon.set_auto_layout {{.*}} : tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding> -> tensor<128x256x!tt.ptr<f32>, [[BLOCKED]]>
    // CHECK: %[[MASK_IN:.+]] = gluon.set_auto_layout {{.*}} : tensor<128x256xi1, #gluon.auto_encoding> -> tensor<128x256xi1, [[BLOCKED]]>
    // CHECK: %[[VALUE:.+]] = tt.load %[[IN_PTRS]], %[[MASK_IN]] : tensor<128x256x!tt.ptr<f32>, [[BLOCKED]]>
    %mask = arith.constant dense<0> : tensor<128x256xi1, #gluon.auto_encoding>
    %in_ptrs_1 = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding>
    %in_ptrs_2 = gluon.set_auto_layout %in_ptrs_1 : tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding> -> tensor<128x256x!tt.ptr<f32>, #gluon.coalesced_encoding>
    %mask_in = gluon.set_auto_layout %mask : tensor<128x256xi1, #gluon.auto_encoding> -> tensor<128x256xi1, #gluon.coalesced_encoding>
    %value = tt.load %in_ptrs_2, %mask_in : tensor<128x256x!tt.ptr<f32>, #gluon.coalesced_encoding>

    // CHECK: %[[SIN:.+]] = math.sin %[[VALUE]] : tensor<128x256xf32, [[BLOCKED]]>
    // CHECK: %[[MAX:.+]] = arith.maxnumf %[[SIN]], {{.*}} : tensor<128x256xf32, [[BLOCKED]]>
    %value_2 = math.sin %value : tensor<128x256xf32, #gluon.coalesced_encoding>
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #gluon.coalesced_encoding>
    %value_3 = arith.maxnumf %value_2, %cst : tensor<128x256xf32, #gluon.coalesced_encoding>

    // CHECK: %[[OUT_PTRS:.+]] = gluon.set_auto_layout {{.*}} : tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding> -> tensor<128x256x!tt.ptr<f32>, [[BLOCKED]]>
    // CHECK: %[[MASK_OUT:.+]] = gluon.set_auto_layout {{.*}} : tensor<128x256xi1, #gluon.auto_encoding> -> tensor<128x256xi1, [[BLOCKED]]>
    // CHECK: tt.store %[[OUT_PTRS]], %[[MAX]], %[[MASK_OUT]] : tensor<128x256x!tt.ptr<f32>, [[BLOCKED]]>
    %out_ptrs_1 = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding>
    %out_ptrs_2 = gluon.set_auto_layout %out_ptrs_1 : tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding> -> tensor<128x256x!tt.ptr<f32>, #gluon.coalesced_encoding>
    %mask_out = gluon.set_auto_layout %mask : tensor<128x256xi1, #gluon.auto_encoding> -> tensor<128x256xi1, #gluon.coalesced_encoding>
    tt.store %out_ptrs_2, %value_3, %mask_out : tensor<128x256x!tt.ptr<f32>, #gluon.coalesced_encoding>
    tt.return
  }
}



// -----

#tmem_root = #ttng.tensor_memory_linear<{row = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]], col = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]]}>
#tmem_reshape = #ttng.tensor_memory_linear<{row = [[0, 1, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0], [1, 0, 0, 0]], col = [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 4], [0, 0, 0, 8], [0, 0, 0, 16], [0, 0, 0, 32], [0, 0, 1, 0]]}>
#tmem_squeezed = #ttng.tensor_memory_linear<{row = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0]], col = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0]]}>
#tmem_subsliced = #ttng.tensor_memory_linear<{row = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], col = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0]]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @preserve_tmem_subslice_active_encoding
  tt.func public @preserve_tmem_subslice_active_encoding() {
    // CHECK: %[[ROOT:.+]] = ttng.tmem_alloc
    // CHECK: %[[RESHAPE:.+]] = ttg.memdesc_reshape %[[ROOT]]
    // CHECK: %[[FIRST:.+]] = ttg.memdesc_subslice %[[RESHAPE]][1, 0, 0, 0]
    // CHECK-SAME: -> !ttg.memdesc<1x64x2x64xf32, [[SQUEEZED:#[a-zA-Z0-9_]+]]
    // CHECK: ttg.memdesc_subslice %[[FIRST]][0, 0, 0, 0]
    // CHECK-SAME: : !ttg.memdesc<1x64x2x64xf32, [[SQUEEZED]]
    // CHECK-SAME: -> !ttg.memdesc<1x32x2x64xf32, [[SUBSLICED:#[a-zA-Z0-9_]+]]
    %root = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem_root, #ttng.tensor_memory, mutable>
    %reshape = ttg.memdesc_reshape %root : !ttg.memdesc<128x128xf32, #tmem_root, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x64x2x64xf32, #tmem_reshape, #ttng.tensor_memory, mutable>
    %first = ttg.memdesc_subslice %reshape[1, 0, 0, 0] : !ttg.memdesc<2x64x2x64xf32, #tmem_reshape, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x64x2x64xf32, #tmem_squeezed, #ttng.tensor_memory, mutable, 2x64x2x64>
    %second = ttg.memdesc_subslice %first[0, 0, 0, 0] : !ttg.memdesc<1x64x2x64xf32, #tmem_squeezed, #ttng.tensor_memory, mutable, 2x64x2x64> -> !ttg.memdesc<1x32x2x64xf32, #tmem_subsliced, #ttng.tensor_memory, mutable, 2x64x2x64>
    tt.return
  }
}
