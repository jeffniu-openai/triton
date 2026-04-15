// RUN: triton-opt --split-input-file %s --verify-diagnostics

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @alloc_tensor_memory() {
    // expected-error @+1 {{uninitialized alloc must have a mutable memdesc type}}
    %0 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
    tt.return
  }
}

// -----
#tmem_linear_64 = #ttng.tensor_memory_linear<{row = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]], col = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tmem_alloc_result_alloc_shape_mismatch() {
    // expected-error @+1 {{result shape and its alloc shape must match}}
    %0 = ttng.tmem_alloc : () -> !ttg.memdesc<64x64xf32, #tmem_linear_64, #ttng.tensor_memory, mutable, 128x128>
    tt.return
  }
}

// -----

#blocked_128x64 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tmem_alloc_source_shape_mismatch() {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked_128x64>
    // expected-error @+1 {{source shape [128, 64] must match}}
    %0 = ttng.tmem_alloc %cst : (tensor<128x64xf32, #blocked_128x64>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem_f16 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 2>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tmem_alloc_source_element_type_mismatch() {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // expected-error @+1 {{source element type}}
    %0 = ttng.tmem_alloc %cst : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf16, #tmem_f16, #ttng.tensor_memory>
    tt.return
  }
}

// -----


#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tmem_layout_cta_mismatch() {
    // expected-error @+1 {{Layout has 1 CTAs per CGA, but the context requires 2 CTAs per CGA.}}
    %0 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// expected-error @+2 {{Expected basis of 'col' not found}}
#tmem_linear_missing_col = #ttng.tensor_memory_linear<{row = [[1, 0]]}>
module attributes {"ttg.num-warps" = 1 : i32} {
  tt.func @dummy_linear_missing_col() {
    tt.return
  }
}

// -----

// expected-error @+2 {{Expected basis of 'row' not found}}
#tmem_linear_missing_row = #ttng.tensor_memory_linear<{col = [[0, 1]]}>
module attributes {"ttg.num-warps" = 1 : i32} {
  tt.func @dummy_linear_missing_row() {
    tt.return
  }
}

// -----

// expected-error @+1 {{twoCTAs requires a non-empty 'block' basis sequence}}
#tmem_linear_twoctas_missing_block = #ttng.tensor_memory_linear<{row = [[1, 0]], col = [[0, 1]]}, twoCTAs = true>
module attributes {"ttg.num-warps" = 1 : i32} {
  tt.func @dummy_linear_twoctas_missing_block() {
    tt.return
  }
}

// -----

// expected-error @+1 {{twoCTAs requires at least one non-zero 'block' basis}}
#tmem_linear_twoctas_bad_block_basis = #ttng.tensor_memory_linear<{row = [[1, 0]], col = [[0, 1]], block = [[0, 0]]}, twoCTAs = true>
module attributes {"ttg.num-warps" = 1 : i32} {
  tt.func @dummy_linear_twoctas_bad_block_basis() {
    tt.return
  }
}

// -----

// expected-error @+1 {{expected a boolean value for twoCTAs}}
#tmem_linear_twoctas_non_bool = #ttng.tensor_memory_linear<{row = [[1, 0]], col = [[0, 1]], block = [[1, 0]]}, twoCTAs = 1>
module attributes {"ttg.num-warps" = 1 : i32} {
  tt.func @dummy_linear_twoctas_non_bool() {
    tt.return
  }
}

// -----

#tmem_linear_twoctas = #ttng.tensor_memory_linear<{row = [[2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]], col = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], block = [[1, 0]]}, twoCTAs = true>
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tmem_load_twoctas_result_layout_mismatch(%arg0: !ttg.memdesc<128x128xf32, #tmem_linear_twoctas, #ttng.tensor_memory, mutable>) {
    // expected-error @+1 {{Result has an invalid layout}}
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xf32, #tmem_linear_twoctas, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    tt.return
  }
}

// -----

#tmem_linear_bad_row_basis_syntax = #ttng.tensor_memory_linear<{row = 7, col = [[0, 1]]}>
// expected-error @+1 {{Expected array of arrays for basis of 'row'}}
module attributes {"ttg.num-warps" = 1 : i32} {
  tt.func @dummy_linear_bad_row_basis_syntax() {
    tt.return
  }
}

// -----

#tmem_linear_bad_basis_rank = #ttng.tensor_memory_linear<{row = [[1, 0], [2, 0, 0]], col = [[0, 1], [0, 2]]}>
// expected-error @+1 {{Invalid bases passed to LinearLayout.}}
module attributes {"ttg.num-warps" = 1 : i32} {
  tt.func @dummy_linear_bad_basis_rank() {
    tt.return
  }
}

// -----

// expected-error @+1 {{After removing zero bases the layout must be injective}}
#tmem_linear_not_bijective = #ttng.tensor_memory_linear<{row = [[1, 0]], col = [[1, 0]]}>
module attributes {"ttg.num-warps" = 1 : i32} {
  tt.func @dummy_linear_not_bijective() {
    tt.return
  }
}

// -----

#tmem_bad_shape = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // expected-error @+1 {{cannot be canonicalized for shape [64, 64]: shape per CTA 64x64 is smaller than the TMEM tile 128x64}}
  tt.func @bad_legacy_tmem_arg(%arg0: !ttg.memdesc<64x64xf32, #tmem_bad_shape, #ttng.tensor_memory, mutable>) {
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#tmem_f32 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_linear_mma_bad_cga = #ttng.tensor_memory_linear<{row = [[2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]], col = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], block = [[1, 0]]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @tcgen5_linear_ret_layout_bad_cga(%a: !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>,
                                            %b: !ttg.memdesc<128x128xf16, #shared1, #ttg.shared_memory>,
                                            %c: !ttg.memdesc<128x128xf32, #tmem_linear_mma_bad_cga, #ttng.tensor_memory, mutable>,
                                            %useAcc: i1,
                                            %pred: i1) {
    // expected-error @+1 {{Layout has 2 CTAs per CGA, but the context requires 1 CTAs per CGA}}
    ttng.tc_gen5_mma %a, %b, %c, %useAcc, %pred :
       !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>,
       !ttg.memdesc<128x128xf16, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x128xf32, #tmem_linear_mma_bad_cga, #ttng.tensor_memory, mutable>
    tt.return
  }

  tt.func @tcgen5_linear_lhs_layout_bad_cga(%a: !ttg.memdesc<128x128xf16, #tmem_linear_mma_bad_cga, #ttng.tensor_memory, mutable>,
                                            %b: !ttg.memdesc<128x128xf16, #shared1, #ttg.shared_memory>,
                                            %c: !ttg.memdesc<128x128xf32, #tmem_f32, #ttng.tensor_memory, mutable>,
                                            %useAcc: i1,
                                            %pred: i1) {
    // expected-error @+1 {{Layout has 2 CTAs per CGA, but the context requires 1 CTAs per CGA}}
    ttng.tc_gen5_mma %a, %b, %c, %useAcc, %pred :
       !ttg.memdesc<128x128xf16, #tmem_linear_mma_bad_cga, #ttng.tensor_memory, mutable>,
       !ttg.memdesc<128x128xf16, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x128xf32, #tmem_f32, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#sharedT = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
#tmem_linear_mma_bad_cga_64 = #ttng.tensor_memory_linear<{row = [[2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]], col = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], block = [[1, 0]]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @tcgen5_scaled_linear_layout_bad_cga(
      %a: !ttg.memdesc<128x256xi8, #shared, #ttg.shared_memory>,
      %b: !ttg.memdesc<256x64xi8, #sharedT, #ttg.shared_memory>,
      %c: !ttg.memdesc<128x64xf32, #tmem_linear_mma_bad_cga_64, #ttng.tensor_memory, mutable>,
      %scale_a: !ttg.memdesc<128x8xf8E4M3FN, #shared1, #ttg.shared_memory>,
      %scale_b: !ttg.memdesc<64x8xf8E4M3FN, #shared1, #ttg.shared_memory>,
      %useAcc: i1,
      %pred: i1) {
    // expected-error @+1 {{Layout has 2 CTAs per CGA, but the context requires 1 CTAs per CGA}}
    ttng.tc_gen5_mma_scaled %a, %b, %c, %scale_a, %scale_b, %useAcc, %pred lhs = e2m1 rhs = e2m1 :
      !ttg.memdesc<128x256xi8, #shared, #ttg.shared_memory>,
      !ttg.memdesc<256x64xi8, #sharedT, #ttg.shared_memory>,
      !ttg.memdesc<128x64xf32, #tmem_linear_mma_bad_cga_64, #ttng.tensor_memory, mutable>,
      !ttg.memdesc<128x8xf8E4M3FN, #shared1, #ttg.shared_memory>,
      !ttg.memdesc<64x8xf8E4M3FN, #shared1, #ttg.shared_memory>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#sharedT = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#shared_scale = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#sharedT = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
#tmem_linear_tile_perm_128_32 = #ttng.tensor_memory_linear<{row = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]], col = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 64], [0, 32]]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @tcgen5_scaled_tile_permuted_accumulator_not_directly_supported(
      %a: !ttg.memdesc<128x32xi8, #shared, #ttg.shared_memory>,
      %b: !ttg.memdesc<32x128xi8, #sharedT, #ttg.shared_memory>,
      %c: !ttg.memdesc<128x128xf32, #tmem_linear_tile_perm_128_32, #ttng.tensor_memory, mutable>,
      %scale_a: !ttg.memdesc<128x2xi8, #tmem_scales, #ttng.tensor_memory>,
      %scale_b: !ttg.memdesc<128x2xi8, #tmem_scales, #ttng.tensor_memory>,
      %useAcc: i1,
      %pred: i1) {
    // expected-error @+1 {{direct block-scaled MMAv5 does not support repeated N=32 instructions along N}}
    ttng.tc_gen5_mma_scaled %a, %b, %c, %scale_a, %scale_b, %useAcc, %pred lhs = e5m2 rhs = e5m2 :
      !ttg.memdesc<128x32xi8, #shared, #ttg.shared_memory>,
      !ttg.memdesc<32x128xi8, #sharedT, #ttg.shared_memory>,
      !ttg.memdesc<128x128xf32, #tmem_linear_tile_perm_128_32, #ttng.tensor_memory, mutable>,
      !ttg.memdesc<128x2xi8, #tmem_scales, #ttng.tensor_memory>,
      !ttg.memdesc<128x2xi8, #tmem_scales, #ttng.tensor_memory>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @alloc_tensor_memory() {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %0 = ttng.tmem_alloc %cst : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
    // expected-error @+1 {{Cannot store into an immutable alloc}}
    ttng.tmem_store %cst, %0, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
    tt.return
  }
}

// -----

#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#scales = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#tmem = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @alloc_tensor_memory() {
    %cst = arith.constant dense<0> : tensor<128x4xi8, #scales>
    %0 = ttng.tmem_alloc %cst : (tensor<128x4xi8, #scales>) -> !ttg.memdesc<128x4xi8, #tmem, #ttng.tensor_memory>
    tt.return
  }

  tt.func public @tmem_load_scales_memdesc_invalid(%arg: !ttg.memdesc<128x4xi8, #tmem, #ttng.tensor_memory, mutable>) {
    %0 = ttng.tmem_load %arg : !ttg.memdesc<128x4xi8, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x4xi8, #scales>
    tt.return
  }
}

// -----

#scales_bad = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tmem_alloc_scales_source_layout_rejected(%arg: tensor<128x8xi8, #scales_bad>) {
    %0 = ttng.tmem_alloc %arg : (tensor<128x8xi8, #scales_bad>) -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
    tt.return
  }
}

// -----

#shared_scales_warpx2_candidate = #ttg.shared_linear<{offset = [[32, 0], [0, 1], [1, 0], [0, 2], [0, 4], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]]}, alignment = 16>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
#shared_cp_warpx2_candidate = #ttg.shared_linear<{offset = [[32, 0], [0, 1], [0, 2], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [64, 0]]}, alignment = 16>
#tmem_linear_cp_128x4 = #ttng.tensor_memory_linear<{row = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]], col = [[0, 1], [0, 2]]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tmem_copy_scales_descriptor_family_clean_unsupported(
      %src: !ttg.memdesc<64x16xi8, #shared_scales_warpx2_candidate, #ttg.shared_memory, mutable>,
      %dst: !ttg.memdesc<64x16xi8, #tmem_scales, #ttng.tensor_memory, mutable>) {
    // expected-error @+4 {{'ttng.tmem_copy' op The source shared layout maps to tcgen05.copy.warpx4.32x128b, but Triton could not synthesize a compatible shared-memory descriptor plan for tensor memory scales.}}
    // expected-note @+3 {{tcgen05.copy.warpx4.32x128b descriptor message 0 has no representable MMAv5 shared-memory descriptor; tried 1 candidate layout(s) for descriptor shape [32, 16] and instruction shape [32, 16].}}
    // expected-note @+2 {{Use a shared layout that lowers to tcgen05.copy.warpx4.32x128b, or reshape / permute the shared tile until it lowers to the same descriptor family.}}
    // expected-note @+1 {{This is reported as cleanly unsupported instead of falling through to late LLVM lowering.}}
    ttng.tmem_copy %src, %dst : !ttg.memdesc<64x16xi8, #shared_scales_warpx2_candidate, #ttg.shared_memory, mutable>, !ttg.memdesc<64x16xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared_cp_warpx2_candidate = #ttg.shared_linear<{offset = [[32, 0], [0, 1], [0, 2], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [64, 0]]}, alignment = 16>
#tmem_linear_cp_128x4 = #ttng.tensor_memory_linear<{row = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]], col = [[0, 1], [0, 2]]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tmem_copy_no_scales_warpx2_candidate(
      %src: !ttg.memdesc<128x4xi32, #shared_cp_warpx2_candidate, #ttg.shared_memory, mutable>,
      %dst: !ttg.memdesc<128x4xi32, #tmem_linear_cp_128x4, #ttng.tensor_memory, mutable>) {
    // expected-error @+5 {{The source shared layout maps to tcgen05.copy.128x128b, but Triton could not synthesize a compatible shared-memory descriptor plan for it.}}
    // expected-note @+4 {{tcgen05.copy.128x128b descriptor message 0 has no representable MMAv5 shared-memory descriptor; tried 2 candidate layout(s) for descriptor shape [32, 4] and instruction shape [32, 4].}}
    // expected-note @+3 {{tcgen05.copy.128x128b descriptor message 0 has no representable MMAv5 shared-memory descriptor; tried 3 candidate layout(s) for descriptor shape [64, 4] and instruction shape [64, 4].}}
    // expected-note @+2 {{Use the canonical shared layout for tcgen05.copy.128x128b, or reshape / permute the shared tile until it lowers to the same descriptor family.}}
    // expected-note @+1 {{This is reported as cleanly unsupported instead of falling through to late LLVM lowering.}}
    ttng.tmem_copy %src, %dst : !ttg.memdesc<128x4xi32, #shared_cp_warpx2_candidate, #ttg.shared_memory, mutable>, !ttg.memdesc<128x4xi32, #tmem_linear_cp_128x4, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared_cp_warpx2_02_13_twocta = #ttg.shared_linear<{offset = [[32, 0], [0, 1], [0, 2], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [64, 0]], block = [[128, 0]]}, alignment = 16>
#tmem_cp_warpx2_02_13_twocta = #ttng.tensor_memory_linear<{row = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 0]], col = [[0, 1], [0, 2]], block = [[128, 0]], out = [256, 4]}, twoCTAs = true>
module attributes {"ttg.num-ctas" = 2 : i32, "ttng.two-ctas" = true, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tmem_copy_no_scales_warpx2_02_13_twocta_clean_unsupported(
      %src: !ttg.memdesc<256x4xi32, #shared_cp_warpx2_02_13_twocta, #ttg.shared_memory, mutable>,
      %dst: !ttg.memdesc<256x4xi32, #tmem_cp_warpx2_02_13_twocta, #ttng.tensor_memory, mutable>) {
    // expected-error @+6 {{The source shared layout maps to tcgen05.copy.warpx2::02_13.64x128b, but Triton could not synthesize a compatible shared-memory descriptor plan for it.}}
    // expected-note @+5 {{tcgen05.copy.warpx2::02_13.64x128b descriptor message 0 has no representable MMAv5 shared-memory descriptor; tried 95 candidate layout(s) for descriptor shape [64, 4] and instruction shape [64, 4].}}
    // expected-note @+4 {{tcgen05.copy.warpx2::02_13.64x128b descriptor message 0 has no representable MMAv5 shared-memory descriptor; tried 161 candidate layout(s) for descriptor shape [32, 4] and instruction shape [64, 4].}}
    // expected-note @+3 {{The two-CTA warpx2::02_13 path remains unsupported until Triton can synthesize a cta_group::2 descriptor/address schedule that preserves the high source-column bit; decomposing this tensor-memory view into cta_group::1 copies is not valid because two-CTA TMEM allocation uses cta_group::2 granularity.}}
    // expected-note @+2 {{Use the canonical shared layout for tcgen05.copy.warpx2::02_13.64x128b, or reshape / permute the shared tile until it lowers to the same descriptor family.}}
    // expected-note @+1 {{This is reported as cleanly unsupported instead of falling through to late LLVM lowering.}}
    ttng.tmem_copy %src, %dst : !ttg.memdesc<256x4xi32, #shared_cp_warpx2_02_13_twocta, #ttg.shared_memory, mutable>, !ttg.memdesc<256x4xi32, #tmem_cp_warpx2_02_13_twocta, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared_cp_4x256b = #ttg.shared_linear<{offset = [[1, 0], [2, 0], [0, 1], [0, 2], [0, 4]]}, alignment = 16>
#tmem_linear_cp_4x256b = #ttng.tensor_memory_linear<{row = [[1, 0], [2, 0]], col = [[0, 1], [0, 2], [0, 4]]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tmem_copy_no_scales_4x256b_clean_unsupported(
      %src: !ttg.memdesc<4x8xi32, #shared_cp_4x256b, #ttg.shared_memory, mutable>,
      %dst: !ttg.memdesc<4x8xi32, #tmem_linear_cp_4x256b, #ttng.tensor_memory, mutable>) {
    // expected-error @+4 {{The source shared layout maps to tcgen05.copy.4x256b, but Triton could not synthesize a compatible shared-memory descriptor plan for it.}}
    // expected-note @+3 {{tcgen05.copy.4x256b is recognized by the ISA, but Triton does not yet have a validated descriptor/address schedule for it. The previous four-row descriptor candidate placed source row values into a single destination row}}
    // expected-note @+2 {{Use the canonical shared layout for tcgen05.copy.4x256b, or reshape / permute the shared tile until it lowers to the same descriptor family.}}
    // expected-note @+1 {{This is reported as cleanly unsupported instead of falling through to late LLVM lowering.}}
    ttng.tmem_copy %src, %dst : !ttg.memdesc<4x8xi32, #shared_cp_4x256b, #ttg.shared_memory, mutable>, !ttg.memdesc<4x8xi32, #tmem_linear_cp_4x256b, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared_cp_4x256b_twocta = #ttg.shared_linear<{offset = [[1, 0], [2, 0], [0, 1], [0, 2], [0, 4]], block = [[4, 0]]}, alignment = 16>
#tmem_linear_cp_4x256b_twocta = #ttng.tensor_memory_linear<{row = [[1, 0], [2, 0]], col = [[0, 1], [0, 2], [0, 4]], block = [[4, 0]], out = [8, 8]}, twoCTAs = true>
module attributes {"ttg.num-ctas" = 2 : i32, "ttng.two-ctas" = true, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tmem_copy_no_scales_4x256b_twocta_clean_unsupported(
      %src: !ttg.memdesc<8x8xi32, #shared_cp_4x256b_twocta, #ttg.shared_memory, mutable>,
      %dst: !ttg.memdesc<8x8xi32, #tmem_linear_cp_4x256b_twocta, #ttng.tensor_memory, mutable>) {
    // expected-error @+4 {{The source shared layout maps to tcgen05.copy.4x256b, but Triton could not synthesize a compatible shared-memory descriptor plan for it.}}
    // expected-note @+3 {{tcgen05.copy.4x256b is recognized by the ISA, but Triton does not yet have a validated descriptor/address schedule for it. The previous four-row descriptor candidate placed source row values into a single destination row}}
    // expected-note @+2 {{Use the canonical shared layout for tcgen05.copy.4x256b, or reshape / permute the shared tile until it lowers to the same descriptor family.}}
    // expected-note @+1 {{This is reported as cleanly unsupported instead of falling through to late LLVM lowering.}}
    ttng.tmem_copy %src, %dst : !ttg.memdesc<8x8xi32, #shared_cp_4x256b_twocta, #ttg.shared_memory, mutable>, !ttg.memdesc<8x8xi32, #tmem_linear_cp_4x256b_twocta, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared_f32 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#tmem_linear_m64 = #ttng.tensor_memory_linear<{row = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]], col = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tmem_copy_linear_blockm64_not_supported(%src: !ttg.memdesc<64x128xf32, #shared_f32, #ttg.shared_memory>,
                                                   %dst: !ttg.memdesc<64x128xf32, #tmem_linear_m64, #ttng.tensor_memory, mutable>) {
    // expected-error @+3 {{The source shared layout does not match any recognized tcgen05.copy family for non-scales tensor memory copies.}}
    // expected-note @+2 {{Recognized tcgen05.copy families are 4x256b, 128x128b, 128x256b, warpx2::01_23.64x128b, warpx2::02_13.64x128b, and warpx4.32x128b.}}
    // expected-note @+1 {{Use the canonical shared layout for your intended family, or reshape / permute the shared tile until it lowers to one of those families.}}
    ttng.tmem_copy %src, %dst : !ttg.memdesc<64x128xf32, #shared_f32, #ttg.shared_memory>, !ttg.memdesc<64x128xf32, #tmem_linear_m64, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared_f32 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#tmem_linear_copy_mixed = #ttng.tensor_memory_linear<{row = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1], [0, 2]], col = [[32, 0], [64, 0], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tmem_copy_linear_mixed_not_supported(%src: !ttg.memdesc<128x128xf32, #shared_f32, #ttg.shared_memory>,
                                                 %dst: !ttg.memdesc<128x128xf32, #tmem_linear_copy_mixed, #ttng.tensor_memory, mutable>) {
    // expected-error @+4 {{The source shared layout maps to tcgen05.copy.128x256b, but Triton could not synthesize a compatible shared-memory descriptor plan for it.}}
    // expected-note @+3 {{Use the canonical shared layout for tcgen05.copy.128x256b, or reshape / permute the shared tile until it lowers to the same descriptor family.}}
    // expected-note @+2 {{This is reported as cleanly unsupported instead of falling through to late LLVM lowering.}}
    // expected-note @+1 {{direct tcgen05.copy does not support TMEM row bases that mix row and column contributions.}}
    ttng.tmem_copy %src, %dst : !ttg.memdesc<128x128xf32, #shared_f32, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem_linear_copy_mixed, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared_f16 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_f16_t = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#tmem_linear_mixed = #ttng.tensor_memory_linear<{row = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1], [0, 2]], col = [[32, 0], [64, 0], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]]}>
#tmem_linear_rowcol_permuted = #ttng.tensor_memory_linear<{row = [[1, 0], [4, 0], [16, 0], [64, 0], [2, 0], [8, 0], [32, 0]], col = [[0, 1], [0, 4], [0, 16], [0, 64], [0, 2], [0, 8], [0, 32]]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @tcgen5_linear_layout_not_mmav5_compatible(%a: !ttg.memdesc<128x128xf16, #shared_f16, #ttg.shared_memory>,
                                                      %b: !ttg.memdesc<128x128xf16, #shared_f16_t, #ttg.shared_memory>,
                                                      %c: !ttg.memdesc<128x128xf32, #tmem_linear_mixed, #ttng.tensor_memory, mutable>,
                                                      %useAcc: i1,
                                                      %pred: i1) {
    // expected-error @+1 {{return operand must have a MMAv5-compatible tensor memory layout}}
    ttng.tc_gen5_mma %a, %b, %c, %useAcc, %pred :
       !ttg.memdesc<128x128xf16, #shared_f16, #ttg.shared_memory>,
       !ttg.memdesc<128x128xf16, #shared_f16_t, #ttg.shared_memory>,
       !ttg.memdesc<128x128xf32, #tmem_linear_mixed, #ttng.tensor_memory, mutable>
    tt.return
  }

  tt.func @tcgen5_linear_rowcol_permuted_layout_not_mmav5_compatible(
      %a: !ttg.memdesc<128x128xf16, #shared_f16, #ttg.shared_memory>,
      %b: !ttg.memdesc<128x128xf16, #shared_f16_t, #ttg.shared_memory>,
      %c: !ttg.memdesc<128x128xf32, #tmem_linear_rowcol_permuted, #ttng.tensor_memory, mutable>,
      %useAcc: i1,
      %pred: i1) {
    // expected-error @+1 {{return operand must have a MMAv5-compatible tensor memory layout}}
    ttng.tc_gen5_mma %a, %b, %c, %useAcc, %pred :
       !ttg.memdesc<128x128xf16, #shared_f16, #ttg.shared_memory>,
       !ttg.memdesc<128x128xf16, #shared_f16_t, #ttg.shared_memory>,
       !ttg.memdesc<128x128xf32, #tmem_linear_rowcol_permuted, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared_f16_t = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#shared_i8_t = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#tmem_linear = #ttng.tensor_memory_linear<{row = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]], col = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]]}>
#tmem_linear_small = #ttng.tensor_memory_linear<{row = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]], col = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]]}>
#tmem_linear_tiny = #ttng.tensor_memory_linear<{row = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]], col = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]]}>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
#tmem_linear_interleaved_bm64 = #ttng.tensor_memory_linear<{row = [[1, 0], [2, 0], [4, 0], [8, 0], [64, 0], [16, 0], [32, 0]], col = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]]}>

#blocked_tmem_impossible = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#tmem_linear_exotic_impossible = #ttng.tensor_memory_linear<{row = [[2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]], col = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0]]}>
module attributes {"ttg.target" = "cuda:100", "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tmem_load_exotic_view_has_no_fallback_layout(
      %arg0: !ttg.memdesc<128x32xf32, #tmem_linear_exotic_impossible, #ttng.tensor_memory, mutable>) {
    // expected-error @+4 {{result has no supported register layout}}
    // expected-note @+3 {{Got: #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}}
    // expected-note @+2 {{requested layout direct-lowering details:}}
    // expected-note @+1 {{No TMEM-compatible register layout exists for this operand.}}
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x32xf32, #tmem_linear_exotic_impossible, #ttng.tensor_memory, mutable> -> tensor<128x32xf32, #blocked_tmem_impossible>
    tt.return
  }

  tt.func @tcgen5_linear_interleaved_bm64_rejected(
      %a: !ttg.memdesc<128x64xf16, #tmem_linear_interleaved_bm64, #ttng.tensor_memory>,
      %b: !ttg.memdesc<64x64xf16, #shared_f16_t, #ttg.shared_memory>,
      %c: !ttg.memdesc<128x64xf32, #tmem_linear_interleaved_bm64, #ttng.tensor_memory, mutable>,
      %useAcc: i1,
      %pred: i1) {
    // expected-error @+1 {{does not support blockM=64 with interleaved blocks in TMEM layout}}
    ttng.tc_gen5_mma %a, %b, %c, %useAcc, %pred :
      !ttg.memdesc<128x64xf16, #tmem_linear_interleaved_bm64, #ttng.tensor_memory>,
      !ttg.memdesc<64x64xf16, #shared_f16_t, #ttg.shared_memory>,
      !ttg.memdesc<128x64xf32, #tmem_linear_interleaved_bm64, #ttng.tensor_memory, mutable>
    tt.return
  }

}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @init_barrier_zero_count() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // expected-error @+1 {{count must be greater than or equal to 1}}
    ttng.init_barrier %bar, 0 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
tt.func @async_tma_gather(%desc: !tt.tensordesc<tensor<1x128xbf16, #shared>>, %x_offsets: tensor<32xi32, #blocked>, %y_offset: i32,
                          %bar: !ttg.memdesc<2xi32, #shared1, #ttg.shared_memory, mutable>,
                          %result: !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory, mutable>,
                          %pred: i1) {
  // expected-error @below {{barrier allocation must be a descriptor of Nxi64 type with N <= number of CTAs}}
  ttng.async_tma_gather %desc[%x_offsets, %y_offset] %result, %bar, %pred : !tt.tensordesc<tensor<1x128xbf16, #shared>>, tensor<32xi32, #blocked>, i32, !ttg.memdesc<2xi32, #shared1, #ttg.shared_memory, mutable>, !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory, mutable>, i1
  tt.return
}
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32} {
tt.func @async_tma_gather(%desc: !tt.tensordesc<tensor<1x128xbf16, #shared>>, %x_offsets: tensor<32xi32, #blocked>, %y_offset: i32,
                          %bar: !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>,
                          %result: !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory>,
                          %pred: i1) {
  // expected-error @below {{cannot store into immutable memory}}
  ttng.async_tma_gather %desc[%x_offsets, %y_offset] %result, %bar, %pred : !tt.tensordesc<tensor<1x128xbf16, #shared>>, tensor<32xi32, #blocked>, i32, !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory, mutable>, !ttg.memdesc<32x128xbf16, #shared, #ttg.shared_memory>, i1
  tt.return
}
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32} {
tt.func @wgmma(%a: tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>, %b: !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, %c: tensor<128x128xf16, #mma>) {
  // expected-error @below {{in-register LHS operand must have a kWidth of 2 but got 1}}
  %0 = ttng.warp_group_dot %a, %b, %c : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory> -> tensor<128x128xf16, #mma>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @async_tma_copy_global_to_local(%arg0: !tt.tensordesc<tensor<1x256x32xf32, #shared>>) -> tensor<256x32xf32, #blocked> {
    %true = arith.constant true
    %c32_i32 = arith.constant 32 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<256x32xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // expected-error @below {{TMA descriptor must have NVMMA shared layout}}
    ttng.async_tma_copy_global_to_local %arg0[%c32_i32, %c32_i32, %c32_i32] %0, %1, %true : !tt.tensordesc<tensor<1x256x32xf32, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<256x32xf32, #shared, #smem, mutable>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @async_tma_copy_global_to_local(%arg0: !tt.tensordesc<tensor<1x256x32xf32, #shared>>) -> tensor<256x32xf32, #blocked> {
    %true = arith.constant true
    %c32_i32 = arith.constant 32 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<256x32xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    // expected-error @below {{TMA descriptor layout must not be transposed}}
    ttng.async_tma_copy_global_to_local %arg0[%c32_i32, %c32_i32, %c32_i32] %0, %1, %true : !tt.tensordesc<tensor<1x256x32xf32, #shared>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<256x32xf32, #shared, #smem, mutable>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#nvmma32 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#nvmma64 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared_mbar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @async_tma_copy_global_to_local(%arg0: !tt.tensordesc<tensor<1x256x64xf32, #nvmma32>>) {
    %true = arith.constant true
    %c32_i32 = arith.constant 32 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<256x64xf32, #nvmma64, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared_mbar, #smem, mutable>
    // expected-error @below {{TMA descriptor layout must match shared layout}}
    ttng.async_tma_copy_global_to_local %arg0[%c32_i32, %c32_i32, %c32_i32] %0, %1, %true : !tt.tensordesc<tensor<1x256x64xf32, #nvmma32>>, !ttg.memdesc<1xi64, #shared_mbar, #smem, mutable> -> !ttg.memdesc<256x64xf32, #nvmma64, #smem, mutable>
    tt.return
  }
}
// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tma_im2col_missing_offsets(%arg0: !ttng.tensordesc_im2col<tensor<64x128xf16, #nvmma_128>>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    // expected-error @below {{IM2COL mode requires offsets to be provided}}
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32, %c0_i32, %c0_i32] %0, %1, %true : !ttng.tensordesc_im2col<tensor<64x128xf16, #nvmma_128>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tma_im2col_wrong_offset_count(%arg0: !ttng.tensordesc_im2col<tensor<64x128xf16, #nvmma_128>>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i16 = arith.constant 1 : i16
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    // expected-error @below {{IM2COL mode with 4D coordinates requires 2 offsets, but got 1}}
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32, %c0_i32, %c0_i32] offsets = [%c1_i16] %0, %1, %true : !ttng.tensordesc_im2col<tensor<64x128xf16, #nvmma_128>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tma_tiled_with_offsets(%arg0: !tt.tensordesc<tensor<64x128xf16, #nvmma_128>>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i16 = arith.constant 1 : i16
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    // expected-error @below {{TILED mode does not support offsets}}
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] offsets = [%c1_i16] %0, %1, %true : !tt.tensordesc<tensor<64x128xf16, #nvmma_128>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tma_im2col_2d_invalid(%arg0: !ttng.tensordesc_im2col<tensor<64x128xf16, #nvmma_128>>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    // expected-error @below {{IM2COL mode requires at least 3D coordinates, but got 2D}}
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %1, %true : !ttng.tensordesc_im2col<tensor<64x128xf16, #nvmma_128>>, !ttg.memdesc<1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma_128, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem_f16 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 2>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @tcgen5(%a: !ttg.memdesc<128x128xbf16, #shared, #ttg.shared_memory>,
                  %b: !ttg.memdesc<128x256xbf16, #shared1, #ttg.shared_memory>,
                  %c: !ttg.memdesc<128x256xf16, #tmem_f16, #ttng.tensor_memory, mutable>,
                  %accUse: i1,
                  %pred: i1,
                  %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
                  %barrierPred: i1) {
    // expected-error @below {{unsupported accumulator dtype for operand types 'bf16' and 'bf16', accumulator dtype is 'f16' but must be one of ['f32']}}
    ttng.tc_gen5_mma %a, %b, %c, %accUse, %pred, %barrier[%barrierPred] {is_async} :
       !ttg.memdesc<128x128xbf16, #shared, #ttg.shared_memory>,
       !ttg.memdesc<128x256xbf16, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf16, #tmem_f16, #ttng.tensor_memory, mutable>,
       !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16, CGALayout = [[0, 1]]}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, colStride = 1, CGALayout = [[0, 1]]>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 8 : i32} {
  tt.func @tcgen5_completion_barrier_cga_layout(
      %a: !ttg.memdesc<128x16xf16, #shared, #ttg.shared_memory>,
      %b: !ttg.memdesc<16x128xf16, #shared1, #ttg.shared_memory>,
      %c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
      %accUse: i1,
      %pred: i1,
      %bar: !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory>,
      %barPred: i1) {
    // expected-error @below {{completion barrier cga_layout must be}}
    ttng.tc_gen5_mma %a, %b, %c, %accUse, %pred, %bar[%barPred] {is_async} :
       !ttg.memdesc<128x16xf16, #shared, #ttg.shared_memory>,
       !ttg.memdesc<16x128xf16, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
       !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory>
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @tcgen5_commit_completion_barrier_cga_layout(
      %bar: !ttg.memdesc<1xi64, #barrier, #smem, mutable>, %pred: i1) {
    // expected-error @below {{completion barrier cga_layout must be}}
    ttng.tc_gen5_commit %bar, %pred : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    tt.return
  }
}

// -----
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @tcgen5_commit_too_many_descs(
      %bar: !ttg.memdesc<1xi64, #barrier, #smem, mutable>,
      %desc0: !ttg.memdesc<128x128xf16, #shared, #smem>,
      %desc1: !ttg.memdesc<128x128xf16, #shared, #smem>,
      %desc2: !ttg.memdesc<128x128xf16, #shared, #smem>,
      %pred: i1) {
    // expected-error @below {{expected 0, 1, or 2 descriptors, got 3}}
    ttng.tc_gen5_commit %bar, %pred descs %desc0, %desc1, %desc2 :
      !ttg.memdesc<1xi64, #barrier, #smem, mutable>,
      !ttg.memdesc<128x128xf16, #shared, #smem>,
      !ttg.memdesc<128x128xf16, #shared, #smem>,
      !ttg.memdesc<128x128xf16, #shared, #smem>
    tt.return
  }
}

// -----

#tmem_barrier = #ttng.tensor_memory_linear<{row = [[1, 0]], col = [[0, 1]]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @tcgen5_commit_barrier_memory_space(
      %bar: !ttg.memdesc<2x2xi64, #tmem_barrier, #ttng.tensor_memory, mutable>,
      %pred: i1) {
    // expected-error @below {{barrier allocation must be a shared memory descriptor}}
    ttng.tc_gen5_commit %bar, %pred : !ttg.memdesc<2x2xi64, #tmem_barrier, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @tcgen5_commit_desc_memory_space(
      %bar: !ttg.memdesc<1xi64, #barrier, #smem, mutable>,
      %desc: !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>,
      %pred: i1) {
    // expected-error @below {{descriptor operands must be shared memory descriptors}}
    ttng.tc_gen5_commit %bar, %pred descs %desc :
      !ttg.memdesc<1xi64, #barrier, #smem, mutable>,
      !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>
    tt.return
  }
}


// -----


#shared_tmembad = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32, CGALayout = [[1, 0]]}>
#tmem_linear_bad = #ttng.tensor_memory_linear<{row = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]], col = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], block = [[128, 0]]}, twoCTAs = true>
#barrier_bad_layout = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tmem_copy_multi_cta_barrier_cga_layout(
      %src: !ttg.memdesc<256x128xf32, #shared_tmembad, #ttg.shared_memory, mutable>,
      %dst: !ttg.memdesc<256x128xf32, #tmem_linear_bad, #ttng.tensor_memory, mutable>,
      %bar: !ttg.memdesc<1xi64, #barrier_bad_layout, #ttg.shared_memory>) {
    // expected-error @below {{completion barrier cga_layout must be}}
    ttng.tmem_copy %src, %dst, %bar :
       !ttg.memdesc<256x128xf32, #shared_tmembad, #ttg.shared_memory, mutable>,
       !ttg.memdesc<256x128xf32, #tmem_linear_bad, #ttng.tensor_memory, mutable>,
       !ttg.memdesc<1xi64, #barrier_bad_layout, #ttg.shared_memory>
    tt.return
  }
}

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#sharedT = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @tcgen5_mma_scaled_sync_with_barrier(
      %a: !ttg.memdesc<128x256xi8, #shared, #ttg.shared_memory>,
      %b: !ttg.memdesc<256x64xi8, #sharedT, #ttg.shared_memory>,
      %c: !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>,
      %scale_a: !ttg.memdesc<128x8xf8E4M3FN, #shared1, #ttg.shared_memory>,
      %scale_b: !ttg.memdesc<64x8xf8E4M3FN, #shared1, #ttg.shared_memory>,
      %useAcc: i1,
      %pred: i1,
      %bar: !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>,
      %barPred: i1) {
    // expected-error @below {{The op is synchronous but a barrier is present.}}
    ttng.tc_gen5_mma_scaled %a, %b, %c, %scale_a, %scale_b, %useAcc, %pred lhs = e2m1 rhs = e2m1, %bar[%barPred] :
      !ttg.memdesc<128x256xi8, #shared, #ttg.shared_memory>,
      !ttg.memdesc<256x64xi8, #sharedT, #ttg.shared_memory>,
      !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>,
      !ttg.memdesc<128x8xf8E4M3FN, #shared1, #ttg.shared_memory>,
      !ttg.memdesc<64x8xf8E4M3FN, #shared1, #ttg.shared_memory>,
      !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

#shared_clc = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0], [0]]}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0], [0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @clc_try_cancel_completion_barrier_cga_layout(
      %result: !ttg.memdesc<2xi64, #shared_clc, #smem>,
      %mbar: !ttg.memdesc<1xi64, #barrier, #smem>) {
    // expected-error @below {{completion barrier cga_layout must be}}
    ttng.clc_try_cancel %result, %mbar {multicast = false} :
      !ttg.memdesc<2xi64, #shared_clc, #smem>, !ttg.memdesc<1xi64, #barrier, #smem>
    tt.return
  }
}

// -----

#shared_clc_bad = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @clc_try_cancel_result_cga_layout_bases_nonzero(
      %result: !ttg.memdesc<2xi64, #shared_clc_bad, #smem>,
      %mbar: !ttg.memdesc<1xi64, #barrier, #smem>) {
    // expected-error @below {{Expected CLC result buffer cga_layout bases to be all zeros. Got [[1]]}}
    ttng.clc_try_cancel %result, %mbar {multicast = false} :
      !ttg.memdesc<2xi64, #shared_clc_bad, #smem>, !ttg.memdesc<1xi64, #barrier, #smem>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
  tt.func @fence_mbarrier_init_release_cluster_invalid() {
    // expected-error @below {{requires ttg.num-ctas > 1}}
    ttng.fence_mbarrier_init_release_cluster
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
  tt.func @cluster_arrive_invalid() {
    // expected-error @below {{requires ttg.num-ctas > 1}}
    ttng.cluster_arrive
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
  tt.func @cluster_wait_invalid() {
    // expected-error @below {{requires ttg.num-ctas > 1}}
    ttng.cluster_wait
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
  tt.func @cluster_arrive_in_default_region_invalid() {
    ttg.warp_specialize()
    default {
      // expected-error @below {{cannot be used inside `ttg.warp_specialize`}}
      ttng.cluster_arrive
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
  tt.func @cluster_wait_in_partition_invalid() {
    ttg.warp_specialize()
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      // expected-error @below {{cannot be used inside `ttg.warp_specialize`}}
      ttng.cluster_wait
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
  tt.func @cluster_barrier_invalid() {
    // expected-error @below {{requires ttg.num-ctas > 1}}
    ttng.cluster_barrier
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
  tt.func @cluster_barrier_in_default_region_invalid() {
    ttg.warp_specialize()
    default {
      // expected-error @below {{cannot be used inside `ttg.warp_specialize`}}
      ttng.cluster_barrier
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
  tt.func @cluster_barrier_in_partition_invalid() {
    ttg.warp_specialize()
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      // expected-error @below {{cannot be used inside `ttg.warp_specialize`}}
      ttng.cluster_barrier
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
  tt.func @init_barrier_in_default_region_invalid() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.warp_specialize()
    default {
      // expected-error @below {{cannot be used inside `ttg.warp_specialize`}}
      ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

// expected-error @+1 {{After removing the zero bases the layout must be bijective}}
#linear = #ttg.linear<{register = [[0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1]], warp = [[16, 0], [8, 0]], block = []}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @invalid_linear_layout(%arg0: tensor<32x64xi32, #linear>) {
    tt.return
  }
}

// -----

// Test that reduction with warps split across N dimension is rejected
// 128x256 with 8 warps -> warpsPerCTA = [4, 2] (2 warps in N)
#blocked_split = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#blocked_red = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#tmem_warp_split = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:107", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_memory_ld_red_warp_split_rejected() {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked_split>
    %0 = ttng.tmem_alloc %cst_0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x256xf32, #blocked_split>) -> !ttg.memdesc<128x256xf32, #tmem_warp_split, #ttng.tensor_memory, mutable>
    // expected-error @+3 {{tmem_load reduction with N dimension sharded across threads is not supported.}}
    // expected-note @+2 {{Reduction requires all N elements to reside in the register dimension and M to be unsharded.}}
    // expected-note @+1 {{Got register layout:}}
    %result, %red = ttng.tmem_load %0 {redOp = #ttng.redOp<min>} : !ttg.memdesc<128x256xf32, #tmem_warp_split, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked_split>, tensor<128xf32, #blocked_red>
    tt.return
  }
}

// -----

// Test that reduction with N shared across threads is rejected
#blocked_split = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked_red = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#bm64_bn128 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:107", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_memory_ld_red_16x32bx2_atom_rejected() {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked_split>
    %0 = ttng.tmem_alloc %cst_0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<64x128xf32, #blocked_split>) -> !ttg.memdesc<64x128xf32, #bm64_bn128, #ttng.tensor_memory, mutable>
    // expected-error @+1 {{tmem_load reduction source layout is not directly tcgen05.ld.red-compatible; use tmem.load(...)+tt.reduce(...) explicitly for software reduction}}
    %result, %red = ttng.tmem_load %0 {redOp = #ttng.redOp<min>} : !ttg.memdesc<64x128xf32, #bm64_bn128, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #blocked_split>, tensor<64xf32, #blocked_red>
    tt.return
  }
}

// -----

// Test: abs requires redOp to be set
#blocked_abs = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem_abs = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:107", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_memory_ld_abs_requires_redop() {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked_abs>
    %0 = ttng.tmem_alloc %cst_0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked_abs>) -> !ttg.memdesc<128x128xf32, #tmem_abs, #ttng.tensor_memory, mutable>
    // expected-error @below {{'abs' requires 'redOp' to be set}}
    %result = ttng.tmem_load %0 {abs = true} : !ttg.memdesc<128x128xf32, #tmem_abs, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked_abs>
    tt.return
  }
}

// -----

// Test: NaN requires redOp to be set
#blocked_nan = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem_nan = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:107", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_memory_ld_nan_requires_redop() {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked_nan>
    %0 = ttng.tmem_alloc %cst_0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked_nan>) -> !ttg.memdesc<128x128xf32, #tmem_nan, #ttng.tensor_memory, mutable>
    // expected-error @below {{'NaN' requires 'redOp' to be set}}
    %result = ttng.tmem_load %0 {NaN = true} : !ttg.memdesc<128x128xf32, #tmem_nan, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked_nan>
    tt.return
  }
}

// -----

// Test: reduction itself currently requires f32 element type
#blocked_red_i32 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked_red_i32_out = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#tmem_red_i32 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:107", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_memory_ld_red_requires_f32() {
    %cst_0 = arith.constant dense<0> : tensor<128x128xi32, #blocked_red_i32>
    %0 = ttng.tmem_alloc %cst_0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xi32, #blocked_red_i32>) -> !ttg.memdesc<128x128xi32, #tmem_red_i32, #ttng.tensor_memory, mutable>
    // expected-error @below {{tmem_load reduction currently requires f32 element type}}
    %result, %red = ttng.tmem_load %0 {redOp = #ttng.redOp<min>} : !ttg.memdesc<128x128xi32, #tmem_red_i32, #ttng.tensor_memory, mutable> -> tensor<128x128xi32, #blocked_red_i32>, tensor<128xi32, #blocked_red_i32_out>
    tt.return
  }
}

// -----

// Test: abs requires f32 element type
#blocked_abs_i32 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked_red_abs_i32 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#tmem_abs_i32 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:107", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_memory_ld_abs_requires_f32() {
    %cst_0 = arith.constant dense<0> : tensor<128x128xi32, #blocked_abs_i32>
    %0 = ttng.tmem_alloc %cst_0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xi32, #blocked_abs_i32>) -> !ttg.memdesc<128x128xi32, #tmem_abs_i32, #ttng.tensor_memory, mutable>
    // expected-error @below {{'abs' requires floating-point element type (f32)}}
    %result, %red = ttng.tmem_load %0 {redOp = #ttng.redOp<min>, abs = true} : !ttg.memdesc<128x128xi32, #tmem_abs_i32, #ttng.tensor_memory, mutable> -> tensor<128x128xi32, #blocked_abs_i32>, tensor<128xi32, #blocked_red_abs_i32>
    tt.return
  }
}

// -----

// Test: NaN requires f32 element type
#blocked_nan_i32 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked_red_nan_i32 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#tmem_nan_i32 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:107", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensor_memory_ld_nan_requires_f32() {
    %cst_0 = arith.constant dense<0> : tensor<128x128xi32, #blocked_nan_i32>
    %0 = ttng.tmem_alloc %cst_0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xi32, #blocked_nan_i32>) -> !ttg.memdesc<128x128xi32, #tmem_nan_i32, #ttng.tensor_memory, mutable>
    // expected-error @below {{'NaN' requires floating-point element type (f32)}}
    %result, %red = ttng.tmem_load %0 {redOp = #ttng.redOp<min>, NaN = true} : !ttg.memdesc<128x128xi32, #tmem_nan_i32, #ttng.tensor_memory, mutable> -> tensor<128x128xi32, #blocked_nan_i32>, tensor<128xi32, #blocked_red_nan_i32>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#nvmma_no_broadcast = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[1, 0]]}>
#shared_bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @async_tma_copy_multicast_requires_broadcast(%arg0: !tt.tensordesc<tensor<64x128xf16, #nvmma_no_broadcast>>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmma_no_broadcast, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    // expected-error @below {{multicast requires the shared layout to broadcast across CTAs}}
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %1, %true {multicast} : !tt.tensordesc<tensor<64x128xf16, #nvmma_no_broadcast>>, !ttg.memdesc<1xi64, #shared_bar, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmma_no_broadcast, #smem, mutable>
    tt.return
  }
}

// -----

// Test invalid TensorDescIm2ColType: rank-3 blockType (must be rank-2)
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  // expected-error @below {{TensorDescIm2ColType requires rank-2 blockType, got rank 3}}
  tt.func @tensordesc_im2col_wrong_rank(%desc: !ttng.tensordesc_im2col<tensor<32x64x128xf16>>) {
    tt.return
  }
}

// -----

#shared_bad = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0], [2], [1]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 8 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @wait_barrier_invalid_cga_layout(%bar: !ttg.memdesc<4xi64, #shared_bad, #smem, mutable>, %phase: i32) {
    // expected-error @below {{broadcasted cluster barriers require bases to be the sequence}}
    ttng.wait_barrier %bar, %phase : !ttg.memdesc<4xi64, #shared_bad, #smem, mutable>
    tt.return
  }
}

// -----

#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @tmem_subslice_non_tmem_source() {
    %md = ttg.local_alloc : () -> !ttg.memdesc<128x128xi32, #shared1, #ttg.shared_memory, mutable>
    // expected-error @+1 {{The source must be a tensor memory buffer.}}
    %sub = ttng.tmem_subslice %md {N = 0 : i32} : !ttg.memdesc<128x128xi32, #shared1, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x128xi32, #shared1, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @tmem_subslice_rank_not_2() {
    %md = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // expected-error @+1 {{The result must be a 2D tensor memory buffer.}}
    %sub = ttng.tmem_subslice %md {N = 0 : i32} : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @tmem_subslice_rows_mismatch() {
    %md = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // expected-error @+1 {{The result must have the same number of rows as the source.}}
    %sub = ttng.tmem_subslice %md {N = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @tmem_subslice_element_type_mismatch() {
    %md = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable>
    // expected-error @+1 {{The source and result must have the same element type.}}
    %sub = ttng.tmem_subslice %md {N = 0 : i32} : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 128x128>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @tmem_subslice_alloc_shape_mismatch() {
    %md = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // expected-error @+1 {{The source and result must have the same alloc shape.}}
    %sub = ttng.tmem_subslice %md {N = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @tmem_subslice_offset_exceed() {
    %md = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // expected-error @+1 {{The split offset may not exceed the source shape}}
    %sub = ttng.tmem_subslice %md {N = 128 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
    tt.return
  }
}

// -----

#tmem_linear = #ttng.tensor_memory_linear<{row = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]], col = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]]}>
#tmem_linear_t = #ttng.tensor_memory_linear<{row = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], col = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func public @tmem_subslice_layout_mismatch_linear() {
    %md = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem_linear, #ttng.tensor_memory, mutable>
    // expected-error @+1 {{'ttng.tmem_subslice' op tensor memory view is not representable as a standalone TMEM linear layout}}
    %sub = ttng.tmem_subslice %md {N = 64 : i32} : !ttg.memdesc<128x128xf32, #tmem_linear, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem_linear_t, #ttng.tensor_memory, mutable, 128x128>
    tt.return
  }
}
