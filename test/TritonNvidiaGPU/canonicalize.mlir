// RUN: triton-opt %s -canonicalize | FileCheck %s

#tmem_scales = #ttng.tensor_memory_scales_encoding<>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.num-ctas" = 1 : i32, "ttg.target" = "cuda:80"} {

// CHECK-LABEL: @test_dce_tmem_alloc
tt.func @test_dce_tmem_alloc() {
  // CHECK-NOT: ttng.tmem_alloc
  %a = ttng.tmem_alloc : () -> !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>
  // CHECK-NEXT: tt.return
  tt.return
}

// CHECK-LABEL: @reinterpret_fold
tt.func @reinterpret_fold(%arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> {
  %0 = ttg.memdesc_reinterpret %arg0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
  // CHECK-NEXT: return %arg0
  tt.return %0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
}

}  // end module
