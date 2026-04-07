import torch

from triton_kernels.tensor import RaggedTensorMetadata, Tensor
from triton_kernels.matmul import PrecisionConfig, FusedComm, FusedActivation, Epilogue


def matmul(
    a: torch.Tensor | Tensor,
    b: torch.Tensor | Tensor,
    bias: torch.Tensor | None,
    a_ragged_metadata: RaggedTensorMetadata | None = None,
    b_ragged_metadata: RaggedTensorMetadata | None = None,
    gather_indx: torch.Tensor | None = None,
    scatter_indx: torch.Tensor | None = None,
    precision_config: PrecisionConfig | None = None,
    betas: torch.Tensor | None = None,
    gammas: torch.Tensor | None = None,
    out_alpha: float | None = None,
    c: torch.Tensor | None = None,
    fused_comm: FusedComm | None = None,
    fused_activation: FusedActivation | None = None,
    epilogue: Epilogue | None = None,
    c_acc_in: torch.Tensor | None = None,
):
    assert a_ragged_metadata is not None
    assert b_ragged_metadata is None
    assert bias is not None
    assert gather_indx is not None
    assert scatter_indx is None
    assert precision_config is not None
    assert betas is None
    assert gammas is None
    assert out_alpha is None
    assert c is not None
    assert fused_comm is None
    assert fused_activation is not None
    assert epilogue is None
    assert c_acc_in is None

    specs = fused_activation.specs
    assert specs.name == "swiglu"
    reduction_n = specs.reduction_n
    swiglu_alpha, swiglu_limit = fused_activation.fn_args

    b_mx_scales = precision_config.b_mx_scale

    assert a.ndim == 2
    assert b.ndim == 2
    print(f"{a.shape=}")
    print(f"{b.shape=}")
    K = a.shape[-1]
    _, N = b.shape[-2:]
