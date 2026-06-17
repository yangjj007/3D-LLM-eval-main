import torch
import torch.nn as nn
from . import SparseTensor
import sys

__all__ = [
    'SparseLinear'
]

_DTYPE_DEBUG_SEEN = set()


class SparseLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__(in_features, out_features, bias)

    def forward(self, input: SparseTensor) -> SparseTensor:
        feats = input.feats
        weight_dtype = self.weight.dtype
        if feats.dtype != weight_dtype:
            key = (id(self), feats.dtype, weight_dtype)
            if key not in _DTYPE_DEBUG_SEEN:
                _DTYPE_DEBUG_SEEN.add(key)
                bias_dtype = self.bias.dtype if self.bias is not None else None
                print(
                    "[SparseLinear][debug] casting sparse feats to layer weight dtype\n"
                    f"  module={self.__class__.__name__}\n"
                    f"  in_features={self.in_features} out_features={self.out_features}\n"
                    f"  feats_dtype={feats.dtype} weight_dtype={weight_dtype} bias_dtype={bias_dtype}\n"
                    f"  feats_shape={tuple(feats.shape)}",
                    file=sys.stderr,
                    flush=True,
                )
            feats = feats.to(dtype=weight_dtype)
        return input.replace(super().forward(feats))
