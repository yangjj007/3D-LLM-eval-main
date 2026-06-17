from typing import *
import torch
from .. import SparseTensor
from .. import DEBUG, ATTN

if ATTN == 'xformers':
    import xformers.ops as xops
elif ATTN == 'flash_attn':
    import flash_attn
else:
    raise ValueError(f"Unknown attention module: {ATTN}")


__all__ = [
    'sparse_scaled_dot_product_attention',
]


def _is_sparse_tensor(x: object) -> bool:
    """Accept SparseTensor instances created before/after sparse backend reloads."""
    return isinstance(x, SparseTensor) or (
        not isinstance(x, torch.Tensor)
        and hasattr(x, "feats")
        and hasattr(x, "coords")
        and hasattr(x, "layout")
        and hasattr(x, "replace")
    )


@overload
def sparse_scaled_dot_product_attention(qkv: SparseTensor) -> SparseTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        qkv (SparseTensor): A [N, *, 3, H, C] sparse tensor containing Qs, Ks, and Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: SparseTensor, kv: Union[SparseTensor, torch.Tensor]) -> SparseTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (SparseTensor): A [N, *, H, C] sparse tensor containing Qs.
        kv (SparseTensor or torch.Tensor): A [N, *, 2, H, C] sparse tensor or a [N, L, 2, H, C] dense tensor containing Ks and Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: torch.Tensor, kv: SparseTensor) -> torch.Tensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (SparseTensor): A [N, L, H, C] dense tensor containing Qs.
        kv (SparseTensor or torch.Tensor): A [N, *, 2, H, C] sparse tensor containing Ks and Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: SparseTensor, k: SparseTensor, v: SparseTensor) -> SparseTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (SparseTensor): A [N, *, H, Ci] sparse tensor containing Qs.
        k (SparseTensor): A [N, *, H, Ci] sparse tensor containing Ks.
        v (SparseTensor): A [N, *, H, Co] sparse tensor containing Vs.

    Note:
        k and v are assumed to have the same coordinate map.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: SparseTensor, k: torch.Tensor, v: torch.Tensor) -> SparseTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (SparseTensor): A [N, *, H, Ci] sparse tensor containing Qs.
        k (torch.Tensor): A [N, L, H, Ci] dense tensor containing Ks.
        v (torch.Tensor): A [N, L, H, Co] dense tensor containing Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: torch.Tensor, k: SparseTensor, v: SparseTensor) -> torch.Tensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (torch.Tensor): A [N, L, H, Ci] dense tensor containing Qs.
        k (SparseTensor): A [N, *, H, Ci] sparse tensor containing Ks.
        v (SparseTensor): A [N, *, H, Co] sparse tensor containing Vs.
    """
    ...

def sparse_scaled_dot_product_attention(*args, **kwargs):
    arg_names_dict = {
        1: ['qkv'],
        2: ['q', 'kv'],
        3: ['q', 'k', 'v']
    }
    num_all_args = len(args) + len(kwargs)
    assert num_all_args in arg_names_dict, f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    for key in arg_names_dict[num_all_args][len(args):]:
        assert key in kwargs, f"Missing argument {key}"

    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs['qkv']
        assert _is_sparse_tensor(qkv), f"qkv must be a SparseTensor, got {type(qkv)}"
        assert len(qkv.shape) == 4 and qkv.shape[1] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"
        device = qkv.device

        s = qkv
        q_seqlen = [qkv.layout[i].stop - qkv.layout[i].start for i in range(qkv.shape[0])]
        kv_seqlen = q_seqlen
        qkv = qkv.feats     # [T, 3, H, C]

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs['q']
        kv = args[1] if len(args) > 1 else kwargs['kv']
        assert _is_sparse_tensor(q) and (_is_sparse_tensor(kv) or isinstance(kv, torch.Tensor)) or \
               isinstance(q, torch.Tensor) and _is_sparse_tensor(kv), \
               f"Invalid types, got {type(q)} and {type(kv)}"
        assert q.shape[0] == kv.shape[0], f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"
        device = q.device

        if _is_sparse_tensor(q):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, C]"
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats     # [T_Q, H, C]
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, C]"
            s = None
            N, L, H, C = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, C)   # [T_Q, H, C]

        if _is_sparse_tensor(kv):
            assert len(kv.shape) == 4 and kv.shape[1] == 2, f"Invalid shape for kv, got {kv.shape}, expected [N, *, 2, H, C]"
            kv_seqlen = [kv.layout[i].stop - kv.layout[i].start for i in range(kv.shape[0])]
            kv = kv.feats     # [T_KV, 2, H, C]
        else:
            assert len(kv.shape) == 5, f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
            N, L, _, H, C = kv.shape
            kv_seqlen = [L] * N
            kv = kv.reshape(N * L, 2, H, C)   # [T_KV, 2, H, C]

    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs['q']
        k = args[1] if len(args) > 1 else kwargs['k']
        v = args[2] if len(args) > 2 else kwargs['v']
        assert _is_sparse_tensor(q) and ((_is_sparse_tensor(k) and _is_sparse_tensor(v)) or (isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor))) or \
               isinstance(q, torch.Tensor) and _is_sparse_tensor(k) and _is_sparse_tensor(v), \
               f"Invalid types, got {type(q)}, {type(k)}, and {type(v)}"
        assert q.shape[0] == k.shape[0] == v.shape[0], f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
        device = q.device

        if _is_sparse_tensor(q):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, Ci]"
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats     # [T_Q, H, Ci]
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
            s = None
            N, L, H, CI = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, CI)  # [T_Q, H, Ci]

        if _is_sparse_tensor(k):
            assert len(k.shape) == 3, f"Invalid shape for k, got {k.shape}, expected [N, *, H, Ci]"
            assert len(v.shape) == 3, f"Invalid shape for v, got {v.shape}, expected [N, *, H, Co]"
            kv_seqlen = [k.layout[i].stop - k.layout[i].start for i in range(k.shape[0])]
            k = k.feats     # [T_KV, H, Ci]
            v = v.feats     # [T_KV, H, Co]
        else:
            assert len(k.shape) == 4, f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
            assert len(v.shape) == 4, f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"
            N, L, H, CI, CO = *k.shape, v.shape[-1]
            kv_seqlen = [L] * N
            k = k.reshape(N * L, H, CI)     # [T_KV, H, Ci]
            v = v.reshape(N * L, H, CO)     # [T_KV, H, Co]

    if DEBUG:
        if s is not None:
            for i in range(s.shape[0]):
                coord_slice = s.coords[s.layout[i]]
                batch_col = coord_slice[:, 0] if coord_slice.numel() else coord_slice.new_empty((0,))
                if not (batch_col == i).all():
                    unique_batches, counts = torch.unique(batch_col, return_counts=True)
                    print(
                        "[SparseScaledDotProductSelfAttention][debug] batch index mismatch\n"
                        f"  i={i}\n"
                        f"  shape={s.shape}\n"
                        f"  layout={s.layout}\n"
                        f"  slice={s.layout[i]}\n"
                        f"  coord_slice_shape={tuple(coord_slice.shape)}\n"
                        f"  batch_unique={unique_batches.detach().cpu().tolist()}\n"
                        f"  batch_counts={counts.detach().cpu().tolist()}\n"
                        f"  coords_head={coord_slice[:8].detach().cpu().tolist()}",
                        flush=True,
                    )
                    raise AssertionError("SparseScaledDotProductSelfAttention: batch index mismatch")
        if num_all_args in [2, 3]:
            q_expected = int(sum(q_seqlen))
            if int(q.shape[0]) != q_expected:
                print(
                    "[SparseScaledDotProductSelfAttention][debug] q shape mismatch\n"
                    f"  num_all_args={num_all_args}\n"
                    f"  q_shape={tuple(q.shape)}\n"
                    f"  q_seqlen={q_seqlen}\n"
                    f"  q_expected_tokens={q_expected}",
                    flush=True,
                )
                raise AssertionError("SparseScaledDotProductSelfAttention: q shape mismatch")
        if num_all_args == 2:
            kv_expected = int(sum(kv_seqlen))
            if int(kv.shape[0]) != kv_expected:
                print(
                    "[SparseScaledDotProductSelfAttention][debug] kv shape mismatch\n"
                    f"  kv_shape={tuple(kv.shape)}\n"
                    f"  kv_seqlen={kv_seqlen}\n"
                    f"  kv_expected_tokens={kv_expected}",
                    flush=True,
                )
                raise AssertionError("SparseScaledDotProductSelfAttention: kv shape mismatch")
        if num_all_args == 3:
            kv_expected = int(sum(kv_seqlen))
            if int(k.shape[0]) != kv_expected or int(v.shape[0]) != kv_expected:
                print(
                    "[SparseScaledDotProductSelfAttention][debug] k/v shape mismatch\n"
                    f"  k_shape={tuple(k.shape)}\n"
                    f"  v_shape={tuple(v.shape)}\n"
                    f"  kv_seqlen={kv_seqlen}\n"
                    f"  kv_expected_tokens={kv_expected}",
                    flush=True,
                )
                raise AssertionError("SparseScaledDotProductSelfAttention: k/v shape mismatch")

    if ATTN == 'xformers':
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
        out = xops.memory_efficient_attention(q, k, v, mask)[0]
    elif ATTN == 'flash_attn':
        cu_seqlens_q = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(q_seqlen), dim=0)]).int().to(device)
        if num_all_args in [2, 3]:
            cu_seqlens_kv = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(kv_seqlen), dim=0)]).int().to(device)
        if num_all_args == 1:
            out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens_q, max(q_seqlen))
        elif num_all_args == 2:
            out = flash_attn.flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen))
        elif num_all_args == 3:
            out = flash_attn.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen))
    else:
        raise ValueError(f"Unknown attention module: {ATTN}")
    
    if s is not None:
        return s.replace(out)
    else:
        return out.reshape(N, L, H, -1)
