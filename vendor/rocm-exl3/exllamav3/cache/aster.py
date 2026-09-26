"""ASTER-packed 5-bit KV cache layer (experimental research prototype).

Layout per 32-value group (same packed shapes as the uniform 5-bit cache):
- words 0..3 hold high 4 bits (index >> 1) of values 8*w..8*w+7 as nibbles,
  nibble (j % 8) at bits (j % 8) * 4.
- word 4 holds low bit of value j at bit position j.
- one FP16 scale per group: FP32 absmax + 1e-10, rounded on store.

Encode: H32 (float32, five butterfly stages, /sqrt(32)) -> absmax (+1e-10,
unrounded for assignment) -> nearest of 32 centroids via midpoint thresholds,
ties to lower index. Decode: codebook[q] * scale -> H32 (/sqrt(32)). Attention
evaluates the default cubic grid analytically; custom grids use indexed loads.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from ..constants import PAGE_SIZE
from .aster_codebook import ASTER5_CENTROIDS, ASTER5_CUBIC
from .quant import CacheLayer_quant

_GROUP = 32
_GROUPS_PER_PROGRAM = 4
_CODEBOOK_BYTES = 128  # 32 float32 centroids
_SYMM_TOL = 1e-6


def _validate_centroids(centroids) -> tuple:
    """Validate 32 finite strictly-increasing symmetric values in [-1, 1]."""
    try:
        vals = tuple(float(c) for c in centroids)
    except TypeError:
        raise ValueError("aster5 centroids must be a sequence of 32 numbers")
    if len(vals) != _GROUP:
        raise ValueError(f"aster5 centroids must hold exactly 32 values, got {len(vals)}")
    for c in vals:
        if not math.isfinite(c):
            raise ValueError("aster5 centroids must all be finite")
        if not -1.0 <= c <= 1.0:
            raise ValueError("aster5 centroids must lie in [-1, 1]")
    for a, b in zip(vals, vals[1:]):
        if not b > a:
            raise ValueError("aster5 centroids must be strictly increasing")
    for i in range(_GROUP // 2):
        if abs(vals[i] + vals[_GROUP - 1 - i]) > _SYMM_TOL:
            raise ValueError("aster5 centroids must be symmetric about 0")
    return vals


@triton.jit
def aster_quant_paged(
    k_in, k_out, k_sc, v_in, v_out, v_sc,
    cb, seqlens, block_table,
    blocks_per_seq, groups_per_token, length, page_size, chunks_per_token,
    IN_CONTIGUOUS: tl.constexpr, GROUPS: tl.constexpr,
):
    # One program covers GROUPS groups of one token; grid dim 1 selects K/V.
    pid0 = tl.program_id(0)
    side = tl.program_id(1)
    tok_lin = pid0 // chunks_per_token
    chunk = pid0 % chunks_per_token
    b = tok_lin // length
    t = tok_lin % length
    seq = tl.load(seqlens + b)
    logic = seq + t
    page_idx = logic // page_size
    slot = logic % page_size
    phys_page = tl.load(block_table + b * blocks_per_seq + page_idx)
    phys_tok = phys_page * page_size + slot
    if IN_CONTIGUOUS:
        in_tok = b * length + t
    else:
        in_tok = phys_tok
    tok_vals = groups_per_token * 32
    in_ptr = tl.where(side == 0, k_in, v_in)
    out_ptr = tl.where(side == 0, k_out, v_out)
    sc_ptr = tl.where(side == 0, k_sc, v_sc)
    # Vectorized [GROUPS, 32] tile.
    g1 = tl.arange(0, GROUPS)[:, None]
    d1 = tl.arange(0, 32)[None, :]
    gid1 = chunk * GROUPS + g1
    gid_full = gid1 + d1 * 0
    gact_full = gid_full < groups_per_token
    cols_full = d1 + g1 * 0
    base_in = in_tok * tok_vals + gid1 * 32
    v = tl.load(in_ptr + base_in + d1, mask=gact_full, other=0.0).to(tl.float32)
    # H32 butterfly on axis 1 (same Sylvester matrix as inherited kernels).
    for bit in tl.static_range(0, 5):
        peer_idx = cols_full ^ (1 << bit)
        peer = tl.gather(v, peer_idx, axis=1)
        cond = (cols_full & (1 << bit)) == 0
        v = tl.where(cond, v + peer, peer - v)
    v = v * 0.17677669529663688110
    s = tl.max(tl.abs(v), axis=1) + 1e-10
    xn = v / s[:, None]
    # Five-step binary search over 31 midpoints; ties fall lower via strict >.
    # mid never reaches 31 within 5 steps from [0, 31], so mid+1 stays in bounds.
    low = tl.zeros((GROUPS, 32), tl.int32)
    high = tl.full((GROUPS, 32), 31, tl.int32)
    for _ in tl.static_range(0, 5):
        mid = (low + high) // 2
        thr = (tl.load(cb + mid) + tl.load(cb + mid + 1)) * 0.5
        gt = xn > thr
        low = tl.where(gt, mid + 1, low)
        high = tl.where(gt, high, mid)
    idx = low
    hi = idx >> 1
    lo = idx & 1
    hi_r = tl.reshape(hi, (GROUPS, 4, 8))
    sh = tl.arange(0, 8)[None, None, :] * 4
    w_high = tl.sum(hi_r << sh, axis=2)
    w4 = tl.sum(lo << cols_full, axis=1)
    base_out = phys_tok * groups_per_token * 5 + gid1 * 5
    base_sc = phys_tok * groups_per_token + gid1
    w1 = tl.arange(0, 4)[None, :]
    mask4 = (gid1 + w1 * 0) < groups_per_token
    mask1 = gid1 < groups_per_token
    tl.store(out_ptr + base_out + w1, w_high, mask=mask4)
    tl.store(out_ptr + base_out + 4, w4[:, None], mask=mask1)
    tl.store(sc_ptr + base_sc, s[:, None].to(tl.float16), mask=mask1)


@triton.jit
def aster_dequant_paged(
    k_in, k_sc, k_out, v_in, v_sc, v_out,
    cb, seqlens, block_table,
    blocks_per_seq, groups_per_token, toks_per_seq, page_size, chunks_per_token,
    window,
    GROUPS: tl.constexpr,
):
    # Full-physical FP16 fallback; invalid rows stay uninitialized.
    pid0 = tl.program_id(0)
    side = tl.program_id(1)
    tok_lin = pid0 // chunks_per_token
    chunk = pid0 % chunks_per_token
    b = tok_lin // toks_per_seq
    t = tok_lin % toks_per_seq
    seq = tl.load(seqlens + b)
    valid = t < seq
    valid = valid & ((window <= 0) | (t >= seq - window))
    page_idx = t // page_size
    slot = t % page_size
    phys_page = tl.load(block_table + b * blocks_per_seq + page_idx, mask=valid, other=0)
    phys_tok = phys_page * page_size + slot
    q_ptr = tl.where(side == 0, k_in, v_in)
    sc_ptr = tl.where(side == 0, k_sc, v_sc)
    out_ptr = tl.where(side == 0, k_out, v_out)
    g1 = tl.arange(0, GROUPS)[:, None]
    d1 = tl.arange(0, 32)[None, :]
    gid1 = chunk * GROUPS + g1
    gact1 = (gid1 < groups_per_token) & valid
    gact_full = ((gid1 + d1 * 0) < groups_per_token) & valid
    cols_full = d1 + g1 * 0
    base_q = phys_tok * groups_per_token * 5 + gid1 * 5
    base_sc = phys_tok * groups_per_token + gid1
    # Load each lane's word directly. A 4-to-32 gather triggers an unsupported
    # cross-warp layout in the AMD Triton compiler; repeated addresses coalesce.
    wq = tl.load(q_ptr + base_q + d1 // 8, mask=gact_full, other=0)
    w4 = tl.load(q_ptr + base_q + 4, mask=gact1, other=0)
    sc = tl.load(sc_ptr + base_sc, mask=gact1, other=0.0).to(tl.float32)
    sh_full = ((d1 % 8) * 4) + g1 * 0
    hi = (wq >> sh_full) & 15
    w4_full = w4 + d1 * 0
    lo = (w4_full >> cols_full) & 1
    q = (hi << 1) | lo
    vv = tl.load(cb + q)
    vv = vv * sc
    vv = vv * 0.17677669529663688110
    for bit in tl.static_range(0, 5):
        peer_idx = cols_full ^ (1 << bit)
        peer = tl.gather(vv, peer_idx, axis=1)
        cond = (cols_full & (1 << bit)) == 0
        vv = tl.where(cond, vv + peer, peer - vv)
    out = vv.to(tl.float16)
    base_out = phys_tok * groups_per_token * 32 + gid1 * 32
    tl.store(out_ptr + base_out + d1, out, mask=gact_full)


class CacheLayer_aster(CacheLayer_quant):
    """Experimental five-bit cache: H32 groups and a shared 32-entry layer grid.

    Supports only k_bits == v_bits == 5, compand_a == 0, head_dim a multiple
    of 32, and PAGE_SIZE 256. Packed shapes match the parent; only the
    5-bit payload semantics differ (codebook indices instead of uniform bins).
    Requires Triton and CUDA/ROCm; CPU reference lives in
    kernels/exl3/asterkv_reference.py.
    """

    cache_format = "aster5"

    def __init__(
        self,
        config,
        attention,
        cache_id: int,
        max_num_tokens: int,
        k_bits: int = 5,
        v_bits: int = 5,
        compand_a: float = 0.0,
        centroids=None,
    ):
        if k_bits != 5 or v_bits != 5:
            raise ValueError(f"CacheLayer_aster supports only k_bits == v_bits == 5, got {k_bits}/{v_bits}")
        if float(compand_a) != 0.0:
            raise ValueError(f"CacheLayer_aster supports only compand_a == 0, got {compand_a}")
        if PAGE_SIZE != 256:
            raise ValueError(f"CacheLayer_aster requires PAGE_SIZE 256, got {PAGE_SIZE}")
        if attention is not None and attention.head_dim % _GROUP != 0:
            raise ValueError(f"CacheLayer_aster requires head_dim a multiple of 32, got {attention.head_dim}")
        self.polynomial = ASTER5_CUBIC if centroids is None else None
        if centroids is None:
            centroids = ASTER5_CENTROIDS
        self.centroids = _validate_centroids(centroids)
        super().__init__(config, attention, cache_id, max_num_tokens, 5, 5, 0.0)
        self.codebook_tensor = None


    def alloc(self, device: torch.device):
        super().alloc(device)
        self.device = self.qk.device if self.qk is not None else torch.device(device)
        self.codebook_tensor = torch.tensor(self.centroids, dtype=torch.float32, device=self.device)


    def free(self):
        super().free()
        self.codebook_tensor = None


    def get_qkv(self):
        self._require_alloc()
        return (*super().get_qkv(), self.codebook_tensor, self.polynomial)


    def _require_alloc(self):
        if (
            self.qk is None or self.qv is None or self.sk is None
            or self.sv is None or self.codebook_tensor is None
        ):
            raise RuntimeError("CacheLayer_aster.alloc(device) must run before cache use")


    def _require_gpu(self):
        if self.device is None or getattr(self.device, "type", None) != "cuda":
            raise RuntimeError(
                "CacheLayer_aster requires Triton and CUDA/ROCm; "
                "CPU reference is kernels/exl3/asterkv_reference.py"
            )


    def _check_paging(self, cache_seqlens: torch.Tensor, block_table: torch.Tensor) -> tuple:
        if not isinstance(cache_seqlens, torch.Tensor) or cache_seqlens.dtype != torch.int32:
            raise ValueError("aster5 cache_seqlens must be an int32 tensor")
        if not isinstance(block_table, torch.Tensor) or block_table.dtype != torch.int32:
            raise ValueError("aster5 block_table must be an int32 tensor")
        if not cache_seqlens.is_contiguous() or not block_table.is_contiguous():
            raise ValueError("aster5 cache_seqlens/block_table must be contiguous")
        if cache_seqlens.device != self.device or block_table.device != self.device:
            raise ValueError("aster5 cache_seqlens/block_table must live on the cache device")
        if cache_seqlens.dim() != 1 or block_table.dim() != 2:
            raise ValueError("aster5 cache_seqlens must be 1D and block_table 2D")
        if block_table.shape[0] != cache_seqlens.shape[0]:
            raise ValueError("aster5 block_table batch must match cache_seqlens batch")
        if cache_seqlens.shape[0] == 0 or block_table.shape[1] == 0:
            raise ValueError("aster5 cache_seqlens/block_table must be non-empty")
        return cache_seqlens.shape[0], block_table.shape[1]


    def _check_new_rows(self, x: torch.Tensor, name: str, bsz: int, length: int):
        if not isinstance(x, torch.Tensor) or x.dtype != torch.half:
            raise ValueError(f"aster5 {name} must be an fp16 tensor")
        if x.device != self.device:
            raise ValueError(f"aster5 {name} must live on the cache device")
        if x.dim() != 4:
            raise ValueError(f"aster5 {name} must be [batch, length, kv_heads, head_dim], got {x.dim()}D")
        if x.shape[0] != bsz or x.shape[1] < length:
            raise ValueError(f"aster5 {name} shape {tuple(x.shape)} cannot supply batch {bsz} length {length}")
        if x.shape[2] != self.attention.num_kv_heads or x.shape[3] != self.attention.head_dim:
            raise ValueError(f"aster5 {name} trailing dims {tuple(x.shape[2:])} mismatch attention geometry")


    def _check_paged_input(self, x: torch.Tensor, name: str):
        if not isinstance(x, torch.Tensor) or x.dtype != torch.half:
            raise ValueError(f"aster5 {name} must be an fp16 tensor")
        if x.device != self.device:
            raise ValueError(f"aster5 {name} must live on the cache device")
        num_pages = self.max_num_tokens // PAGE_SIZE
        ok_4d = x.dim() == 4 and tuple(x.shape) == (num_pages, PAGE_SIZE, self.attention.num_kv_heads, self.attention.head_dim)
        ok_3d = x.dim() == 3 and tuple(x.shape) == (num_pages, PAGE_SIZE, self.token_dim)
        if not (ok_4d or ok_3d):
            raise ValueError(
                f"aster5 {name} needs full physical paged fp16 geometry "
                f"({num_pages}, 256, heads, dim) or ({num_pages}, 256, {self.token_dim}), "
                f"got {tuple(x.shape)}"
            )


    def _chunks_per_token(self) -> int:
        return (self.token_dim // _GROUP + _GROUPS_PER_PROGRAM - 1) // _GROUPS_PER_PROGRAM


    def update_kv_direct(
        self,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        length: int
    ):
        self._require_alloc()
        bsz, blocks_per_seq = self._check_paging(cache_seqlens, block_table)
        self._check_new_rows(k, "k", bsz, length)
        self._check_new_rows(v, "v", bsz, length)
        if tuple(k.shape) != tuple(v.shape):
            raise ValueError(f"aster5 k/v shapes differ: {tuple(k.shape)} vs {tuple(v.shape)}")
        if length < 0:
            raise ValueError(f"aster5 length must be >= 0, got {length}")
        if length == 0:
            return
        self._require_gpu()
        # Slice to length so batch stride matches the kernel's length indexing.
        k = k[:, :length].contiguous()
        v = v[:, :length].contiguous()
        chunks = self._chunks_per_token()
        grid = (bsz * length * chunks, 2)
        aster_quant_paged[grid](
            k, self.qk, self.sk, v, self.qv, self.sv,
            self.codebook_tensor, cache_seqlens, block_table,
            blocks_per_seq, self.token_dim // _GROUP, length, PAGE_SIZE, chunks,
            1, _GROUPS_PER_PROGRAM,
            num_warps=4,
        )


    def update_kv(
        self,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        length: int
    ):
        self._require_alloc()
        bsz, blocks_per_seq = self._check_paging(cache_seqlens, block_table)
        self._check_paged_input(k, "k")
        self._check_paged_input(v, "v")
        if tuple(k.shape) != tuple(v.shape):
            raise ValueError(f"aster5 k/v shapes differ: {tuple(k.shape)} vs {tuple(v.shape)}")
        if length < 0:
            raise ValueError(f"aster5 length must be >= 0, got {length}")
        if length == 0:
            return
        self._require_gpu()
        k = k.contiguous()
        v = v.contiguous()
        chunks = self._chunks_per_token()
        grid = (bsz * length * chunks, 2)
        aster_quant_paged[grid](
            k, self.qk, self.sk, v, self.qv, self.sv,
            self.codebook_tensor, cache_seqlens, block_table,
            blocks_per_seq, self.token_dim // _GROUP, length, PAGE_SIZE, chunks,
            0, _GROUPS_PER_PROGRAM,
            num_warps=4,
        )


    def get_kv(self, cache_seqlens: torch.Tensor, block_table: torch.Tensor, sliding_window: int = -1):
        self._require_alloc()
        bsz, pages_per_seq = self._check_paging(cache_seqlens, block_table)
        self._require_gpu()
        k = torch.empty(self.shape, dtype=torch.half, device=self.device)
        v = torch.empty(self.shape, dtype=torch.half, device=self.device)
        chunks = self._chunks_per_token()
        toks_per_seq = pages_per_seq * PAGE_SIZE
        grid = (bsz * toks_per_seq * chunks, 2)
        aster_dequant_paged[grid](
            self.qk, self.sk, k, self.qv, self.sv, v,
            self.codebook_tensor, cache_seqlens, block_table,
            pages_per_seq, self.token_dim // _GROUP, toks_per_seq, PAGE_SIZE, chunks,
            sliding_window,
            _GROUPS_PER_PROGRAM,
            num_warps=4,
        )
        return k, v


    def copy_page(self, source, from_page: int, to_page: int, num_tokens: int):
        if getattr(source, "cache_format", None) != "aster5" or not isinstance(source, CacheLayer_aster):
            raise ValueError("aster5 copy_page requires an aster5 source (no uniform q5 <-> aster5 reinterpretation)")
        if self.qshape_k != source.qshape_k or self.qshape_v != source.qshape_v:
            raise ValueError("aster5 copy_page requires identical packed bit geometry")
        if self.centroids != getattr(source, "centroids", None):
            raise ValueError("aster5 copy_page requires the same codebook")
        if not 0 < num_tokens <= PAGE_SIZE:
            raise ValueError(f"aster5 copy_page needs 1..{PAGE_SIZE} tokens, got {num_tokens}")
        num_pages = self.max_num_tokens // PAGE_SIZE
        if not (0 <= from_page < num_pages and 0 <= to_page < num_pages):
            raise ValueError("aster5 copy_page page index out of range")
        if self.qk is None or source.qk is None:
            raise RuntimeError("aster5 copy_page requires allocated source and destination caches")
        super().copy_page(source, from_page, to_page, num_tokens)


    def get_tensors(self):
        # The generator moves/defragments these tensors by page index.
        return super().get_tensors()


    def get_metadata_tensors(self):
        # Immutable layer-wide state must never be moved as a cache page.
        return [self.codebook_tensor]


    def storage_size(self):
        return super().storage_size() + _CODEBOOK_BYTES


    def tp_export(self, plan):
        raise RuntimeError("CacheLayer_aster tp_export is explicitly unsupported until validated")


__all__ = ["CacheLayer_aster"]
