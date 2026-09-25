"""Explicit experimental fallback for the pinned noncooperative WSL runtime.

Set EXL3_BC_ATTN=0 before importing ExLlamaV3. Call install for each config,
then prepare_loaded_module after every load/reload and before its forward pass.
This changes process-local dispatch only; it does not establish model support.
"""

import os
import sys
from functools import wraps


def smallm_supported(layer, rows=None):
    """Match the verified binary's codebook, bit width and row envelope."""
    mul1 = getattr(layer, "mul1", None)
    codebook = 2 if mul1 is True else 0
    bits = getattr(layer, "K", None)
    bit_support = bits in (2, 3, 4) or (
        bits in (5, 6) and mul1 is True and rows in (2, 3, 5)
        and getattr(layer, '_quantlab_smallm_highbit', False)
        and os.environ.get('EXL3_SMALLM_WMMA', '0') == '0')
    return (bit_support
            and getattr(layer, "mcg", True) is False
            and (mul1 is True or mul1 is False)
            and codebook in getattr(layer, "_quantlab_smallm_codebooks", (0,))
            and getattr(layer, "in_features", 0) > 0
            and getattr(layer, "out_features", 0) > 0
            and layer.in_features % 128 == 0 and layer.out_features % 128 == 0)


def install(config, *, native_smallm=False, native_smallm_max_rows=3, native_attention=False,
            native_smallm_codebooks=(0,), native_smallm_highbit=False):
    """Use safe fallback, optionally with the separately built small-M binary.

    The caller must verify the experimental binary hash before enabling this.
    Unsupported quantizers/shapes retain the rowwise fallback.
    """
    if native_smallm_max_rows not in (3, 5, 9):
        raise ValueError('Native small-M maximum must be 3, 5 or 9, matching the verified binary')
    codebooks = tuple(native_smallm_codebooks)
    if codebooks not in ((0,), (0, 2)) or any(type(cb) is not int for cb in codebooks):
        raise ValueError('Native small-M codebooks must be (0,) or (0, 2), matching the verified binary')
    if native_attention and (not native_smallm or native_smallm_max_rows < 5):
        raise ValueError('Native attention experiment requires the verified small-M graph build')
    if native_attention and os.environ.get("EXL3_BC_ATTN") != "1":
        raise RuntimeError('Set EXL3_BC_ATTN=1 before importing for the native attention experiment')
    if not native_attention and os.environ.get("EXL3_BC_ATTN") != "0":
        raise RuntimeError("Set EXL3_BC_ATTN=0 before importing ExLlamaV3")
    attn = sys.modules.get("exllamav3.modules.attn")
    if attn is not None and getattr(attn, "_bc_attn_enable", False) != native_attention:
        raise RuntimeError("ExLlamaV3 attention was imported before EXL3_BC_ATTN=0")

    from exllamav3.modules.quant.exl3 import LinearEXL3
    # Like the existing dispatch switches, capabilities are process-local and
    # refreshed on every install, including after the wrapper already exists.
    LinearEXL3._quantlab_smallm_codebooks = codebooks
    LinearEXL3._quantlab_smallm_highbit = native_smallm_highbit

    # Mode 2 accepts every eligible GEMV shape instead of declining to GEMM
    # based on profitability. This is re-read by the pinned native dispatcher.
    os.environ["EXL3_GEMV"] = "2"
    os.environ["EXL3_SMALLM"] = "1" if native_smallm else "0"
    os.environ["EXL3_SMALLM_MAX_ROWS"] = str(native_smallm_max_rows)
    config.infer_params.no_reconstruct = False
    config.infer_params.use_mgemm = lambda *args, **kwargs: False

    original = LinearEXL3.forward
    if getattr(original, "_quantlab_noncooperative", False):
        return

    @wraps(original)
    def forward(self, x, params, out_dtype=None):
        # Delegate overrides as one complete call; splitting them could change
        # a replacement module's semantics. The replacement EXL3 instance, if
        # any, goes through this wrapper normally.
        if "ovr" in params:
            overrides = params["ovr"]
            if self.key in overrides and overrides[self.key].inner is not self:
                return original(self, x, params, out_dtype)
        if params.get("reconstruct"):
            return original(self, x, params, out_dtype)

        assert x.is_contiguous(), f"LinearEXL3 {self.key}: non-contiguous input {tuple(x.shape)}"
        rows = x.numel() // x.shape[-1]
        if rows <= 1:
            return original(self, x, params, out_dtype)
        max_rows = int(os.environ.get('EXL3_SMALLM_MAX_ROWS', '3')) if os.environ.get('EXL3_SMALLM') == '1' else 3
        if rows > max_rows:
            return original(self, x, {**params, "reconstruct": True}, out_dtype)
        if os.environ.get("EXL3_SMALLM") == "1" and smallm_supported(self, rows):
            self._quantlab_smallm_calls = getattr(self, "_quantlab_smallm_calls", 0) + 1
            return original(self, x, params, out_dtype)

        flat = x.view(rows, x.shape[-1])
        first = original(self, flat[:1], params, out_dtype)
        output = first.new_empty((rows, self.out_features))
        output[:1] = first
        for row in range(1, rows):
            output[row:row + 1] = original(self, flat[row:row + 1], params, out_dtype)
        return output.view(*x.shape[:-1], self.out_features)

    forward._quantlab_noncooperative = True
    LinearEXL3.forward = forward


def prepare_loaded_module(module):
    """Disable native projection bypasses on this loaded module and children.

The pinned Module.__iter__ already traverses descendants. Do not remove
LinearEXL3.bc: single-row calls still use its ordinary GEMV dispatch.
"""
    from exllamav3.modules.attn import Attention
    from exllamav3.modules.gated_delta_net import GatedDeltaNet
    from exllamav3.modules.mlp import GatedMLP, MLP

    for child in module:
        if isinstance(child, (MLP, GatedMLP)):
            child.bc = None
        if isinstance(child, GatedMLP):
            child.multi_gu = [None for _ in child.multi_gu]
        if isinstance(child, GatedDeltaNet):
            child.bc = None
            child.bc_split = False
        if isinstance(child, Attention):
            child.multi_kv = None
            child.multi_qg = None
