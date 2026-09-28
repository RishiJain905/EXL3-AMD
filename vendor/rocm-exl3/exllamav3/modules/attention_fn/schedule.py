"""Measured gfx1101 attention schedules, selected without device readbacks."""


def decode_options(*, arch, batch, query_rows, query_heads, kv_heads, head_dim,
                   occupied_bound, k_bits, v_bits):
    """Choose launch geometry, never cache precision or attention semantics.

    The bound is the job page-table width capped by physical cache pages.
    Device lengths remain authoritative for masking, including graph replay.
    Unmeasured GPUs, shapes and short contexts retain the inherited schedule.
    """
    if (arch != 'gfx1101' or batch != 1 or not 1 <= query_rows <= 8
            or not 16384 <= occupied_bound <= 131072):
        return {}
    geometry = (query_heads, kv_heads, head_dim)
    uniform = k_bits == v_bits
    group = query_heads // kv_heads if kv_heads else 0
    options = dict(block_n=16, num_splits=64, parallel_combine=True)
    if geometry in ((16, 4, 256), (24, 4, 256)):
        if uniform and k_bits in (0, 4, 5, 6, 8):
            if query_rows > 4 and k_bits == 4:
                return {}
            if k_bits == 0:
                options['num_splits'] = 64 if group == 6 and query_rows >= 3 else 32
            elif query_rows >= 5:
                options['head_block'] = 4 if group == 4 or k_bits in (5, 6) else 8
            elif k_bits in (5, 6) and group == 6 and query_rows >= 3:
                options['head_block'] = 8
            elif k_bits in (4, 8) and group == 6 and query_rows >= 3:
                options.update(block_n=32, head_block=8)
            elif k_bits == 4:
                options['block_n'] = 32
            elif k_bits == 8 and occupied_bound <= 16384 and query_rows <= 2:
                options.update(block_n=32, num_splits=32)
        elif query_rows <= 4 and (k_bits, v_bits) in ((8, 4), (6, 4), (8, 6), (5, 8)):
            if group == 6 and query_rows >= 3:
                options['head_block'] = 8
            else:
                options.update(block_n=32 if v_bits == 4 else 16, num_splits=32)
        else:
            return {}
    elif (geometry in ((32, 8, 128), (16, 2, 128), (32, 4, 128), (32, 8, 256))
          and query_rows <= 4 and occupied_bound <= 65536
          and uniform and k_bits in (0, 8)):
        if group == 8 and query_rows >= 3:
            options['head_block'] = 8
        elif head_dim == 128 and k_bits == 8:
            options['block_n'] = 32
        elif group == 8 and query_rows <= 2:
            options.update(block_n=32, num_splits=32 if kv_heads == 2 else 64)
        elif head_dim == 256 and k_bits == 8 and query_rows <= 2:
            options['num_splits'] = 64
        else:
            options['num_splits'] = 32
    else:
        return {}
    return options


def prefill_tiles(*, arch, head_dim, query_rows):
    """(block_m, block_n, num_warps, num_stages) for the FP16 paged prefill kernel, or None.

    quantlab: measured on gfx1101 at head_dim 256 over 17-2048 query rows and 8K-128K
    context: 2.0-2.5x the inherited (64, 32, 8, 2) tile with the same FP32-reference error.
    Both keep 16 query rows per warp. Other GPUs and head sizes keep the inherited tile.
    """
    if arch != 'gfx1101' or not 128 < head_dim <= 256 or query_rows < 1:
        return None
    return (64, 64, 4, 1) if query_rows <= 128 else (128, 64, 8, 1)
