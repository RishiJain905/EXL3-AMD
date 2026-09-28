"""GDN recurrence against a float64 NumPy oracle, including history and replay.

The oracle uses tensor contractions, independent of the kernel's thread layout
and partial reductions. Output bounds include one BF16 step plus float32
reduction error; an analytical fixture separately enforces truncating output.
"""
import math

import numpy as np


def reference(qkv, gate, beta, initial, slots, nk, nv, dk, dv, history):
    batch, length, _ = qkv.shape
    query = qkv[..., :nk * dk].reshape(batch, length, nk, dk).astype(np.float64)
    key = qkv[..., nk * dk:2 * nk * dk].reshape(batch, length, nk, dk).astype(np.float64)
    value = qkv[..., 2 * nk * dk:].reshape(batch, length, nv, dv).astype(np.float64)
    query /= np.sqrt(np.sum(query * query, axis=-1, keepdims=True) + 1e-6)
    key /= np.sqrt(np.sum(key * key, axis=-1, keepdims=True) + 1e-6)
    query = np.repeat(query, nv // nk, axis=2)
    key = np.repeat(key, nv // nk, axis=2)
    expected = initial.astype(np.float64).copy()
    output = np.empty((batch, length, nv, dv), dtype=np.float64)
    touched = set()
    for b, slot in enumerate(slots):
        state = initial[slot, 0].astype(np.float64).copy()
        for token in range(length):
            decay = np.exp(gate[b, token].astype(np.float64))[:, None, None]
            delta = value[b, token] - np.einsum('hkv,hk->hv', state, key[b, token]) * decay[:, 0]
            state = (state * decay + key[b, token, :, :, None] * delta[:, None, :]
                     * beta[b, token, :, None, None].astype(np.float64))
            output[b, token] = np.einsum('hkv,hk->hv', state, query[b, token]) / math.sqrt(dk)
            index = token + 1 if history and token < length - 1 else 0
            expected[slot, index] = state
            touched.add((int(slot), index))
    return output, expected, touched


def truncate_bf16(values):
    values = np.asarray(values, dtype=np.float32)
    return (values.view(np.uint32) & np.uint32(0xffff0000)).view(np.float32)


def run_checks(torch, extension):
    records = []
    rng = np.random.default_rng(92731)

    def check(batch, nk, nv, dk, dv, length, history, *, use_slots=True,
              graph=False, special=None):
        count = batch + 2
        stride = length + 2 if history else 3
        selected = np.arange(batch, dtype=np.int32)
        if use_slots:
            selected = np.arange(batch, dtype=np.int32)[::-1] + 1

        def inputs():
            q = rng.normal(0, 0.4, (batch, length, 2 * nk * dk + nv * dv)).astype(np.float32)
            g = rng.uniform(-1.5, 0, (batch, length, nv)).astype(np.float32)
            beta = rng.uniform(0, 1, (batch, length, nv)).astype(np.float32)
            state = rng.normal(0, 0.12, (count, stride, nv, dk, dv)).astype(np.float32)
            if special == 'zero-qk':
                q[..., :2 * nk * dk] = 0
            elif special == 'zero-beta':
                beta.fill(0)
            elif special == 'tiny-qk':
                q[..., :2 * nk * dk] *= 1e-5
            elif special == 'strong-decay':
                g.fill(-80)
            elif special == 'truncate':
                q.fill(0)
                q[..., np.arange(nk) * dk] = 1
                g.fill(0)
                beta.fill(0)
                state.fill(0)
                state[:, 0, :, 0, :] = 1.5
            # Quantize inputs before the reference; the interface accepts BF16.
            q = torch.from_numpy(q).to(torch.bfloat16).float().numpy()
            beta = torch.from_numpy(beta).to(torch.bfloat16).float().numpy()
            return q, g, beta, state

        cpu_q, cpu_g, cpu_beta, cpu_state = inputs()
        q = torch.from_numpy(cpu_q).to(device='cuda', dtype=torch.bfloat16)
        g = torch.from_numpy(cpu_g).cuda()
        beta = torch.from_numpy(cpu_beta).to(device='cuda', dtype=torch.bfloat16)
        slots = torch.from_numpy(selected.copy()).cuda() if use_slots else None
        buffers = []

        def guarded(shape, dtype):
            backing = torch.full((math.prod(shape) + 32,), 23, device='cuda', dtype=dtype)
            buffers.append(backing)
            return backing[16:-16].view(shape)

        state = guarded(cpu_state.shape, torch.float32)
        state.copy_(torch.from_numpy(cpu_state))
        output = guarded((batch, length, nv, dv), torch.bfloat16)
        arguments = (q, g, beta, state, output, nk, nv, dk, dv, slots, history)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            extension.cuda_recurrent_gated_delta_rule(*arguments)
            if graph:
                captured = torch.cuda.CUDAGraph()
                with torch.cuda.graph(captured, stream=stream):
                    extension.cuda_recurrent_gated_delta_rule(*arguments)
                # Change all mutable inputs, state contents and slot values.
                cpu_q, cpu_g, cpu_beta, cpu_state = inputs()
                selected = np.arange(batch, dtype=np.int32)
                q.copy_(torch.from_numpy(cpu_q))
                g.copy_(torch.from_numpy(cpu_g))
                beta.copy_(torch.from_numpy(cpu_beta))
                state.copy_(torch.from_numpy(cpu_state))
                slots.copy_(torch.from_numpy(selected))
                captured.replay()
        stream.synchronize()
        expected_out, expected_state, touched = reference(
            cpu_q, cpu_g, cpu_beta, cpu_state, selected, nk, nv, dk, dv, history)
        actual_state = state.cpu().numpy()
        actual_out = output.float().cpu().numpy()
        assert np.isfinite(actual_state).all() and np.isfinite(actual_out).all()
        state_error = 0.0
        for slot in range(count):
            for step in range(stride):
                actual, expected = actual_state[slot, step], expected_state[slot, step]
                if (slot, step) in touched:
                    error = float(np.max(np.abs(actual - expected)))
                    state_error = max(state_error, error)
                    assert np.allclose(actual, expected, rtol=3e-5, atol=3e-6), (
                        'state/reference', batch, nk, nv, dk, dv, length, history, special, error)
                else:
                    assert np.array_equal(actual, cpu_state[slot, step]), ('untouched state', slot, step)
        # BF16 spacing varies by exponent; near zero permit only the FP32 error floor.
        _, exponent = np.frexp(np.abs(expected_out))
        spacing = np.where(expected_out == 0, 0, np.ldexp(np.ones_like(expected_out), exponent - 8))
        bound = spacing * 1.01 + 5e-7
        out_error = np.abs(actual_out.astype(np.float64) - expected_out)
        assert np.all(out_error <= bound), ('output/reference', nk, nv, length, float(out_error.max()))
        if special == 'zero-qk':
            assert not np.any(actual_out), 'Zero queries must produce zero output'
        if special == 'truncate':
            assert np.array_equal(actual_out, truncate_bf16(expected_out)), 'BF16 output must truncate toward zero'
            nearest = torch.from_numpy(expected_out.astype(np.float32)).to(torch.bfloat16).float().numpy()
            assert np.any(nearest != actual_out), 'Rounding fixture must distinguish RN from RZ'
        assert np.array_equal(q.float().cpu().numpy(), cpu_q)
        assert np.array_equal(g.cpu().numpy(), cpu_g)
        assert np.array_equal(beta.float().cpu().numpy(), cpu_beta)
        if slots is not None:
            assert np.array_equal(slots.cpu().numpy(), selected)
        for backing in buffers:
            assert bool((backing[:16] == 23).all() and (backing[-16:] == 23).all()), 'Canary overwritten'
        records.append(dict(batch=batch, key_heads=nk, value_heads=nv, key_dim=dk, value_dim=dv,
                            length=length, history=history, slots=use_slots, graph=graph, special=special,
                            state_max_abs=state_error, output_max_abs=float(out_error.max()),
                            nondefault_stream=True, untouched_state_exact=True))

        if history and length > 1:
            # Model rewind keeps the prefix after `kept` tokens. Then replay the
            # rejected suffix and compare to the uninterrupted reference above.
            kept = max(1, length // 2)
            jobs = [extension.StateRewindJob(state[int(slot), kept].data_ptr(),
                    state[int(slot), 0].data_ptr(), state[int(slot), 0].numel()) for slot in selected]
            with torch.cuda.stream(stream):
                extension.batched_state_rewind(jobs, q.device.index)
            stream.synchronize()
            for slot in selected:
                assert torch.equal(state[int(slot), 0], state[int(slot), kept]), 'Rewind copy mismatch'
            suffix = tuple(t[:, kept:].contiguous() for t in (q, g, beta))
            suffix_out = torch.empty((batch, length - kept, nv, dv), device='cuda', dtype=torch.bfloat16)
            with torch.cuda.stream(stream):
                stream.wait_stream(torch.cuda.default_stream())
                extension.cuda_recurrent_gated_delta_rule(
                    *suffix, state, suffix_out, nk, nv, dk, dv, slots, False)
            stream.synchronize()
            replay = state.cpu().numpy()
            for slot in selected:
                assert np.allclose(replay[int(slot), 0], expected_state[int(slot), 0], rtol=3e-5, atol=3e-6)
            suffix_error = np.abs(suffix_out.float().cpu().numpy() - expected_out[:, kept:])
            assert np.all(suffix_error <= bound[:, kept:])
            for backing in buffers:
                assert bool((backing[:16] == 23).all() and (backing[-16:] == 23).all()), 'Rewind canary overwritten'
            records.append(dict(rewind=True, key_heads=nk, value_heads=nv, key_dim=dk, value_dim=dv,
                                length=length, kept=kept, suffix_replay=True))

    for nv in (32, 48):
        for length in (1, 2, 3, 5, 9):
            for history in (False, True):
                check(1, 16, nv, 128, 128, length, history)
        check(1, 16, nv, 128, 128, 3, True, graph=True)
        check(1, 16, nv, 128, 128, 1, False, graph=True)
        check(1, 16, nv, 128, 128, 5, True, use_slots=False)
        check(1, 16, nv, 128, 128, 17, False)
        check(1, 16, nv, 128, 128, 17, True)
        check(1, 16, nv, 128, 128, 65, False)
    for special in ('zero-qk', 'tiny-qk', 'zero-beta', 'strong-decay', 'truncate'):
        check(1, 2, 4, 128, 128, 1 if special == 'truncate' else 5, True, special=special)
    check(1, 2, 4, 128, 128, 3, True, special='truncate')
    # Different group ratios and head counts within the candidate geometry.
    for nk, nv in ((1, 1), (2, 6), (8, 16), (32, 64)):
        check(1, nk, nv, 128, 128, 3, True, graph=True)
    # Existing generic and unsplit fallbacks remain exercised.
    for batch, nk, nv, dk, dv in ((2, 4, 8, 128, 128), (1, 16, 80, 128, 128),
                                 (1, 4, 8, 64, 128), (1, 4, 8, 128, 64), (1, 2, 4, 256, 256)):
        check(batch, nk, nv, dk, dv, 3, True)
    return records
