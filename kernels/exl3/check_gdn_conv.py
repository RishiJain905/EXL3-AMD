"""Native GDN convolution against an independent CPU window reference."""
import hashlib
import math

import numpy as np


def reference(x, weight, bias, initial, slots, activation, history):
    batch, channels, length = x.shape
    width = weight.shape[1]
    state = initial.copy()
    output = np.empty((batch, length, channels), dtype=np.float32)
    for b, slot in enumerate(slots):
        sequence = np.concatenate((initial[slot, :, :width], x[b]), axis=1)
        for token in range(length):
            acc = bias.copy() if bias is not None else np.zeros(channels, dtype=np.float32)
            for k in range(width):
                # Round once per multiply-add, as the native FP32 FMA does.
                acc = (weight[:, k].astype(np.float64)
                       * sequence[:, token + k + 1].astype(np.float64)
                       + acc.astype(np.float64)).astype(np.float32)
            if activation:
                acc = (acc.astype(np.float64) / (1 + np.exp(-acc.astype(np.float64)))).astype(np.float32)
            output[b, token] = acc
        if history:
            size = min(state.shape[2], sequence.shape[1])
            state[slot, :, -size:] = sequence[:, -size:]
        else:
            state[slot, :, :width] = sequence[:, -width:]
    return output, state


def run_checks(torch, extension):
    rng = np.random.default_rng(92734)
    records = []

    def check(channels, width, length, history, activation=True, bias_enabled=False,
              batch=1, use_slots=True, graph=False, short_state=False):
        state_size = width if short_state else width + length + 3
        selected = np.arange(batch, dtype=np.int32)
        if use_slots:
            selected = selected[::-1] + 1

        def bf16(array):
            return torch.from_numpy(array.astype(np.float32)).to(torch.bfloat16).float().numpy()

        def inputs():
            return (bf16(rng.normal(0, 0.5, (batch, channels, length))),
                    bf16(rng.normal(0, 0.2, (channels, width))),
                    bf16(rng.normal(0, 0.1, channels)) if bias_enabled else None,
                    bf16(rng.normal(0, 0.3, (batch + 2, channels, state_size))))

        cpu_x, cpu_w, cpu_bias, cpu_state = inputs()
        x = torch.from_numpy(cpu_x).to('cuda', torch.bfloat16)
        weight = torch.from_numpy(cpu_w).to('cuda', torch.bfloat16)
        bias = torch.from_numpy(cpu_bias).to('cuda', torch.bfloat16) if bias_enabled else None
        slots = torch.from_numpy(selected.copy()).cuda() if use_slots else None
        buffers = []

        def guarded(shape):
            backing = torch.full((math.prod(shape) + 32,), 23, device='cuda', dtype=torch.bfloat16)
            buffers.append(backing)
            return backing[16:-16].view(shape)

        state = guarded(cpu_state.shape)
        state.copy_(torch.from_numpy(cpu_state))
        out = guarded((batch, length, channels))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        args = (x, state, slots, weight, bias, out, activation, history)
        with torch.cuda.stream(stream):
            extension.cuda_causal_conv1d_update(*args)
            if graph:
                captured = torch.cuda.CUDAGraph()
                with torch.cuda.graph(captured, stream=stream):
                    extension.cuda_causal_conv1d_update(*args)
                cpu_x, cpu_w, cpu_bias, cpu_state = inputs()
                selected = np.arange(batch, dtype=np.int32)
                for tensor, cpu in ((x, cpu_x), (weight, cpu_w), (state, cpu_state)):
                    tensor.copy_(torch.from_numpy(cpu))
                if bias_enabled:
                    bias.copy_(torch.from_numpy(cpu_bias))
                slots.copy_(torch.from_numpy(selected))
                captured.replay()
        stream.synchronize()
        expected_out, expected_state = reference(
            cpu_x, cpu_w, cpu_bias, cpu_state, selected, activation, history)
        actual_out = out.float().cpu().numpy()
        actual_state = state.float().cpu().numpy()
        assert np.isfinite(actual_out).all()
        assert np.array_equal(actual_state, expected_state), 'Convolution state/window mismatch'
        _, exponent = np.frexp(np.abs(expected_out.astype(np.float64)))
        spacing = np.where(expected_out == 0, 0, np.ldexp(np.ones_like(exponent, dtype=np.float64), exponent - 8))
        error = np.abs(actual_out.astype(np.float64) - expected_out)
        assert np.all(error <= spacing + 5e-7), ('Convolution output/reference', float(error.max()))
        if not activation:
            assert np.array_equal(actual_out, bf16(expected_out)), 'Linear convolution must round to nearest BF16'
        for tensor, cpu in ((x, cpu_x), (weight, cpu_w)):
            assert np.array_equal(tensor.float().cpu().numpy(), cpu)
        if bias_enabled:
            assert np.array_equal(bias.float().cpu().numpy(), cpu_bias)
        if slots is not None:
            assert np.array_equal(slots.cpu().numpy(), selected)

        if history and length > 1 and not short_state:
            kept = max(1, length // 2)
            end = state_size - (length - kept)
            expected_window = expected_state[selected, :, end - width:end].copy()
            jobs = [extension.ConvRewindJob(state[int(slot), 0, end - width].data_ptr(),
                    state[int(slot), 0, 0].data_ptr(), channels, width, state.stride(1)) for slot in selected]
            with torch.cuda.stream(stream):
                extension.batched_conv_rewind(jobs, x.device.index)
            stream.synchronize()
            assert np.array_equal(state.float().cpu().numpy()[selected, :, :width], expected_window)
            suffix = x[:, :, kept:].contiguous()
            suffix_out = torch.empty((batch, length - kept, channels), device='cuda', dtype=torch.bfloat16)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                extension.cuda_causal_conv1d_update(
                    suffix, state, slots, weight, bias, suffix_out, activation, False)
            stream.synchronize()
            assert torch.equal(suffix_out, out[:, kept:]), 'Convolution rewind/suffix replay mismatch'
        for backing in buffers:
            assert bool((backing[:16] == 23).all() and (backing[-16:] == 23).all()), 'Canary overwritten'
        records.append(dict(channels=channels, width=width, length=length, history=history,
                            activation=activation, bias=bias_enabled, batch=batch, slots=use_slots,
                            graph=graph, state_size=state_size, output_max_abs=float(error.max()),
                            output_sha256=hashlib.sha256(actual_out.tobytes()).hexdigest(),
                            state_sha256=hashlib.sha256(actual_state.tobytes()).hexdigest(),
                            rewind=history and length > 1 and not short_state))

    for channels in (8192, 10240):
        for length in (1, 2, 3, 5, 9):
            for history in (False, True):
                check(channels, 4, length, history)
        check(channels, 4, 5, True, bias_enabled=True, graph=True)
        check(channels, 4, 1, False, graph=True)
    for width in (1, 2, 3, 4, 5, 8, 16):
        for history in (False, True):
            for activation in (False, True):
                check(257, width, 5, history, activation, bias_enabled=True, batch=2)
        check(17, width, 9, True, short_state=True, use_slots=False)
    check(127, 4, 33, False, use_slots=False)
    return records
