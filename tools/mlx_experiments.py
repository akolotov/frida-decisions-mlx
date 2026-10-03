"""Isolated MLX experiments. Production inference does not import this module."""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import mlx.core as mx

from frida_decisions.mlx_modeling import FridaMlxAttention, FridaMlxDecisionModel, T5Norm


def manual_attention(q, k, v, bias):
    scores = q @ k.transpose(0, 1, 3, 2) + bias
    probabilities = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
    return probabilities @ v


def fast_attention(q, k, v, bias):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=bias)


@contextmanager
def fast_norm_path():
    original = T5Norm.__call__

    def norm(self, x):
        if x.dtype == mx.float32 and self.weight.dtype == mx.float32:
            return mx.fast.rms_norm(x, self.weight, self.eps)
        # T5 rounds before multiplying BF16 weights. Preserve that order.
        return original(self, x)

    with patch.object(T5Norm, '__call__', norm):
        yield


@contextmanager
def attention_path(attend):
    """Replace only attention arithmetic, including state and cached paths."""
    def attention(self, x, bias):
        b, n, _ = x.shape
        shape = (b, n, self.num_heads, self.d_kv)
        q, k, v = [p(x).reshape(shape).transpose(0, 2, 1, 3)
                   for p in (self.q, self.k, self.v)]
        out = attend(q, k, v, bias).transpose(0, 2, 1, 3).reshape(b, n, -1)
        return self.o(out)

    def cached(self, input_ids, buckets, allowed, ks, vs):
        bias = self.attention_bias(buckets, allowed)
        x = self.embed(input_ids)
        b, n, _ = x.shape
        for i, layer in enumerate(self.layers):
            attn = layer.attention
            h = layer.ln1(x)
            shape = (b, n, attn.num_heads, attn.d_kv)
            q, k, v = [p(h).reshape(shape).transpose(0, 2, 1, 3)
                       for p in (attn.q, attn.k, attn.v)]
            k = mx.concatenate([mx.broadcast_to(ks[i], (b, *ks[i].shape[1:])), k], axis=2)
            v = mx.concatenate([mx.broadcast_to(vs[i], (b, *vs[i].shape[1:])), v], axis=2)
            out = attend(q, k, v, bias).transpose(0, 2, 1, 3).reshape(b, n, -1)
            x = x + attn.o(out)
            x = x + layer.dense(layer.ln2(x))
        return self.final_norm(x)

    with patch.object(FridaMlxAttention, '__call__', attention), \
            patch.object(FridaMlxDecisionModel, 'forward_cached', cached):
        yield


class CompiledModel:
    """Compile complete packed and cached-query encoders with fixed weights."""

    def __init__(self, model):
        self.model = model
        self.traces = {'packed': [], 'cached': []}
        traces = self.traces

        def traced(name, fun):
            def call(*args):
                # Python executes once per specialization, not per GPU call.
                traces[name].append([list(a.shape) for a in args if isinstance(a, mx.array)])
                return fun(*args)
            return mx.compile(call)

        self.packed = traced('packed', model.__call__)
        self.forward_cached = traced('cached', model.forward_cached)

    def __call__(self, *args):
        return self.packed(*args)

    def __getattr__(self, name):
        return getattr(self.model, name)
