"""Reproduce FP32 latency, parity, phase timings, attention shapes, and captures."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack, nullcontext
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from conftest import load_cases
from frida_decisions.protocol import decision


def summary(samples):
    return {'seconds': samples, 'median_seconds': float(np.median(samples)),
            'p10_seconds': float(np.percentile(samples, 10)),
            'p90_seconds': float(np.percentile(samples, 90)),
            'min_seconds': min(samples), 'max_seconds': max(samples)}


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')


def cases_for(args, all_cases=False):
    cases = load_cases()
    if not all_cases:
        names = ('choice/support-topic', 'generated/intent-catalog-40', 'generated/long-state')
        cases = [next(c for c in cases if c['name'] == name) for name in names]
    if args.real_request:
        cases.insert(0, {'name': 'private/17-tags', 'request': json.loads(args.real_request.read_text())})
    if args.primary_only:
        cases = cases[:1]
    if args.case:
        cases = [case for case in cases if case['name'] in args.case]
    return cases


def margins(response):
    return np.array([m for q in response['margins'].values() for m in q.values()])


def compare(response, reference):
    actual, expected = margins(response), margins(reference)
    assert actual.shape == expected.shape and np.isfinite(actual).all()
    drift = float(np.max(np.abs(actual - expected)))
    assert drift <= 1e-3, drift
    assert {q: decision(a) for q, a in response['answers'].items()} == {
        q: decision(a) for q, a in reference['answers'].items()}
    for q, answer in response['answers'].items():
        if answer['type'] == 'ranking':
            assert answer['ranking'] == reference['answers'][q]['ranking']
    return drift


def environment(args):
    def version(package):
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return None
    return {'platform': platform.platform(), 'python': platform.python_version(),
            'MTL_CAPTURE_ENABLED': os.environ.get('MTL_CAPTURE_ENABLED'),
            'mlx': version('mlx'), 'torch': version('torch'),
            'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'baseline_revision': subprocess.check_output(['git', 'rev-parse', 'profile-mlx-baseline-20261003'], cwd=ROOT, text=True).strip(),
            'model_revision': model_revision(args.model),
            'model_sha256': {p.name: file_hash(p)
                for p in sorted(args.model.iterdir()) if p.suffix in ('.safetensors', '.json')},
            'request_sha256': hashlib.sha256(args.real_request.read_bytes()).hexdigest() if args.real_request else None,
            'dtype': 'float32', 'rows_per_forward': args.rows_per_forward,
            'state_max': {'private/17-tags': 512, 'other': 384}, 'state_cache_mb': 512}


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def model_revision(folder):
    metadata = folder / '.cache/huggingface/download/model.safetensors.metadata'
    return metadata.read_text().splitlines()[0] if metadata.exists() else 'local export, identified by file hashes'


def set_mode(judge, mode, mx=None):
    if judge.backend == 'mlx':
        from frida_decisions.mlx_modeling import StateCache
        judge.state_cache = None if mode == 'off' else StateCache()
    else:
        from frida_decisions.modeling import StateCache
        judge.state_cache = None if mode == 'off' else StateCache(512 * 2**20)
    # Diagnostic miss/hit modes use explicit caching for single-row cases too.
    # This does not change the application's cache policy.
    judge.use_state_cache = lambda requests: judge.state_cache is not None and len(requests) == 1


def clear_state(judge):
    judge.state_cache.clear()


def benchmark(args, judge, sync, report, mx=None):
    for case in cases_for(args):
        req = case['request']
        judge.state_max = 512 if case['name'].startswith('private/') else 384
        row = {'name': case['name'], 'counts': judge.token_counts(req), 'modes': {}}
        report['scenarios'].append(row)
        reference = None
        for mode in args.modes:
            set_mode(judge, mode, mx)
            sync()
            if mx:
                mx.clear_cache()
            warm, peaks, times, drifts, statuses = [], [], [], [], []
            for _ in range(args.warmup):
                if mode == 'miss':
                    clear_state(judge)
                sync()
                started = time.perf_counter()
                judge(req)
                sync()
                warm.append(time.perf_counter() - started)
            for _ in range(args.runs):
                if mode == 'miss':
                    clear_state(judge)
                sync()
                if mx:
                    mx.reset_peak_memory()
                started = time.perf_counter()
                response = judge(req)
                sync()
                times.append(time.perf_counter() - started)
                if mx:
                    peaks.append(mx.get_peak_memory())
                reference = response if reference is None else reference
                drifts.append(compare(response, reference))
                statuses.append(response['usage']['state_cache'])
                assert statuses[-1] == mode, (mode, statuses[-1])
            row['modes'][mode] = {**summary(times), 'warmup_seconds': warm,
                'peak_mlx_bytes': max(peaks) if peaks else None, 'cache_status': statuses,
                'max_repeat_margin_drift': max(drifts), 'response': response}
            if hasattr(judge.model, 'traces'):
                report['compile_traces'] = judge.model.traces
            save(args.output, report)
            print(json.dumps({'case': case['name'], 'mode': mode, 'variant': args.variant,
                'median': np.median(times), 'peak': max(peaks) if peaks else None}), flush=True)


def parity(args, judge, sync, report):
    reference = json.loads(args.reference.read_text()) if args.reference else None
    worst, decisions = 0.0, 0
    for case in cases_for(args, all_cases=True):
        req, name = case['request'], case['name']
        judge.state_max = 512 if name.startswith('private/') else 384
        row = {'name': name, 'modes': {}}
        report['scenarios'].append(row)
        if reference:
            ref = next(c for c in reference['scenarios'] if c['name'] == name)
        for mode in ('off', 'miss', 'hit'):
            set_mode(judge, mode)
            if mode == 'hit':
                judge.margins_cached(req)
            if mode == 'miss':
                # Use explicit caching even for one-row requests.
                parsed, candidates, tok = judge.compile(req)
                from frida_decisions.protocol import aggregate
                values, usage = judge._score_cached(tok)
                by_question = {}
                for c, value in zip(candidates, values):
                    by_question.setdefault(c.question_id, {})[c.option_id] = value
                response = {'answers': aggregate(parsed, candidates, values), 'margins': by_question, 'usage': usage}
            else:
                response = judge(req)
            sync()
            drift = compare(response, ref['modes'][mode]['response']) if reference else 0.0
            if mode != 'off':
                drift = max(drift, compare(response, row['modes']['off']['response']))
            worst = max(worst, drift)
            decisions += len(response['answers'])
            row['modes'][mode] = {'response': response, 'max_margin_drift': drift}
        report['parity'] = {'requests': len(report['scenarios']), 'decisions': decisions,
                            'max_margin_drift': worst, 'decision_mismatches': 0}
        save(args.output, report)
        print(json.dumps({'case': name, 'worst_drift': worst}), flush=True)


class Timers:
    """Synchronized exclusive diagnostic times. These alter GPU scheduling."""

    def __init__(self, mx):
        self.mx = mx
        self.totals = defaultdict(float)
        self.calls = defaultdict(int)
        self.stack = []

    def wrap(self, name, fun):
        def call(*args, **kwargs):
            self.mx.synchronize()
            start = time.perf_counter()
            self.stack.append(0.0)
            result = fun(*args, **kwargs)
            # Only evaluate tensor trees, never CPU packing dataclasses.
            def arrays(value):
                if isinstance(value, self.mx.array):
                    return [value]
                if isinstance(value, (list, tuple)):
                    return [a for v in value for a in arrays(v)]
                return []
            leaves = arrays(result)
            if leaves:
                self.mx.eval(*leaves)
            self.mx.synchronize()
            elapsed = time.perf_counter() - start
            children = self.stack.pop()
            self.totals[name] += elapsed - children
            self.calls[name] += 1
            if self.stack:
                self.stack[-1] += elapsed
            return result
        return call


def phases(args, judge, mx, report):
    from frida_decisions import base, mlx_backend, mlx_modeling
    import mlx.nn as nn
    from mlx_experiments import attention_path, CompiledModel
    encoder = judge.model.model if isinstance(judge.model, CompiledModel) else judge.model
    timers = Timers(mx)
    shapes = {}
    projections = {}
    for layer in judge.model.layers:
        for name in ('q', 'k', 'v', 'o'):
            projections[id(getattr(layer.attention, name))] = 'attention_projections'
        for name in ('wi_0', 'wi_1', 'wo'):
            projections[id(getattr(layer.dense, name))] = 'FFN_matrices_and_gate'
    projections[id(judge.model.head)] = 'decision_head_matrix'
    linear = nn.Linear.__call__

    def project(module, x):
        return timers.wrap(projections.get(id(module), 'other_matrix'), linear)(module, x)

    def attend(q, k, v, bias):
        key = (q.shape, k.shape, bias.shape)
        shapes[str(key)] = {'B': q.shape[0], 'heads': q.shape[1], 'Tq': q.shape[2],
            'Tkv': k.shape[2], 'D': q.shape[3], 'dtype': str(q.dtype), 'mask_shape': list(bias.shape)}
        timers.wrap('attention_layout_and_KV_copies', lambda: (q, k, v, bias))()
        scores = timers.wrap('attention_scores', lambda: q @ k.transpose(0, 1, 3, 2))()
        scores = timers.wrap('attention_bias_add', lambda: scores + bias)()
        p = timers.wrap('softmax', lambda: mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype))()
        return timers.wrap('attention_values', lambda: p @ v)()

    for case in cases_for(args):
        judge.state_max = 512 if case['name'].startswith('private/') else 384
        row = {'name': case['name'], 'modes': {}}
        report['scenarios'].append(row)
        for mode in args.modes:
            set_mode(judge, mode, mx)
            if mode == 'hit':
                judge.margins_cached(case['request'])
            judge(case['request'])
            if mode == 'miss':
                clear_state(judge)
            timers.totals.clear()
            timers.calls.clear()
            shapes.clear()
            with ExitStack() as stack:
                stack.enter_context(attention_path(attend))
                stack.enter_context(patch.object(nn.Linear, '__call__', project))
                targets = [(base, 'parse_request', 'request_parse'),
                    (base, 'compile_request', 'candidate_construction'), (base, 'tokenize', 'tokenization'),
                    (base, 'aggregate', 'answer_aggregation'), (mlx_backend, 'build_rows', 'packing'),
                    (mlx_backend, 'pack_rows', 'packing'), (mlx_backend, 'pack_queries', 'packing'),
                    (mlx_backend, 'state_buckets', 'packing'),
                    (mlx_modeling.FridaMlxDecisionModel, 'encode_state', 'state_encoding_other'),
                    (mlx_modeling.FridaMlxDecisionModel, 'forward_cached', 'cached_query_other'),
                    (mlx_modeling.FridaMlxDecisionModel, '__call__', 'packed_encoder_other'),
                    (mlx_modeling.FridaMlxDecisionModel, 'pool', 'candidate_pooling'),
                    (mlx_modeling.FridaMlxDecisionModel, 'margins', 'decision_head'),
                    (mlx_modeling.T5Norm, '__call__', 'normalization_and_residual'),
                    (mlx_modeling.FridaMlxDenseActivation, '__call__', 'FFN_elementwise'),
                    (mlx_modeling, 'gelu_new', 'gelu_new')]
                for owner, attr, name in targets:
                    stack.enter_context(patch.object(owner, attr, timers.wrap(name, getattr(owner, attr))))
                # Rebind after all wrappers; phase timers require eager encoder calls.
                stack.enter_context(patch.object(judge, '_forward_encoder', encoder.__call__))
                stack.enter_context(patch.object(judge, '_forward_cached_encoder', encoder.forward_cached))
                response = judge(case['request'])
            row['modes'][mode] = {'exclusive_diagnostic_seconds': dict(timers.totals),
                'calls': dict(timers.calls), 'attention_shapes': list(shapes.values()),
                'usage': response['usage']}
            save(args.output, report)
            print(json.dumps({'case': case['name'], 'mode': mode, 'phases': dict(timers.totals)}), flush=True)


def attention_benchmark(args, judge, mx, report):
    from frida_decisions.packing import build_rows, pack_rows, pack_queries, state_buckets
    from mlx_experiments import manual_attention, fast_attention
    for case in cases_for(args):
        judge.state_max = 512 if case['name'].startswith('private/') else 384
        tok = judge.compile(case['request'])[2]
        rows = build_rows([tok], judge.config)[0]
        n = len(tok.state)
        sb, sa = state_buckets(n, judge.config)
        inputs = [('state', np.array([tok.state + [0] * (sb.shape[0] - n)]), sb[None], sa[None])]
        for i, row in enumerate(rows):
            b = pack_rows([row], judge.config)
            inputs.append((f'packed-{i}', b.input_ids, b.buckets, b.allowed))
        qr = pack_queries(tok, judge.config)
        for i in range(len(qr.rows)):
            inputs.append((f'cached-{i}', qr.input_ids[i:i+1], qr.buckets[i:i+1], qr.allowed[i:i+1]))
        seen = set()
        for path, ids, buckets, allowed in inputs:
            shape_key = (ids.shape, buckets.shape)
            if shape_key in seen:
                continue
            seen.add(shape_key)
            # Use real first-layer projections, real bias, and real state KV.
            layer = judge.model.layers[0]
            h = layer.ln1(judge.model.embed(mx.array(ids, dtype=mx.int32)))
            shape = (*h.shape[:2], layer.attention.num_heads, layer.attention.d_kv)
            q, k, v = [p(h).reshape(shape).transpose(0, 2, 1, 3)
                       for p in (layer.attention.q, layer.attention.k, layer.attention.v)]
            if path.startswith('cached'):
                sh = layer.ln1(judge.model.embed(mx.array([tok.state], dtype=mx.int32)))
                ss = (1, n, layer.attention.num_heads, layer.attention.d_kv)
                sk, sv = [p(sh).reshape(ss).transpose(0, 2, 1, 3)
                          for p in (layer.attention.k, layer.attention.v)]
                k, v = mx.concatenate([sk, k], axis=2), mx.concatenate([sv, v], axis=2)
            bias = judge.model.attention_bias(mx.array(buckets), mx.array(allowed))
            mx.eval(q, k, v, bias)
            ref = manual_attention(q, k, v, bias)
            mx.eval(ref)
            row = {'name': case['name'], 'path': path, 'q': list(q.shape), 'k': list(k.shape),
                   'mask': list(bias.shape), 'dtype': str(q.dtype), 'variants': {}}
            report['scenarios'].append(row)
            for name, fun in [('manual', manual_attention), ('sdpa', fast_attention),
                    ('force_fused', lambda q, k, v, b: mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=b, force_fused=True))]:
                try:
                    for _ in range(args.warmup):
                        mx.eval(fun(q, k, v, bias))
                    times, peaks = [], []
                    for _ in range(args.runs):
                        mx.synchronize()
                        mx.reset_peak_memory()
                        started = time.perf_counter()
                        out = fun(q, k, v, bias)
                        mx.eval(out)
                        mx.synchronize()
                        times.append(time.perf_counter() - started)
                        peaks.append(mx.get_peak_memory())
                    delta = np.abs(np.array(out) - np.array(ref))
                    row['variants'][name] = {**summary(times), 'max_absolute_difference': float(delta.max()),
                        'max_relative_difference': float((delta / np.maximum(np.abs(np.array(ref)), 1e-12)).max()),
                        'peak_mlx_bytes': max(peaks)}
                except Exception as error:
                    row['variants'][name] = {'error': str(error)}
            save(args.output, report)
            print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['benchmark', 'parity', 'phases', 'attention', 'capture', 'kernels'], default='benchmark')
    parser.add_argument('--backend', choices=['mlx', 'mps', 'cpu'], default='mlx')
    parser.add_argument('--variant', choices=['baseline', 'sdpa', 'compile', 'sdpa-compile', 'rms', 'production'], default='baseline')
    parser.add_argument('--model', type=Path, default=ROOT / '_export/FRIDA-Decisions')
    parser.add_argument('--real-request', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--modes', nargs='+', choices=['off', 'miss', 'hit'], default=['off', 'miss', 'hit'])
    parser.add_argument('--rows-per-forward', type=int, default=1)
    parser.add_argument('--primary-only', action='store_true')
    parser.add_argument('--case', nargs='+', help='Restrict results to these case names')
    parser.add_argument('--counter-library', type=Path, help='Compiled tools/metal_kernel_counts.mm library')
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 1:
        parser.error('Use positive run and warmup counts')
    if args.backend != 'mlx' and (args.task not in ('benchmark', 'parity') or args.variant != 'baseline'):
        parser.error('PyTorch supports only baseline benchmark and parity tasks')
    report = {'environment': environment(args), 'task': args.task, 'backend': args.backend,
              'variant': args.variant, 'runs': args.runs, 'scenarios': []}
    mx = None
    if args.backend == 'mlx':
        import mlx.core as mx
        from frida_decisions import MlxJudge
        from mlx_experiments import attention_path, fast_attention, manual_attention, fast_norm_path, CompiledModel
        if args.task == 'kernels':
            if not args.counter_library:
                parser.error('Use --counter-library for kernel counting')
            import ctypes
            counters = ctypes.CDLL(str(args.counter_library.resolve()))
            counters.frida_profile_stop.argtypes = [ctypes.c_char_p]
            counters.frida_profile_install()
        judge = MlxJudge.from_pretrained(args.model, state_max=512, rows_per_forward=args.rows_per_forward,
                                        compile_encoder=args.variant == 'production')
        sync = mx.synchronize
        report['environment']['device'] = mx.device_info()
        context = nullcontext() if args.variant == 'production' else attention_path(
            fast_attention if args.variant.startswith('sdpa') else manual_attention)
    else:
        import torch
        from frida_decisions import Judge
        torch.set_num_threads(6)
        torch.set_num_interop_threads(1)
        judge = Judge.from_pretrained(args.model, device=args.backend, dtype=torch.float32,
                                     state_max=512, rows_per_forward=args.rows_per_forward)
        sync = torch.mps.synchronize if args.backend == 'mps' else lambda: None
        report['environment']['cpu_threads'] = 6
        context = torch.inference_mode()
    with context:
        with ExitStack() as norms:
            if args.backend == 'mlx' and args.variant.endswith('rms'):
                norms.enter_context(fast_norm_path())
            run_tasks(args, judge, sync, report, mx, parser, counters if args.task == 'kernels' else None)
    save(args.output, report)


def run_tasks(args, judge, sync, report, mx, parser, counters):
    if 'compile' in args.variant:
        from mlx_experiments import CompiledModel
        judge.model = CompiledModel(judge.model)
    if args.backend == 'mlx' and args.variant != 'production':
        judge._forward_encoder = judge.model.__call__
        judge._forward_cached_encoder = judge.model.forward_cached
    if args.task == 'benchmark':
        benchmark(args, judge, sync, report, mx)
    elif args.task == 'parity':
        parity(args, judge, sync, report)
    elif args.task == 'phases':
        phases(args, judge, mx, report)
    elif args.task == 'attention':
        attention_benchmark(args, judge, mx, report)
    elif args.task in ('capture', 'kernels'):
        if args.task == 'capture' and os.environ.get('MTL_CAPTURE_ENABLED') != '1':
            parser.error('Set MTL_CAPTURE_ENABLED=1 for capture')
        request = cases_for(args)[0]['request']
        judge(request)
        judge(request)
        sync()
        trace = args.output.with_suffix('.gputrace').resolve()
        if args.task == 'capture':
            mx.metal.start_capture(str(trace))
        else:
            counters.frida_profile_start()
        try:
            response = judge(request)
            sync()
        finally:
            if args.task == 'capture':
                mx.metal.stop_capture()
            else:
                counts_path = args.output.with_suffix('.counts.json').resolve()
                counters.frida_profile_stop(str(counts_path).encode())
                report['kernel_counts'] = json.loads(counts_path.read_text())
                report['total_dispatches'] = sum(report['kernel_counts'].values())
                assert report['total_dispatches'] > 0 and 'unknown' not in report['kernel_counts']
        report.update(trace=str(trace) if args.task == 'capture' else None, usage=response['usage'])
    if hasattr(judge.model, 'traces'):
        report['compile_traces'] = judge.model.traces

if __name__ == '__main__':
    main()
