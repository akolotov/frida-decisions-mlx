"""Compare four warmed cache-hit variants in a counterbalanced order."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / '_export/FRIDA-Decisions')
    parser.add_argument('--real-request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--runs', type=int, default=12)
    parser.add_argument('--rows-per-forward', type=int, default=1)
    args = parser.parse_args()
    if args.runs < 10:
        parser.error('Use at least ten repetitions')
    import mlx.core as mx
    from frida_decisions import MlxJudge
    from frida_decisions.mlx_modeling import FridaMlxDecisionModel
    from mlx_experiments import attention_path, manual_attention, fast_attention
    from profile_mlx import compare, environment, save, summary

    request = json.loads(args.real_request.read_text())
    model = FridaMlxDecisionModel.from_folder(args.model)
    variants = ['baseline', 'sdpa', 'compile', 'production']
    judges, samples, warmups = {}, {}, {}
    reference = None
    if args.reference:
        refs = json.loads(args.reference.read_text())
        reference = next(c for c in refs['scenarios'] if c['name'] == 'private/17-tags')['modes']['hit']['response']
    report = {'environment': environment(args), 'device': mx.device_info(),
              'runs': args.runs, 'order': [], 'variants': {}, 'max_margin_drift': 0.0,
              'method': 'Shared immutable weights, independent state caches, rotating and reversed order. Memory is measured in separate processes.'}
    for name in variants:
        context = nullcontext() if name == 'production' else attention_path(
            fast_attention if name == 'sdpa' else manual_attention)
        with context:
            judge = MlxJudge(args.model, model, state_max=512,
                             rows_per_forward=args.rows_per_forward,
                             compile_encoder=name in ('compile', 'production'))
            judges[name] = judge
            samples[name], warmups[name] = [], []
            for _ in range(2):
                mx.synchronize()
                start = time.perf_counter()
                response = judge(request)
                mx.synchronize()
                warmups[name].append(time.perf_counter() - start)
            assert response['usage']['state_cache'] == 'hit'
            if reference is None:
                reference = response
            compare(response, reference)
    report['counts'] = judges['production'].token_counts(request)
    for repetition in range(args.runs):
        offset = repetition % len(variants)
        order = variants[offset:] + variants[:offset]
        if (repetition // len(variants)) % 2:
            order = list(reversed(order))
        report['order'].append(order)
        for name in order:
            mx.synchronize()
            start = time.perf_counter()
            response = judges[name](request)
            mx.synchronize()
            samples[name].append(time.perf_counter() - start)
            assert response['usage']['state_cache'] == 'hit'
            drift = compare(response, reference)
            report['max_margin_drift'] = max(report['max_margin_drift'], drift)
            report['variants'][name] = {**summary(samples[name]),
                'warmup_seconds': warmups[name], 'cache_status': 'hit'}
        save(args.output, report)
        print(json.dumps({'round': repetition + 1,
                         'seconds': {name: samples[name][-1] for name in variants}}), flush=True)


if __name__ == '__main__':
    main()
