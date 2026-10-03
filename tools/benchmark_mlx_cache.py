"""Compare exact state caching with packed inference on saved requests."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import mlx.core as mx
import numpy as np

from frida_decisions import MlxJudge
from frida_decisions.mlx_modeling import StateCache
from frida_decisions.protocol import decision


class ExplicitCachedJudge(MlxJudge):
    """Measure the explicit cached path even for a first single-row request."""

    def use_state_cache(self, requests):
        return self.state_cache is not None and len(requests) == 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('_export/FRIDA-Decisions'))
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--real-request', type=Path, help='Path to an optional private request JSON file')
    parser.add_argument('--output', type=Path, default=Path('tests/_results/mlx-cache-benchmark.json'))
    args = parser.parse_args()
    if args.runs < 5:
        parser.error('--runs must be at least five')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'tests'))
    from conftest import load_cases
    cases = load_cases()
    selected = [next(c for c in cases if c['name'] == name) for name in (
        'choice/support-topic', 'mixed/ticket-triage',
        'generated/intent-catalog-40', 'generated/long-state')]
    real, request_sha256 = None, None
    schedule = [(1, selected)]
    if args.real_request is not None:
        request_bytes = args.real_request.read_bytes()
        real = {'name': 'private/real-request', 'request': json.loads(request_bytes)}
        request_sha256 = hashlib.sha256(request_bytes).hexdigest()
        selected.insert(0, real)
        schedule.append((None, [real]))
    judge = ExplicitCachedJudge.from_pretrained(args.model, state_max=512, state_cache_mb=0)
    report = {'environment': {'machine': platform.machine(), 'platform': platform.platform(),
        'python': platform.python_version(), 'mlx': importlib.metadata.version('mlx')}, 'runs': args.runs,
        'dtype': 'float32', 'request_sha256': request_sha256,
        'model_sha256': {p.name: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest() for p in args.model.glob('*.safetensors')},
        'scenarios': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for step, selected_cases in schedule:
        judge.rows_per_forward = step
        for case in selected_cases:
            req = case['request']
            judge.state_max = 512 if case is real else 384
            counts = judge.token_counts(req)
            row = {'name': case['name'], 'rows_per_forward': step, 'counts': counts,
                   'questions': len(req['questions']), 'state_max': judge.state_max, 'modes': {}}
            reference = None
            for mode in ('off', 'miss', 'hit'):
                judge.state_cache = None if mode == 'off' else StateCache()
                mx.synchronize()
                mx.clear_cache()
                if mode == 'hit':
                    judge(req)
                # Warm kernels before measurements, without warming miss entries.
                judge(req)
                times, peaks, usages = [], [], []
                for _ in range(args.runs):
                    if mode == 'miss':
                        judge.state_cache.clear()
                    mx.synchronize()
                    mx.reset_peak_memory()
                    started = time.perf_counter()
                    response = judge(req)
                    mx.synchronize()
                    times.append(time.perf_counter() - started)
                    peaks.append(mx.get_peak_memory())
                    usages.append(response['usage']['state_cache'])
                    margins = np.array([v for q in response['margins'].values() for v in q.values()])
                    decisions = {q: decision(a) for q, a in response['answers'].items()}
                    if reference is None:
                        reference = (margins, decisions)
                    assert decisions == reference[1]
                    assert np.isfinite(margins).all()
                    assert float(np.max(np.abs(margins - reference[0]))) <= 1e-3
                row['modes'][mode] = {'median_seconds': statistics.median(times),
                    'seconds': times, 'peak_mlx_bytes': max(peaks), 'cache_status': usages,
                    'rows': response['usage']['rows'], 'encoder_tokens': response['usage']['encoder_tokens'],
                    'retained_cache_bytes': judge.state_cache.bytes if judge.state_cache is not None else 0,
                    'max_margin_drift': float(np.max(np.abs(margins - reference[0])))}
            report['scenarios'].append(row)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(row), flush=True)
    print(f'Results: {args.output}', flush=True)


if __name__ == '__main__':
    main()
