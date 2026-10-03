"""Measure evaluated MLX inference for four regression request shapes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import mlx.core as mx

from frida_decisions import MlxJudge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ai-forever/FRIDA-Decisions')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error('--runs must be positive')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'tests'))
    from conftest import load_cases
    cases = load_cases()
    selected = [cases[0], next(c for c in cases if c['name'].startswith('mixed/')),
                cases[-2], cases[-1]]
    started = time.perf_counter()
    judge = MlxJudge.from_pretrained(args.model, dtype=getattr(mx, args.dtype))
    mx.synchronize()
    result = {'dtype': args.dtype, 'cold_load_seconds': time.perf_counter() - started,
              'rows_per_forward': judge.rows_per_forward, 'scenarios': []}
    for case in selected:
        mx.clear_cache()
        mx.reset_peak_memory()
        started = time.perf_counter()
        response = judge(case['request'])
        mx.synchronize()
        first = (time.perf_counter() - started) * 1000
        times = []
        for _ in range(args.runs):
            started = time.perf_counter()
            judge(case['request'])
            mx.synchronize()
            times.append((time.perf_counter() - started) * 1000)
        counts = judge.token_counts(case['request'])
        row = {'name': case['name'], 'state_tokens': counts['state_tokens'],
               'questions': len(case['request']['questions']), 'candidates': counts['candidates'],
               'packed_rows': counts['packed_rows'], 'dtype': args.dtype,
               'first_request_ms': first, 'repeated_median_ms': statistics.median(times),
               'peak_mlx_bytes': mx.get_peak_memory()}
        result['scenarios'].append(row)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + '\n')


if __name__ == '__main__':
    main()
