# Exact MLX state cache

The native MLX backend now reuses state tensors across rows and later requests with identical state token IDs.
The cache preserves all tested decisions.
The real 17-tag request runs faster, but the short-state 40-intent request runs slower.

## Changes

`frida_decisions/mlx_modeling.py` adds `encode_state`, `forward_cached`, and `StateCache`.
K/V tensors are the attention keys and values.
Each layer stores only the real state tokens in native MLX arrays.
The final layer produces state K/V without state attention or feed-forward work.
An empty state produces empty K/V arrays because MLX cannot evaluate these projection kernels with zero tokens.
The packed encoder, masks, position buckets, normalization, activation, candidate pooling, and decision head retain their existing computation.

`StateCache` uses the exact token tuple as its key.
LRU eviction removes the least recently used entry first.
The cache counts hits, misses, evictions, and retained bytes.
Tensor `nbytes` determines storage, including the actual dtype.
An oversized state is usable for the current request but does not enter the cache.
`clear()` removes entries and resets retained bytes, while diagnostic counters remain cumulative.

`frida_decisions/mlx_backend.py` adds automatic selection and row-chunk evaluation.
The default cache budget is 512 MiB, and `state_cache_mb=0` disables automatic caching.
A new single-row state uses packed inference and does not enter the cache.
A multi-row request encodes its state once.
A later single request with an existing state entry reuses it, even with different questions or candidates.
Batches with multiple requests retain packed inference.

The response reports `state_cache` as `off`, `miss`, or `hit`.
Its `rows` count describes the path that actually runs.
Its `encoder_tokens` count includes state tokens only on a miss.
The default `rows_per_forward=1` also limits cached inference, and `None` evaluates all query rows together.
The optional cache argument follows existing positional arguments to preserve existing calls.

`tests/test_mlx_cache.py` covers exact reuse, changed tokens, different questions, disabled mode, eviction, empty states, row limits, and released weights.
`examples/mlx_quickstart.py` also exercises a miss and an automatic hit in the clean runtime.
`tools/benchmark_mlx_cache.py` records synchronized measurements for four public regression shapes.
The optional `--real-request` argument adds a private request.
The older benchmark script defaults to cache-disabled inference to preserve its baseline behavior.
The local `.ai/vorec-example/analyze.py` also explicitly disables the cache for historical reproduction.

## Parity

FP32 means 32-bit floating point.
The recorded cache comparison uses nine public regression requests plus the private 17-tag request.
The public cache test runs independently of the private request.
It compares candidate margins in their original order, final decisions, and complete ranking order.
A margin is the decision head score for one candidate.
All measured outputs are finite.

```text
cached vs packed requests: 10
decisions compared: 32
max FP32 margin drift: 1.192092896e-05
decision mismatches: 0
```

The largest drift across the additional benchmark runs is 1.215934753e-05.
Both results stay below the 0.001 limit.
The cache test uses `state_max=512`, while the external regression suite retains its default of 384.

The existing PyTorch CPU FP32 comparison also passes on 9 requests and 15 decisions.
Its maximum FP32 drift is 2.408027649e-05, with zero decision mismatches.
The BF16 comparison also preserves all tested decisions.
BF16 means 16-bit floating point with a wide exponent range.
Its maximum drift from PyTorch FP32 is 0.09635639191, so BF16 does not meet the FP32 numerical limit.

The clean `.venv-mlx-only` runtime contains no PyTorch installation.
The real request succeeds with backend `mlx`, and no PyTorch module enters the process.
The repeated request reports a hit, and the retained tensors are native MLX arrays.

## Cache behavior

```text
released parity cache hits: 10
released parity cache misses: 10
evictions tested: yes
disabled mode tested: yes
clear tested: yes
changed state tested: yes
state encoding count tested: yes
```

Instrumentation observes one state encoding across a miss and a hit.
Small tests also change the questions while retaining the exact same state.
The deterministic eviction test fills a 64-byte cache, accesses the older entry, and inserts a third entry.
The cache evicts the entry that now has the oldest access.
It also rejects an oversized entry and tests replacement accounting.

## Real workload

The recorded benchmark uses the private 17-tag request from `.ai/vorec-example/request.json`.
The repository does not include this request.
Use `--real-request` to supply a local copy when you have access to the private data.
The saved summary, tag descriptions, candidate wording, model weights, and FP32 dtype remain the same.
The model file hashes match the original tagging experiment.
The fixed threshold remains 0.5, and FRIDA still selects only `mira`.
This optimization does not fix the missed `bedtime` tag.

```text
state tokens: 267
questions: 17
candidates: 34
packed rows: 4
cached query rows: 3

cache disabled: 2.864617 seconds
cache miss: 2.434070 seconds
cache hit: 2.243905 seconds

speedup on miss: 1.177x
speedup on hit: 1.277x
latency reduction on miss: 15.0%
latency reduction on hit: 21.7%
peak memory delta on miss or hit: 163.14 MiB
retained state cache: 75.09 MiB
```

This direct comparison uses the existing default, `rows_per_forward=1`.
The disabled result reproduces the earlier 2.867-second median within 0.003 seconds.
The cache reduces repeated encoder tokens from 3868 to 2698.
A miss processes 2965 encoder tokens, including the state once.
The query rows decrease from four to three because their token budget excludes the cached state.

## Other benchmarks

All latency values below are median milliseconds across five measured runs.
Every row uses MLX FP32 on the Apple M4.
The final row is a separate experiment with `rows_per_forward=None`.
The default remains one row per forward, so the all-row experiment does not replace the direct comparison.

| Request | State tokens | Questions | Candidates | Packed / cached rows | Disabled ms | Miss ms | Hit ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vorec/tags-17 | 267 | 17 | 34 | 4 / 3 | 2864.6 | 2434.1 | 2243.9 |
| choice/support-topic | 27 | 1 | 4 | 1 / 1 | 75.8 | 109.0 | 65.5 |
| mixed/ticket-triage | 44 | 4 | 13 | 1 / 1 | 217.5 | 231.7 | 183.3 |
| generated/intent-catalog-40 | 18 | 1 | 40 | 3 / 3 | 379.0 | 467.4 | 425.1 |
| generated/long-state | 372 | 3 | 8 | 1 / 1 | 370.3 | 379.1 | 125.7 |
| vorec/tags-17 (all rows) | 267 | 17 | 34 | 4 / 3 | 2996.6 | 2442.7 | 2252.2 |

The 40-intent request contains only 18 state tokens.
Its cache miss takes 1.23 times the disabled latency.
Its cache hit takes 1.12 times the disabled latency.
Saved state computation does not offset the cached attention overhead for this workload.
The cache policy still follows the existing PyTorch policy, rather than adding a new workload heuristic.

The long-state hit runs 2.95 times faster than disabled inference.
Its miss has little benefit because that request occupies only one row.
Single-row miss and hit measurements explicitly select the cached path through a benchmark subclass.
The subclass changes only path selection and uses the same inference and response code.
These measurements describe a previously seeded cache entry.
Automatic single-row requests with a new state remain packed, including later repeats unless another eligible request seeds the cache.

Peak memory includes model weights, retained state tensors, and temporary MLX arrays.
It excludes Python and NumPy allocations.
GiB and MiB use powers of 1024.

| Request | Disabled peak GiB | Miss peak GiB | Hit peak GiB | Retained state MiB |
| --- | ---: | ---: | ---: | ---: |
| vorec/tags-17 | 4.039 | 4.198 | 4.198 | 75.09 |
| choice/support-topic | 3.323 | 3.270 | 3.270 | 7.59 |
| mixed/ticket-triage | 3.642 | 3.624 | 3.624 | 12.38 |
| generated/intent-catalog-40 | 3.484 | 3.453 | 3.453 | 5.06 |
| generated/long-state | 3.839 | 3.768 | 3.602 | 104.62 |
| vorec/tags-17 (all rows) | 5.168 | 4.923 | 4.923 | 75.09 |

## Reproduction

Run these commands from `/Users/agent/projects/mlx-frida-decisions` on an Apple Silicon Mac with GPU access.
The existing environments and local released model files supply the dependencies.
The timer includes tokenization, row layout, encoder evaluation, and response aggregation.
Model loading, file hashing, and downloads do not enter the timer.
Each mode warms the model kernels before five measured calls.
Every miss measurement clears the state entries before its request, while hit measurements retain them.
MLX synchronization completes GPU work before the timer stops.
Tests and benchmarks run separately.

```bash
FD_MLX_ONLY_PYTHON="$PWD/.venv-mlx-only/bin/python" .venv/bin/python -m pytest -q -rs
.venv/bin/python tools/benchmark_mlx_cache.py --model _export/FRIDA-Decisions --runs 5
```

These commands do not need `.ai/vorec-example`.
The benchmark runs four public scenarios and writes new results to `tests/_results/mlx-cache-benchmark.json` by default.
The private cache test skips when `FD_MLX_REAL_REQUEST` is not set.
If you set that variable, the test reads the named file and reports an error if the file is absent.

If you have access to the private request, set `FD_MLX_REAL_REQUEST` to its local path before the private test.
Supply the same path through `--real-request` for the private benchmark.
The benchmark includes the private request with both row limits and labels it `private/real-request`.

```bash
FD_MLX_REAL_REQUEST="/path/to/private/request.json" \
  .venv/bin/python -m pytest tests/test_mlx_cache.py::test_private_cache_parity -q -rs
.venv/bin/python tools/benchmark_mlx_cache.py --model _export/FRIDA-Decisions --runs 5 \
  --real-request "/path/to/private/request.json"
```

Raw measurements appear in [mlx-cache-benchmark.json](mlx-cache-benchmark.json).
Per-request numerical results and runtime evidence appear in [mlx-cache-validation.json](mlx-cache-validation.json).
These saved results include private measurements that require the original request to reproduce.
The recorded full suite passed 68 tests and skipped seven tests before the private cache test became a separate test.
The skipped inputs are the forbidden-word list, ONNX graph, and original adapter reference.
All MLX cache, external parity, batch, loading, and clean-runtime tests pass.

## Remaining limitations

The 40-intent workload slows down despite exact reuse.
The cache budget limits retained state storage, rather than the entire process or temporary inference storage.
Multiple-request batches do not use the state cache.
The cache key uses the truncated tokenized state, as in PyTorch, rather than the original text bytes.
The cache assumes that model weights and dtype remain fixed for the judge lifetime.
The cache has no concurrency controls, as in the existing PyTorch implementation.

Older MLX versions, other Mac models, and classification quality outside these requests remain untested.
No quantization, compiled graphs, custom kernels, thresholds, or request semantics change in this task.
