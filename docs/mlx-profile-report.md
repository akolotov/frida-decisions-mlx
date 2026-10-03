# MLX FP32 profiling results

The saved 17-tag request drops from 2.239 seconds to 1.740 seconds with a state cache hit.
This reduces latency by 22.3% and beats the reproduced MPS baseline by 10.8%.
Peak MLX memory drops from 4.198 GiB to 3.864 GiB.
All tested decisions and ranking orders match the references.

FP32 means 32-bit floating point.
SDPA is a combined attention operation.
Compilation combines compatible array operations before execution.
A margin is a score for one candidate.
GiB means 2^30 bytes.
The selected MLX implementation uses fast SDPA and compiles both encoder paths.

## Conditions and baseline

The host is an Apple M4 Mac mini with 32 GiB memory.
It uses macOS 27.0.1, Python 3.12.12, MLX 0.32.3, and PyTorch 2.14.1.
The model revision is `0096b5384e821c68791cb2b3b7292c2b937dcec8`.
The baseline code revision is `6cf11648c71864d487dc8b0097313ab00ae656d8`.
The tag `profile-mlx-baseline-20261003` preserves that implementation.

A token is one text unit from the tokenizer.
The primary input is the saved `.ai/vorec-example/request.json` request.
Its state contains 267 tokens, and its 17 questions produce 34 candidates.
The request uses FP32, `state_max=512`, `rows_per_forward=1`, a 512 MiB cache budget, and the existing 0.5 threshold.
The state, descriptions, questions, tokenizer, model files, and packing remain unchanged.
The recorded SHA-256 hashes establish the exact input and model files.

The original full suite passes 69 tests and skips 7 tests.
The original MLX measurements reproduce the planned 2.238-second cache hit.
The reproduced MPS median is 1.950 seconds, close to the previous 1.951-second reference.
The raw baseline results remain in `.ai/profile/raw/baseline.json` and `.ai/profile/raw/mps.json`.
The production changes are separate commits: `c2682e4` for SDPA and `65f0b9d` for compilation.

## End-to-end measurements

The timers include tokenization, candidate construction, packing, model work, and answer aggregation.
Each timer starts after synchronization and stops after synchronization.
Two initial requests precede ten measured requests for each shape and cache mode.
Model loading, downloads, hashing, report writing, and compilation during initial requests remain outside repeated latency.
P10 is the 10th percentile.
P90 is the 90th percentile.

| Implementation | Median seconds | P10 to P90 seconds | Peak MLX GiB | GPU launches |
| --- | ---: | ---: | ---: | ---: |
| A: Original manual attention | 2.238533 | 2.237152 to 2.239789 | 4.198181 | 3006 |
| B: SDPA only | 1.874773 | 1.848564 to 1.897833 | 3.982845 | 2718 |
| C: Compilation only | 2.099461 | 2.089138 to 2.107553 | 4.194984 | 1842 |
| D: SDPA and compilation, production | 1.740165 | 1.739507 to 1.740572 | 3.864211 | 1554 |
| Fast RMSNorm only, rejected | 2.297938 | 2.295935 to 2.299170 | 4.165184 | 2124 |
| PyTorch MPS, original control | 1.949976 | 1.949263 to 1.950873 | N/A | N/A |

The following matrix uses milliseconds.
One millisecond is 0.001 seconds.
Miss and hit measurements explicitly select caching, including cases with one row.
The production policy still leaves a new request with one row uncached.
The baseline secondary results come from `baseline-secondary.json`, which exercises actual cache misses and hits.

| Request and cache mode | A | B | C | D | RMSNorm, rejected |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17-tag, disabled | 2863.5 | 2447.2 | 2579.6 | 2298.3 | 2885.0 |
| 17-tag, miss | 2434.5 | 2035.4 | 2230.1 | 1938.3 | 2494.5 |
| 17-tag, hit | 2238.5 | 1874.8 | 2099.5 | 1740.2 | 2297.9 |
| Support topic, disabled | 76.1 | 77.5 | 73.3 | 72.3 | 78.5 |
| Support topic, miss | 108.9 | 110.7 | 108.6 | 107.6 | 112.4 |
| Support topic, hit | 65.5 | 66.9 | 64.2 | 62.9 | 68.3 |
| 40 intents, disabled | 378.9 | 385.2 | 368.2 | 359.4 | 395.4 |
| 40 intents, miss | 467.5 | 469.0 | 458.0 | 443.9 | 487.2 |
| 40 intents, hit | 424.0 | 429.2 | 413.1 | 400.8 | 443.3 |
| Long state, disabled | 369.9 | 352.6 | 347.2 | 315.4 | 380.7 |
| Long state, miss | 379.3 | 371.7 | 388.9 | 362.9 | 393.8 |
| Long state, hit | 125.3 | 119.9 | 121.0 | 110.5 | 130.2 |

SDPA alone slightly slows the small support and intent cases.
Compilation alone slightly slows the long-state miss.
The selected combination improves every measured mode against its baseline.
The small changes near 2% remain sensitive to measurement variation.
The implementation uses the normal MLX dispatch and adds no shape threshold.

Later controls record MPS at 2.054 seconds and manual MLX at 2.316 seconds.
A second experiment rotates and reverses variant order across twelve measured rounds.
Its median seconds are 2.243 for A, 1.852 for B, 2.033 for C, and 1.644 for D.
That experiment shares immutable model weights and keeps four independent caches in one process.
Its allocation state differs from the separate-process matrix.
The separate-process result remains the reported production latency.

Latency rises in the final rounds of the rotating experiment.
These observations establish variation, but they do not establish its cause.
The selected combination wins throughout that experiment and beats both measured MPS controls.
Early diagnostic runs enabled Metal capture during latency measurement.
The clean SDPA rerun replaces those timings in the tables above.
All original samples remain available in the raw results.

## Where the time goes

An encoder transforms tokens into contextual model vectors.
The diagnostic run evaluates each projection and phase before its timer stops.
These evaluations change scheduling, so the following values describe diagnosis only.
The normal synchronized request timer determines the performance result.
The raw diagnostics cover disabled caching, misses, and hits for all four workloads.
FFN means the feed-forward part of an encoder layer.
K/V are the attention keys and values.
GELU is the encoder activation function.

| Phase in the original 17-tag cache hit | Diagnostic milliseconds |
| --- | ---: |
| FFN matrices and the activation gate | 957.3 |
| Attention projections and their output layout | 525.6 |
| Attention bias addition | 259.8 |
| GELU activation | 220.5 |
| Softmax | 182.0 |
| Attention score multiplication | 149.7 |
| Normalization, residuals, and initial embedding materialization | 126.9 |
| Attention layout, cached K/V copies, and bias materialization | 52.9 |
| Attention multiplication by values | 124.7 |
| Query encoder remainder | 17.1 |
| Packing | 7.6 |
| Tokenization | 2.4 |
| Candidate pooling | 1.3 |
| Decision head | 0.9 |
| Request parsing and candidate construction | 0.1 |
| Answer aggregation | 0.04 |

The FFN matrices and activation form the largest diagnostic cost.
Manual attention also consumes substantial time in bias addition, softmax, and matrix multiplication.
Host preparation and answer aggregation are small in this request.

A kernel is a GPU program.
The optional Objective-C++ counter intercepts actual Metal launches in a separate profiling process.
It records kernel names and counts between a start marker and a stop marker.
The following groups follow those names and the encoder structure.
The runtime does not load this counter.

| GPU work | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| Projection, FFN, and head matrix operations | 507 | 507 | 507 | 507 |
| Attention arithmetic and output layout | 360 | 72 | 360 | 72 |
| Normalization | 1029 | 1029 | 441 | 441 |
| GELU and its gate | 648 | 648 | 72 | 72 |
| Residual additions | 144 | 144 | 144 | 144 |
| Cached K/V concatenation copies | 288 | 288 | 288 | 288 |
| Embedding, bias construction, pooling, and other work | 30 | 30 | 30 | 30 |
| Total | 3006 | 2718 | 1842 | 1554 |

Compilation removes 1164 launches through normalization and activation fusion.
SDPA removes another 288 launches through combined attention and a different output layout.
The matrix operations and cached K/V copies remain substantial work after these changes.
The rejected RMSNorm experiment shows that fewer launches do not guarantee lower latency.

## Attention shapes and numerical behavior

Every measured attention shape uses batch size 1, 24 heads, head dimension 64, and FP32.
The additive mask has shape `[1, 24, Tq, Tkv]`.
Tq counts query tokens.
Tkv counts key and value tokens.
The primary state has 267 real tokens and a padded state width of 272.
Its three cached query rows each have width 957 and key width 1224.

The microbenchmark uses real first-layer projections and the real learned bias.
It evaluates Q, K, V, and bias before timing attention.
Thirty synchronized repetitions follow five initial calls for each operation and shape.
Rows with identical shapes share a measurement.
The table contains every distinct measured state, packed, and cached shape.

| Workload and path | Tq x Tkv | Manual ms | SDPA ms | Maximum absolute difference |
| --- | --- | ---: | ---: | ---: |
| 17-tag state | 272 x 272 | 1.141 | 0.753 | 1.07e-6 |
| 17-tag packed row 1 | 992 x 992 | 7.360 | 2.954 | 1.07e-6 |
| 17-tag packed row 2 | 1008 x 1008 | 7.570 | 3.148 | 1.07e-6 |
| 17-tag packed row 3 | 1016 x 1016 | 7.734 | 3.171 | 1.79e-6 |
| 17-tag packed row 4 | 864 x 864 | 5.672 | 2.346 | 1.19e-6 |
| 17-tag cached queries | 957 x 1224 | 8.830 | 3.504 | 1.07e-6 |
| Support state | 32 x 32 | 0.255 | 0.231 | 4.77e-7 |
| Support packed row | 120 x 120 | 0.320 | 0.295 | 4.77e-7 |
| Support cached queries | 93 x 120 | 0.308 | 0.285 | 4.17e-7 |
| 40-intent state | 24 x 24 | 0.250 | 0.217 | 2.38e-7 |
| 40-intent packed row 1 | 232 x 232 | 0.636 | 0.473 | 5.96e-7 |
| 40-intent packed row 2 | 208 x 208 | 0.579 | 0.414 | 4.77e-7 |
| 40-intent packed row 3 | 144 x 144 | 0.399 | 0.309 | 4.17e-7 |
| 40-intent cached queries | 214 x 232 | 0.612 | 0.443 | 5.96e-7 |
| Long-state encoding | 376 x 376 | 1.404 | 0.733 | 1.01e-6 |
| Long-state packed row | 552 x 552 | 2.604 | 1.276 | 1.07e-6 |
| Long-state cached queries | 180 x 552 | 1.079 | 0.616 | 9.54e-7 |

The primary cached shape improves by about 2.5 times in isolation.
Its maximum relative output difference is 0.01166 with denominator `max(abs(reference), 1e-12)`.
Across all shapes, the largest absolute difference is 1.79e-6 and the largest relative difference is 0.07192.
The relative maxima occur near zero, so request margins provide the release gate.
The long-state regression case contains 372 real state tokens under the unchanged 384-token cap.

FRIDA uses `scale=1.0` and preserves the supplied additive mask.
MLX supports this mask and computes the SDPA softmax in FP32. See the [MLX SDPA documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html).
The implementation leaves `force_fused` disabled.
The forced primary kernel measures 3.486 ms against the normal 3.504 ms, below a repeatable 2% benefit.
Some short shapes slow down with forced selection.

## Compilation and rejected changes

Compilation covers the complete packed encoder and the complete cached query encoder.
Each function includes embedding, bias construction, every encoder layer, and the final normalization.
State encoding remains eager, which means direct execution without compilation.
Candidate pooling and the decision head retain FP32 outside those functions.
The GELU tanh formula and T5 normalization retain their original operations and casts.

The primary packed request creates four function versions for widths 992, 1008, 1016, and 864.
The cached primary request creates one version for its three equal row shapes.
Miss and hit reuse that same cached version.
The complete benchmark creates nine packed versions and four cached versions across four workloads.
Repeated measurements create no additional versions.

The first compiled packed request takes 2.634 seconds, and the second takes 2.581 seconds in the compile-only experiment.
Those initial calls include compilation and kernel initialization.
The code warms every measured shape before repeated timing.
The first hit-mode setup call also populates the cleared state cache, so it includes a miss.
The existing stable shapes do not justify `shapeless=True`. See the [MLX compilation documentation](https://ml-explore.github.io/mlx/build/html/usage/compile.html).

RMSNorm normalizes vectors with their mean squared value.
The installed `nn.RMSNorm` already calls `mx.fast.rms_norm`.
FRIDA uses a separate `T5Norm` to preserve its precision order.
An isolated FP32 fast RMSNorm experiment passes numerical tests but takes 2.298 seconds and slows the secondary workloads.
That experiment remains available only in profiling tools.

An isolated FP32 cast creates a new Python object but launches zero GPU kernels.
Active MLX memory remains 262144 bytes before and after that cast.
The existing casts remain because BF16 needs their rounding order.
Encoder intermediates stay in MLX arrays.
The existing evaluation and list conversion at each row boundary preserve the memory limit.
The measurements do not justify further changes to those boundaries.

## Correctness and memory

The final full suite passes 69 tests and skips 7 tests.
The skips cover optional word restrictions, unavailable ONNX graphs, and the unavailable original adapter reference.
Every independent experiment also passes the existing suite.
The additional parity run covers ten requests and 32 distinct decisions through packed, miss, and hit paths.
That produces 96 decision comparisons per variant.

| Variant | Maximum margin drift against prior MLX | Maximum margin drift against CPU FP32 | Decision mismatches |
| --- | ---: | ---: | ---: |
| SDPA | 1.1444e-5 | 1.9670e-5 | 0 |
| Compilation | 1.1921e-5 | 2.1815e-5 | 0 |
| Final production | 1.0967e-5 | 2.1994e-5 | 0 |
| RMSNorm, rejected | 1.4067e-5 | 2.1517e-5 | 0 |

Every output is finite, and complete ranking orders match.
The final cache paths differ from the packed path by at most 1.3351e-5.
The maximum final FP32 drift remains far below the 1e-3 limit and within the earlier numerical range.
The clean `.venv-mlx-only` environment contains no PyTorch installation and imports no PyTorch module.
Two calls to the exact saved request succeed with backend `mlx`.
The first call records a cache miss, and the second records a cache hit.
The repeated margin difference in that clean run is 2.3842e-7.

The exact token tuple remains the cache key.
LRU removes the least recently used cache entry.
The budget, eviction rules, disabled mode, and selection policy remain unchanged.
The primary cache entry retains 75.094 MiB in native MLX arrays.
The final peak is 3.864 GiB against the original 4.198 GiB, a 7.96% reduction.
The MLX memory counter excludes Python and NumPy allocations.

Two earlier test processes pass their assertions but abort during shutdown.
The macOS crash stacks identify the ONNX Runtime telemetry HTTP worker.
The profiling test runner disables that unrelated telemetry and completes with exit code zero.
The production backend adds no ONNX dependency or shutdown workaround.

## GPU captures and reproduction

The warm captures contain one representative 17-tag cache hit for each variant.
They use `MTL_CAPTURE_ENABLED=1` and `mx.metal.start_capture`, followed by synchronization before capture stops.
Their files remain under `.ai/profile/raw/*-capture.gputrace`.
Metadata and kernel names were inspected, and live Metal counters establish the launch counts above.
Xcode and `xctrace` are absent on this host, so Xcode replay and individual GPU timestamps remain unverified.
The phase table supplies synchronized diagnostic wall times, without a claim of individual GPU timestamps.
The saved captures, live kernel counts, diagnostic timings, isolated benchmarks, and correctness results establish the selected optimization.
Xcode replay remains optional research into the remaining latency.

Use the existing `.venv` development environment on a Mac with Metal access.
Make sure that `_export/FRIDA-Decisions` contains the released model and tokenizer files.
The private saved request remains outside version control.
Run the following commands from the repository root.
The first command benchmarks production, and the second compares twelve rounds of the four variants.

```bash
.venv/bin/python tools/profile_mlx.py --variant production \
  --real-request .ai/vorec-example/request.json --runs 10 \
  --output tests/_results/profile-production.json
.venv/bin/python tools/benchmark_mlx_variants.py \
  --real-request .ai/vorec-example/request.json --runs 12 \
  --output tests/_results/profile-variants.json
```

The profiler also accepts `--variant baseline`, `sdpa`, `compile`, and `rms` for isolated experiments.
Use `--task phases` for synchronized diagnostics.
Use `--task attention --runs 30` for attention measurements.
A capture path must be new for each capture.
Run the following capture command with Metal capture enabled.
Open the resulting trace in Xcode on a host that provides Xcode. See the [MLX Metal capture instructions](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html).

```bash
MTL_CAPTURE_ENABLED=1 .venv/bin/python tools/profile_mlx.py \
  --variant production --task capture --primary-only \
  --real-request .ai/vorec-example/request.json \
  --output tests/_results/profile-capture.json
```

The optional counter uses the installed Apple command line compiler.
It changes only its own profiling process.
Compile the host counter before the kernel measurement.
Use these commands on a Mac with that compiler and Metal access.

```bash
clang++ -std=c++17 -dynamiclib -fobjc-arc -framework Foundation \
  -framework Metal tools/metal_kernel_counts.mm \
  -o /private/tmp/frida-metal-counts.dylib
.venv/bin/python tools/profile_mlx.py --task kernels --variant production \
  --primary-only --real-request .ai/vorec-example/request.json \
  --counter-library /private/tmp/frida-metal-counts.dylib \
  --output tests/_results/profile-kernels.json
```

Set `FD_MLX_REAL_REQUEST` to the private request for its parity test.
Set `FD_MLX_ONLY_PYTHON` to the existing interpreter without PyTorch.
Run the complete production suite with the following command.
The test runner disables optional ONNX telemetry before test collection.
Set `compile_encoder=False` when a caller needs direct encoder execution without compilation.

```bash
FD_MLX_REAL_REQUEST=.ai/vorec-example/request.json \
FD_MLX_ONLY_PYTHON="$PWD/.venv-mlx-only/bin/python" \
  .venv/bin/python tools/test_mlx_variant.py --variant production -q
```

The [published measurements](mlx-profile-results.json) contain raw latency samples, diagnostics, hashes, launch counts, and numerical results without private text or tag names.
The complete local artifacts also preserve failed process logs and the original capture-enabled timing runs.
The selected implementation is faster than MPS for this saved request on the tested Mac.
Its remaining cost lies mainly in matrix work and the preserved cached K/V copies.
These measurements do not establish performance for other models, devices, or request shapes.
