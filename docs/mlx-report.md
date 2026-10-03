# MLX engineering report

The native MLX backend loads the original released files and executes all four decision types without PyTorch.
The implementation reuses `BaseJudge`, tokenization, packing, protocol parsing, and answer aggregation.
FP32 (32-bit floating point) is the default encoder precision.
BF16 (16-bit floating point) is available explicitly.
The decision head and candidate pooling use FP32 in both modes.

## Changes

`frida_decisions/mlx_modeling.py` adds the encoder, supplied attention bias, T5 normalization, gated `gelu_new`, candidate pooling, and strict weight loading.
`frida_decisions/mlx_backend.py` adds `MlxJudge` and memory control through `rows_per_forward`.
`frida_decisions/__init__.py` exposes the backend lazily, and `pyproject.toml` adds the MLX extra without PyTorch.
The README and `examples/mlx_quickstart.py` describe the API.
The three MLX test files cover loading, arithmetic, request parity, batches, and inference without PyTorch.
`tools/benchmark_mlx.py` measures load time, latency, and memory, and `.gitignore` excludes the separate environment.

## Correctness

The reference uses the released weights upcast to FP32 with PyTorch on the CPU.
Both backends use the same token IDs, candidate order, supplied position buckets, and visibility masks.
Tests compare response structure and decisions, including the complete ranking order.
The regression set contains seven repository requests and two generated requests, including English, Russian, mixed questions, and multiple packed rows.

| Measurement | MLX FP32 | MLX BF16 |
| --- | ---: | ---: |
| Parity requests | 9 | 9 |
| Decisions compared | 15 | 15 |
| Maximum absolute margin difference | 2.413988113e-05 | 0.07992219925 |
| Decision mismatches | 0 | 0 |

All outputs are finite.
The FP32 difference is below the 0.001 target and the 0.01 release limit.
BF16 preserves these decisions but produces larger numerical differences.
FP32 is the default because it gives closer agreement at a modest latency cost.

The intermediate comparison uses the first repository request.
Embeddings and attention bias match exactly.
Selected residual states reach approximately 725,000, with maximum absolute differences of 0.75 and relative differences near 0.000001.
Final normalization reduces the maximum hidden-state difference to 0.00000268.
Pooled vectors differ by at most 0.000000417, and head margins differ by at most 0.00000483.
Tests independently compare `gelu_new` and normalization with Hugging Face implementations.

The existing suite passed 40 tests and skipped 7 tests because optional inputs were absent.
Those inputs are the original adapter reference, the ONNX export, and the forbidden-word list.
All 19 MLX tests passed, including the real request in the separate environment.
The original CPU quickstart also passed, including its 243-intent catalog.

## Native runtime

The separate environment contains the package and its MLX extra, with no PyTorch, torchvision, or torchaudio installation.
A real request selects `port` and reports backend `mlx`.
The example asserts that no PyTorch module or `frida_decisions.torch_backend` enters the process.
The backend performs inference directly in MLX.

| Runtime requirement | Result |
| --- | --- |
| PyTorch installed | No |
| PyTorch imported | No |
| Reported backend | mlx |
| Real request succeeded | Yes |

From the repository root on the tested Apple Silicon Mac, reproduce the isolated test:

```bash
python3 -m venv .venv-mlx-only
.venv-mlx-only/bin/python -m pip install -e '.[mlx]'
.venv-mlx-only/bin/python examples/mlx_quickstart.py --assert-no-torch \
  --revision 0096b5384e821c68791cb2b3b7292c2b937dcec8
```

The recorded run used `--model _export/FRIDA-Decisions` with the same original files already downloaded.
The encoder file SHA256 is `70a8cb915323c072f549f46b0f89c57a83b54128458e3cc7178a7bc3c013de5f`.

## Environment

Tests ran on a Mac mini, model Mac16,10, with an Apple M4 and 32 GB memory.
The host uses macOS 27.0.1 and Python 3.12.12.
The development environment uses MLX 0.32.3, PyTorch 2.14.1, and Transformers 5.18.0.
The clean runtime environment uses MLX 0.32.3 and no PyTorch or Transformers.

The FRIDA code baseline is v0.2.0, commit `67379d37b7cb2989d9bfaac6831b08d088681cb5`.
The Hugging Face model revision is `0096b5384e821c68791cb2b3b7292c2b937dcec8`.
The official MLX example reference is commit `796f5b53cab69a3d48a44233ce21aae889e94a08`.
Upstream `main` at `e968beb567cc99c392ce3953ff445cd202a1d096` changes documentation and author metadata, with no relevant inference changes.

## Performance

Measurements use the Apple M4 GPU and evaluate all lazy MLX operations before timing ends.
Each repeated latency is the median of five requests after the first request.
Each encoder call uses one packed row.
Peak MLX memory includes model weights and temporary arrays, but excludes Python and NumPy allocations.
These measurements run separately from the test suite.

Fresh-process model load took 0.207 seconds in FP32 and 0.131 seconds in BF16.
The filesystem cache was warm, and these times exclude downloading files.
Single-request latency includes tokenization, packing, inference, and aggregation.

| Request | State tokens | Questions | Candidates | Rows | Precision | First request ms | Repeated ms | Peak MLX GiB |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| choice/support-topic | 27 | 1 | 4 | 1 | float32 | 85.4 | 76.1 | 3.323 |
| mixed/ticket-triage | 44 | 4 | 13 | 1 | float32 | 220.8 | 217.6 | 3.642 |
| generated/intent-catalog-40 | 18 | 1 | 40 | 3 | float32 | 385.4 | 378.0 | 3.484 |
| generated/long-state | 372 | 3 | 8 | 1 | float32 | 375.5 | 368.8 | 3.839 |
| choice/support-topic | 27 | 1 | 4 | 1 | bfloat16 | 77.7 | 63.5 | 1.730 |
| mixed/ticket-triage | 44 | 4 | 13 | 1 | bfloat16 | 185.6 | 182.7 | 2.004 |
| generated/intent-catalog-40 | 18 | 1 | 40 | 3 | bfloat16 | 324.2 | 320.4 | 1.863 |
| generated/long-state | 372 | 3 | 8 | 1 | bfloat16 | 316.0 | 312.5 | 2.156 |

After installing the development dependencies, reproduce parity and performance from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,mlx]'
FD_MODEL_DIR="$PWD/_export/FRIDA-Decisions" \
FD_MLX_ONLY_PYTHON="$PWD/.venv-mlx-only/bin/python" \
.venv/bin/python -m pytest tests/test_mlx.py tests/test_mlx_parity.py tests/test_mlx_runtime.py -q
.venv/bin/python tools/benchmark_mlx.py --model _export/FRIDA-Decisions --runs 5
.venv/bin/python tools/benchmark_mlx.py --model _export/FRIDA-Decisions --dtype bfloat16 --runs 5
```

The parity suite requires original model files in `_export/FRIDA-Decisions`, or the folder named by `FD_MODEL_DIR`.
Local measurements and logs remain in the ignored `tests/_results/` directory.

## Limitations

These baseline measurements predate the state cache.
The backend now supports exact state reuse.
See [the state cache report](mlx-cache-report.md) for current correctness and performance results.
The initial implementation uses explicit matrix operations without compiled graphs, fused attention, quantization, or custom Metal kernels.
BF16 produces the recorded numerical differences, and decision agreement outside this regression set is untested.
Older MLX versions and other Mac models are untested.
The optional extra requires the tested MLX version, 0.32.3, or newer.
