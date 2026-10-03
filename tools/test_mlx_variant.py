"""Run the existing suite against one isolated MLX experiment."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['sdpa', 'compile', 'sdpa-compile', 'rms', 'production'], required=True)
    args, pytest_args = parser.parse_known_args()
    try:
        import onnxruntime
        # macOS crash reports identify ONNX telemetry's HTTP response worker
        # during teardown. Keep unrelated network telemetry out of this suite.
        onnxruntime.disable_telemetry_events()
    except ImportError:
        pass
    import pytest
    if args.variant == 'production':
        raise SystemExit(pytest.main(pytest_args or ['-q']))
    from frida_decisions import MlxJudge
    from mlx_experiments import attention_path, manual_attention, fast_attention, fast_norm_path, CompiledModel

    init = MlxJudge.__init__

    def initialize(self, *args, **kwargs):
        kwargs['compile_encoder'] = False
        init(self, *args, **kwargs)
        if 'compile' in args_variant:
            self.model = CompiledModel(self.model)
        self._forward_encoder = self.model.__call__
        self._forward_cached_encoder = self.model.forward_cached

    args_variant = args.variant
    with attention_path(fast_attention if args.variant.startswith('sdpa') else manual_attention), \
            patch.object(MlxJudge, '__init__', initialize):
        with ExitStack() as norms:
            if args.variant.endswith('rms'):
                norms.enter_context(fast_norm_path())
            raise SystemExit(pytest.main(pytest_args or ['-q']))


if __name__ == '__main__':
    main()
