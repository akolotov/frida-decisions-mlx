"""Packed FRIDA inference with MLX on Apple Silicon."""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from .base import BaseJudge, resolve_model_dir
from .constants import DEFAULT_REPO_ID
from .mlx_modeling import FridaMlxDecisionModel
from .packing import TokenizedRequest, build_rows, pack_rows

_MLX_FILES = ['*.json', 'model.safetensors', 'head.safetensors']


class MlxJudge(BaseJudge):
    """The shared decision API with a native MLX encoder and FP32 head."""

    backend = 'mlx'

    def __init__(self, folder: Path, encoder: FridaMlxDecisionModel,
                 state_max: int | None = None, rows_per_forward: int | None = 1):
        super().__init__(folder, state_max)
        if rows_per_forward is not None and (not isinstance(rows_per_forward, int)
                                             or rows_per_forward < 1):
            raise ValueError('rows_per_forward must be a positive integer or None')
        self.model = encoder
        self.dtype = encoder.embed.weight.dtype
        self.rows_per_forward = rows_per_forward

    @classmethod
    def from_pretrained(cls, path_or_repo: str | Path = DEFAULT_REPO_ID,
                        dtype=mx.float32, state_max: int = 384,
                        rows_per_forward: int | None = 1,
                        revision: str | None = None) -> 'MlxJudge':
        """Load original released weights. FP32 is the default encoder precision.

        dtype accepts mx.float32 or mx.bfloat16. The head stays in FP32.
        rows_per_forward bounds the attention memory. None uses all rows.
        revision pins a Hugging Face revision. Local folders also work.
        """
        folder = resolve_model_dir(path_or_repo, _MLX_FILES, revision)
        return cls(folder, FridaMlxDecisionModel.from_folder(folder, dtype),
                   state_max, rows_per_forward)

    def _score(self, requests: list[TokenizedRequest]):
        return self._score_packed(requests)

    def _score_packed(self, requests: list[TokenizedRequest]):
        rows, per_request = build_rows(requests, self.config)
        if not rows:
            return [[] for _ in requests], {'rows': 0, 'encoder_tokens': 0, 'state_cache': 'off'}
        step = self.rows_per_forward or len(rows)
        margins, tokens = [], 0
        for i in range(0, len(rows), step):
            batch = pack_rows(rows[i:i + step], self.config)
            margins += self._forward_packed(batch)
            tokens += sum(batch.lengths)
        out, start = [], 0
        for n in per_request:
            out.append(margins[start:start + n])
            start += n
        return out, {'rows': len(rows), 'encoder_tokens': tokens, 'state_cache': 'off'}

    def _forward_packed(self, batch) -> list[float]:
        hidden = self.model(mx.array(batch.input_ids, dtype=mx.int32),
                            mx.array(batch.buckets, dtype=mx.int32), mx.array(batch.allowed))
        r = batch.readout
        margins = self.model.margins(hidden, mx.array(r.row, dtype=mx.int32),
                                     mx.array(r.col, dtype=mx.int32),
                                     mx.array(r.slot, dtype=mx.int32), r.count)
        mx.eval(margins)
        return margins.tolist()

    def margins_packed(self, request: dict) -> list[float]:
        """Return margins in candidate order through the packed encoder."""
        return self._score_packed([self.compile(request)[2]])[0][0]
