"""Torch-backed adapter serving a JointIntentSlotModel over the protocol.

The only adapter that runs real inference: log-mel front end, causal
encoder, greedy joint decision. Greedy frame-tag streams can violate BIO
structure (a head may emit a stray ``I-``), so the adapter applies the
``promote`` repair policy before responding — the same policy the
streaming session documents — guaranteeing protocol-valid responses.
Quality is the evaluation package's business, not the adapter's.
Requires the torch extra; the model runs in eval mode under ``no_grad``.
"""

from __future__ import annotations

import numpy as np
import torch

from slotvox.adapters.protocol import InferRequest, InferResponse
from slotvox.errors import AdapterError
from slotvox.features.frontend import log_mel
from slotvox.models.joint import JointIntentSlotModel
from slotvox.tagging.bio import repair_sequence


class JointAdapter:
    """Serves greedy joint decisions from an in-process model."""

    kind = "joint-model"

    def __init__(self, model: JointIntentSlotModel):
        if not isinstance(model, JointIntentSlotModel):
            raise AdapterError(f"model must be a JointIntentSlotModel, got {type(model).__name__}")
        self._model = model
        self._model.eval()

    @property
    def model(self) -> JointIntentSlotModel:
        """The served model (forced to eval mode at construction)."""
        return self._model

    @property
    def model_hash(self) -> str:
        """Provenance hash pinned into every response."""
        return self._model.spec.model_hash

    def infer(self, request: InferRequest) -> InferResponse:
        """Understand one request's audio; strictly validated on both ends."""
        if not isinstance(request, InferRequest):
            raise AdapterError(f"request must be an InferRequest, got {type(request).__name__}")
        expected = self._model.spec.feature_config.sample_rate
        if request.sample_rate != expected:
            raise AdapterError(
                f"request sample rate {request.sample_rate} does not match the model's {expected}"
            )
        samples = np.asarray(request.samples, dtype=np.float64)
        mel = log_mel(samples, self._model.spec.feature_config)
        if mel.shape[0] == 0:
            raise AdapterError("audio is shorter than one analysis frame; nothing to decide")
        with torch.no_grad():
            intent_logits, slot_logits = self._model(torch.from_numpy(mel).unsqueeze(0))
            posterior = torch.softmax(intent_logits[0], dim=-1)
            best = int(posterior.argmax())
            greedy = tuple(
                self._model.spec.tags[index] for index in slot_logits.argmax(dim=-1)[0].tolist()
            )
        return InferResponse(
            request_id=request.request_id,
            intent=self._model.spec.intents[best],
            posterior=float(posterior[best]),
            frame_tags=repair_sequence(greedy, "promote"),
            model_hash=self.model_hash,
        )
