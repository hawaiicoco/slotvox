"""slotvox: end-to-end spoken language understanding on fully synthetic speech.

The package pairs a NumPy-only signal/feature/labeling core with an optional
CPU-PyTorch joint intent-slot model, a streaming inference session, an
instruction-data pipeline, protocol adapters, and offline evaluation tools.
All data is synthetic and deterministic; the project makes no claims about
real speech corpora or pretrained models.
"""

from slotvox._version import __version__

__all__ = ["__version__"]
