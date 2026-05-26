"""Pytest configuration — make src/ and the cacheblend submodule importable."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
# Project sources
sys.path.insert(0, str(_ROOT / "src"))
# Submodule: cacheblend-hf-v7 ships its package under src/cacheblend
_V7_SRC = _ROOT / "src" / "external" / "cacheblend-hf-v7" / "src"
if _V7_SRC.exists():
    sys.path.insert(0, str(_V7_SRC))


def _cuda_available() -> bool:
    try:
        import torch  # noqa: WPS433 — local import keeps collect-time cheap
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def pytest_collection_modifyitems(config, items):
    """Auto-skip @pytest.mark.gpu tests when CUDA is absent."""
    if _cuda_available():
        return
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
