"""Frozen numerical outputs captured independently before module consolidation.

Reference revision: 000401e6a9c30e7f6f9d1e917b5f872689ace6d6.
Captured with Torch 2.11.0+cu128 on CPU. See THIRD_PARTY_NOTICES for attribution.
Keep these fixtures fixed: implementation changes must not regenerate expectations.
"""
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def oracle():
    with np.load(Path(__file__).parent / "fixtures" / "numerical_reference.npz", allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}
