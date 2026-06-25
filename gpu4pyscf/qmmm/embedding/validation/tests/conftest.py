# Copyright 2025 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared pytest fixtures / helpers for the embedding-validation tests.

The heavy tests require a working GPU (gpu4pyscf only runs on CUDA).  We probe
the GPU once and expose a module-level ``GPU_AVAILABLE`` flag plus a
``requires_gpu`` skip marker so the suite degrades gracefully to the pure-CPU
math tests on machines without a usable CUDA driver.
"""

import sys
import os

import pytest

# Make the validation package importable both as a package and as loose scripts.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _probe_gpu():
    """Return True only if gpu4pyscf can actually run a GPU calculation."""
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            return False
        # Importing gpu4pyscf touches the CUDA runtime; guard the whole thing.
        import gpu4pyscf  # noqa: F401
        from gpu4pyscf.dft import rks  # noqa: F401
        return True
    except Exception:
        return False


GPU_AVAILABLE = _probe_gpu()

requires_gpu = pytest.mark.skipif(
    not GPU_AVAILABLE,
    reason="No usable GPU / gpu4pyscf runtime; skipping GPU embedding tests.")
