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

"""Pure-CPU unit tests for the backend-agnostic analysis helpers.

These run everywhere -- they exercise the basis-transformation maths, the PES
fit, the xyz parsing / fragment heuristics and the dimension guards WITHOUT
needing a GPU.
"""

import os
import json
import tempfile

import numpy as np
import pytest

import embedding_analysis as ea
import generate_inputs as gen
import run_single_job as single


# ---------------------------------------------------------------------------
# Helper: build an S-orthonormal projector B for synthetic tests.
# ---------------------------------------------------------------------------
def _random_overlap(nao, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((nao, nao))
    return M @ M.T + nao * np.eye(nao)          # SPD overlap-like matrix


def _s_orthonormal_B(nao, nemb, S, seed=1):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((nao, nemb))
    G = A.T @ S @ A
    w, v = np.linalg.eigh(G)
    G_inv_sqrt = v @ np.diag(w ** -0.5) @ v.T
    return A @ G_inv_sqrt                        # satisfies B^T S B = I


class TestBasisTransforms:
    def test_emb_to_ao_dimension(self):
        nao, nemb = 7, 4
        S = _random_overlap(nao)
        B = _s_orthonormal_B(nao, nemb, S)
        dm_emb = np.eye(nemb)
        dm_ao = ea.project_dm_emb_to_ao(dm_emb, B)
        assert dm_ao.shape == (nao, nao)

    def test_metric_round_trip(self):
        # (B^T S) [B X B^T] (S B) == X  for S-orthonormal B.
        nao, nemb = 8, 5
        S = _random_overlap(nao, seed=3)
        B = _s_orthonormal_B(nao, nemb, S, seed=4)
        rng = np.random.default_rng(5)
        X = rng.standard_normal((nemb, nemb))
        X = X + X.T
        dm_ao = ea.project_dm_emb_to_ao(X, B)
        X_back = ea.project_dm_ao_to_emb(dm_ao, B, S)
        assert np.allclose(X_back, X, atol=1e-9)

    def test_mo_projection_dimension(self):
        nao, nemb, nmo = 6, 4, 4
        S = _random_overlap(nao)
        B = _s_orthonormal_B(nao, nemb, S)
        C_emb = np.eye(nemb)[:, :nmo]
        C_ao = ea.project_mo_emb_to_ao(C_emb, B)
        assert C_ao.shape == (nao, nmo)

    def test_assert_full_ao_guard(self):
        nao = 5
        good = np.zeros((nao, nao))
        assert ea.assert_full_ao(good, nao)
        bad = np.zeros((3, 3))                    # embedding-sized -> must raise
        with pytest.raises(AssertionError):
            ea.assert_full_ao(bad, nao, "test dm")


class TestBondShift:
    """Single-point bond-shift behaviour (no PES scan / no equilibrium fit)."""

    def test_scale_bond_moves_only_target(self):
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        out = ea.scale_bond(coords, 0, 1, 1.5)
        assert np.allclose(out[0], coords[0])      # anchor unchanged
        assert np.allclose(out[2], coords[2])      # untouched atom unchanged
        assert np.allclose(out[1], [0.0, 0.0, 1.5])

    def test_resolve_geometry_shift_disabled(self):
        # bond_shift_flag false -> original geometry, applied=False.
        config = {
            "structure": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            "fragment_id": [0, 1], "bond_test_id": [0, 1],
            "bond_shift_flag": False, "bond_shift_scale": 1.2,
        }
        coords, info = single.resolve_bond_geometry(config)
        assert info["applied"] is False
        assert np.allclose(coords, config["structure"])

    def test_resolve_geometry_shift_enabled(self):
        # bond_shift_flag true -> exactly the target bond is rescaled once.
        config = {
            "structure": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            "fragment_id": [0, 1], "bond_test_id": [0, 1],
            "bond_shift_flag": True, "bond_shift_scale": 1.2,
        }
        coords, info = single.resolve_bond_geometry(config)
        assert info["applied"] is True
        assert info["bond_atoms"] == [0, 1]
        assert abs(info["r0_angstrom"] - 1.0) < 1e-12
        assert abs(info["r_shifted_angstrom"] - 1.2) < 1e-12
        assert np.allclose(coords[1], [0.0, 0.0, 1.2])  # only the bond atom moved
        assert np.allclose(coords[2], config["structure"][2])

    def test_resolve_geometry_no_bond_pair(self):
        # Shift requested but no usable bond pair -> falls back to original.
        config = {
            "structure": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            "fragment_id": [0, 1], "bond_test_id": [],
            "bond_shift_flag": True, "bond_shift_scale": 1.5,
        }
        coords, info = single.resolve_bond_geometry(config)
        assert info["applied"] is False
        assert np.allclose(coords, config["structure"])


class TestXYZParsing:
    def _write(self, text):
        fd, path = tempfile.mkstemp(suffix=".xyz")
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        return path

    def test_parse_standard_xyz(self):
        path = self._write("2\ncomment\nH 0 0 0\nH 0 0 0.74\n")
        elems, coords = gen.parse_xyz(path)
        os.remove(path)
        assert elems == ["H", "H"]
        assert coords.shape == (2, 3)
        assert abs(coords[1, 2] - 0.74) < 1e-12

    def test_parse_no_count_line(self):
        path = self._write("O 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.24\n")
        elems, coords = gen.parse_xyz(path)
        os.remove(path)
        assert elems == ["O", "H", "H"]

    def test_fragment_and_bonds(self):
        # Ethane-like: heavy C0 with three bonded H -> fragment {0,2,3,4}.
        path = self._write(
            "8\nethane\n"
            "C -0.76 0 0\nC 0.76 0 0\n"
            "H -1.16 1.02 0\nH -1.16 -0.51 -0.88\nH -1.16 -0.51 0.88\n"
            "H 1.16 -1.02 0\nH 1.16 0.51 0.88\nH 1.16 0.51 -0.88\n")
        elems, coords = gen.parse_xyz(path)
        os.remove(path)
        bonds = gen.perceive_bonds(elems, coords)
        frag = gen.auto_fragment(elems, bonds)
        assert 0 in frag and len(frag) >= 2
        bond_ids = gen.auto_bond_test_ids(bonds, elems)
        # The single heavy-heavy bond is C0-C1.
        assert bond_ids == [0, 1]

    def test_generate_writes_json(self):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "h2.xyz"), "w") as fh:
            fh.write("2\nh2\nH 0 0 0\nH 0 0 0.74\n")
        out = os.path.join(d, "test_systems.json")
        systems = gen.generate(d, out, basis_set="sto-3g",
                               bond_shift_flag=True, bond_shift_scale=1.1)
        assert os.path.exists(out)
        with open(out) as fh:
            loaded = json.load(fh)
        assert "h2" in loaded
        assert loaded["h2"]["basis_set"] == "sto-3g"
        assert loaded["h2"]["charge"] == 0
        # New single-point bond-shift annotations are written through.
        assert loaded["h2"]["bond_shift_flag"] is True
        assert abs(loaded["h2"]["bond_shift_scale"] - 1.1) < 1e-12
        for key in ("element", "structure", "fragment_id", "bond_test_id",
                    "energy_flag", "bond_shift_flag", "bond_shift_scale"):
            assert key in loaded["h2"]


class TestSpinGuess:
    def test_even_electron_singlet(self):
        assert gen.guess_spin(["H", "H"], 0) == 0          # 2 e- -> singlet
        assert gen.guess_spin(["O", "H", "H"], 0) == 0     # 10 e- -> singlet

    def test_odd_electron_doublet(self):
        assert gen.guess_spin(["H"], 0) == 1               # 1 e- -> doublet
        assert gen.guess_spin(["H", "H"], 1) == 1          # cation, 1 e-


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
