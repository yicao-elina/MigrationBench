"""The Fig. 4 DFT-NEB reference must come from QE neb.out and match the submitted curves."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "migrationbench"))
import dft_neb_reference as ref  # noqa: E402

# Values plotted in the submitted Revision 1 Fig. 4 (formerly hard-coded arrays).
SUBMITTED = {
    "in_gap": [0.0, 0.14998279, 0.45337452, 0.84019660, 1.23871102, 1.49900233, 1.52429883, 1.32639253, 1.01828651, 0.84891322],
    "gap_to_ql": [0.0, -0.00599650, -0.01751910, -0.02938830, -0.04046770, -0.05133610, -0.06176651, -0.07202330, -0.08241070, -0.09594731],
    "within_ql": [0.0, -0.28038152, -0.31615703, -0.30312301, 0.69680110, 1.63002741, 1.84489982, 0.41361450, -0.75147823, -0.50192181],
}


@pytest.mark.parametrize("path_key", sorted(SUBMITTED))
def test_parsed_profile_matches_submitted_curve(path_key):
    profile = ref.load_path(path_key)
    assert profile["relative_energy_eV"] == pytest.approx(SUBMITTED[path_key], abs=1e-6)
    assert profile["x"][0] == 0.0 and profile["x"][-1] == 1.0


def test_in_gap_barrier_is_canonical_1p52_eV():
    assert max(ref.load_path("in_gap")["relative_energy_eV"]) == pytest.approx(1.524, abs=1e-3)


def test_convergence_status_is_reported_not_hidden():
    for key in SUBMITTED:
        assert ref.load_path(key)["converged"] is False


def test_unknown_path_raises():
    with pytest.raises(KeyError):
        ref.load_path("sliding")
