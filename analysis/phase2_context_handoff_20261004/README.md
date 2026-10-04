# Phase 2 context handoff — 460000 / 462200

This additive campaign closes the original P0 scope gap. The canonical audit
already covers 461800 and 462000; this package runs the same four-model,
2-ps handoff protocol from the retained neighboring source snapshots 460000
and 462200.

Protocol is identical to the canonical handoff: no velocity resampling,
600 K Langevin, friction 0.01 fs^-1, dt 1 fs, 2000 integration steps,
metrics every 10 steps, and the same MACE model paths and finite-state
requirements. Outputs use a date-isolated Rockfish root and are not merged
with the canonical retry2 results until input/model hashes and frame counts
are verified.

The 462200 source is a stress/context state. Any resulting runaway is a
diagnostic counterfactual and does not enter diffusion or formal DFT claims.

The first submission `31591668_[0-5]` failed before integration because the
wrapper pre-created directories that `handoff_md.py` intentionally requires
to be absent. Its logs are retained remotely as a launcher audit; no output
from that submission is treated as a trajectory. `retry1` removes that
pre-creation and uses a new result root.

Retry array: `31591778_[0-7]`, result root
`/scratch16/pclancy3/yi/phase2_context_handoff_20261004_retry1`.
