# Unified nine-path monitoring — 2026-10-16

The final scientific scope is nine production paths: three Rockfish Cr-from-Cu
restart continuations and six Discovery Cr-from-Li/Na geometry-template paths.
The two Discovery smoke jobs (`6966067`, Li gap-to-QL; `6966068`, Na gap-to-QL)
are launcher/input/I/O gates and are tracked as prerequisites, not as two of
the nine production results.

The unified heartbeat is `monitor-rockfish-cr-cr-neb-audit`, active every six
hours. It owns the final 9-path monitor, the smoke-gated Discovery production
submission, and the eventual status-aware merge. The two previous overlapping
Discovery heartbeats are paused to prevent duplicate monitoring or submission.

No production result is publication-qualified until its platform-specific
scientific gates pass. Completion requires finite ten-image output, correct
81-atom Cr/Sb/Te identity and cell/PBC/order, normal termination, explicit
forward/reverse barriers, complete provenance hashes, and no remaining SCF
warning. The Rockfish first-run raw directories remain immutable.

After all nine pass, the heartbeat must update the existing comparison tables,
render the affected bidirectional and Figure 1 artifacts through the Plot
Atlas workflow, and produce a claim-evidence/release bundle. Pending,
warning-contaminated, or failed paths remain in the exclusion ledger and do
not enter final barrier rankings or paper claims.
