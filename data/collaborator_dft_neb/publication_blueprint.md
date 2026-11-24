# Collaborator DFT/NEB: publication blueprint

## Acceptance target

The deliverable is not a directory listing. It is a provenance-linked SQLite
dataset in which each calculation can be traced as:

`source path -> input/script/output artifacts -> parsed system/method -> quality gates -> normalized structures/trajectory -> result -> manuscript/reviewer claim`.

Only records passing the applicable quality gates may be used as DFT reference
or NEB barrier evidence. Failed, partial, duplicate, and unknown-method records
remain in the database with explicit status and are never silently discarded.

## Processing stages

1. **Access and inventory**: preserve remote absolute paths, ownership, mtime,
   size, and SHA256; create source manifest. Joshitha is currently blocked by
   permissions and is not counted as available evidence.
2. **Discovery**: identify QE inputs/outputs, NEB image directories, scripts,
   pseudopotentials, CIF/XYZ/extXYZ and scheduler logs. Do not infer a result
   from a file name or Slurm `COMPLETED` state.
3. **Parsing**: extract formula, cell/PBC, mobile/dopant species, site/pathway,
   XC functional, SOC, D3, cutoffs, k-points, pseudopotentials, images,
   energies, forces, convergence text and failure signatures.
4. **Quality control**: apply Q1-Q10; cross-check input/output identity; detect
   duplicate/restarted calculations; classify `accepted`, `partial`, `failed`,
   `unknown`, or `duplicate`.
5. **Processed release**: export only compact normalized structures/trajectories
   and result tables needed to reproduce figures. Keep raw files on Rockfish;
   publish scripts, schema, manifests, processed data and checksums in GitHub.
6. **MACE comparison**: run the same accepted endpoints/NEB paths with the
   MigrationBench MACE protocol, preserving path IDs and system IDs. Compare
   DFT-vs-MACE only when method and convergence gates pass.

## Current evidence-driven hypotheses

- Victor's `nebProject` and `V-ST` are the highest-priority sources for direct
  Sb2Te3 migration comparators: they contain relax/NEB branches, SOC/D3 and
  pseudopotential directories, but acceptance is not yet established.
- A targeted read-only inventory has confirmed a structured
  `nebProject/120_interstitial` subproject with TMi2 octahedral and TMi4
  tetrahedral relax/vcrelax variants, batch outputs, an extraction script, and
  `calculation_results.csv`. These are candidate endpoint/reference records,
  not yet accepted barriers; the `old/` subtree is explicitly history/duplicate
  material until content hashes and convergence parsing resolve it.
- Victor's `ST-other` has Cr/Mn/Mo defect relaxations and is a candidate for
  dopant generalization, not yet a NEB dataset.
- Victor's `flare/sb2te3` contains DFT xyz/output and convergence material that
  may support training-data provenance or validation, not automatically a
  migration barrier.
- Joshitha's data may add independent DFT/NEB systems, but no scientific claim
  should include it until `/data/jedison2/jpilla1` is readable and parsed.

## Reviewer and manuscript mapping

| Evidence product | Planned target | Status |
|---|---|---|
| accepted DFT endpoints and method table | A1; main text 3.2; SI endpoint table | placeholder until parser/acceptance |
| accepted QE-NEB barriers with convergence flags | A1; Fig 3d/Fig S3 comparison | placeholder; do not replace historical 0.16 eV yet |
| independent V/Cr/Mn/Mo pathways | A4 generalizability; SI validation table | candidate only |
| DFT trajectory/path manifest and identity audit | A1/A2 evidence supplement | planned |
| DFT-vs-MACE endpoint and barrier comparison | A1, A3 model disagreement context | planned after accepted records |
| grouped processed data + reproducibility scripts | A2/C10/A3 supporting audit infrastructure | planned; separate from training split claims |

## Gap register and executable closure

| Gap | Closure | Exit evidence |
|---|---|---|
| Joshitha unreadable | owner/group read permission or shared copy | successful recursive read test + manifest |
| no normalized collaborator catalog | run source-specific inventory/parser | SQLite rows + artifact hashes |
| result validity unknown | parse convergence/fatal-error/image identity | Q3-Q8 evidence per record |
| systems not comparable to Cr@Sb2Te3 | canonical system/path schema and units | crosswalk table + accepted MACE input bundle |
| manuscript integration absent | create generated tables/figure placeholders and response snippets | committed `.tex` inputs/placeholder labels |
| raw data cannot be published | retain remote raw URI; publish processed release only | README lineage + checksums + rerun command |

## Stop conditions

Do not report a count of valid DFT/NEB results, a barrier, or a reviewer-claim
closure until the corresponding quality gates and evidence artifacts are in the
SQLite database. A count of discovered files is not a count of scientific
results.
