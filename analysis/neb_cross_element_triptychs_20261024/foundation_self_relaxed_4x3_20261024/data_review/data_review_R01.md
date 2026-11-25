FAIL

状态：`pre-execution/partial`。formal array `32459413` 已提交且在审计快照时仍运行；当前只看到 5 个 attempt-level `endpoint_only_manifest.json`，对应 4 个唯一 row（`FSELF-20261024-000001` 同时有 smoke 与 formal 两个 attempt），不是 108 行正式完成结果。本报告为只读数据验收，不修改执行脚本、原始输入、注册表或运行产物。

审计快照：本地 `2026-10-25T01:49:33Z`；Rockfish 约 `2026-10-25T01:49:39Z`。主目录：`/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/neb_cross_element_triptychs_20261024/foundation_self_relaxed_4x3_20261024/`。Rockfish 目录：`/scratch16/pclancy3/yi/foundation_self_relaxed_4x3_20261024/`。

## 1. 验收结论摘要

| 门 | 结果 | 证据 |
|---|---|---|
| 4 元素 × 3 path × 9 model 的计划矩阵 | PASS（计划层）/ FAIL（发布层） | `JOB_TABLE_IDPP.tsv` 有 108 个唯一 row，全部仍为 `PLANNED`；当前仅 4 个唯一 row 有完成 manifest，且都为 `Cr__in_gap`。 |
| 9 个 Foundation checkpoint、SHA-256、head | PASS（registry 与远端文件） | 9 行唯一 model；远端 9 个文件均存在，实测 SHA-256 与 `MODEL_REGISTRY.tsv` 一致；7 个 `default`，2 个 `omat_pbe`。 |
| Cr 的 `2L-Cu` legacy stem 仅作 alias | PASS | Cr 三行 `source_kind=CANONICAL_ENDPOINT_INPUT`，`intermediate_source_frames_used_by_runner=NO`；运行命令只传 canonical initial/final endpoint。 |
| endpoint-only / IDPP / 10-image / seed | PASS（计划与当前已完成 attempt） | 108 行均记录 `ENDPOINTS_ONLY`、`IDPP`、10、42；已完成 attempt 的 command manifest 只有 initial/final endpoint，profile 为 10 个有序 image。 |
| 元素、81 atoms、cell、PBC、endpoint hash | PASS（几何与文件 hash） | 24 个 endpoint 文件均为 81 原子、各含 1 个目标迁移元素、坐标有限；initial/final 的 symbols、cell、PBC 一致；实际文件 SHA-256 与 path registry 一致。 |
| endpoint 内部 provenance header | FAIL | 24/24 的 campaign/path metadata 不是当前 canonical 标识；9 个非 Cr final 文件的 `role` 仍是 `initial`；18/24 缺 `endpoint_only_input`；24/24 的嵌入/缺失 `cell_sha256` 不等于 registry。 |
| 当前已完成 profile 的有限性与收敛门 | PASS（仅当前 4 个唯一 row） | 5 个 attempt manifest 的 profile 均 10 点、有序、有限，trajectory image hash 与几何检查通过；4 个唯一 row 均 `target_model_convergence=true`。 |
| 失败保留、全量 row audit、派生 profile | FAIL / 未形成证据 | 当前 local 与 remote `data/` 均无 `ROW_RESULTS.tsv`、`IMAGE_PROFILES.tsv`、`FAILURE_REGISTRY.tsv`、class profile 或 collection/derived manifest。 |
| fixed-path 与 self-relaxed 分开 | PASS（设计层）；全量发布仍未验收 | manifest 标记 `fixed_path_data_excluded=true`，job observable 是 `SELF_RELAXED_ENDPOINT_ONLY_NEB`；旧 campaign 结果未计入本 campaign。 |

因此，当前不能给出 release-level `PASS`，也不能把计划、运行中任务或已完成的少量 row 外推成 108-row 完成 campaign。

## 2. 已验证的正面证据

### 2.1 矩阵、模型与 canonical path

- `MODEL_REGISTRY.tsv:2-10` 有 9 个唯一 `model_id`、9 个唯一 model hash。family 分布为 MACE-MP 5、MACE-OMAT-0 2、MACE-MH 2；没有 MPA、fine-tuned、scratch、Blender 或 legacy model row。
- 在 Rockfish 上对 9 个 `model_path` 做了只读存在性和 SHA-256 核验，9/9 与 registry 相等。MACE-MH-0/1 使用 `omat_pbe`，其余 7 个使用 `default`。
- `PATH_REGISTRY.tsv:2-13` 有 12 个 canonical path，元素为 Cr/Cu/Li/Na，各三类 `in_gap`、`gap_to_ql`、`within_ql`。所有行声明 81 atoms、`T T T` PBC，且 endpoint-only runner 不使用中间 source frame。
- Cr 行中的 `2L-Cu__neb-*` 只出现在 `legacy_source_stem`；当前命令通过 `Cr__in_gap` 等 canonical path 和两个 endpoint 文件启动，没有将 legacy stem 当作元素或 path identity。
- `JOB_TABLE_IDPP.tsv:2-109` 的计数为 108 个唯一 record：Cr/Cu/Li/Na 各 27，三类 path 各 36，9 个 model；108/108 记录 `ENDPOINTS_ONLY`、`IDPP`、10 images、seed 42、`fmax=0.05`、`steps=500`。

### 2.2 endpoint 几何与已完成 Rockfish rows

对 `inputs/endpoints/` 的 24 个文件逐文件做了只读解析：

- 24/24 实际文件 hash 与 `PATH_REGISTRY.tsv` 一致；由 header 的 lattice/PBC 计算的 registry cell hash 也一致。
- 24/24 为 81 个 atom，目标迁移元素计数为 1，数据行的数值坐标有限；initial/final 的 atom symbol 顺序、lattice 和 PBC 成对一致。
- 当前 Rockfish completed outputs（截至快照）为：
  - `FSELF-20261024-000001` / `foundation_agnesi_large` / `Cr__in_gap`：smoke attempt 与 formal attempt 各一份；
  - `FSELF-20261024-000002` / `foundation_agnesi_medium_density` / `Cr__in_gap`；
  - `FSELF-20261024-000003` / `foundation_agnesi_small_density` / `Cr__in_gap`；
  - `FSELF-20261024-000004` / `foundation_agnesi_medium` / `Cr__in_gap`。
- 上述 5 个 attempt manifest 均报告 `return_code=0`、`scientific_status=CONVERGED`、`target_model_convergence=true`、`publication_status=PRIMARY_EVIDENCE`；其 profile 均为 image index 0–9 的 10 个有限点，`mlff_neb_images.extxyz` 的 hash 与 engine manifest 相符，原始文本检查显示 10×81 原子且坐标/forces 数值有限。
- 代表性最终门值：large 为 `0.039679/0.038040 eV/A`（NEB force/residual），medium-density 为 `0.047825/0.035441`，medium 为 `0.048243/0.041536`，均低于 0.05；这只证明已完成的少量 self-relaxed model observables，不是 DFT ground truth。

## 3. 编号反馈与精确修复建议

### F-01 [BLOCKER] 全量 campaign 尚未完成，必须标为 `pre-execution/partial`

证据：`CAMPAIGN_MANIFEST.json:25-31` 记录 formal array `0-107%4`、job `32459413`、状态 `RUNNING`；`CAMPAIGN_MANIFEST.json:58-60` 仍保留 `READY_FOR_FULL_ARRAY` 与目标 108。Rockfish scheduler 快照显示 tasks 4–7 running、8–107 pending（受 array limit 限制）。当前仅 4 个唯一 row 有完成 manifest，且全部属于 Cr/in-gap；Cu、Li、Na 和另外两个 path class 尚无当前 campaign 完成结果。

精确修复：

1. 等待 array 完成或明确终止后，再以 108 个 `record_id` 为单位收集结果；运行中和 pending row 记为 `PENDING/RUNNING`，不能记为失败或成功。
2. 对每个 row 生成一个明确状态：`CONVERGED`、`NONCONVERGED`、`FAILURE_MODE`、`MISSING_OUTPUT` 或 `PENDING`，并使 campaign 状态与正式 array 状态一致；在全量 gates 通过前保持 diagnostic/incomplete。
3. 只有 108/108 row 都有完整 provenance、10-image profile、有限值和收敛门通过后，才允许 `release_ready=true` 或发布 class mean/SD。

### F-02 [BLOCKER] 没有 row-level collection、failure ledger 或派生数据

证据：local 与 Rockfish 当前 campaign 的 `data/` 都没有文件；缺少 `ROW_RESULTS.tsv`、`IMAGE_PROFILES.tsv`、`FAILURE_REGISTRY.tsv`、`COLLECTION_MANIFEST.json`、`CLASS_PROFILES.tsv`、`CLASS_SUMMARY.tsv`、`PROFILE_INCLUSION_AUDIT.tsv` 和 `DERIVED_PROFILE_MANIFEST.json`。因此当前没有可审计的 denominator、excluded-row ledger 或失败保留记录。

精确修复：

1. array 结束后生成上述 collection/failure/derived artifacts，并为每个 108 row 写入 attempt path、job/array task、manifest hash、model/endpoint/runner hash、profile image count、有限性、收敛门、scientific/publication status 和 failure reason。
2. class summary 必须明确 4×3×3 family/path 组合、每组 accepted/expected denominator、被排除 row 及原因；任何缺失组合都保持 `INCOMPLETE`，不能用空 panel、插值或平滑隐藏。
3. 派生 profile 只从通过 model/data gates 的 self-relaxed rows 生成；fixed-path 数据继续保持排除。

### F-03 [BLOCKER] local、remote 和 smoke 使用了三个不同的 campaign manifest hash

证据：

- 当前 local `CAMPAIGN_MANIFEST.json` SHA-256 为 `58f7f7c2bff77eda2b1ea1650640f520558699ceccc5364fa4f52b772104afff`，包含 formal submission 字段。
- Rockfish 正式执行所读取的 `CAMPAIGN_MANIFEST.json` SHA-256 为 `9b23025dd472b942e0b7f67a25c510e5fef5bf8f93171f9fa96d6d45f643a04a`，该远端文件没有 local 的 `formal_submission` 字段；local 的 `remote_execution_manifest_sha256` 指向此 hash。
- smoke attempt `32459224_0` 的 endpoint/engine manifest 绑定旧 hash `3529255e5fed705d6f3c23deb2d77fa1b86210ebd099ad59d8d361538f1b31f8`；4 个 formal completed attempt 绑定 `9b23025...`。

这不是结果数值本身的失败，但违反了最终 release 需要唯一、可回溯 launch manifest 的 provenance 要求。

精确修复：保留三份 hash 及其角色（pre-smoke、formal-launch remote、post-submit local），不要覆盖历史文件；将正式 array 的唯一输入 manifest 路径和 hash写入一个不可变的 execution manifest，并在每个 row manifest 中记录 `manifest_role=FORMAL_LAUNCH`。完成后在 final release manifest 中同时列出 criteria hash、formal-launch hash、local bookkeeping hash 及同步时间。

### F-04 [HIGH] endpoint 文件的内部 header provenance 不是 canonical campaign identity

证据：几何和文件 hash 是正确的，但 24/24 endpoint 文件的 header `campaign_id` 不是 `foundation_self_relaxed_4x3_20261024`；24/24 的 header `path_id` 不是 `Cr__in_gap` 等 canonical ID；9 个非 Cr final 文件的 `role` 仍为 `initial`；非 Cr 的 18 个文件缺少 `endpoint_only_input=T`；24/24 的嵌入 `cell_sha256` 为旧值 `b8c90c...` 或缺失，而不是 registry cell hash。当前 smoke/formal 的 `mlff_neb_images.extxyz` 也继承了这类旧 campaign/path/cell header。

精确修复：不要覆盖或删除 raw source。新增版本化的 canonical endpoint representation，header 至少应明确写入当前 `campaign_id`、canonical `path_id`、正确 `role`、`endpoint_only_input=T`、registry `cell_sha256`、lattice 和 PBC；保留 `source_path`、`source_sha256` 与 `legacy_source_stem` 作为 alias/provenance。对 canonical representation 的任何修改都必须重新计算 24 个 endpoint hash，并同步更新 path/job registry、remote copy 及受影响 run 的 endpoint hashes；既有结果不能在未重新绑定 hash 的情况下继续作为当前输入证据。

### F-05 [HIGH] smoke 与 formal 共用了同一个 record_id，attempt 选择会静默丢掉历史 attempt

证据：`FSELF-20261024-000001` 同时出现在 `attempt_32459224_0`（`mbfssmoke`）和 `attempt_32459419_0`（`mbfself`），两者都报告 `CONVERGED/PRIMARY_EVIDENCE`。raw attempt 目录被保留，但 `code/collect_results.py:29-32` 只选择按名称排序的最后一个 attempt；`collect_results.py:41-52` 和 `130-143` 没有把同一 row 的其它 attempt 纳入 row-level attempt ledger。

精确修复：新增 attempt-level 表（至少 `record_id`、`attempt_id`、`run_role`、scheduler job、array task、manifest hash、状态和 selection reason）；将 smoke 明确标为 `DIAGNOSTIC_ONLY/SMOKE`，正式 array attempt 作为 row 的正式候选；collector 必须保留所有 attempts 和失败/重试记录，只在 row result 中记录经过审计的 selected attempt，不能仅靠 `attempts[-1]` 静默覆盖。

### F-06 [MEDIUM] manifest 状态语义落后于实际执行状态

当前 local manifest 已有 `formal_submission.status=RUNNING`，但顶层 `status` 仍为 `READY_FOR_FULL_ARRAY`；远端 formal-launch manifest 没有 submission 状态字段。此状态不能作为“已完成”使用，但会使下游读者误解当前阶段。

精确修复：在不改变冻结 acceptance criteria 的前提下，维护一个 additive execution/review status，明确 `RUNNING`、`PARTIAL`、`COMPLETE`、`RELEASE_READY` 的转换条件；本报告阶段应保持 `pre-execution/partial`，array 完成且 collection QA 通过后才允许进入下一状态。

## 4. 不应被误判的边界

- 计划矩阵的 108 行不等于 108 个成功结果；当前 `JOB_TABLE_IDPP.tsv` 的 108 行仍为 `PLANNED`，而 Rockfish 正在运行的 task 不应被当作成功或失败。
- 当前已完成的 4 个 unique row 可以作为有 provenance 的诊断/候选 observables，但不能代表 Cu/Li/Na、其它 path 或 9-model 全量结果。
- 已找到的旧 campaign `neb_fixed_vs_self_matrix_20261019` 及其它 fixed-path profile 不计入本 campaign；这与 `prior_outputs_reused=false`、`fixed_path_data_excluded=true` 的边界一致。
- MLFF self-relaxed profile 不是 DFT ground truth；若后续显示 QE/DFT reference，必须保留其既有 SCF-warning boundary。

## 5. Re-review 条件

本报告在当前快照返回 `FAIL`。重新审查至少需要：

1. formal array 结束，108 row 的结果/失败/缺失状态全部可追溯；
2. smoke/formal duplicate attempt 已在 attempt ledger 中明确分层；
3. local/remote formal-launch manifest hash 已固定并可读取；
4. endpoint header provenance 修复或有等价的、hash-bound canonical endpoint layer；
5. collection、failure ledger、10-image profile integrity、geometry/PBC、finite-value 和 convergence audit 全部生成；
6. class denominator、排除项和 self-relaxed/fixed-path separation 在 derived manifest 中明确记录。

