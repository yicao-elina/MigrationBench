FAIL

状态：pre-execution/partial。formal array 32459698 在审计快照时仍运行；本报告不把计划、运行中任务、smoke 或少量已完成 row 视为 release PASS。只读审查未修改执行代码、registries、inputs/endpoints 或 Rockfish raw runs；本次只新增本报告文件。

审计快照：Rockfish 2026-10-25T02:03:12Z；远端根目录 /scratch16/pclancy3/yi/foundation_self_relaxed_4x3_20261024_r02。该时刻 array tasks 0、4、5、6 为 RUNNING，7–107 为 PENDING（array limit 4）。远端已有 4 个 endpoint_only_manifest.json / 4 个 mlff_neb_manifest.json，其中 1 个是 smoke，3 个是 formal completed；不能外推为 108 个完成结果。

## 1. 总结判定

| 验收门 | R02 结果 | 只读证据 |
|---|---|---|
| 4 元素 × 3 path × 9 model = 108 | PASS（计划层）；FAIL（发布层） | JOB_TABLE_IDPP.tsv:1-109 共 108 个唯一 record_id；当前 formal array 仍运行，收集快照仅 3 个 formal row 被选为 CONVERGED。 |
| 9 个 Foundation checkpoint / SHA-256 / head | PASS（registry 与远端 checkpoint） | MODEL_REGISTRY.tsv:1-10 共 9 行；远端 9/9 文件存在且 SHA-256 一致；head 为 7 个 default、2 个 omat_pbe。 |
| 24 个 canonical endpoint header | PASS（F-04 输入层） | 24/24 header 为 canonical_endpoint_v2；campaign/path/role/endpoint_only_input/cell_sha256/PBC 与 R02 registry 一致；本地和远端 endpoint 文件 hash 逐一一致。 |
| Cr legacy 2L-Cu stem 仅作 alias | PASS | PATH_REGISTRY.tsv:2-4 的 Cr 行为 CANONICAL_ENDPOINT_INPUT；2L-Cu stem 只在 legacy_source_stem；Cr 三行均声明不使用 intermediate source frames。 |
| endpoint-only / IDPP / 10-image / seed | PASS（设计与当前已完成 manifest） | JOB_TABLE_IDPP.tsv:2-109 的 108/108 行为 ENDPOINTS_ONLY、IDPP、10、42、fmax 0.05、steps 500；run_one_job.py:87-127 只传 initial/final endpoint。 |
| immutable CAMPAIGN_MANIFEST hash | PASS（launch binding）；PARTIAL（release proof） | 本地与远端 CAMPAIGN_MANIFEST hash 均为 e226280c…；EXECUTION_BOOKKEEPING_R02.json:5-6 明确绑定且标记 immutable；已观察的 4 个 row manifest 也绑定同一 hash。 |
| 失败保留、attempt 分层、派生数据 | FAIL | 当前 collection 快照有 108 row，但把 105 个运行中/排队或尚无终态的 row 记为 MISSING_OUTPUT；8 条 attempt ledger 全部标为 FORMAL_ARRAY，smoke 未被识别为 SMOKE。 |
| finite profile 与收敛门 | PASS（代码设计与 3 个 formal completed row）；FAIL（全量发布） | run_one_job.py:201-222 明确要求 optimizer_converged 且 force/residual <= 0.05；当前仅 3 个 formal completed row 各有 10 点 profile。 |
| self-relaxed 与 fixed-path 分开 | PASS（设计层） | CAMPAIGN_MANIFEST.json:25、37-39 与 JOB_TABLE_IDPP.tsv 的 observable 明确排除 fixed-path、prior outputs 不复用；全量 release 仍未验收。 |

## 2. F-04：canonical endpoint header 修复

结论：F-04 的输入层设计已修复，当前没有发现 endpoint header/hash 的残余不一致。

- PATH_REGISTRY.tsv:1-13 有 12 个 canonical path，4 个元素各 3 个 path class；所有行的 header_schema 为 canonical_endpoint_v2，endpoint_header_campaign_id 为 foundation_self_relaxed_4x3_20261024_r02，endpoint_header_roles 为 initial,final，endpoint_header_endpoint_only_input 为 T。
- inputs/endpoints/ 下 24 个文件均存在；实际文件 SHA-256 与 PATH_REGISTRY.tsv 的 initial_endpoint_sha256 / final_endpoint_sha256 逐一相等。local/remote 24 个文件的 hash 也逐一相等。
- 24 个文件均为 81 原子；每个 endpoint 的元素计数为目标迁移元素 1、Sb 32、Te 48；initial/final 的 symbol 顺序、cell、PBC 均一致，PBC 为 T T T。header 中的 cell_sha256 与 path registry 一致，source_path 存在且 24/24 的 source_sha256 可复核。
- Cr 的 2L-Cu__neb-gap-diffuse、2L-Cu__neb-gap2QL、2L-Cu__neb-QLs 只保留在 legacy_source_stem；Cr 的 element/path identity 来自 Cr__* canonical path_id，不来自 legacy stem。Cu/Li/Na 的 legacy linear source 也只作为 source provenance，PATH_REGISTRY.tsv 明确 intermediate_source_frames_used_by_runner=NO。
- 当前已完成的 smoke 与 3 个 formal output 的 mlff_neb_images.extxyz 继承了上述 canonical campaign_id、path_id、cell_sha256、PBC 和 endpoint-only header；已生成的 output 文件均为 10×81 原子布局。

## 3. F-03：immutable launch manifest 与 execution bookkeeping

结论：F-03 的核心 hash-binding 设计已修复，但还不能称为 release-level 完整修复。

已修复的部分：

- CAMPAIGN_MANIFEST.json 的实际 SHA-256 为 e226280ce571cc7df286824a627772a1b8c3ebe7808fde5ca7805e82d1f0bd4c；EXECUTION_BOOKKEEPING_R02.json:5-6 记录相同 hash，并声明 formal_launch_manifest_is_immutable=true。
- 远端 CAMPAIGN_MANIFEST.json 的 SHA-256 与 local 完全相同；远端 JOB_TABLE_IDPP.tsv、PATH_REGISTRY.tsv、MODEL_REGISTRY.tsv 也分别与 local 完全相同，避免了 R01 的 local/remote formal manifest 分叉。
- 当前观察到的 4 个 endpoint_only_manifest.json 均记录 campaign_manifest_sha256=e226280c…；其 endpoint/model hash、model head、campaign/path identity 均与 R02 registry 相符。run_one_job.py:146-191 也会记录 campaign manifest、runner、wrapper、endpoint 和 model provenance。

仍未闭合的部分：

- EXECUTION_BOOKKEEPING_R02.json:13 明确写 local_bookkeeping_only=true；该文件没有远端副本，也没有自身的不可变归档 hash。远端仅能看到 CAMPAIGN_MANIFEST 和 row-level manifest，不能独立重建 smoke/formal submission bookkeeping。
- CAMPAIGN_MANIFEST.json:26 的 formal_launch_job_id 仍为 null，这是冻结 manifest 不应在提交后回写的合理结果，但父 array job 32459698、child job、array task 的完整映射尚未进入每个 row manifest；当前 row manifest 只记录 slurm_job_id 与 slurm_array_task_id。
- CAMPAIGN_MANIFEST.json:47 的 launch-time status 仍为 READY_FOR_FORMAL_ARRAY，而 bookkeeping:12 为 RUNNING。这个状态分离可以成立，但最终必须由不可变的 additive execution/release manifest 给出 COMPLETE、INCOMPLETE 或 RELEASE_READY，不能以 launch-time status 代替结果状态。

因此 F-03 的精确判定是：launch hash 绑定 PASS；execution bookkeeping 的 release 归档和终态证明仍为 blocker。不要在 array 运行中修改已绑定的 CAMPAIGN_MANIFEST；应在 array 结束后生成不可变 final bookkeeping/release manifest，记录 criteria hash、formal launch hash、local bookkeeping hash、parent array job、全部 child/task 映射、collection/derived hash 和终态。

## 4. 剩余 blocker 与精确修复建议

### R02-B01 [BLOCKER] formal array 尚未完成，必须保持 pre-execution/partial

证据：formal array 32459698 仍有 RUNNING/PENDING tasks；截至 2026-10-25T02:03:12Z 只有 3 个 formal endpoint/engine manifest 完成，另有 1 个 smoke manifest。远端 data/ 只有当前时点的部分 collection artifacts，没有 108-row release evidence。

修复：等待 array 结束或明确终止；对 108 个 record_id 分别记录 CONVERGED、NONCONVERGED、FAILURE_MODE、MISSING_OUTPUT、RUNNING 或 PENDING。只有 108/108 row 完成 row-level provenance、finite/geometry gate、10-image gate 和 convergence gate 后，才可进入 release review。

### R02-B02 [BLOCKER，F-05 残留] smoke 与 formal 仍复用同一 row identity，角色未真正分层

证据：smoke job 32459612_0 使用 FSELF-R02-20261024-000001；其 endpoint_only_manifest.json 的 execution_role=FORMAL_ARRAY、manifest_role=FORMAL_LAUNCH，与正式 array 语义相同。data/ATTEMPT_LEDGER.tsv 的 8 条记录全部 run_role=FORMAL_ARRAY，smoke 没有被标为 SMOKE/DIAGNOSTIC_ONLY。collect_results.py:44-47 仍只识别 R01 的旧 attempt 前缀 32459224、32459413，不能识别当前 R02 smoke 32459612。

修复：为 smoke 使用独立 record/attempt role，或至少在 submission metadata 中写入不可变的 SMOKE role、smoke job id、formal parent array job id；collector 必须按 manifest metadata 选择正式 attempt，不能依赖硬编码 job 前缀。保留所有 attempts、失败和重试；smoke 只能是 diagnostic，不得成为正式 row 的 PRIMARY_EVIDENCE。

### R02-B03 [BLOCKER] partial collection 把 pending/running 当作 MISSING_OUTPUT，不能作为失败 ledger 或 release denominator

证据：远端 data/COLLECTION_MANIFEST.json 创建于 2026-10-25T02:02:54Z，target_rows=108、rows_with_manifests=3、converged_rows=3、profile_rows=30、failure_rows=105；ROW_RESULTS.tsv 的 105 条为 MISSING_OUTPUT，FAILURE_REGISTRY.tsv 的 105 条也全部为 MISSING_OUTPUT。此时 scheduler 明确仍有 running/pending tasks，因此这些不是已判定失败的科学结果。当前 ATTEMPT_LEDGER 保留了 8 条 attempts，这是正面证据，但 row-level status 语义仍不正确。

修复：array 运行期间不得把无终态的 attempt 写入 FAILURE_REGISTRY；使用 PENDING/RUNNING 状态并附 scheduler evidence。array 完成后再生成最终 failure ledger，明确“无 output”与“运行中/排队”及真实 FAILURE_MODE 的区别；重建 collection manifest 和所有 derived artifacts。

### R02-B04 [HIGH] acceptance criteria 的 campaign identity 与 R02 manifest identity 不一致

证据：ACCEPTANCE_CRITERIA.md:3 仍写 foundation_self_relaxed_4x3_20261024；CAMPAIGN_MANIFEST.json:6、PATH_REGISTRY.tsv:2-13、JOB_TABLE_IDPP.tsv:2-109 和 EXECUTION_BOOKKEEPING_R02.json:3 使用 foundation_self_relaxed_4x3_20261024_r02。criteria SHA-256 本身是已正确绑定的 70e2fc23c876721d321138afd3a49ecbb79208878f0b36af848d12bffde2c5a8，因此不能静默改 criteria 或回写已提交 manifest。

修复：保留当前冻结 criteria 与 hash；在 additive bookkeeping/final release manifest 中明确 campaign_base_id=foundation_self_relaxed_4x3_20261024、campaign_revision_id=foundation_self_relaxed_4x3_20261024_r02 的唯一映射，并把该映射纳入 final hash。以后每个 frozen criteria 文件应直接声明适用的 revision 或采用明确的 version policy。

### R02-B05 [HIGH] 当前 collection/derived evidence 尚未形成 release package

证据：远端目前只有 ATTEMPT_LEDGER.tsv、ROW_RESULTS.tsv、IMAGE_PROFILES.tsv、FAILURE_REGISTRY.tsv、COLLECTION_MANIFEST.json；没有 CLASS_PROFILES.tsv、CLASS_SUMMARY.tsv、PROFILE_INCLUSION_AUDIT.tsv 或 DERIVED_PROFILE_MANIFEST.json。当前 30 profile rows 只来自 3 个 formal completed rows，不能形成 4×3 path、9-model 的 class denominator 或 mean/SD。

修复：array 结束后生成全量 row results、attempt ledger、failure registry、10-image profiles、class summary、profile inclusion audit 和 derived manifest；derived profile 只能来自通过 gates 的 self-relaxed rows，并报告 accepted/expected denominator 与所有 excluded rows。任何缺失组合都保持 INCOMPLETE。

### R02-B06 [HIGH] 全量 geometry/profile/convergence QA 尚未完成

证据：代码层面的收敛门已明确：run_one_job.py:201-222 要求 optimizer_converged=true、final_max_internal_neb_force_eV_A<=0.05、final_max_neb_residual_eV_A<=0.05；collect_results.py:130-149 检查 profile 恰为 10 个有序 image 且 energy/relative-energy/fmax 有限。当前 3 个 formal completed row 的 10-point profile 通过这些现有字段门，但 105 个 row 尚无可审计终态，也没有全量 output 的 atom count/cell/PBC/finite coordinate-force audit。

修复：array 结束后逐 row 验证 10×81、image order、symbol order、cell/PBC、坐标/energy/force finite、trajectory/image hash、optimizer convergence 和两个 <=0.05 门；任何异常保留原始 logs、stderr/stdout、runtime provenance 和 failure reason。

## 5. 不应误判的边界

- 108 行 JOB_TABLE_IDPP.tsv 是 execution plan，不是 108 个成功结果。
- 4 个 endpoint manifest 中的 smoke 和 3 个 formal completed row 只能作为当前诊断快照；不能代表 Cu、Li、Na、其他 path class 或全部 9 model。
- canonical endpoint header 修复不等于 formal scientific release；F-04 已修复输入层，但 F-01/F-02/F-05 及终态 provenance 仍阻塞 release。
- MLFF self-relaxed profile 是 model observable / preconditioner，不是 DFT ground truth；fixed-path 与 self-relaxed 仍须保持为不同 observable。

## 6. Re-review 条件

1. 32459698 完成或被明确终止，108 row 状态可逐 row 追溯，且不把 PENDING/RUNNING 当作失败。
2. smoke/formal 角色、parent array job、child job、array task 和所有 attempts 在 ledger 中明确分层。
3. F-03 的 formal launch hash、criteria hash、bookkeeping hash、remote copy/归档和最终终态可复核。
4. F-04 的 24 canonical endpoint header/hash 继续保持一致；任何 endpoint 改动都必须同步更新 registry 和重新执行受影响 row。
5. 全量 collection/failure/derived artifacts 生成，profile inclusion、denominator、finite/geometry/PBC 和 convergence QA 全部通过。
6. final release manifest 明确 self-relaxed/fixed-path separation，并在 independent data review 与 visual review 都返回 PASS 前保持 INCOMPLETE。

