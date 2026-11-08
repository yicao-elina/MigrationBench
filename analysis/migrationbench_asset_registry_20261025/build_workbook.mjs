#!/usr/bin/env node
// Build the reviewable spreadsheet mirror of the SQLite asset registry.
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = path.dirname(fileURLToPath(import.meta.url));
const data = JSON.parse(await fs.readFile(path.join(root, "registry_data.json"), "utf8"));
const outputPath = path.join(root, "migrationbench_asset_registry.xlsx");

const colors = {
  navy: "#17365D",
  teal: "#0F766E",
  paleBlue: "#EAF2F8",
  paleTeal: "#E8F3F1",
  paleGold: "#FFF4D6",
  paleRed: "#FDECEC",
  paleGray: "#F4F6F8",
  line: "#D6DEE8",
  ink: "#203040",
  white: "#FFFFFF",
};

const fields = {
  models: ["model_id", "display_name", "family_id", "identity_kind", "training_regime", "training_data_label", "training_temperature_label", "training_seed", "representation_label", "checkpoint_sha256", "artifact_path", "lineage_status", "release_status", "scientific_status", "active_scope", "notes"],
  model_groups: ["model_group_id", "display_name", "major_class", "group_level", "sort_order", "description"],
  model_name_map: ["model_id", "preferred_name", "plot_label", "short_label", "major_class", "subgroup_label", "search_text", "naming_status", "notes"],
  model_aliases: ["alias_id", "model_id", "model_group_id", "alias", "alias_context", "preferred"],
  coverage: ["coverage_id", "model_id", "task_id", "system_id", "campaign_id", "dataset_id", "scope_label", "expected_count", "completed_count", "run_status", "scientific_status", "dft_reference_status", "result_summary", "source_id"],
  datasets: ["dataset_id", "display_name", "data_layer", "task_id", "system_id", "campaign_id", "record_count", "unit_of_record", "scientific_status", "dft_reference_status", "notes"],
  tasks: ["task_id", "category", "operator", "comparison_layer", "description", "publication_boundary"],
  task_groups: ["task_group_id", "display_name", "task_layer", "sort_order", "description"],
  task_group_members: ["task_group_id", "task_id", "membership_role"],
  task_aliases: ["alias_id", "task_id", "alias", "alias_context", "preferred"],
  task_group_aliases: ["alias_id", "task_group_id", "alias", "alias_context", "preferred"],
  systems: ["system_id", "material_family", "chemistry", "atom_count", "defect_or_geometry", "notes"],
  campaigns: ["campaign_id", "title", "campaign_path", "campaign_date", "status", "notes"],
  assets: ["asset_id", "asset_type", "display_name", "local_path", "campaign_id", "existence_at_build", "scientific_status", "notes"],
  gaps: ["gap_id", "priority", "entity_type", "entity_id", "gap_type", "description", "recommended_next_action", "status"],
  sources: ["source_id", "title", "source_path", "source_kind", "evidence_level", "as_of_date", "notes"],
  audit_log: ["checked_at", "check_name", "result", "details"],
  charts: ["chart_id", "title", "chart_kind", "chart_status", "asset_path", "asset_mime", "asset_exists", "campaign_id", "source_id", "data_query", "notes"],
  chart_panels: ["chart_id", "panel_id", "panel_label", "panel_title", "model_id", "task_id", "system_id", "dataset_id", "data_query", "notes"],
  chart_data_files: ["chart_id", "file_id", "display_name", "local_path", "file_format", "file_exists", "notes"],
  chart_provenance: ["chart_id", "manifest_path", "manifest_exists", "output_sha256", "input_file_hashes", "processing_command", "provenance_status", "notes"],
};

const labels = {
  model_id: "Model ID", display_name: "Display name", family_id: "Model family", identity_kind: "Identity kind",
  training_regime: "Training regime", training_data_label: "Training data / protocol", training_temperature_label: "Temperature protocol",
  training_seed: "Training seed", representation_label: "Architecture / representation", checkpoint_sha256: "Checkpoint SHA256",
  artifact_path: "Artifact path", lineage_status: "Lineage status", release_status: "Release status", scientific_status: "Scientific status",
  active_scope: "Active scope", notes: "Notes", coverage_id: "Coverage ID", task_id: "Task ID", system_id: "System ID",
  campaign_id: "Campaign ID", dataset_id: "Dataset ID", scope_label: "Coverage scope", expected_count: "Expected", completed_count: "Completed",
  run_status: "Run status", dft_reference_status: "DFT reference status", result_summary: "Result / boundary", source_id: "Source ID",
  data_layer: "Data layer", record_count: "Records", unit_of_record: "Unit of record", category: "Category", operator: "Operator",
  comparison_layer: "Comparison layer", description: "Description", publication_boundary: "Publication boundary", material_family: "Material family",
  chemistry: "Chemistry", atom_count: "Atom count", defect_or_geometry: "Defect / geometry", title: "Title", campaign_path: "Campaign path",
  campaign_date: "Campaign date", status: "Status", asset_id: "Asset ID", asset_type: "Asset type", local_path: "Local path",
  existence_at_build: "Exists at build", gap_id: "Gap ID", priority: "Priority", entity_type: "Entity type", entity_id: "Entity ID",
  gap_type: "Gap type", recommended_next_action: "Recommended next action", source_path: "Source path", source_kind: "Source kind",
  evidence_level: "Evidence level", as_of_date: "As-of date", checked_at: "Checked at", check_name: "Check", result: "Result", details: "Details",
  model_group_id: "Model group ID", group_level: "Group level", sort_order: "Sort order", preferred_name: "Preferred name",
  plot_label: "Plot label", short_label: "Short label", major_class: "Major class", subgroup_label: "Subgroup",
  search_text: "Search text", naming_status: "Naming status", alias_id: "Alias ID", alias: "Alias",
  alias_context: "Alias context", preferred: "Preferred", task_group_id: "Task group ID", task_layer: "Task layer",
  membership_role: "Membership role", chart_id: "Chart ID", chart_kind: "Chart kind", chart_status: "Chart status",
  asset_path: "Asset path", asset_mime: "Asset MIME", asset_exists: "Asset exists", data_query: "Data query",
  panel_id: "Panel ID", panel_label: "Panel label", panel_title: "Panel title", file_id: "File ID",
  file_format: "File format", file_exists: "File exists", manifest_path: "Manifest path", manifest_exists: "Manifest exists",
  output_sha256: "Output SHA256", input_file_hashes: "Input file hashes", processing_command: "Processing command",
  provenance_status: "Provenance status",
};

const sheetDefinitions = [
  { name: "Models", key: "models", title: "Model Registry", subtitle: "Canonical identities, historical variants, campaign labels, seeds, and lineage status." },
  { name: "Model Groups", key: "model_groups", title: "Model Groups", subtitle: "Major classes and protocol subgroups used for filtering and plot naming." },
  { name: "Model Name Map", key: "model_name_map", title: "Model Name Map", subtitle: "Canonical names, standardized plot labels, subgroup labels, and search text." },
  { name: "Model Aliases", key: "model_aliases", title: "Model Aliases", subtitle: "Exact model aliases and group-level shorthand; aliases never replace canonical IDs." },
  { name: "Task Matrix", key: "coverage", title: "Model × Task Coverage", subtitle: "Fact grain: model identity × task/operator × material/system × campaign. Aggregate rows remain explicit." },
  { name: "Data Assets", key: "datasets", title: "Data Assets", subtitle: "DFT reference, MLIP prediction, MD, NEB, and OOD data sets with qualification boundaries." },
  { name: "Tasks", key: "tasks", title: "Task Taxonomy", subtitle: "Operator definitions and non-negotiable publication boundaries." },
  { name: "Task Groups", key: "task_groups", title: "Task Groups", subtitle: "Normalized DFT, MLIP, MD, NEB, interlayer, and OOD categories." },
  { name: "Task Group Members", key: "task_group_members", title: "Task Group Members", subtitle: "Fine-grained task IDs assigned to each normalized task group." },
  { name: "Task Aliases", key: "task_aliases", title: "Task Aliases", subtitle: "Search aliases for individual task/operator records." },
  { name: "Task Group Aliases", key: "task_group_aliases", title: "Task Group Aliases", subtitle: "Human shorthand for normalized task groups." },
  { name: "Systems", key: "systems", title: "Materials & Systems", subtitle: "Separate Cr81, Cr61, Cu/Li/Na, interlayer, and OOD scopes." },
  { name: "Campaigns", key: "campaigns", title: "Campaigns", subtitle: "Campaign paths and current acceptance state." },
  { name: "Assets", key: "assets", title: "Artifact Locators", subtitle: "Linked files/directories and whether they existed when this registry was built." },
  { name: "Gaps", key: "gaps", title: "Open Gaps & Repairs", subtitle: "A live repair queue. Do not close an item without auditable evidence." },
  { name: "Sources", key: "sources", title: "Evidence Sources", subtitle: "Input documents and machine-readable registry locations." },
  { name: "Audit", key: "audit_log", title: "Build Audit", subtitle: "Structural checks performed during registry creation." },
  { name: "Charts", key: "charts", title: "Chart Registry", subtitle: "Stable chart IDs, rendered assets, status, and read-only data queries." },
  { name: "Chart Panels", key: "chart_panels", title: "Chart Panels", subtitle: "Panel-level task, system, dataset, and model lineage." },
  { name: "Chart Data Files", key: "chart_data_files", title: "Chart Data Files", subtitle: "CSV/TSV/JSON source data links for each chart." },
  { name: "Chart Provenance", key: "chart_provenance", title: "Chart Provenance", subtitle: "Manifest locators and provenance qualification for each chart." },
];

function columnLetter(index) {
  let n = index + 1;
  let output = "";
  while (n > 0) {
    const remainder = (n - 1) % 26;
    output = String.fromCharCode(65 + remainder) + output;
    n = Math.floor((n - 1) / 26);
  }
  return output;
}

function valueForCell(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  return value;
}

function columnWidth(field) {
  if (["notes", "description", "publication_boundary", "result_summary", "recommended_next_action", "details"].includes(field)) return 48;
  if (["local_path", "source_path", "artifact_path", "campaign_path"].includes(field)) return 42;
  if (["model_id", "coverage_id", "campaign_id", "dataset_id", "system_id", "asset_id", "gap_id"].includes(field)) return 27;
  if (["display_name", "title", "scope_label", "defect_or_geometry", "representation_label"].includes(field)) return 30;
  if (field.includes("status") || ["identity_kind", "training_regime", "data_layer", "evidence_level", "comparison_layer", "gap_type"].includes(field)) return 21;
  if (["expected_count", "completed_count", "record_count", "atom_count", "training_seed", "existence_at_build"].includes(field)) return 12;
  return 18;
}

function statusFill(value) {
  const text = String(value || "");
  if (/FAIL|EXCLUDE|UNSTABLE|ERROR/.test(text)) return colors.paleRed;
  if (/WARN|PARTIAL|PENDING|OPEN|NEEDS|PROVISIONAL|UNAUDITED|SCF/.test(text)) return colors.paleGold;
  if (/COMPLETE|VERIFIED|CONVERGED|SUPPORTED|PASS/.test(text)) return colors.paleTeal;
  return colors.paleGray;
}

function addDataSheet(workbook, definition) {
  const sheet = workbook.worksheets.add(definition.name);
  sheet.showGridLines = false;
  sheet.tabColor = definition.key === "gaps" ? "#C2410C" : definition.key === "models" ? colors.teal : colors.navy;
  const records = data[definition.key] || [];
  const keys = fields[definition.key];
  const lastColumn = columnLetter(keys.length - 1);
  sheet.getRange(`A1:${lastColumn}1`).format = { fill: colors.navy, font: { name: "Aptos", size: 16, bold: true, color: colors.white }, verticalAlignment: "center" };
  sheet.getRange("A1").values = [[definition.title]];
  sheet.getRange(`A2:${lastColumn}2`).format = { fill: colors.paleBlue, font: { name: "Aptos", size: 10, color: colors.ink, italic: true }, verticalAlignment: "center", wrapText: true };
  sheet.getRange("A2").values = [[definition.subtitle]];
  sheet.getRange(`A4:${lastColumn}4`).format = { fill: colors.paleTeal, font: { name: "Aptos", size: 10, bold: true, color: colors.ink } };
  sheet.getRange("A4").values = [[`${records.length} rows • filter directly in the table below`]];
  const headerRow = keys.map((field) => labels[field] || field);
  const values = records.map((record) => keys.map((field) => valueForCell(record[field])));
  sheet.getRange(`A5:${lastColumn}${5 + values.length}`).values = [headerRow, ...values];
  const table = sheet.tables.add(`A5:${lastColumn}${5 + values.length}`, true, `${definition.key.replace(/[^a-zA-Z0-9]/g, "")}Table`);
  table.style = "TableStyleMedium2";
  table.showFilterButton = true;
  sheet.freezePanes.freezeRows(5);
  sheet.getRange(`A5:${lastColumn}${5 + values.length}`).format.font = { name: "Aptos", size: 10, color: colors.ink };
  sheet.getRange(`A6:${lastColumn}${5 + values.length}`).format.wrapText = true;
  sheet.getRange(`A1:${lastColumn}2`).format.rowHeight = 24;
  sheet.getRange(`A2:${lastColumn}2`).format.rowHeight = 34;
  sheet.getRange(`A5:${lastColumn}5`).format.rowHeight = 28;
  for (let index = 0; index < keys.length; index += 1) {
    sheet.getRangeByIndexes(0, index, values.length + 5, 1).format.columnWidth = columnWidth(keys[index]);
  }
  for (const statusField of ["scientific_status", "run_status", "lineage_status", "release_status", "status", "priority", "result"]) {
    const index = keys.indexOf(statusField);
    if (index >= 0 && values.length > 0) {
      for (let row = 0; row < values.length; row += 1) {
        const cell = sheet.getCell(row + 5, index);
        cell.format.fill = statusFill(values[row][index]);
        cell.format.font = { name: "Aptos", size: 10, bold: true, color: colors.ink };
      }
    }
  }
  if (values.length > 0) sheet.getRange(`A6:${lastColumn}${5 + values.length}`).format.autofitRows();
  return sheet;
}

const workbook = Workbook.create();
const overview = workbook.worksheets.add("Overview");
overview.showGridLines = false;
overview.tabColor = colors.teal;
overview.getRange("A1:J1").format = { fill: colors.navy, font: { name: "Aptos", size: 18, bold: true, color: colors.white }, verticalAlignment: "center" };
overview.getRange("A1").values = [["MigrationBench Asset Registry"]];
overview.getRange("A2:J2").format = { fill: colors.paleBlue, font: { name: "Aptos", size: 10, color: colors.ink, italic: true }, verticalAlignment: "center" };
overview.getRange("A2").values = [["Evidence-aware inventory • initialized 2026-10-25 • SQLite is the source of truth; this workbook is the review and Google Sheets import mirror."]];
overview.getRange("A4:B4").values = [["Registry scope", "Count"]];
overview.getRange("A4:B4").format = { fill: colors.teal, font: { name: "Aptos", size: 11, bold: true, color: colors.white } };
overview.getRange("A5:A11").values = [["Models"], ["Model families"], ["Task definitions"], ["Systems / materials"], ["Data assets"], ["Coverage records"], ["Open gaps"]];
overview.getRange("B5:B11").formulas = [["=COUNTA(Models!A6:A1000)"], ["=COUNTA('Task Matrix'!A6:A1000)"], ["=COUNTA(Tasks!A6:A1000)"], ["=COUNTA(Systems!A6:A1000)"], ["=COUNTA('Data Assets'!A6:A1000)"], ["=COUNTA('Task Matrix'!A6:A1000)"], ["=COUNTA(Gaps!A6:A1000)"]];
// The model-family count uses a source data constant because no pivot helper is needed in the exchange workbook.
overview.getRange("B6").values = [[data.model_families.length]];
overview.getRange("A5:B11").format = { font: { name: "Aptos", size: 11, color: colors.ink }, borders: { preset: "all", style: "thin", color: colors.line } };
overview.getRange("B5:B11").format = { fill: colors.paleTeal, font: { name: "Aptos", size: 11, bold: true, color: colors.ink }, horizontalAlignment: "center" };
overview.getRange("D4").format = { fill: colors.teal, font: { name: "Aptos", size: 11, bold: true, color: colors.white } };
overview.getRange("D4").values = [["How to read this registry"]];
overview.getRange("D5:D11").values = [
  ["1. Model identities are immutable: family name, checkpoint path/SHA, training seed, and MD seed are not interchangeable."],
  ["2. The central matrix is model × task/operator × material/system × campaign. Filter the Task Matrix first."],
  ["3. Fixed-path NEB, self-relaxed NEB, and MIC are separate observables; never pool them into one barrier ranking."],
  ["4. Cr61, Cr81, Cu/Li/Na, interlayer sliding, and OOD explosion probes are separate system partitions."],
  ["5. Scheduler completion, finite values, convergence, and publication acceptance are different gates; status columns preserve that distinction."],
  ["6. Open gaps are intentional records, not omissions. Close only after adding an auditable locator/source."],
  ["7. The legacy compiled OMAT artifact remains visible as EXCLUDE_FROM_PUBLICATION rather than being silently removed."],
];
overview.getRange("D5:D11").format = { fill: colors.paleGray, font: { name: "Aptos", size: 10, color: colors.ink }, wrapText: true, verticalAlignment: "center", borders: { preset: "all", style: "thin", color: colors.line } };
overview.getRange("A13:J13").format = { fill: colors.navy, font: { name: "Aptos", size: 11, bold: true, color: colors.white } };
overview.getRange("A13").values = [["Status guide"]];
overview.getRange("A14:C18").values = [
  ["Verified / complete", "Green", "Evidence and technical gate available; still respect task-specific boundary."],
  ["Partial / warning / pending", "Amber", "Useful but cannot support an unqualified scientific claim."],
  ["Failure / exclude", "Red", "Retained as evidence; do not use as accepted result."],
  ["Campaign aggregate", "Gray", "Source reports a total but the per-model mapping remains intentionally unresolved."],
  ["Source of truth", "Teal", "SQLite database; spreadsheet can be imported into Google Sheets for collaborative review."],
];
overview.getRange("A14:C18").format = { font: { name: "Aptos", size: 10, color: colors.ink }, wrapText: true, borders: { preset: "all", style: "thin", color: colors.line } };
overview.getRange("A14:C14").format.fill = colors.paleTeal;
overview.getRange("A15:C15").format.fill = colors.paleGold;
overview.getRange("A16:C16").format.fill = colors.paleRed;
overview.getRange("A17:C17").format.fill = colors.paleGray;
overview.getRange("A18:C18").format.fill = colors.paleBlue;
overview.getRange("A:A").format.columnWidth = 27;
overview.getRange("B:B").format.columnWidth = 14;
overview.getRange("C:C").format.columnWidth = 40;
overview.getRange("D:D").format.columnWidth = 106;
overview.getRange("E:J").format.columnWidth = 8;
overview.getRange("A1:J1").format.rowHeight = 30;
overview.getRange("A2:J2").format.rowHeight = 32;
overview.getRange("D5:D11").format.rowHeight = 34;
overview.freezePanes.freezeRows(3);

for (const definition of sheetDefinitions) addDataSheet(workbook, definition);

const readme = workbook.worksheets.add("Read Me");
readme.showGridLines = false;
readme.tabColor = colors.navy;
readme.getRange("A1:H1").format = { fill: colors.navy, font: { name: "Aptos", size: 16, bold: true, color: colors.white } };
readme.getRange("A1").values = [["Registry operating guide"]];
readme.getRange("A3:B9").values = [
  ["Purpose", "A clean, queryable bridge between current MigrationBench evidence and future campaigns."],
  ["Authoritative store", "migrationbench_asset_registry.sqlite. This workbook is intentionally a mirror, not the master database."],
  ["Google Sheets", "Import this .xlsx as a native Google Sheet after the Google Drive connection is enabled. Keep the SQLite file as the durable source of truth."],
  ["Add a model", "Create a canonical model ID first. Record family, base/lineage, training seed, checkpoint SHA256, artifact path, and aliases separately."],
  ["Add an evaluation", "Add a coverage row at model × task/operator × system × campaign. Keep run status distinct from scientific status."],
  ["Add a data package", "Create source, campaign, dataset, and asset-locator records before attaching results."],
  ["Resolve a gap", "Update the Gaps tab/database only after linking a concrete manifest, hash, audit, or accepted result source."],
];
readme.getRange("A3:B9").format = { font: { name: "Aptos", size: 11, color: colors.ink }, wrapText: true, verticalAlignment: "center", borders: { preset: "all", style: "thin", color: colors.line } };
readme.getRange("A3:A9").format = { fill: colors.paleTeal, font: { name: "Aptos", size: 11, bold: true, color: colors.ink } };
readme.getRange("A:A").format.columnWidth = 24;
readme.getRange("B:B").format.columnWidth = 100;
readme.getRange("A3:B9").format.rowHeight = 42;

workbook.recalculate();
const inspect = await workbook.inspect({ kind: "workbook,sheet,table", include: "id,name,range", tableMaxRows: 2, tableMaxCols: 6, maxResults: 200 });
await fs.writeFile(path.join(root, "workbook_inspect.json"), JSON.stringify(inspect, null, 2));
const formulaErrors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 100 }, summary: "formula error scan" });
await fs.writeFile(path.join(root, "workbook_formula_errors.json"), JSON.stringify(formulaErrors, null, 2));

// Rendering every sheet can block when the workspace is on a cloud-mounted path.
// Keep previews available for an explicit visual-QA run, but make the normal
// workbook build deterministic and independent of that optional step.
if (process.env.MIGRATIONBENCH_RENDER_PREVIEWS === "1") {
  const previewDir = path.join(root, "previews");
  await fs.mkdir(previewDir, { recursive: true });
  for (const sheetName of ["Overview", ...sheetDefinitions.map((x) => x.name), "Read Me"]) {
    const preview = await workbook.render({ sheetName, range: "A1:J20", scale: 1.25, format: "png" });
    await fs.writeFile(path.join(previewDir, `${sheetName.replace(/[^a-z0-9]+/gi, "_").toLowerCase()}.png`), new Uint8Array(await preview.arrayBuffer()));
  }
}

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
console.log(JSON.stringify({ outputPath, sheetCount: 12, workbookInspect: "workbook_inspect.json", formulaErrors: "workbook_formula_errors.json" }, null, 2));
