-- Allow incomplete/blocked recipe rows to express an unknown command without a fake placeholder.
PRAGMA foreign_keys=OFF;
BEGIN IMMEDIATE;

DROP TRIGGER IF EXISTS cleanup_reproducibility_guard_insert;
DROP TRIGGER IF EXISTS cleanup_reproducibility_guard_update;

CREATE TABLE reproduction_recipe_replacement (
  recipe_id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES workflow_run(run_id),
  recipe_version TEXT NOT NULL,
  exact_command TEXT,
  working_directory TEXT NOT NULL,
  executable_version TEXT,
  environment_description TEXT,
  module_list TEXT,
  random_seed TEXT,
  script_artifact_id INTEGER REFERENCES artifact(artifact_id),
  scheduler_script_artifact_id INTEGER REFERENCES artifact(artifact_id),
  input_manifest_artifact_id INTEGER REFERENCES artifact(artifact_id),
  recipe_status TEXT NOT NULL DEFAULT 'DOCUMENTED'
    CHECK (recipe_status IN ('DOCUMENTED','REVIEWED','REPRODUCED','INCOMPLETE','BLOCKED')),
  last_reproduced_utc TEXT,
  validation_note TEXT NOT NULL DEFAULT '',
  UNIQUE(run_id, recipe_version),
  CHECK (
    (recipe_status IN ('DOCUMENTED','REVIEWED','REPRODUCED')
      AND exact_command IS NOT NULL AND trim(exact_command) <> '')
    OR recipe_status IN ('INCOMPLETE','BLOCKED')
  )
);

INSERT INTO reproduction_recipe_replacement (
  recipe_id,run_id,recipe_version,exact_command,working_directory,executable_version,
  environment_description,module_list,random_seed,script_artifact_id,scheduler_script_artifact_id,
  input_manifest_artifact_id,recipe_status,last_reproduced_utc,validation_note
)
SELECT
  recipe_id,run_id,recipe_version,exact_command,working_directory,executable_version,
  environment_description,module_list,random_seed,script_artifact_id,scheduler_script_artifact_id,
  input_manifest_artifact_id,recipe_status,last_reproduced_utc,validation_note
FROM reproduction_recipe;

DROP TABLE reproduction_recipe;
ALTER TABLE reproduction_recipe_replacement RENAME TO reproduction_recipe;

CREATE TRIGGER cleanup_reproducibility_guard_insert
BEFORE INSERT ON cleanup_item
WHEN NEW.disposition = 'DELETE' AND
     (SELECT retention_class FROM artifact WHERE artifact_id = NEW.artifact_id) = 'REPRODUCIBLE_INTERMEDIATE' AND
     NOT EXISTS (
       SELECT 1 FROM run_artifact ra
       JOIN reproduction_recipe rr ON rr.run_id = ra.run_id
       WHERE ra.artifact_id = NEW.artifact_id
         AND ra.direction = 'INTERMEDIATE'
         AND rr.recipe_status IN ('DOCUMENTED','REVIEWED','REPRODUCED')
         AND rr.script_artifact_id IS NOT NULL
         AND rr.scheduler_script_artifact_id IS NOT NULL
         AND rr.input_manifest_artifact_id IS NOT NULL
         AND EXISTS (SELECT 1 FROM artifact_location l WHERE l.artifact_id = rr.script_artifact_id AND l.availability IN ('PRESENT','ARCHIVED'))
         AND EXISTS (SELECT 1 FROM artifact_location l WHERE l.artifact_id = rr.scheduler_script_artifact_id AND l.availability IN ('PRESENT','ARCHIVED'))
         AND EXISTS (SELECT 1 FROM artifact_location l WHERE l.artifact_id = rr.input_manifest_artifact_id AND l.availability IN ('PRESENT','ARCHIVED'))
         AND EXISTS (SELECT 1 FROM run_artifact x JOIN artifact_location l USING (artifact_id) WHERE x.run_id = ra.run_id AND x.direction = 'INPUT' AND l.availability IN ('PRESENT','ARCHIVED'))
         AND EXISTS (SELECT 1 FROM run_artifact x JOIN artifact_location l USING (artifact_id) WHERE x.run_id = ra.run_id AND x.direction = 'OUTPUT' AND l.availability IN ('PRESENT','ARCHIVED'))
     )
BEGIN
  SELECT RAISE(ABORT, 'delete blocked: reproducible-intermediate evidence chain incomplete');
END;

CREATE TRIGGER cleanup_reproducibility_guard_update
BEFORE UPDATE OF disposition, deletion_state ON cleanup_item
WHEN NEW.disposition = 'DELETE' AND
     (SELECT retention_class FROM artifact WHERE artifact_id = NEW.artifact_id) = 'REPRODUCIBLE_INTERMEDIATE' AND
     NOT EXISTS (
       SELECT 1 FROM run_artifact ra
       JOIN reproduction_recipe rr ON rr.run_id = ra.run_id
       WHERE ra.artifact_id = NEW.artifact_id
         AND ra.direction = 'INTERMEDIATE'
         AND rr.recipe_status IN ('DOCUMENTED','REVIEWED','REPRODUCED')
         AND rr.script_artifact_id IS NOT NULL
         AND rr.scheduler_script_artifact_id IS NOT NULL
         AND rr.input_manifest_artifact_id IS NOT NULL
         AND EXISTS (SELECT 1 FROM artifact_location l WHERE l.artifact_id = rr.script_artifact_id AND l.availability IN ('PRESENT','ARCHIVED'))
         AND EXISTS (SELECT 1 FROM artifact_location l WHERE l.artifact_id = rr.scheduler_script_artifact_id AND l.availability IN ('PRESENT','ARCHIVED'))
         AND EXISTS (SELECT 1 FROM artifact_location l WHERE l.artifact_id = rr.input_manifest_artifact_id AND l.availability IN ('PRESENT','ARCHIVED'))
         AND EXISTS (SELECT 1 FROM run_artifact x JOIN artifact_location l USING (artifact_id) WHERE x.run_id = ra.run_id AND x.direction = 'INPUT' AND l.availability IN ('PRESENT','ARCHIVED'))
         AND EXISTS (SELECT 1 FROM run_artifact x JOIN artifact_location l USING (artifact_id) WHERE x.run_id = ra.run_id AND x.direction = 'OUTPUT' AND l.availability IN ('PRESENT','ARCHIVED'))
     )
BEGIN
  SELECT RAISE(ABORT, 'delete blocked: reproducible-intermediate evidence chain incomplete');
END;

INSERT INTO schema_migration(version,migration_file,applied_utc,description)
VALUES (14,'014_incomplete_recipes_allow_missing_command.sql',
        strftime('%Y-%m-%dT%H:%M:%fZ','now'),
        'Allow INCOMPLETE/BLOCKED recipes to preserve absent exact commands without placeholders');

COMMIT;
PRAGMA foreign_keys=ON;
