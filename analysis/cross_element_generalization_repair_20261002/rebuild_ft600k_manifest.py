#!/usr/bin/env python3
"""Emit a new-checkpoint repair manifest; no training or source mutation."""
from __future__ import annotations
import json, hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parent
TARGET=["Cr","Cu","Li","Na","Sb","Te"]
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def main():
 foundation=(ROOT/'../task2_cross_element/models/Foundation-MACE-matpes-pbe-omat-ft.model').resolve()
 ft=(ROOT/'../task2_cross_element/models/FT-600K.model').resolve()
 out={"schema_version":"ft600k-broad-roster-repair-plan-1.0","status":"plan_only_no_checkpoint_created","target_elements":TARGET,"base_foundation":{"path":str(foundation),"sha256":sha(foundation)},"reference_ft600k":{"path":str(ft),"sha256":sha(ft)},"required_training_design":{"initialization":"Foundation checkpoint","task_data":"Cr/Sb/Te AMD fine-tuning trajectory with original labels and provenance","replay_data":"held-out broad-element Foundation/MPA0 anchor frames for Cr, Cu, Li, Na, Sb, Te","loss":"energy + forces with explicit replay weighting","validation":"element-wise probes plus held-out energy/force metrics and catastrophic-forgetting comparison"},"forbidden_shortcut":"Do not edit FT-600K atomic_numbers or append elements without allocating/training the missing embeddings and verifying outputs.","new_output_namespace":str(ROOT/'outputs/ft600k_broad_roster_repaired.model')}
 (ROOT/'ft600k_repair_manifest.json').write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
