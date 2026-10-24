# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""Production-context declaration: the questions, their allowed answers and labels.
Shared by the Production Context node, the assessment and the report."""

YNU = ["unknown", "yes", "no"]
YNUNA = ["unknown", "yes", "no", "not applicable"]
YN = ["no", "yes"]
YNNA = ["no", "yes", "not applicable"]

PURPOSES = ["unknown", "ideation / moodboard", "temporary reference", "research and development",
            "internal production asset", "marketing / publicity", "final deliverable", "other"]
LOW_RISK_PURPOSES = {"ideation / moodboard", "temporary reference", "research and development"}

ENVIRONMENTS = ["auto (from workflow)", "local workstation", "private studio server", "enterprise-secured cloud",
                "public / consumer cloud", "mixed", "unknown"]
SECURE_ENVIRONMENTS = {"local workstation", "private studio server", "enterprise-secured cloud"}

# (key, label, answers, default) – text fields use answers=None
TEXT_FIELDS = [
    ("production_title", "Production"),
    ("project_episode", "Project / episode"),
    ("shot_asset", "Shot / asset"),
    ("vendor_department", "Vendor / department"),
    ("audit_requester", "Audit requested by"),
]
CHOICE_FIELDS = [
    ("purpose", "Purpose", PURPOSES, "unknown"),
    ("in_final_cut", "Appears in final cut", YNU, "unknown"),
    ("story_critical", "Story-critical / prominent in the output", YNU, "unknown"),
    ("execution_environment", "Execution environment", ENVIRONMENTS, "auto (from workflow)"),
    ("unreleased_material", "Unreleased production material in inputs", YNU, "unknown"),
    ("personal_data", "Personal data in inputs", YNU, "unknown"),
    ("talent_likeness_or_voice", "Talent likeness or voice in inputs", YNU, "unknown"),
    ("third_party_copyright", "Third-party copyrighted material in inputs", YNU, "unknown"),
    ("trademarks_brands", "Trademarks / branded assets in inputs", YNU, "unknown"),
    ("union_covered_work", "Union-covered performance or work", YNU, "unknown"),
    ("external_tools_train", "External tools store / reuse / train on data", YNUNA, "unknown"),
    ("input_rights_documented", "Input-rights documentation available", YN, "no"),
    ("talent_consent", "Talent consent available", YNNA, "no"),
    ("union_approval", "Union / guild approval available", YNNA, "no"),
    ("studio_approval", "Studio / production approval available", YN, "no"),
    ("legal_approval", "Legal approval available", YNNA, "no"),
]
REVIEWER_FIELDS = [
    ("creative_reviewer", "Creative reviewer"),
    ("technical_reviewer", "Technical / QC reviewer"),
    ("rights_reviewer", "Rights / ethics reviewer"),
]

TOOLTIPS = {
    "purpose": "What the generated material is for. 'final deliverable' and 'final cut = yes' usually need "
               "written approval in studio productions.",
    "in_final_cut": "Whether generated material (or material derived from it) appears in the finished film / "
                    "deliverable. 'no' for moodboards, previs and references that are replaced later.",
    "story_critical": "Whether the generated elements carry the story or are prominently visible (a key character, "
                      "a hero prop, a main set). Only relevant if they appear in the final cut – a moodboard for a "
                      "key scene is not a risk.",
    "execution_environment": "Where the workflow ran. 'auto' = local workstation, or 'mixed' when API / cloud "
                             "nodes are in the run.",
    "external_tools_train": "Whether any external service used here may store, reuse or train on inputs or "
                            "outputs (check its terms). 'not applicable' if everything ran locally.",
    "talent_consent": "Documented consent of the people whose likeness or voice is used. 'not applicable' if no "
                      "talent is involved.",
}

# Departments a finding is routed to
ROUTES = {
    "legal": "Legal / rights",
    "talent": "Talent & consent",
    "security": "Data protection / IT security",
    "production": "Production / studio approval",
    "vfx": "VFX supervision / QC",
    "docs": "Documentation",
}
SEVERITIES = ["critical", "high", "medium", "low", "info"]

TRIAGE = {
    "A": "LOW-RISK CANDIDATE",
    "B": "REVIEW REQUIRED",
    "C": "WRITTEN APPROVAL INDICATED",
    "D": "INDETERMINATE",
}
TRIAGE_LEGEND = {
    "A": "declared as ideation / temporary reference / R&D, not in the final cut, no trigger, nothing open of "
         "medium severity or higher",
    "B": "open points must be reviewed before the material is used",
    "C": "at least one typical trigger for written approval (final cut, talent, union work, data leaving the "
         "machine, third-party material without rights, credential found …)",
    "D": "no production context connected, or purpose and final cut both undeclared",
}
EVIDENCE_LEGEND = ("observed = seen in the workflow · derived = concluded from it · declared = your statement "
                   "(model_info file, Production Context) · external evidence required = cannot be shown by the "
                   "workflow, needs a document or an answer")
PRINCIPLES_NOTE = ("YES only with evidence or an explicit declaration – never because nothing contrary was found. "
                   "UNKNOWN = not declared or not provable from the workflow.")
TRIAGE_NOTE = ("Indicative first assessment derived only from the declared production context and what the "
               "workflow shows. It is not an approval, not legal advice and not an official studio assessment.")

PRINCIPLES = [
    ("P1", "Output does not replicate unowned or copyrighted material"),
    ("P2", "Tools do not store, reuse or train on production data"),
    ("P3", "Runs in a secured environment"),
    ("P4", "Generated material is temporary, not a final deliverable"),
    ("P5", "No talent performance or union work replaced without consent"),
]
