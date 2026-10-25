# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""Production-context declaration: the questions, their allowed answers, labels and tooltips.
Shared by the Production Context node, the assessment and the report."""

# Date of the rule set (findings, triage, principles). Shown in every report; change it whenever the rules change.
RULES_DATE = "2026-10-25"

YNU = ["unknown", "yes", "no"]
YNUNA = ["unknown", "yes", "no", "not applicable"]
# Evidence fields: is a document on file? Old values ("no", "yes", "not applicable") stay valid.
EVIDENCE = ["unknown", "yes", "no", "pending", "not applicable"]

PURPOSES = ["unknown", "ideation / moodboard", "temporary reference", "research and development",
            "internal production asset", "marketing / publicity", "final deliverable", "other"]
LOW_RISK_PURPOSES = {"ideation / moodboard", "temporary reference", "research and development"}

ENVIRONMENTS = ["auto (from workflow)", "local workstation", "private studio server", "enterprise-secured cloud",
                "public / consumer cloud", "mixed", "unknown"]
SECURE_ENVIRONMENTS = {"local workstation", "private studio server", "enterprise-secured cloud"}

# (key, label) for text fields; (key, label, answers, default) for choices
TEXT_FIELDS = [
    ("production_title", "Production"),
    ("project_episode", "Project / episode"),
    ("shot_asset", "Shot / asset"),
    ("vendor_department", "Vendor / department"),
    ("audit_requester", "Audit requested by"),
]
CHOICE_FIELDS = [
    ("purpose", "Purpose", PURPOSES, "unknown"),
    ("in_final_cut", "Appears in final cut / published result", YNU, "unknown"),
    ("story_critical", "Story-critical / prominent in the output", YNU, "unknown"),
    ("execution_environment", "Execution environment", ENVIRONMENTS, "auto (from workflow)"),
    ("unreleased_material", "Unreleased production material in inputs", YNU, "unknown"),
    ("personal_data", "Personal data in inputs", YNU, "unknown"),
    ("talent_likeness_or_voice", "Talent likeness or voice in inputs", YNU, "unknown"),
    ("third_party_copyright", "Third-party copyrighted material in inputs", YNU, "unknown"),
    ("trademarks_brands", "Trademarks / brands in inputs or output", YNU, "unknown"),
    ("union_covered_work", "Union / guild-covered performance or work", YNU, "unknown"),
    ("external_tools_train", "External tools store / reuse / train on data", YNUNA, "unknown"),
    ("input_rights_documented", "Input-rights documentation on file", EVIDENCE, "unknown"),
    ("talent_consent", "Written talent consent on file", EVIDENCE, "unknown"),
    ("union_approval", "Union / guild approval on file", EVIDENCE, "unknown"),
    ("studio_approval", "Studio / client GenAI approval on file", EVIDENCE, "unknown"),
    ("legal_approval", "Legal clearance on file", EVIDENCE, "unknown"),
]
REVIEWER_FIELDS = [
    ("creative_reviewer", "Creative reviewer"),
    ("technical_reviewer", "Technical / QC reviewer"),
    ("rights_reviewer", "Rights / ethics reviewer"),
]

_EVIDENCE_TIP = (" 'yes' only if you can show a document (contract clause, e-mail, signed form). Verbal OK = 'no' "
                 "or 'pending' plus a note in context_notes. 'unknown' = not checked.")
TOOLTIPS = {
    "production_title": "Film, series, campaign or product the work is for.",
    "project_episode": "Project, episode, campaign or job.",
    "shot_asset": "Shot or asset this run produces (e.g. sh010, label design).",
    "vendor_department": "Company or department doing the work.",
    "audit_requester": "Who asked for this audit (person or role).",
    "purpose": "What the output of THIS workflow is for. Multi-step work: audit every step – an intermediate step "
               "is 'internal production asset', the step that makes the finished result is 'final deliverable'. "
               "Low risk: ideation / moodboard, temporary reference, research and development.",
    "in_final_cut": "Whether the generated material – or anything derived from it – appears in the finished, "
                    "published result (film, episode, ad, packaging, print). 'no' for moodboards, previs and "
                    "references that are replaced later. 'yes' usually needs written approval (triage C).",
    "story_critical": "Whether the generated elements carry the story or are prominently visible (a key character, "
                      "a hero product, a main set). Only relevant if they appear in the final result – a moodboard "
                      "for a key scene is not a risk.",
    "execution_environment": "Where the workflow ran. 'auto' = local workstation, or 'mixed' when API / cloud "
                             "nodes are in the run.",
    "unreleased_material": "Unpublished footage, scripts, designs or other confidential material of the "
                           "production in the inputs. Locally (no API / cloud) this is only a contract question; "
                           "sent to a cloud service it is a data-protection trigger.",
    "personal_data": "Images, voices or data of identifiable real people in the inputs (data-protection law "
                     "applies even locally).",
    "talent_likeness_or_voice": "Face, body, performance or voice of a real performer used or changed. 'no' for "
                                "original generated characters.",
    "third_party_copyright": "Material you do not own in the inputs (stock, film stills, artwork, photos by "
                             "others). Your own sketches and plates of the production = 'no'.",
    "trademarks_brands": "Logos, brand names, products or packaging of third parties – in the inputs or likely "
                         "in the output. 'unknown' if a name or design might be close to an existing brand.",
    "union_covered_work": "Performance or work covered by a union or guild agreement (e.g. SAG-AFTRA actors, WGA "
                          "writers). These agreements require consent for AI use. 'no' for original generated "
                          "characters and non-union work.",
    "external_tools_train": "Whether any external service used here may store, reuse or train on inputs or "
                            "outputs (check its terms). 'not applicable' if everything ran locally.",
    "input_rights_documented": "Licence, purchase receipt or permission for third-party input material."
                               + _EVIDENCE_TIP,
    "talent_consent": "Consent of the people whose likeness or voice is used. 'not applicable' if no talent is "
                      "involved." + _EVIDENCE_TIP,
    "union_approval": "Approval under the union / guild agreement. 'not applicable' for non-union work."
                      + _EVIDENCE_TIP,
    "studio_approval": "The production or client agreed that GenAI may be used for THIS asset. A job order alone "
                       "('design the label') is not approval for GenAI." + _EVIDENCE_TIP,
    "legal_approval": "A legal department or lawyer reviewed and cleared the use (e.g. trademark clearance). "
                      "'pending' while the review is open." + _EVIDENCE_TIP,
    "creative_reviewer": "Who checked the result creatively (name or role). Empty = not reviewed yet.",
    "technical_reviewer": "Who checked the result technically / QC (name or role). Empty = not reviewed yet.",
    "rights_reviewer": "Who checked rights and ethics (name or role). Empty = not reviewed yet.",
    "context_notes": "Anything else about rights, approvals or restrictions – e.g. what is pending and with whom.",
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
    "X": "BLOCKED",
}
GRADE_ORDER = ["A", "B", "C", "D", "X"]
TRIAGE_LEGEND = {
    "A": "declared as ideation / temporary reference / R&D, not in the final result, no trigger, nothing open "
         "of medium severity or higher",
    "B": "open points must be reviewed before the material is used",
    "C": "at least one typical trigger for written approval (final result, talent, union work, data leaving the "
         "machine, third-party material without rights, model license UNKNOWN / NOT VERIFIED for published "
         "material …)",
    "X": "a CRITICAL finding – the use as declared is not permitted (e.g. the model license does not allow "
         "commercial use of published material). An approval does not fix it; only a license or a change does",
    "D": "not assessable, because purpose and final result are both undeclared",
}
SEVERITY_LEGEND = ("Severity (critical > high > medium > low) rates a single finding. The grade (A, B, C, D, X) "
                   "says what has to happen before the material is used: CRITICAL always means X.")
EVIDENCE_LEGEND = ("observed = seen in the workflow · derived = concluded from it · declared = your statement "
                   "(model_info file, Production Context) · external evidence required = cannot be shown by the "
                   "workflow, needs a document or an answer")
UNKNOWN_LABEL = "UNKNOWN / NOT VERIFIED"
PRINCIPLES_NOTE = ("YES only with evidence or an explicit declaration – never because nothing contrary was found. "
                   f"{UNKNOWN_LABEL} = not declared or not provable from the workflow.")
TRIAGE_NOTE = ("Indicative first assessment derived only from the declared production context and what the "
               "workflow shows. It is not an approval, not legal advice and not an official studio assessment.")
NOT_ASSESSED = "not assessed – no Production Context connected"

PRINCIPLES = [
    ("P1", "Output does not replicate unowned or copyrighted material"),
    ("P2", "Tools do not store, reuse or train on production data"),
    ("P3", "Runs in a secured environment"),
    ("P4", "Generated material is temporary, not a final deliverable"),
    ("P5", "No talent performance or union work replaced without consent"),
]
