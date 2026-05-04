"""Feedback loop: capture reviewer corrections and inject them into future
extraction prompts as few-shot examples.

This is "in-context learning from human feedback" — every time a reviewer
edits a field, the system stores (form_id, field_id, model_value,
corrected_value, reason). Future extractions of the same form type prepend
recent lessons to the prompt so the model can apply learned patterns.
"""
from __future__ import annotations

import json
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from config import LESSONS_DIR

# Reviewer-facing reason codes. Keep this list short; long lists go unused.
REASON_OPTIONS = [
    {"id": "handwriting_unclear", "label": "Handwriting unclear"},
    {"id": "model_misread",       "label": "Model misread"},
    {"id": "wrong_format",        "label": "Wrong format"},
    {"id": "schema_gap",          "label": "Schema gap (field doesn't fit)"},
    {"id": "policy_correction",   "label": "Policy / business rule"},
    {"id": "other",               "label": "Other"},
]
VALID_REASON_IDS = {r["id"] for r in REASON_OPTIONS}

MAX_LESSONS_IN_PROMPT = 8  # cap injected examples to keep prompts lean


def _path(form_id: str) -> Path:
    return LESSONS_DIR / f"{form_id}.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(form_id: str) -> list[dict]:
    p = _path(form_id)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        return []
    return data.get("lessons", []) if isinstance(data, dict) else []


def _save(form_id: str, lessons: list[dict]) -> None:
    _path(form_id).write_text(json.dumps({"form_id": form_id, "lessons": lessons}, indent=2))


def record_lessons_from_review(record: dict, edit_reasons: dict[str, str], reviewer: str) -> int:
    """For each edited field on this approved record, store a lesson.
    Returns the number of lessons captured."""
    edits = record.get("edits") or {}
    if not edits:
        return 0
    form_id = record.get("form_id")
    if not form_id:
        return 0
    extraction_fields = (record.get("extraction") or {}).get("fields") or {}

    lessons = _load(form_id)
    captured = 0
    for field_id, corrected_value in edits.items():
        original = extraction_fields.get(field_id, {}).get("value")
        if original == corrected_value:
            continue  # no actual change
        reason = edit_reasons.get(field_id, "other")
        if reason not in VALID_REASON_IDS:
            reason = "other"
        lessons.append({
            "id": uuid.uuid4().hex[:10],
            "field_id": field_id,
            "model_value": original,
            "corrected_value": corrected_value,
            "reason": reason,
            "reviewer": reviewer,
            "from_record_id": record.get("id"),
            "at": _now(),
        })
        captured += 1
    if captured:
        _save(form_id, lessons)
    return captured


def get_recent_lessons(form_id: str, limit: int = MAX_LESSONS_IN_PROMPT) -> list[dict]:
    lessons = _load(form_id)
    lessons.sort(key=lambda l: l.get("at", ""), reverse=True)
    return lessons[:limit]


def build_few_shot_block(form_id: str, limit: int = MAX_LESSONS_IN_PROMPT) -> str:
    """Format recent lessons as a prompt fragment. Empty string if no lessons."""
    recent = get_recent_lessons(form_id, limit=limit)
    if not recent:
        return ""

    # Group by field to surface patterns rather than raw events
    by_field: dict[str, list[dict]] = {}
    for l in recent:
        by_field.setdefault(l["field_id"], []).append(l)

    lines = [
        "PAST REVIEWER CORRECTIONS — apply these patterns to your extraction below:",
        "",
    ]
    reason_label = {r["id"]: r["label"] for r in REASON_OPTIONS}
    for field_id, entries in by_field.items():
        lines.append(f"  Field `{field_id}`:")
        for e in entries[:3]:  # cap per-field
            mv = e.get("model_value")
            cv = e.get("corrected_value")
            r = reason_label.get(e.get("reason", "other"), "other")
            mv_str = "null" if mv in (None, "") else f"\"{mv}\""
            cv_str = "null" if cv in (None, "") else f"\"{cv}\""
            lines.append(f"    - model said {mv_str} → reviewer corrected to {cv_str}  [{r}]")
        lines.append("")
    return "\n".join(lines)


def lessons_count(form_id: str | None = None) -> int:
    if form_id:
        return len(_load(form_id))
    total = 0
    for p in LESSONS_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text())
            total += len(data.get("lessons", []))
        except json.JSONDecodeError:
            continue
    return total


def lessons_summary(form_id: str | None = None) -> dict:
    """Aggregate stats for the dashboard widget."""
    if form_id:
        all_lessons = _load(form_id)
    else:
        all_lessons: list[dict] = []
        for p in LESSONS_DIR.glob("*.json"):
            try:
                data = json.loads(p.read_text())
                all_lessons.extend(data.get("lessons", []))
            except json.JSONDecodeError:
                continue

    total = len(all_lessons)
    reason_counter = Counter(l.get("reason", "other") for l in all_lessons)
    field_counter = Counter(l.get("field_id") for l in all_lessons if l.get("field_id"))
    reason_label = {r["id"]: r["label"] for r in REASON_OPTIONS}

    top_reasons = [
        {"id": rid, "label": reason_label.get(rid, rid), "count": cnt}
        for rid, cnt in reason_counter.most_common(5)
    ]
    top_fields = [
        {"field_id": fid, "count": cnt}
        for fid, cnt in field_counter.most_common(5)
    ]

    return {
        "total": total,
        "top_reasons": top_reasons,
        "top_fields": top_fields,
    }


def list_reasons() -> list[dict]:
    return REASON_OPTIONS


def clear_all_lessons() -> int:
    count = 0
    for p in list(LESSONS_DIR.glob("*.json")):
        p.unlink()
        count += 1
    return count
