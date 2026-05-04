from __future__ import annotations

import base64
import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import PROCESSED_DIR, REVIEWERS, SCHEMAS_DIR
from lessons import record_lessons_from_review

STATUS_NEEDS_REVIEW = "needs_review"
STATUS_AUTO_APPROVED = "auto_approved"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"


def _record_path(record_id: str) -> Path:
    return PROCESSED_DIR / f"{record_id}.json"


def _image_path(record_id: str) -> Path:
    return PROCESSED_DIR / f"{record_id}.png"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _next_reviewer() -> dict:
    """Round-robin: pick the reviewer with the fewest open (needs_review) records."""
    counts = {r["id"]: 0 for r in REVIEWERS}
    for record in list_records():
        if record["status"] == STATUS_NEEDS_REVIEW and record.get("assigned_to"):
            if record["assigned_to"] in counts:
                counts[record["assigned_to"]] += 1
    chosen_id = min(counts, key=counts.get)
    return next(r for r in REVIEWERS if r["id"] == chosen_id)


def list_reviewers() -> list[dict]:
    return REVIEWERS


def save_extraction(extraction_result: dict, original_filename: str) -> dict:
    """Persist an extraction. Returns the saved record (without preview image)."""
    record_id = uuid.uuid4().hex[:12]

    flagged = extraction_result.get("flagged_fields", [])
    if flagged:
        status = STATUS_NEEDS_REVIEW
        assigned = _next_reviewer()
    else:
        status = STATUS_AUTO_APPROVED
        assigned = None

    extraction_data = extraction_result.get("extraction") or {}
    overall = extraction_data.get("overall_confidence", 0)
    overall_trust = extraction_data.get("overall_trust_score", overall)
    hallucination_catches = extraction_data.get("hallucination_catches", 0)
    cross_field_issues = extraction_data.get("cross_field_issues") or []

    # Save preview PNG separately
    image_b64 = extraction_result.get("preview_image_b64")
    if image_b64:
        _image_path(record_id).write_bytes(base64.b64decode(image_b64))

    record = {
        "id": record_id,
        "filename": original_filename,
        "form_id": extraction_result["form_id"],
        "form_name": extraction_result["form_name"],
        "extracted_at": _now_iso(),
        "status": status,
        "assigned_to": assigned["id"] if assigned else None,
        "assigned_to_name": assigned["name"] if assigned else None,
        "overall_confidence": overall,
        "overall_trust_score": overall_trust,
        "hallucination_catches": hallucination_catches,
        "cross_field_issues_count": len(cross_field_issues),
        "flagged_fields": flagged,
        "confidence_threshold": extraction_result.get("confidence_threshold"),
        "schema": extraction_result["schema"],
        "extraction": extraction_result["extraction"],
        "edits": {},  # field_id -> edited value (overrides extraction.fields[id].value)
        "history": [
            {"at": _now_iso(), "action": "extracted", "actor": "system",
             "detail": f"Auto-status: {status}"},
        ],
    }
    _record_path(record_id).write_text(json.dumps(record, indent=2))
    return record


def list_records(status: str | None = None, assigned_to: str | None = None,
                 date_from: str | None = None, date_to: str | None = None) -> list[dict]:
    records = []
    for path in PROCESSED_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if status and data.get("status") != status:
            continue
        if assigned_to and data.get("assigned_to") != assigned_to:
            continue
        if date_from and data.get("extracted_at", "") < date_from:
            continue
        if date_to and data.get("extracted_at", "") > date_to:
            continue
        records.append(data)
    records.sort(key=lambda r: r.get("extracted_at", ""), reverse=True)
    return records


def compute_stats(date_from: str | None = None, date_to: str | None = None) -> dict:
    records = list_records(date_from=date_from, date_to=date_to)
    total = len(records)

    by_status = {STATUS_NEEDS_REVIEW: 0, STATUS_AUTO_APPROVED: 0,
                 STATUS_APPROVED: 0, STATUS_REJECTED: 0}
    by_reviewer = {r["id"]: {"name": r["name"], "total": 0, "approved": 0,
                              "rejected": 0, "pending": 0} for r in REVIEWERS}
    confidence_sum = 0.0
    trust_sum = 0.0
    edit_count = 0
    flagged_total = 0
    flagged_field_total = 0
    hallucination_catches_total = 0
    cross_field_issues_total = 0

    for r in records:
        st = r.get("status", "")
        if st in by_status:
            by_status[st] += 1
        confidence_sum += r.get("overall_confidence", 0) or 0
        trust_sum += r.get("overall_trust_score", r.get("overall_confidence", 0)) or 0
        hallucination_catches_total += r.get("hallucination_catches", 0) or 0
        cross_field_issues_total += r.get("cross_field_issues_count", 0) or 0
        if r.get("edits"):
            edit_count += 1
        flagged = r.get("flagged_fields") or []
        if flagged:
            flagged_total += 1
        flagged_field_total += len(flagged)

        rev_id = r.get("assigned_to")
        if rev_id and rev_id in by_reviewer:
            by_reviewer[rev_id]["total"] += 1
            if st == STATUS_APPROVED:
                by_reviewer[rev_id]["approved"] += 1
            elif st == STATUS_REJECTED:
                by_reviewer[rev_id]["rejected"] += 1
            elif st == STATUS_NEEDS_REVIEW:
                by_reviewer[rev_id]["pending"] += 1

    auto_approval_rate = (by_status[STATUS_AUTO_APPROVED] / total) if total else 0
    reviewed = by_status[STATUS_APPROVED] + by_status[STATUS_REJECTED]
    approval_rate = (by_status[STATUS_APPROVED] / reviewed) if reviewed else 0
    avg_confidence = (confidence_sum / total) if total else 0
    avg_trust = (trust_sum / total) if total else 0
    edit_rate = (edit_count / total) if total else 0
    avg_flagged_per_doc = (flagged_field_total / total) if total else 0

    return {
        "total": total,
        "by_status": by_status,
        "by_reviewer": list(by_reviewer.values()),
        "auto_approval_rate": auto_approval_rate,
        "approval_rate": approval_rate,
        "avg_confidence": avg_confidence,
        "avg_trust": avg_trust,
        "edit_rate": edit_rate,
        "avg_flagged_per_doc": avg_flagged_per_doc,
        "needs_review": by_status[STATUS_NEEDS_REVIEW],
        "hallucinations_caught": hallucination_catches_total,
        "cross_field_issues": cross_field_issues_total,
    }


def clear_all() -> int:
    count = 0
    for path in list(PROCESSED_DIR.glob("*.json")) + list(PROCESSED_DIR.glob("*.png")):
        path.unlink()
        if path.suffix == ".json":
            count += 1
    return count


def _load_default_schema() -> dict:
    path = SCHEMAS_DIR / "medical_fax_referral.json"
    return json.loads(path.read_text())


def seed_mock_records(count: int = 18) -> int:
    """Create realistic mock records spanning the last ~30 days for demo purposes."""
    from verifiers import apply_verifications
    from lessons import record_lessons_from_review
    schema = _load_default_schema()
    rng = random.Random(42)

    sample_patients = [
        ("Maria Gonzalez", "06/14/1972", "F", "Aetna PPO #A1042288"),
        ("James Patel", "11/02/1958", "M", "BCBS #BC557812"),
        ("Anna Schmidt", "03/29/1985", "F", "United Health #UH998112"),
        ("David Kim", "08/17/1991", "M", "Cigna #CG441098"),
        ("Linda Thompson", "12/03/1964", "F", "Medicare #MED73821"),
        ("Robert Nguyen", "05/22/1949", "M", "Humana #HU298145"),
        ("Sophia Russo", "09/10/1978", "F", "Kaiser #KP802219"),
        ("Marcus Webb", "01/19/2002", "M", "Aetna HMO #A2298011"),
    ]
    diagnoses = [
        ("Atrial fibrillation", "I48.91", "Cardiology"),
        ("Type 2 diabetes — uncontrolled", "E11.65", "Endocrinology"),
        ("Lumbar disc herniation", "M51.26", "Orthopedics"),
        ("Suspected sleep apnea", "G47.33", "Pulmonology"),
        ("Chronic migraine", "G43.701", "Neurology"),
        ("Hashimoto's thyroiditis", "E06.3", "Endocrinology"),
        ("Anxiety disorder", "F41.1", "Behavioral Health"),
        ("Rotator cuff tear", "M75.101", "Orthopedics"),
    ]
    physicians = [
        ("Dr. Mark Lin", "Sunset Family Clinic", "415-555-0188", "415-555-0189"),
        ("Dr. Priya Shah", "Westside Internal Medicine", "650-555-0234", "650-555-0235"),
        ("Dr. Carla Mendez", "Bay View Primary Care", "510-555-0411", "510-555-0412"),
    ]
    urgencies = ["Routine", "Routine", "Routine", "Urgent", "Urgent", "STAT"]

    def _gen_valid_npi() -> str:
        first_9 = f"{rng.randint(100000000, 999999999):09d}"
        payload = "80840" + first_9
        total = 0
        for i, ch in enumerate(reversed(payload)):
            d = int(ch)
            if i % 2 == 0:
                d *= 2
                if d > 9:
                    d -= 9
            total += d
        checksum = (10 - total % 10) % 10
        return first_9 + str(checksum)

    now = datetime.now(timezone.utc)
    created = 0

    for i in range(count):
        record_id = uuid.uuid4().hex[:12]
        patient = rng.choice(sample_patients)
        dx = rng.choice(diagnoses)
        ref_md = rng.choice(physicians)
        urgency = rng.choice(urgencies)
        days_back = rng.randint(0, 28)
        hours_back = rng.randint(0, 23)
        when = now - timedelta(days=days_back, hours=hours_back)
        when_iso = when.isoformat()

        # Realistic-ish confidence distribution: most high, a tail of low
        base_conf = rng.choices(
            [0.94, 0.91, 0.88, 0.86, 0.82, 0.78, 0.71, 0.65],
            weights=[20, 18, 15, 12, 10, 12, 8, 5],
        )[0]

        fields = {}
        flagged = []

        def _field(field_id, value, conf=None, reasoning=""):
            c = conf if conf is not None else max(0.5, min(0.99, rng.gauss(base_conf, 0.06)))
            fields[field_id] = {
                "value": value,
                "confidence": round(c, 2),
                "reasoning": reasoning if c < 0.85 else "",
            }
            if c < 0.85:
                flagged.append(field_id)

        _field("patient_name", patient[0])
        _field("patient_dob", patient[1])
        _field("patient_sex", patient[2])
        _field("patient_phone", f"555-{rng.randint(100,999):03d}-{rng.randint(1000,9999):04d}")
        _field("patient_address", f"{rng.randint(100,9999)} Main St")
        _field("patient_insurance", patient[3])
        _field("referring_physician", ref_md[0])
        _field("referring_clinic", ref_md[1])
        _field("referring_phone", ref_md[2])
        _field("referring_fax", ref_md[3])
        _field("referring_npi", _gen_valid_npi())
        _field("receiving_physician", f"{dx[2]} Specialty Group")
        _field("receiving_clinic", "City General Hospital")
        _field("reason_for_referral", f"Eval and management of {dx[0].lower()}")
        _field("diagnosis", f"{dx[0]} ({dx[1]})")
        _field("current_medications", "Lisinopril 10mg, Metformin 500mg")
        _field("allergies", "NKDA")
        _field("relevant_history", "See attached chart notes")
        _field("urgency", urgency)
        _field("date_of_referral", when.strftime("%m/%d/%Y"))
        _field("physician_signature_present", "Yes", conf=0.9)

        overall = round(sum(f["confidence"] for f in fields.values()) / len(fields), 2)

        # Inject realistic-looking validator failures into a subset of records
        # so the dashboard shows hallucination-catch metrics out of the box.
        ocr_text_parts = [
            patient[0], patient[1], patient[2], patient[3],
            ref_md[0], ref_md[1], ref_md[2], ref_md[3],
            f"{dx[2]} Specialty Group", "City General Hospital",
            f"Eval and management of {dx[0].lower()}",
            f"{dx[0]} ({dx[1]})", "Lisinopril 10mg, Metformin 500mg",
            "NKDA", "See attached chart notes", urgency, when.strftime("%m/%d/%Y"),
        ]
        if rng.random() < 0.18:
            # Make NPI fail Luhn — model "hallucinated" a plausible-looking but invalid NPI
            fields["referring_npi"]["value"] = f"{rng.randint(1000000000, 9999999999)}"  # random 10 digits, won't pass Luhn
            fields["referring_npi"]["confidence"] = 0.91  # high self-confidence!
        if rng.random() < 0.12:
            # Implausible DOB year (model misread year digit)
            fields["patient_dob"]["value"] = "06/14/1872"
            fields["patient_dob"]["confidence"] = 0.88
        if rng.random() < 0.10:
            # Phone wrong format
            fields["referring_phone"]["value"] = "555-Lin-Office"
            fields["referring_phone"]["confidence"] = 0.78

        # Distribute statuses: roughly 30% auto-approved, 40% approved, 15% rejected, 15% needs_review
        if not flagged and rng.random() < 0.7:
            status = STATUS_AUTO_APPROVED
            assigned = None
        else:
            roll = rng.random()
            if roll < 0.55:
                status = STATUS_APPROVED
            elif roll < 0.75:
                status = STATUS_REJECTED
            else:
                status = STATUS_NEEDS_REVIEW
            assigned = rng.choice(REVIEWERS)

        history = [{"at": when_iso, "action": "extracted", "actor": "system",
                    "detail": f"Auto-status: {STATUS_NEEDS_REVIEW if flagged else STATUS_AUTO_APPROVED}"}]
        if assigned:
            history.append({"at": when_iso, "action": "assigned", "actor": "system",
                            "detail": assigned["name"]})
        if status == STATUS_APPROVED:
            history.append({"at": when_iso, "action": "approved", "actor": assigned["name"] if assigned else "auto"})
        elif status == STATUS_REJECTED:
            history.append({"at": when_iso, "action": "rejected", "actor": assigned["name"] if assigned else "system",
                            "detail": "Illegible / incomplete form"})

        edits = {}
        edit_reasons_mock = {}
        if status == STATUS_APPROVED and flagged and rng.random() < 0.7:
            mock_reasons = ["handwriting_unclear", "model_misread", "wrong_format", "policy_correction"]
            for fid in flagged[:rng.randint(1, len(flagged))]:
                # Mock: reviewer corrected the value to a slightly different version
                orig = fields[fid]["value"] or ""
                edits[fid] = (str(orig) + " (verified)") if orig else "(filled in)"
                edit_reasons_mock[fid] = rng.choice(mock_reasons)

        # Run real verifiers on the mock data — produces trust scores +
        # validator catches automatically.
        ext_payload = {
            "fields": fields,
            "overall_confidence": overall,
            "notes": "Mock record — generated for demo.",
            "_ocr_text": " ".join(str(p) for p in ocr_text_parts if p),
        }
        threshold = schema.get("confidence_threshold", 0.85)
        # Skip NPPES calls during seed — keeps `+ Seed Demo` fast and offline.
        # Real extractions go through the full external_lookups path.
        apply_verifications(ext_payload, schema, threshold, external_lookups=False)

        flagged = ext_payload.get("flagged_fields", flagged)
        # Re-decide status if validators caught new issues
        if flagged and status == STATUS_AUTO_APPROVED:
            status = STATUS_NEEDS_REVIEW
            assigned = rng.choice(REVIEWERS)
            history.append({"at": when_iso, "action": "assigned", "actor": "system",
                            "detail": f"{assigned['name']} (validator catch)"})

        record = {
            "id": record_id,
            "filename": f"fax_referral_{when.strftime('%Y%m%d')}_{i+1:03d}.pdf",
            "form_id": schema["form_id"],
            "form_name": schema["form_name"],
            "extracted_at": when_iso,
            "status": status,
            "assigned_to": assigned["id"] if assigned else None,
            "assigned_to_name": assigned["name"] if assigned else None,
            "overall_confidence": overall,
            "overall_trust_score": ext_payload.get("overall_trust_score", overall),
            "hallucination_catches": ext_payload.get("hallucination_catches", 0),
            "cross_field_issues_count": len(ext_payload.get("cross_field_issues") or []),
            "flagged_fields": flagged,
            "confidence_threshold": threshold,
            "schema": schema,
            "extraction": ext_payload,
            "edits": edits,
            "history": history,
        }
        _record_path(record_id).write_text(json.dumps(record, indent=2))
        created += 1

        # Mock the feedback loop: when a record is approved with edits,
        # capture lessons just like the review flow would.
        if status == STATUS_APPROVED and edits:
            record_lessons_from_review(record, edit_reasons_mock,
                                        assigned["name"] if assigned else "system")
    return created


def get_record(record_id: str) -> dict | None:
    path = _record_path(record_id)
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    image_path = _image_path(record_id)
    if image_path.exists():
        record["preview_image_b64"] = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return record


def _save_record(record: dict) -> dict:
    _record_path(record["id"]).write_text(json.dumps(record, indent=2))
    return record


def update_record(record_id: str, *, edits: dict | None = None,
                  edit_reasons: dict | None = None,
                  status: str | None = None, assigned_to: str | None = None,
                  actor: str = "reviewer") -> dict | None:
    path = _record_path(record_id)
    if not path.exists():
        return None
    record = json.loads(path.read_text())

    if edits is not None:
        record["edits"] = {**record.get("edits", {}), **edits}
        if edit_reasons:
            record["edit_reasons"] = {**record.get("edit_reasons", {}), **edit_reasons}
        record["history"].append({"at": _now_iso(), "action": "edited", "actor": actor,
                                  "detail": f"{len(edits)} field(s) edited"})

    if assigned_to is not None:
        reviewer = next((r for r in REVIEWERS if r["id"] == assigned_to), None)
        record["assigned_to"] = assigned_to if reviewer else None
        record["assigned_to_name"] = reviewer["name"] if reviewer else None
        record["history"].append({"at": _now_iso(), "action": "reassigned", "actor": actor,
                                  "detail": reviewer["name"] if reviewer else "unassigned"})

    if status is not None:
        record["status"] = status
        record["history"].append({"at": _now_iso(), "action": status, "actor": actor})

        # Capture lessons on approval — feedback loop into future extractions
        if status == STATUS_APPROVED and record.get("edits"):
            captured = record_lessons_from_review(
                record,
                record.get("edit_reasons") or {},
                actor,
            )
            if captured:
                record["history"].append({
                    "at": _now_iso(), "action": "lessons_captured", "actor": "system",
                    "detail": f"{captured} lesson(s) saved for future extractions",
                })

    return _save_record(record)


def delete_record(record_id: str) -> bool:
    path = _record_path(record_id)
    image_path = _image_path(record_id)
    existed = path.exists()
    if existed:
        path.unlink()
    if image_path.exists():
        image_path.unlink()
    return existed
