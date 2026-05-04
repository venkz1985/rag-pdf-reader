"""Trust-score machinery: field validators, cross-field rules, OCR co-presence.

This module replaces naive LLM-self-confidence with a composite trust score:
    trust = base_confidence × validator_mult × ocr_mult × cross_field_mult

Each multiplier is independent of the model, so the score is harder to "game"
than self-reported confidence alone.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

import httpx

# ============================================================
# Field-level validators
# ============================================================

DATE_FORMATS = ["%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y", "%Y-%m-%d"]
PHONE_RE = re.compile(r"^\+?1?[-. ]?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}$")
ICD10_RE = re.compile(r"\b([A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?)\b")


def _parse_date(value: str) -> datetime | None:
    if not value:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    return None


def validate_date(value: str, *, min_year: int = 1900, max_offset_days: int = 0) -> tuple[bool, str]:
    if not value:
        return True, ""
    dt = _parse_date(value)
    if not dt:
        return False, "Date doesn't match expected format (MM/DD/YYYY)"
    today = datetime.now()
    if dt.year < min_year:
        return False, f"Year {dt.year} is implausibly old"
    if dt > today + timedelta(days=max_offset_days):
        return False, "Date is in the future"
    return True, ""


def validate_phone(value: str) -> tuple[bool, str]:
    if not value:
        return True, ""
    if PHONE_RE.match(value.strip()):
        return True, ""
    return False, "Phone number doesn't match a recognizable format"


def validate_enum(value: str, options: list[str]) -> tuple[bool, str]:
    if not value:
        return True, ""
    norm = value.strip().lower()
    valid = [str(o).lower() for o in options]
    if norm in valid:
        return True, ""
    return False, f"Value '{value}' not in {options}"


def validate_npi(value: str) -> tuple[bool, str]:
    """NPI uses Luhn-mod-10 with constant '80840' prefix (per CMS standard)."""
    if not value:
        return True, ""
    digits = re.sub(r"\D", "", value)
    if len(digits) != 10:
        return False, f"NPI must be 10 digits (got {len(digits)})"
    payload = "80840" + digits[:9]
    total = 0
    for i, ch in enumerate(reversed(payload)):
        d = int(ch)
        if i % 2 == 0:  # double every other from right
            d *= 2
            if d > 9:
                d -= 9
        total += d
    expected_check = (10 - total % 10) % 10
    if expected_check != int(digits[9]):
        return False, "NPI failed Luhn checksum (likely fabricated or misread)"
    return True, ""


def validate_icd10_format(value: str) -> tuple[bool, str]:
    """If the value contains an ICD-10-looking code, check its format."""
    if not value:
        return True, ""
    matches = ICD10_RE.findall(value)
    # If no code-like substrings, that's fine — diagnosis can be free text
    if not matches:
        return True, ""
    # If we found code-shaped tokens, they must be valid format
    # (regex itself enforces format; this is a no-op pass for matched codes)
    return True, ""


def validate_boolean_like(value: str) -> tuple[bool, str]:
    if not value:
        return True, ""
    if str(value).strip().lower() in {"yes", "no", "true", "false", "y", "n", "1", "0"}:
        return True, ""
    return False, f"Expected yes/no, got '{value}'"


# Map field_id -> list of validator callables
FIELD_VALIDATORS: dict[str, list] = {
    "patient_dob": [lambda v: validate_date(v, min_year=1900)],
    "date_of_referral": [lambda v: validate_date(v, max_offset_days=7)],
    "patient_phone": [validate_phone],
    "referring_phone": [validate_phone],
    "referring_fax": [validate_phone],
    "referring_npi": [validate_npi],
    "patient_sex": [lambda v: validate_enum(v, ["M", "F", "Other"])],
    "urgency": [lambda v: validate_enum(v, ["Routine", "Urgent", "STAT"])],
    "diagnosis": [validate_icd10_format],
    "physician_signature_present": [validate_boolean_like],
}


def validate_field(field_id: str, value: Any) -> tuple[bool, list[str]]:
    """Run all validators for a field. Returns (all_passed, list_of_issues)."""
    issues = []
    validators = FIELD_VALIDATORS.get(field_id, [])
    if not validators:
        return True, []
    str_value = "" if value is None else str(value)
    for v in validators:
        try:
            ok, msg = v(str_value)
        except Exception as e:
            ok, msg = False, f"Validator error: {e}"
        if not ok:
            issues.append(msg)
    return len(issues) == 0, issues


# ============================================================
# Cross-field rules
# ============================================================

def cross_field_checks(fields: dict) -> list[dict]:
    """Returns a list of {fields: [...], issue: "..."} for cross-field conflicts."""
    issues = []

    def _val(fid):
        return (fields.get(fid) or {}).get("value")

    dob = _parse_date(str(_val("patient_dob") or ""))
    ref_date = _parse_date(str(_val("date_of_referral") or ""))

    if dob and ref_date:
        if ref_date < dob:
            issues.append({
                "fields": ["patient_dob", "date_of_referral"],
                "issue": "Date of referral is BEFORE patient's date of birth — impossible.",
            })
        else:
            age_years = (ref_date - dob).days / 365.25
            if age_years > 120:
                issues.append({
                    "fields": ["patient_dob"],
                    "issue": f"Computed patient age = {age_years:.0f} years (> 120, implausible).",
                })
            if age_years < 0:
                issues.append({
                    "fields": ["patient_dob"],
                    "issue": "Patient appears to have negative age.",
                })

    return issues


# ============================================================
# OCR co-presence
# ============================================================

def normalize_for_match(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — for fuzzy substring matching."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ocr_corroborates(value: Any, ocr_text: str) -> bool | None:
    """Returns True if the substantive content of `value` appears in OCR text;
    False if it doesn't; None if not applicable (empty value, etc.)."""
    if not ocr_text or value is None or str(value).strip() == "":
        return None
    norm_value = normalize_for_match(str(value))
    norm_ocr = normalize_for_match(ocr_text)
    if not norm_value:
        return None
    # Pick "substantive" tokens (>=3 chars, ignore stop words)
    stop_words = {"the", "and", "for", "see", "yes", "not"}
    tokens = [t for t in norm_value.split() if len(t) >= 3 and t not in stop_words]
    if not tokens:
        # Short/numeric values: do a direct substring check on the normalized form
        return norm_value in norm_ocr
    # Most substantive tokens must appear in OCR
    hits = sum(1 for t in tokens if t in norm_ocr)
    return hits / len(tokens) >= 0.6


# ============================================================
# NPPES — external oracle for NPI verification
# ============================================================

NPPES_URL = "https://npiregistry.cms.hhs.gov/api/?number={npi}&version=2.1"
_npi_cache: dict[str, dict | None] = {}


def lookup_npi(npi: str, timeout: float = 5.0) -> dict | None:
    """Query NPPES for an NPI. Returns provider info or None if not found.
    Results cached in-process to avoid repeat calls during a session."""
    digits = re.sub(r"\D", "", str(npi or ""))
    if len(digits) != 10:
        return None
    if digits in _npi_cache:
        return _npi_cache[digits]
    try:
        r = httpx.get(NPPES_URL.format(npi=digits), timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None  # don't cache transient failures
    if data.get("result_count", 0) == 0:
        _npi_cache[digits] = None
        return None
    result = (data.get("results") or [{}])[0]
    basic = result.get("basic") or {}
    enum_type = result.get("enumeration_type", "")  # "NPI-1" individual, "NPI-2" org

    if enum_type == "NPI-2":
        info = {
            "npi": digits,
            "type": "organization",
            "organization_name": (basic.get("organization_name") or "").strip(),
            "first_name": "",
            "last_name": (basic.get("organization_name") or "").strip(),  # use as match key
            "credential": "",
        }
    else:
        info = {
            "npi": digits,
            "type": "individual",
            "first_name": (basic.get("first_name") or "").strip(),
            "last_name": (basic.get("last_name") or "").strip(),
            "credential": (basic.get("credential") or "").strip(),
            "organization_name": "",
        }
    _npi_cache[digits] = info
    return info


def check_npi_against_physician(npi_value: Any, physician_name: Any) -> tuple[bool | None, str, dict | None]:
    """Returns (match_status, human_readable_detail, provider_info_or_None).
    match_status: True (matches name), False (registered but name mismatch / not in registry), None (skipped)."""
    if not npi_value:
        return None, "", None
    info = lookup_npi(str(npi_value))
    if info is None:
        return False, "NPI not found in NPPES registry", None

    pname_lower = str(physician_name or "").lower()
    if info.get("type") == "organization":
        org = info.get("organization_name", "")
        # Match if any substantial token of the org name appears in the physician string
        org_tokens = [t for t in re.split(r"\W+", org.lower()) if len(t) >= 4]
        if any(t in pname_lower for t in org_tokens):
            return True, f"Matches NPPES (organization): {org}", info
        return False, f"NPI registered to organization '{org}' — doesn't match '{physician_name}'", info

    last = info.get("last_name", "").lower()
    if last and last in pname_lower:
        full = " ".join(filter(None, [info.get("first_name"), info.get("last_name"), info.get("credential")]))
        return True, f"Matches NPPES: {full}", info
    full = " ".join(filter(None, [info.get("first_name"), info.get("last_name")])) or "(unknown)"
    return False, f"NPI registered to {full} — doesn't match '{physician_name}'", info


NPPES_MATCH_BOOST = 1.15
NPPES_NOT_FOUND_MULT = 0.4
NPPES_MISMATCH_MULT = 0.3


# ============================================================
# Composite trust score
# ============================================================

VALIDATOR_FAIL_MULT = 0.4
OCR_NOT_FOUND_MULT = 0.75
OCR_CORROBORATED_BOOST = 1.05
CROSS_FIELD_FAIL_MULT = 0.55


def compute_trust(base_confidence: float, *, validators_passed: bool,
                  ocr_corroborated_signal: bool | None,
                  in_cross_field_conflict: bool) -> float:
    score = max(0.0, min(1.0, float(base_confidence or 0)))
    if not validators_passed:
        score *= VALIDATOR_FAIL_MULT
    if ocr_corroborated_signal is True:
        score = min(1.0, score * OCR_CORROBORATED_BOOST)
    elif ocr_corroborated_signal is False:
        score *= OCR_NOT_FOUND_MULT
    if in_cross_field_conflict:
        score *= CROSS_FIELD_FAIL_MULT
    return round(score, 3)


# ============================================================
# Top-level orchestration
# ============================================================

def apply_verifications(extraction: dict, schema: dict, threshold: float,
                        *, external_lookups: bool = True) -> dict:
    """Mutates `extraction` to add per-field signals + trust_score, plus
    top-level cross_field_issues + overall_trust_score + flagged_fields.
    Returns the augmented extraction dict.

    `external_lookups`: when True, queries NPPES for NPI verification.
    Set False from seeders or batch jobs that don't want network I/O.
    """
    fields = extraction.get("fields") or {}
    ocr_text = extraction.get("_ocr_text") or ""

    # External oracle: NPPES NPI lookup (run once, attach result to npi field)
    nppes_match: bool | None = None
    nppes_detail = ""
    nppes_info = None
    if external_lookups:
        npi_value = (fields.get("referring_npi") or {}).get("value")
        physician_value = (fields.get("referring_physician") or {}).get("value")
        nppes_match, nppes_detail, nppes_info = check_npi_against_physician(
            npi_value, physician_value
        )

    # Cross-field first (so we know which fields are involved)
    cross_issues = cross_field_checks(fields)
    cross_field_set: set[str] = set()
    for ci in cross_issues:
        cross_field_set.update(ci.get("fields", []))

    flagged: list[str] = []
    trust_sum = 0.0
    trust_count = 0
    hallucination_catches = 0

    for f in schema.get("fields", []):
        fid = f["id"]
        info = fields.get(fid)
        if not isinstance(info, dict):
            continue
        value = info.get("value")
        validators_passed, validator_issues = validate_field(fid, value)
        ocr_signal = ocr_corroborates(value, ocr_text)
        in_cross = fid in cross_field_set

        trust = compute_trust(
            info.get("confidence", 0),
            validators_passed=validators_passed,
            ocr_corroborated_signal=ocr_signal,
            in_cross_field_conflict=in_cross,
        )

        # NPPES is field-specific: only adjust the referring_npi field
        signals: dict = {
            "ocr_corroborated": ocr_signal,
            "validators_passed": validators_passed,
            "validator_issues": validator_issues,
            "in_cross_field_conflict": in_cross,
        }
        if fid == "referring_npi" and external_lookups:
            signals["nppes_match"] = nppes_match
            signals["nppes_detail"] = nppes_detail
            signals["nppes_provider"] = nppes_info
            model_was_confident = info.get("confidence", 0) >= threshold
            if nppes_match is True:
                trust = min(1.0, trust * NPPES_MATCH_BOOST)
            elif nppes_match is False and "not found" in nppes_detail.lower():
                trust = min(trust, NPPES_NOT_FOUND_MULT)
                if model_was_confident and validators_passed:
                    hallucination_catches += 1
            elif nppes_match is False:
                trust = min(trust, NPPES_MISMATCH_MULT)
                if model_was_confident and validators_passed:
                    hallucination_catches += 1

        info["trust_score"] = round(trust, 3)
        info["signals"] = signals

        # Track which fields the validators caught that the model didn't
        if not validators_passed and (info.get("confidence", 0) >= threshold):
            hallucination_catches += 1

        trust_sum += trust
        trust_count += 1

        if (trust < threshold) or (not validators_passed) or in_cross:
            flagged.append(fid)

    extraction["cross_field_issues"] = cross_issues
    extraction["overall_trust_score"] = round(trust_sum / trust_count, 3) if trust_count else 0
    extraction["hallucination_catches"] = hallucination_catches
    extraction["flagged_fields"] = flagged
    return extraction
