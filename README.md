# Glyph

### *The model triages. Humans adjudicate.*

> *A portfolio prototype I built to explore vision-LLM + verifier patterns for clinical document workflows. Not a product — see [Scope](#scope) for what's intentionally not built.*

Verified structured data from scanned, handwritten medical forms. Built for clinics — designed for the front-desk reviewer clearing 30 fax referrals before lunch.

**Try it:** [run locally](#running-locally) (~3 min setup), or ping me for a live walkthrough on [glyph-clinical-ai-1.onrender.com](https://glyph-clinical-ai-1.onrender.com).

---

## The problem

Healthcare, insurance, legal, and government still receive enormous volumes of documents on paper and by fax. People type those into systems by hand. Existing OCR tools handle typed text but struggle with handwriting and varied layouts. LLM-only "AI extractors" hallucinate confidently — they happily invent a plausible-looking NPI, DOB, or diagnosis with 90% self-reported confidence.

The right answer isn't *more* AI. It's:

1. A **vision LLM** that reads handwriting (the part traditional OCR fails at).
2. **Independent verifiers** that check what the model says against authoritative sources — the model can't fake those.
3. A **human-in-the-loop queue** for cases the verifiers can't resolve. The reviewer is the system's safety net, not its bottleneck.

This project demonstrates that pattern end-to-end on **medical fax referrals**.

---

## Scope

### In scope
- Single form type: **Medical Fax Referral** (21 fields — patient info, referring/receiving providers, clinical, logistics).
- Single-page image / PDF upload.
- Schema-as-config: form definitions live in JSON, not code, so adding a second form type is a config change.
- Confidence-gated review queue with mock reviewers and round-robin assignment.
- Composite trust scoring: validators + OCR co-presence + cross-field rules + NPPES external oracle.
- Insights dashboard with workflow + quality metrics.

### Out of scope (deliberate, named)
- Multi-page documents — current code renders page 1 only.
- Real auth / RBAC — reviewers are a hardcoded mock list.
- PHI compliance (HIPAA, BAA, encryption-at-rest) — demo only; do not upload real patient data.
- Multiple form types — schema-as-config makes this trivial, but only one schema is shipped.
- Fine-tuning / model training — feedback loop captures corrections in-prompt, no retraining.

---

## How it works

### End-to-end flow

```
Upload (PDF or image)
    │
    ▼
┌────────────────────────────────────────┐
│ Vision LLM call (Claude Haiku 4.5      │
│ via OpenRouter)                        │
│                                        │
│ Returns: structured fields + confidence│
│ + verbatim transcription (_ocr_text)   │
└────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────┐
│ Verifier pipeline                      │
│  • Field validators (Luhn, regex,      │
│    date plausibility, enums)           │
│  • Cross-field rules (DOB vs referral  │
│    date, age plausibility)             │
│  • OCR co-presence check               │
│  • NPPES external oracle (CMS NPI      │
│    registry, real public API)          │
│                                        │
│ Computes: per-field trust score        │
│ (composite, not self-reported)         │
└────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────┐
│ Routing                                │
│  • All fields above threshold AND      │
│    no validator failures               │
│           → status = auto_approved     │
│  • Anything flagged                    │
│           → status = needs_review      │
│           → assigned to reviewer       │
│             (round-robin, lowest queue)│
└────────────────────────────────────────┘
    │
    ▼
Worklist tab — reviewer opens the record,
edits flagged fields, approves / rejects /
reassigns. (Edits feed back into the
prompt for future extractions.)
```

### Trust score (the differentiator)

Self-reported LLM confidence is unreliable — models hallucinate confidently. This system replaces it with a composite score:

```
trust_score = base_confidence
            × validator_multiplier   (0.4 if any field validator fails, else 1.0)
            × ocr_multiplier         (1.05 if value found in OCR text,
                                      0.75 if not found, 1.0 if N/A)
            × cross_field_multiplier (0.55 if field in any cross-field conflict)
            × nppes_multiplier       (only on referring_npi field:
                                      1.15 if registry match,
                                      0.4 if NPI not in registry,
                                      0.3 if registered to a different name)
```

**Why this matters:** validators, OCR, cross-field rules, and NPPES are *independent* of the LLM. A model that confidently fabricates a 10-digit NPI cannot also fool the NPPES check, because NPPES is the authoritative external registry — not part of the model's output.

A field is **flagged for review** if `trust_score < threshold` OR validators fail OR a cross-field conflict involves it.

### Verifier details

| Verifier | Catches | Source of truth |
|---|---|---|
| **DOB plausibility** | Year before 1900, future dates, malformed dates | Calendar |
| **Phone regex** | Invalid US phone formats | Regex |
| **NPI Luhn checksum** | Fabricated 10-digit numbers | CMS Luhn-mod-10 with `80840` prefix |
| **ICD-10 format** | Malformed code-like substrings | Regex |
| **Enum check** | Wrong values for Sex, Urgency | Schema |
| **Cross-field: DOB ↔ referral date** | Referral before birth, age >120 / <0 | Calendar |
| **OCR co-presence** | Field values that don't appear anywhere in the form's transcription | Same model's verbatim output (weak version) |
| **NPPES NPI lookup** | NPIs not in the official CMS registry; NPI registered to a different doctor than the form claims | Live CMS public API |

### Hallucination catch counter

The dashboard tracks a specific metric: **hallucinations caught** = fields where the model was confident (≥ threshold) but a downstream verifier rejected the value. This is the metric that proves the verifier layer is doing real work.

### Feedback loop (partially shipped)

When a reviewer approves a record with edits, each correction is stored as a "lesson":

```json
{
  "field_id": "patient_dob",
  "model_value": "03/15/85",
  "corrected_value": "03/15/1985",
  "reason": "wrong_format",
  "reviewer": "Dr. Sarah Chen",
  "at": "2026-05-02T..."
}
```

The next extraction of the same form type prepends recent lessons to the prompt as few-shot examples — so the model sees concrete corrections from prior reviews. Implementation status: backend wiring complete (`lessons.py`, prompt injection, record-on-approve), UI for capturing the **why** dropdown is the next thing to ship.

---

### Evaluation

The extraction pipeline is measured against a hand-labeled golden set — not the model's self-reported confidence.

**Golden set.** 10 synthetic medical fax referral forms (`golden/forms/`), each with hand-written ground-truth labels (`golden/labels/`) covering all 21 fields. Deliberately varied: typed and handwritten, fully-completed and partially-blank, plus one degraded scan.

**Scorer.** `golden/score.py` runs each form through the full Glyph pipeline and compares the extraction to its label, field by field. Each field gets one of five verdicts:

| Verdict | Meaning | Counts as |
|---|---|---|
| `correct` | extracted value matches the label | TP |
| `missed` | label has a value, model returned null | FN |
| `hallucinated` | label is null, model invented a value | FP |
| `wrong_value` | model returned a value, but the wrong one | FP **and** FN |
| `correct_blank` | both are null | TN |

`wrong_value` counts as both a false positive and a false negative — the model both asserted something false and failed to capture the true value. The scorer also records a `would_pass_normalized` diagnostic flag, separating genuine misreads from pure formatting differences.

**Baseline.** ~0.99 precision and recall — on 10 clean, synthetic, single-template forms. A controlled baseline, not a real-world accuracy claim; messy, multi-template documents would score lower.

**What it surfaced.** On its first run the harness caught a reproducible model failure: Glyph misreads "NKDA" as "NRDA" on this form font, consistently across multiple forms. Finding a specific, repeatable failure mode — rather than just a headline number — is the point of the eval.

**Run it:** `.venv/bin/python golden/score.py`

**Limitations.** Single form template, n=10, synthetic data, single-run (variance not yet measured). Roadmap: multi-run variance, a second template, an expanded set.

---

## Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python 3.9+, ASGI) |
| Vision LLM | Claude Haiku 4.5 via OpenRouter (multimodal vision + structured JSON output) |
| External oracle | CMS NPPES NPI Registry public API (no auth, free) |
| Image rendering | PyMuPDF (PDF → PNG, ~200 DPI) |
| Frontend | Server-rendered Jinja2 + vanilla JS — intentionally minimal, no SPA build pipeline |
| Vector store (RAG side) | FAISS — currently used by `ingest.py` for the legacy PDF chat (kept for future feedback-loop use) |
| Embeddings | HuggingFace Inference API (BGE-small) |
| Persistence | Filesystem JSON in `processed/` and `lessons/` |
| Hosting | Render (Docker container, see `Dockerfile` + `render.yaml`) |

---

## Repository layout

```
.
├── main.py                 # FastAPI app, all routes
├── forms.py                # Vision LLM call, prompt building, post-extraction wiring
├── verifiers.py            # Validators, cross-field rules, OCR co-presence, NPPES lookup, trust scoring
├── lessons.py              # Reviewer-correction store + few-shot prompt injection
├── worklist.py             # Persistence, round-robin assignment, stats, mock seeder
├── ingest.py / rag.py      # Legacy PDF-chat RAG layer (kept; not currently surfaced in UI)
├── config.py               # Paths, model name, thresholds, mock reviewer list
├── schemas/
│   └── medical_fax_referral.json   # Schema-as-config: 21 fields
├── processed/              # One JSON + PNG per record (gitignored in production)
├── lessons/                # Reviewer corrections by form_id (gitignored in production)
├── templates/
│   └── index.html          # Tabbed UI (Forms Extractor, Worklist) + review modal
├── static/
│   └── style.css
├── samples/
│   └── README.md           # Where to get test images
└── README.md               # ← you are here
```

---

## Running locally

```bash
# 1. Create a venv and install deps
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Set up .env (file already created with empty placeholders)
echo "OPENROUTER_API_KEY=sk-or-v1-..." >> .env
echo "HF_TOKEN=hf_..." >> .env   # optional, only used by legacy RAG

# 3. Run the server
.venv/bin/uvicorn main:app --reload

# 4. Open http://127.0.0.1:7860
#    - Forms Extractor tab: upload an image, extract
#    - Worklist tab: click "+ Seed Demo" for 18 mock records
```

See [`samples/README.md`](samples/README.md) for where to get test images. The fastest path: hand-write a fake referral on paper and photograph it.

---

## Demo script (5 minutes)

1. **Open Worklist tab** — show the 8 widgets, point at "Hallucinations Caught" and explain that this counts cases where the model was confident but a deterministic verifier rejected the value.
2. **Click a flagged record** — show the review modal. Point out the chips per field: 📝 OCR ✓, ✅ Valid, ⚠ Validator, 🔗 Conflict, 📒 NPPES ✓/✗.
3. **Highlight the trust drift chip** on a low-confidence field — "the model said 91%, our composite trust says 41%, because the NPI failed the registry check."
4. **Switch to Forms Extractor**, upload a real handwritten form. Watch extraction (~10s). Show the cross-field issues panel if any fire, the per-field trust scores, and the "Open in Worklist" button.
5. **Land the line:** "We don't trust the model's self-confidence. We verify against independent oracles and route the rest to humans. The model triages; humans adjudicate."

---

## Roadmap / known limitations

| Gap | Mitigation today | Production fix |
|---|---|---|
| Multi-page docs not supported | Renders first page only | Iterate over `doc[:n]`, dedupe extracted entities across pages |
| ICD-10 validator is a no-op | `validate_icd10_format` currently always passes — listed but not enforcing | Implement real ICD-10 code validation, or remove the claim |
| OCR co-presence uses same model | Catches blatant fabrication, weak on subtle errors | Swap to Tesseract (typed text) + Textract / Document AI (handwriting) |
| Single LLM = single point of failure | OpenRouter falls back to other backends, but they share blind spots | Cross-model agreement (Haiku + GPT-4o-mini), shadow mode for new models |
| Reviewer reason capture not yet in UI | Backend ready, dropdown is the next ship | Add reason picker on edit; lessons feed into prompt automatically |
| No PHI controls | Demo-only — README warns not to upload real patient data | BAA with model provider, encryption-at-rest, access logs, audit trail |
| Round-robin assignment is naive | Works for 3 reviewers + 18 docs | Specialization (cardiology → cardiologist), priority queue (STAT first), SLA timers |
| Confidence threshold is fixed | Configured per schema | Per-customer / per-field thresholds, calibrated against historical reviewer outcomes |
| Eval is single-run on a small synthetic set | 10-form hand-labeled golden set + scorer (`golden/score.py`); ~0.99 precision/recall baseline | Expand set + second template; multi-run variance reporting; online eval instrumentation once there's real traffic |
| Extraction and verification are fused | `extract_form` runs both in one call | Refactor into composable stages so callers can run partial pipelines and control external (NPPES) calls |

---

## Senior-PM framing for an interview

> "It's a thin orchestration layer over a vision LLM with deterministic schema enforcement and a confidence-gated HITL queue. The architectural principle is: *use the LLM as a perception system, not a system of record.* The schema lives outside the model. Confidence comes from independent verifiers — validators, cross-field rules, OCR corroboration, and external oracles like NPPES — not from the model's self-report. Reviewer corrections feed back into the prompt as few-shot examples, so the system gets cheaper and more accurate the more it's used. That's the moat."

---

## License

MIT — see [LICENSE](LICENSE). Demo only; do not upload real patient data (see [Scope](#scope) for PHI / HIPAA caveats).
