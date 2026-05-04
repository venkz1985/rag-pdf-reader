from __future__ import annotations

import base64
import json
import re

import fitz  # PyMuPDF
from openai import AsyncOpenAI

from config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    SCHEMAS_DIR,
    VISION_MODEL,
)
from lessons import build_few_shot_block, get_recent_lessons
from verifiers import apply_verifications

client = AsyncOpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)


def list_schemas() -> list[dict]:
    schemas = []
    for path in sorted(SCHEMAS_DIR.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        schemas.append({
            "form_id": data["form_id"],
            "form_name": data["form_name"],
            "description": data.get("description", ""),
        })
    return schemas


def load_schema(form_id: str) -> dict:
    path = SCHEMAS_DIR / f"{form_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Schema not found: {form_id}")
    with open(path) as f:
        return json.load(f)


def file_to_png_b64(file_bytes: bytes, filename: str) -> str:
    """Convert uploaded file to PNG base64. PDFs render first page; images pass through."""
    name = filename.lower()
    if name.endswith(".pdf"):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page = doc[0]
        pix = page.get_pixmap(dpi=200)
        png_bytes = pix.tobytes("png")
        doc.close()
    elif name.endswith((".png", ".jpg", ".jpeg", ".webp")):
        # Re-encode through PyMuPDF to normalize to PNG
        doc = fitz.open(stream=file_bytes, filetype=name.rsplit(".", 1)[-1])
        page = doc[0]
        pix = page.get_pixmap(dpi=200)
        png_bytes = pix.tobytes("png")
        doc.close()
    else:
        raise ValueError(f"Unsupported file type: {filename}")
    return base64.b64encode(png_bytes).decode("ascii")


def build_field_block(schema: dict) -> str:
    lines = []
    for f in schema["fields"]:
        parts = [f"- {f['id']} ({f['label']}, type={f['type']}"]
        if f.get("format"):
            parts.append(f", format={f['format']}")
        if f.get("options"):
            parts.append(f", one of {f['options']}")
        parts.append(f", required={f.get('required', False)})")
        lines.append("".join(parts))
    return "\n".join(lines)


def build_prompt(schema: dict) -> str:
    field_block = build_field_block(schema)
    lessons_block = build_few_shot_block(schema["form_id"])
    lessons_section = (lessons_block + "\n\n") if lessons_block else ""
    return f"""{lessons_section}You are extracting structured data from a scanned/handwritten form: **{schema['form_name']}**.

{schema.get('description', '')}

Extract these fields:
{field_block}

For EACH field, return:
- value: the extracted text (or null if not present, blank, or illegible)
- confidence: float 0.0-1.0 — how certain you are
- reasoning: short note (especially when confidence < 0.85, e.g. "handwriting unclear", "two possible readings", "field blank")

Be CONSERVATIVE. If handwriting is ambiguous, set value=null with low confidence rather than guessing. Downstream validators will check your work — fabricating plausible-looking values (e.g. invented NPIs, made-up dates) WILL be caught and undermines trust.

You must ALSO return a "_ocr_text" field containing a verbatim transcription of EVERY visible word/number/marking on the form, in roughly reading order, with no interpretation or paraphrase. This is used to verify your extractions against fabrication.

Return ONLY valid JSON in exactly this shape (no markdown, no commentary):
{{
  "fields": {{
    "<field_id>": {{"value": <string|null>, "confidence": <float>, "reasoning": "<string>"}},
    ...
  }},
  "overall_confidence": <float>,
  "notes": "<any cross-field observations, e.g. signature legibility, missing sections>",
  "_ocr_text": "<verbatim transcription of every visible token on the form>"
}}"""


def parse_json_response(raw: str) -> dict:
    """Parse LLM output, tolerating accidental code-fence wrapping."""
    text = raw.strip()
    # Strip ```json ... ``` if present
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    return json.loads(text)


async def extract_form(file_bytes: bytes, filename: str, form_id: str) -> dict:
    schema = load_schema(form_id)
    image_b64 = file_to_png_b64(file_bytes, filename)
    prompt = build_prompt(schema)

    response = await client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
            ],
        }],
    )

    raw = response.choices[0].message.content or ""
    try:
        data = parse_json_response(raw)
    except json.JSONDecodeError as e:
        return {
            "error": "Model did not return valid JSON",
            "raw_output": raw,
            "parse_error": str(e),
        }

    threshold = schema.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)

    # Run validators + OCR co-presence + cross-field rules; replaces naive
    # confidence-only flagging with a composite trust score.
    apply_verifications(data, schema, threshold)

    informed_by = len(get_recent_lessons(form_id))

    return {
        "form_id": form_id,
        "form_name": schema["form_name"],
        "schema": schema,
        "extraction": data,
        "flagged_fields": data.get("flagged_fields", []),
        "confidence_threshold": threshold,
        "preview_image_b64": image_b64,
        "informed_by_lessons": informed_by,
    }
