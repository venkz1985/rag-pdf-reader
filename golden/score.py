import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

GOLDEN_DIR = Path(__file__).parent
sys.path.insert(0, str(GOLDEN_DIR.parent))

from forms import extract_form


def load_pair(form_number):
    num = f"{form_number:03d}"
    image_path = GOLDEN_DIR / "forms" / f"form_{num}.png"
    label_path = GOLDEN_DIR / "labels" / f"label_form_{num}.json"
    label = json.loads(label_path.read_text())
    return image_path, label


async def run_glyph(image_path, filename):
    file_bytes = image_path.read_bytes()
    return await extract_form(file_bytes, filename, "medical_fax_referral")


def verdict(glyph_value, label_value):
    if glyph_value is None and label_value is None:
        return "correct_blank", False
    if glyph_value is None:
        return "missed", False
    if label_value is None:
        return "hallucinated", False

    normalized_match = str(glyph_value).strip().lower() == str(label_value).strip().lower()
    if glyph_value == label_value:
        return "correct", True
    return "wrong_value", normalized_match


async def score_form(form_number):
    image_path, label = load_pair(form_number)
    glyph_response = await run_glyph(image_path, image_path.name)
    glyph_fields = glyph_response["extraction"]["fields"]

    results = []
    for field_name, label_value in label.items():
        if field_name == "image":
            continue
        glyph_value = glyph_fields.get(field_name, {}).get("value")
        verdict_name, would_pass_normalized = verdict(glyph_value, label_value)
        results.append((field_name, glyph_value, label_value, verdict_name, would_pass_normalized))
    return results


def tally(results):
    counts = Counter(tup[3] for tup in results)

    tp = counts["correct"]
    tn = counts["correct_blank"]
    fp = counts["hallucinated"] + counts["wrong_value"]
    fn = counts["missed"] + counts["wrong_value"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
    }


def print_form_trace(form_number, results):
    print(f"\n=== Form {form_number:03d} ===")
    failures = 0
    for field_name, glyph_value, label_value, verdict_name, would_pass_normalized in results:
        is_pass = verdict_name in ("correct", "correct_blank")
        marker = "  " if is_pass else "* "
        line = f"{marker}{field_name:<32} {verdict_name}"
        if not is_pass:
            failures += 1
            line += f"    glyph={glyph_value!r}  label={label_value!r}"
            if would_pass_normalized:
                line += "  (would pass normalized)"
        print(line)
    print(f"  -- {len(results) - failures}/{len(results)} correct, {failures} failures")


def print_overall(metrics):
    print("\n=== OVERALL ===")
    print(f"  TP (correct):           {metrics['tp']}")
    print(f"  FP (halluc + wrong):    {metrics['fp']}")
    print(f"  FN (missed + wrong):    {metrics['fn']}")
    print(f"  TN (correct_blank):     {metrics['tn']}")
    print()
    print(f"  Precision:  {metrics['precision']:.3f}")
    print(f"  Recall:     {metrics['recall']:.3f}")


async def main():
    all_results = []
    for form_number in range(1, 11):
        per_form = await score_form(form_number)
        print_form_trace(form_number, per_form)
        all_results.extend(per_form)
    print_overall(tally(all_results))


if __name__ == "__main__":
    asyncio.run(main())
