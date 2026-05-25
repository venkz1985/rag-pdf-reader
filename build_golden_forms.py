import os, random
from PIL import Image, ImageDraw, ImageFont, ImageFilter

random.seed(7)
os.makedirs("golden/forms", exist_ok=True)

FONT_REG = "/System/Library/Fonts/Helvetica.ttc"
FONT_MONO = "/System/Library/Fonts/Courier.ttc"
HAND_CANDS = [
    "/System/Library/Fonts/Supplemental/Bradley Hand.ttc",
    "/System/Library/Fonts/Supplemental/Marker Felt.ttc",
    "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
]
FONT_HAND = next((p for p in HAND_CANDS if os.path.exists(p)), FONT_REG)

def font(path, size, idx=0):
    try:
        if path.endswith(".ttc"):
            return ImageFont.truetype(path, size, index=idx)
        return ImageFont.truetype(path, size)
    except Exception:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            return ImageFont.load_default()

W, H = 1700, 2200

def make_npi(valid=True):
    body = [random.randint(1, 9)] + [random.randint(0, 9) for _ in range(8)]
    digits = [int(d) for d in "80840"] + body
    total = sum(sum(divmod(d * 2, 10)) if i % 2 == 0 else d
                for i, d in enumerate(reversed(digits)))
    check = (10 - total % 10) % 10
    if not valid:
        check = (check + random.randint(1, 9)) % 10
    return "".join(map(str, body)) + str(check)

PATIENTS = [
    ("Margery Quinlivan", "F", "03/14/1962"),
    ("Bertram Hargreaves", "M", "11/22/1978"),
    ("Lucinda Pemberton", "F", "07/05/1955"),
    ("Desmond Okafor", "M", "01/18/1990"),
    ("Priscilla Voss", "F", "09/30/1944"),
    ("Thaddeus Marchetti", "M", "06/12/1985"),
    ("Imogen Yarrow", "F", "12/03/1971"),
    ("Cyril Botham", "M", "04/27/1958"),
    ("Tamsin Lindqvist", "Other", "08/19/1995"),
    ("Reginald Sterling", "M", "02/14/1949"),
]
REF_PHYS = [
    ("Dr. Helga Marchetti, MD", "Crestwood Family Medicine"),
    ("Dr. Aurelius Tinsdale, DO", "Riverbend Primary Care"),
    ("Dr. Octavia Pemberton, MD", "Hollybrook Internal Medicine"),
    ("Dr. Cornelius Yarrow, MD", "Foxglen Health Partners"),
    ("Dr. Beatrix Whitlock, MD", "Larchmont Medical Group"),
    ("Dr. Phineas Drummond, MD", "Sycamore Hill Family Practice"),
    ("Dr. Wilhelmina Frost, DO", "Coppertree Community Clinic"),
    ("Dr. Ignatius Vaughn, MD", "Cedar Bluff Medical"),
    ("Dr. Persephone Calder, MD", "Brookside Primary Care"),
    ("Dr. Mortimer Halligan, MD", "Stonebridge Family Health"),
]
RECV_PHYS = [
    ("Dr. Saoirse Whitcombe, MD", "Northgate Cardiology Associates"),
    ("Dr. Ezra Pendelton, MD", "Vermilion Neurology Group"),
    ("Dr. Carmela Strickland, MD", "Pinehurst Endocrinology"),
    ("Dr. Silas Kindred, MD", "Westview Pulmonology"),
    ("Dr. Marisol Beaumont, MD", "Crescent Bay Gastroenterology"),
    ("Dr. Octavius Larch, MD", "Highland Orthopedic Specialists"),
    ("Dr. Edith Quinwell, MD", "Maple Crest Rheumatology"),
    ("Dr. Bartholomew Reeve, MD", "Bayshore Nephrology Group"),
    ("Dr. Cassius Whitmore, MD", "Eastbrook Dermatology"),
    ("Dr. Daria Underhill, MD", "Foxbridge Oncology Center"),
]
INSURANCES = ["BlueCross Standard PPO", "United Heritage HMO", "Aetna Choice Plus",
              "Cigna Open Access", "Humana Gold PPO", "Anthem Silver PPO",
              "Medicare Part B", "Medicaid (state)", "Kaiser Foundation HMO", "Oscar Bronze EPO"]
REASONS = [
    "Eval of recurrent chest pain w/ exertional dyspnea",
    "Persistent migraine refractory to first-line therapy",
    "Worsening glycemic control despite metformin titration",
    "Chronic cough x 8 weeks; suspect reactive airway disease",
    "Abdominal pain w/ unexplained weight loss",
    "Bilateral knee pain limiting mobility",
    "Eval of suspected autoimmune arthropathy",
    "Declining eGFR; eval for CKD stage progression",
    "New skin lesion, r/o malignancy",
    "Newly noted thyroid nodule on imaging",
]
DIAGNOSES = [
    "Stable angina (I20.9); HTN (I10)",
    "Chronic migraine w/o aura (G43.709)",
    "Type 2 DM uncontrolled (E11.65); HLD",
    "Chronic cough (R05.3); GERD (K21.9)",
    "Unintentional wt loss (R63.4); abd pain (R10.9)",
    "Bilateral primary OA of knee (M17.0)",
    "Polyarthralgia (M25.50); fatigue (R53.83)",
    "CKD stage 3a (N18.31); HTN (I10)",
    "Suspicious pigmented lesion, L forearm (D48.5)",
    "Thyroid nodule (E04.1)",
]
MEDS = [
    "Lisinopril 20mg qd; atorvastatin 40mg qhs; ASA 81mg qd",
    "Topiramate 50mg bid; sumatriptan 100mg prn",
    "Metformin 1000mg bid; empagliflozin 10mg qd",
    "Omeprazole 40mg qd; albuterol HFA prn",
    "Pantoprazole 40mg qd; multivitamin daily",
    "Acetaminophen 1g tid prn; meloxicam 15mg qd",
    "Hydroxychloroquine 200mg bid; ibuprofen prn",
    "Lisinopril 10mg qd; furosemide 20mg qd",
    "None reported",
    "Levothyroxine 50mcg qd",
]
ALLERGIES = ["NKDA", "PCN -- rash", "Sulfa -- hives", "Codeine -- nausea",
             "Latex", "Iodine contrast", "NKDA", "PCN, sulfa", "NKDA", "Aspirin -- GI bleed"]
HISTORY = [
    "HTN x 12 yrs; former smoker (quit 2018); fam hx CAD (father MI age 58)",
    "Migraine since adolescence; failed propranolol; topiramate added 2024",
    "T2DM dx 2011; A1c 9.2 last visit; non-adherent w/ diet",
    "Asthma in childhood; environmental allergies; nonsmoker",
    "20 lb wt loss over 4 mo; appetite preserved; no F/C/N/V",
    "Bilateral knee pain x 3 yrs; XR shows joint space narrowing",
    "Joint pain w/ AM stiffness > 1 hr; +ANA; RF/CCP pending",
    "HTN x 20 yrs; eGFR trending 58 -> 51 -> 47 over 18 mo",
    "Lesion noted 3 mo, increasing pigmentation, irregular border",
    "Incidental finding on neck CT for unrelated complaint",
]
STREETS = ["Larkspur Ln", "Bartholomew Ave", "Kestrel Ct", "Mockingbird Dr",
           "Tinsdale Pkwy", "Old Mill Rd", "Vermilion St", "Hawthorne Pl",
           "Wisteria Way", "Bramblewood Dr"]
CITIES = [("Austin","TX","78745"),("Denver","CO","80212"),("Portland","OR","97214"),
          ("Asheville","NC","28801"),("Madison","WI","53703"),("Boise","ID","83702"),
          ("Burlington","VT","05401"),("Tacoma","WA","98402")]
AREAS = ["512","713","303","503","828","608","208","802","253","415"]

def phone():
    return f"({random.choice(AREAS)}) 555-01{random.randint(10,99):02d}"

def addr():
    c,s,z = random.choice(CITIES)
    return f"{random.randint(100,9999)} {random.choice(STREETS)}, {c}, {s} {z}"

def date_2026():
    return f"{random.randint(1,5):02d}/{random.randint(1,28):02d}/2026"

def sec_header(d, text, x1, y, x2, f):
    d.rectangle([(x1, y), (x2, y+42)], fill=(232, 236, 246))
    d.line([(x1, y), (x2, y)], fill=(40,50,90), width=2)
    d.line([(x1, y+42), (x2, y+42)], fill=(40,50,90), width=2)
    d.text((x1+10, y+6), text, fill=(20, 30, 70), font=f)

def val(d, key, data, x, y, f, color, missing, hand):
    if key in missing:
        d.line([(x, y+34), (x+340, y+34)], fill=(160,160,160), width=1)
        return
    v = data.get(key, "")
    d.text((x, y + (-4 if hand else 0)), str(v), fill=color, font=f)

def multi(d, label, key, data, x, y, fl, fv, color, missing, hand, right=W-80):
    d.text((x, y), label, fill=(0,0,0), font=fl)
    if key in missing:
        d.line([(x, y+58), (right, y+58)], fill=(170,170,170), width=1)
        return
    d.text((x+20, y+32 + (-4 if hand else 0)), data.get(key, ""), fill=color, font=fv)

def checkbox(d, x, y, checked):
    d.rectangle([(x, y), (x+22, y+22)], outline=(0,0,0), width=2)
    if checked:
        d.line([(x+3, y+4), (x+19, y+20)], fill=(0,0,0), width=3)
        d.line([(x+3, y+20), (x+19, y+4)], fill=(0,0,0), width=3)

def draw_form(idx, data, style="typed", quality="clean", missing=None):
    missing = missing or set()
    img = Image.new("RGB", (W, H), (252, 251, 247))
    d = ImageDraw.Draw(img)

    f_hdr   = font(FONT_REG, 38, idx=1)
    f_sec   = font(FONT_REG, 24, idx=1)
    f_lab   = font(FONT_REG, 22, idx=1)
    f_typed = font(FONT_MONO, 24)
    f_hand  = font(FONT_HAND, 32)
    f_small = font(FONT_REG, 17)
    f_sig   = font(FONT_HAND, 42)

    hand = (style == "handwritten")
    f_val = f_hand if hand else f_typed
    val_col = (15, 25, 95) if hand else (15, 15, 15)

    m = 80
    y = 40

    # Fax transmission line
    fax_hdr = (f"FAX TRANSMISSION    {data['date_of_referral']}  09:{random.randint(10,55):02d} AM"
               f"    FROM: {data['referring_phone']}    TO: {phone()}    PAGE 1 OF 1")
    d.text((m, y), fax_hdr, fill=(80,80,80), font=f_small)
    y += 28
    d.line([(m, y), (W-m, y)], fill=(70,70,70), width=1)
    y += 25

    # Letterhead
    d.text((m, y), data['referring_clinic'].upper(), fill=(20, 35, 80), font=f_hdr)
    y += 50
    d.text((m, y), f"Referring Phone: {data['referring_phone']}    |    Fax: {data['referring_fax']}",
           fill=(70,70,70), font=f_small)
    y += 38

    # Title bar
    d.rectangle([(m, y), (W-m, y+56)], fill=(40, 55, 100))
    d.text((m+18, y+12), "PATIENT REFERRAL  /  CONSULTATION REQUEST",
           fill=(255,255,255), font=font(FONT_REG, 26, idx=1))
    y += 85

    # Date + urgency row
    d.text((m, y), "Date of Referral:", fill=(0,0,0), font=f_lab)
    val(d, "date_of_referral", data, m+230, y, f_val, val_col, missing, hand)
    ux = m + 700
    d.text((ux, y), "Urgency:", fill=(0,0,0), font=f_lab)
    cx = ux + 130
    for opt in ["Routine", "Urgent", "STAT"]:
        is_checked = (data.get("urgency") == opt) and ("urgency" not in missing)
        checkbox(d, cx, y+3, is_checked)
        d.text((cx+30, y+1), opt, fill=(0,0,0), font=f_lab)
        cx += 200
    y += 60

    sec_header(d, "PATIENT INFORMATION", m, y, W-m, f_sec)
    y += 60
    d.text((m, y), "Name:", fill=(0,0,0), font=f_lab)
    val(d, "patient_name", data, m+95, y, f_val, val_col, missing, hand)
    d.text((m+800, y), "DOB:", fill=(0,0,0), font=f_lab)
    val(d, "patient_dob", data, m+880, y, f_val, val_col, missing, hand)
    d.text((m+1180, y), "Sex:", fill=(0,0,0), font=f_lab)
    val(d, "patient_sex", data, m+1260, y, f_val, val_col, missing, hand)
    y += 50
    d.text((m, y), "Phone:", fill=(0,0,0), font=f_lab)
    val(d, "patient_phone", data, m+105, y, f_val, val_col, missing, hand)
    d.text((m+550, y), "Insurance:", fill=(0,0,0), font=f_lab)
    val(d, "patient_insurance", data, m+695, y, f_val, val_col, missing, hand)
    y += 50
    d.text((m, y), "Address:", fill=(0,0,0), font=f_lab)
    val(d, "patient_address", data, m+135, y, f_val, val_col, missing, hand)
    y += 70

    sec_header(d, "REFERRING PROVIDER", m, y, W-m, f_sec)
    y += 60
    d.text((m, y), "Physician:", fill=(0,0,0), font=f_lab)
    val(d, "referring_physician", data, m+150, y, f_val, val_col, missing, hand)
    d.text((m+900, y), "NPI:", fill=(0,0,0), font=f_lab)
    val(d, "referring_npi", data, m+980, y, f_val, val_col, missing, hand)
    y += 50
    d.text((m, y), "Clinic:", fill=(0,0,0), font=f_lab)
    val(d, "referring_clinic", data, m+105, y, f_val, val_col, missing, hand)
    y += 50
    d.text((m, y), "Phone:", fill=(0,0,0), font=f_lab)
    val(d, "referring_phone", data, m+105, y, f_val, val_col, missing, hand)
    d.text((m+550, y), "Fax:", fill=(0,0,0), font=f_lab)
    val(d, "referring_fax", data, m+625, y, f_val, val_col, missing, hand)
    y += 70

    sec_header(d, "RECEIVING PROVIDER", m, y, W-m, f_sec)
    y += 60
    d.text((m, y), "Physician:", fill=(0,0,0), font=f_lab)
    val(d, "receiving_physician", data, m+150, y, f_val, val_col, missing, hand)
    y += 50
    d.text((m, y), "Clinic:", fill=(0,0,0), font=f_lab)
    val(d, "receiving_clinic", data, m+105, y, f_val, val_col, missing, hand)
    y += 70

    sec_header(d, "CLINICAL INFORMATION", m, y, W-m, f_sec)
    y += 60
    multi(d, "Reason for Referral:", "reason_for_referral", data, m, y, f_lab, f_val, val_col, missing, hand)
    y += 85
    multi(d, "Diagnosis:", "diagnosis", data, m, y, f_lab, f_val, val_col, missing, hand)
    y += 85
    multi(d, "Current Medications:", "current_medications", data, m, y, f_lab, f_val, val_col, missing, hand)
    y += 85
    multi(d, "Allergies:", "allergies", data, m, y, f_lab, f_val, val_col, missing, hand)
    y += 85
    multi(d, "Relevant History:", "relevant_history", data, m, y, f_lab, f_val, val_col, missing, hand)
    y += 130

    d.text((m, y), "Physician Signature:", fill=(0,0,0), font=f_lab)
    sx = m + 280
    sline = y + 50
    d.line([(sx, sline), (sx+580, sline)], fill=(0,0,0), width=2)
    if data.get("physician_signature_present"):
        sig_text = data["referring_physician"].split(",")[0].replace("Dr. ", "")
        d.text((sx+30, y-2), sig_text, fill=(20, 30, 130), font=f_sig)
    d.text((sx+620, y), "Date:", fill=(0,0,0), font=f_lab)
    d.line([(sx+720, sline), (sx+960, sline)], fill=(0,0,0), width=2)
    if data.get("physician_signature_present"):
        d.text((sx+735, y+8), data["date_of_referral"], fill=(20,30,130), font=f_val)

    if quality == "degraded":
        # Lower contrast (faded toner) + slight blur + slight skew + partial-scan black band
        img = img.point(lambda p: int(min(255, p * 0.55 + 105)))
        img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
        img = img.rotate(-1.7, fillcolor=(245, 244, 240), expand=False, resample=Image.BICUBIC)
        d2 = ImageDraw.Draw(img)
        d2.rectangle([(0, H-160), (W, H)], fill=(15, 15, 15))
        # streaks
        for _ in range(8):
            sy = random.randint(200, H-300)
            d2.line([(0, sy), (W, sy)], fill=(190,190,185), width=1)

    img.save(f"golden/forms/form_{idx:03d}.png", "PNG")

# Build the 10 forms
forms = []
for i in range(10):
    p_name, p_sex, p_dob = PATIENTS[i]
    rp_name, rp_clinic = REF_PHYS[i]
    rcv_name, rcv_clinic = RECV_PHYS[i]
    npi_valid = (i != 9)  # form 10 is the malformed one
    data = {
        "patient_name": p_name,
        "patient_dob": p_dob,
        "patient_sex": p_sex,
        "patient_phone": phone(),
        "patient_address": addr(),
        "patient_insurance": INSURANCES[i],
        "referring_physician": rp_name,
        "referring_clinic": rp_clinic,
        "referring_phone": phone(),
        "referring_fax": phone(),
        "referring_npi": make_npi(valid=npi_valid),
        "receiving_physician": rcv_name,
        "receiving_clinic": rcv_clinic,
        "reason_for_referral": REASONS[i],
        "diagnosis": DIAGNOSES[i],
        "current_medications": MEDS[i],
        "allergies": ALLERGIES[i],
        "relevant_history": HISTORY[i],
        "urgency": ["Routine","Urgent","STAT","Routine","Urgent","Routine","STAT","Urgent","Routine","STAT"][i],
        "date_of_referral": date_2026(),
        "physician_signature_present": True,
    }
    forms.append(data)

styles = {
    1:  ("typed", "clean", set()),
    2:  ("typed", "clean", set()),
    3:  ("typed", "clean", set()),
    4:  ("handwritten", "clean", set()),
    5:  ("typed", "clean", set()),
    6:  ("typed", "clean", set()),
    7:  ("typed", "clean", {"patient_insurance", "allergies", "relevant_history"}),
    8:  ("handwritten", "clean", {"referring_fax", "patient_address", "current_medications"}),
    9:  ("typed", "clean", {"diagnosis", "patient_insurance", "receiving_clinic"}),
    10: ("typed", "degraded", set()),  # degraded + malformed NPI
}

# Mark signature absent on a couple
forms[6]["physician_signature_present"] = False  # form 7
forms[8]["physician_signature_present"] = False  # form 9

for i in range(10):
    style, quality, missing = styles[i+1]
    draw_form(i+1, forms[i], style=style, quality=quality, missing=missing)

print("10 images generated")
