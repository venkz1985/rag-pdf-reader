import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"
SCHEMAS_DIR = BASE_DIR / "schemas"
PROCESSED_DIR = BASE_DIR / "processed"
LESSONS_DIR = BASE_DIR / "lessons"

DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)
SCHEMAS_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)
LESSONS_DIR.mkdir(exist_ok=True)

REVIEWERS = [
    {"id": "sarah_chen", "name": "Dr. Sarah Chen", "role": "Physician Reviewer"},
    {"id": "marcus_wong", "name": "NP Marcus Wong", "role": "Nurse Practitioner"},
    {"id": "lisa_park", "name": "Admin Lisa Park", "role": "Intake Admin"},
]

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_INFERENCE_URL = "https://router.huggingface.co/hf-inference/models/"

LLM_MODEL = "anthropic/claude-haiku-4.5"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5

VISION_MODEL = "anthropic/claude-haiku-4.5"
DEFAULT_CONFIDENCE_THRESHOLD = 0.85
