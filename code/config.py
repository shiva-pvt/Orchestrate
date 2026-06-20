"""
Configuration for the Multi-Modal Evidence Review system.
Paths, allowed values, and constants.
"""
import os
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────
# Base dataset directory (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

CLAIMS_CSV = DATASET_DIR / "claims.csv"
SAMPLE_CLAIMS_CSV = DATASET_DIR / "sample_claims.csv"
USER_HISTORY_CSV = DATASET_DIR / "user_history.csv"
EVIDENCE_REQUIREMENTS_CSV = DATASET_DIR / "evidence_requirements.csv"
IMAGES_DIR = DATASET_DIR  # image_paths are relative to dataset dir
OUTPUT_CSV = PROJECT_ROOT / "output.csv"

# ─── API ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

# ─── Allowed output values ──────────────────────────────────────────
ALLOWED_ISSUE_TYPES = [
    "dent", "scratch", "crack", "broken_part", "stain",
    "water_damage", "crushed_packaging", "torn_packaging",
    "missing_contents", "none", "unknown"
]

ALLOWED_OBJECT_PARTS_CAR = [
    "front_bumper", "rear_bumper", "door", "hood", "windshield",
    "headlight", "taillight", "side_mirror", "unknown"
]

ALLOWED_OBJECT_PARTS_LAPTOP = [
    "screen", "keyboard", "trackpad", "hinge", "corner", "body", "unknown"
]

ALLOWED_OBJECT_PARTS_PACKAGE = [
    "package_corner", "package_side", "seal", "label", "contents", "unknown"
]

ALLOWED_OBJECT_PARTS_ALL = sorted(set(
    ALLOWED_OBJECT_PARTS_CAR + ALLOWED_OBJECT_PARTS_LAPTOP + ALLOWED_OBJECT_PARTS_PACKAGE
))

ALLOWED_CLAIM_STATUSES = ["supported", "contradicted", "not_enough_information"]

ALLOWED_RISK_FLAGS = [
    "none", "blurry_image", "wrong_angle", "wrong_object",
    "claim_mismatch", "damage_not_visible", "cropped_or_obstructed",
    "non_original_image", "text_instruction_present",
    "user_history_risk", "manual_review_required"
]

ALLOWED_SEVERITIES = ["low", "medium", "high", "none", "unknown"]

ALLOWED_EVIDENCE_STANDARD = ["true", "false"]

ALLOWED_VALID_IMAGE = ["true", "false"]

# ─── Output columns (exact order) ───────────────────────────────────
OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason",
    "risk_flags", "issue_type", "object_part", "claim_status",
    "claim_status_justification", "supporting_image_ids",
    "valid_image", "severity"
]

# ─── Logging ─────────────────────────────────────────────────────────
LOG_DIR = Path(os.path.expanduser("~")) / "hackerrank_orchestrate"
LOG_FILE = LOG_DIR / "log.txt"
