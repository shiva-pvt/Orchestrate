"""
Evaluation pipeline: runs system on sample_claims.csv and compares against expected outputs.
Computes per-field accuracy metrics.
"""
import sys
import os
import pandas as pd
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    SAMPLE_CLAIMS_CSV, USER_HISTORY_CSV, EVIDENCE_REQUIREMENTS_CSV,
    OUTPUT_COLUMNS, LOG_DIR, LOG_FILE, DATASET_DIR
)
from claim_processor import ClaimProcessor
import datetime


def setup_logging():
    """Create log directory and file if they don't exist."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE.exists():
        LOG_FILE.touch()


def log_entry(action: str, details: str = ""):
    """Append a log entry."""
    timestamp = datetime.datetime.now().isoformat()
    entry = f"[{timestamp}] ACTION: {action}"
    if details:
        entry += f" | DETAILS: {details}"
    entry += "\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry)


def compute_flag_overlap(predicted: str, expected: str) -> float:
    """Compute Jaccard similarity between two semicolon-separated flag sets."""
    pred_set = set(f.strip() for f in str(predicted).split(';') if f.strip())
    exp_set = set(f.strip() for f in str(expected).split(';') if f.strip())
    
    if not pred_set and not exp_set:
        return 1.0
    if not pred_set or not exp_set:
        return 0.0
    
    intersection = pred_set & exp_set
    union = pred_set | exp_set
    return len(intersection) / len(union) if union else 1.0


def evaluate():
    """Run evaluation on sample_claims.csv and print metrics."""
    print("=" * 60)
    print("Evaluation Pipeline")
    print("=" * 60)
    
    # Load data
    print("\n[1/3] Loading data...")
    sample_df = pd.read_csv(SAMPLE_CLAIMS_CSV)
    user_history_df = pd.read_csv(USER_HISTORY_CSV)
    evidence_req_df = pd.read_csv(EVIDENCE_REQUIREMENTS_CSV)
    
    print(f"  Sample claims: {len(sample_df)}")
    
    # Initialize processor
    print("\n[2/3] Processing sample claims through VLM...")
    processor = ClaimProcessor(
        user_history_df=user_history_df,
        evidence_req_df=evidence_req_df,
    )
    
    # Process each sample claim
    predictions = []
    for idx, row in sample_df.iterrows():
        print(f"\n  Processing sample {idx+1}/{len(sample_df)}: {row['user_id']} ({row['claim_object']})")
        output = processor.process_claim(row.to_dict())
        predictions.append(output)
    
    # Compare predictions to expected
    print(f"\n\n[3/3] Evaluation Results")
    print("=" * 60)
    
    # Fields to evaluate
    exact_match_fields = [
        'evidence_standard_met', 'issue_type', 'object_part',
        'claim_status', 'valid_image', 'severity'
    ]
    
    flag_overlap_fields = ['risk_flags', 'supporting_image_ids']
    
    # Compute exact match accuracy per field
    results = {}
    for field in exact_match_fields:
        correct = 0
        total = len(sample_df)
        details = []
        for i, (pred, (_, expected)) in enumerate(zip(predictions, sample_df.iterrows())):
            pred_val = str(pred.get(field, '')).lower().strip()
            exp_val = str(expected.get(field, '')).lower().strip()
            match = pred_val == exp_val
            if match:
                correct += 1
            else:
                details.append(f"    Row {i+1}: predicted='{pred_val}', expected='{exp_val}'")
        
        acc = correct / total if total > 0 else 0
        results[field] = acc
        status = "✓" if acc >= 0.8 else "✗"
        print(f"\n  {status} {field}: {correct}/{total} = {acc:.1%}")
        if details and acc < 1.0:
            for d in details[:5]:  # Show at most 5 mismatches
                print(d)
            if len(details) > 5:
                print(f"    ... and {len(details) - 5} more mismatches")
    
    # Compute flag overlap for risk_flags and supporting_image_ids
    for field in flag_overlap_fields:
        total_overlap = 0
        total = len(sample_df)
        details = []
        for i, (pred, (_, expected)) in enumerate(zip(predictions, sample_df.iterrows())):
            pred_val = str(pred.get(field, 'none'))
            exp_val = str(expected.get(field, 'none'))
            overlap = compute_flag_overlap(pred_val, exp_val)
            total_overlap += overlap
            if overlap < 1.0:
                details.append(f"    Row {i+1}: predicted='{pred_val}', expected='{exp_val}' (overlap={overlap:.2f})")
        
        avg_overlap = total_overlap / total if total > 0 else 0
        results[field] = avg_overlap
        status = "✓" if avg_overlap >= 0.7 else "✗"
        print(f"\n  {status} {field} (Jaccard): {avg_overlap:.1%}")
        if details and avg_overlap < 1.0:
            for d in details[:5]:
                print(d)
            if len(details) > 5:
                print(f"    ... and {len(details) - 5} more mismatches")
    
    # Overall summary
    print(f"\n{'=' * 60}")
    overall = sum(results.values()) / len(results)
    print(f"Overall Score: {overall:.1%}")
    print(f"{'=' * 60}")
    
    # Save evaluation predictions for comparison
    eval_output_path = Path(__file__).resolve().parent.parent / "evaluation_output.csv"
    pred_rows = []
    for i, (pred, (_, row)) in enumerate(zip(predictions, sample_df.iterrows())):
        pred_row = {
            'user_id': row['user_id'],
            'image_paths': row['image_paths'],
            'user_claim': row['user_claim'],
            'claim_object': row['claim_object'],
        }
        pred_row.update(pred)
        pred_rows.append(pred_row)
    
    eval_df = pd.DataFrame(pred_rows)
    eval_df = eval_df[[c for c in OUTPUT_COLUMNS if c in eval_df.columns]]
    eval_df.to_csv(eval_output_path, index=False, quoting=1)
    print(f"\nPredictions saved to: {eval_output_path}")
    
    log_entry("EVALUATION_COMPLETE", f"Overall score: {overall:.1%}")
    
    return results


if __name__ == "__main__":
    setup_logging()
    log_entry("SESSION_START", "Evaluation run started")
    
    # Check for API key
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set!")
        print("Get a free key from: https://aistudio.google.com/apikey")
        print("Then run: $env:GEMINI_API_KEY = 'your-key-here'")
        sys.exit(1)
    
    evaluate()
