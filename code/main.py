"""
Main entry point: reads claims.csv, processes each claim via VLM, writes output.csv.
Also sets up logging per AGENTS.md.
"""
import sys
import os
import datetime
import pandas as pd
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    CLAIMS_CSV, SAMPLE_CLAIMS_CSV, USER_HISTORY_CSV,
    EVIDENCE_REQUIREMENTS_CSV, OUTPUT_CSV, OUTPUT_COLUMNS,
    LOG_DIR, LOG_FILE, DATASET_DIR
)
from claim_processor import ClaimProcessor


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


def run_claims(claims_csv_path: Path, output_csv_path: Path):
    """Process all claims and write output CSV."""
    print("=" * 60)
    print("Multi-Modal Evidence Review System")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading data files...")
    claims_df = pd.read_csv(claims_csv_path)
    user_history_df = pd.read_csv(USER_HISTORY_CSV)
    evidence_req_df = pd.read_csv(EVIDENCE_REQUIREMENTS_CSV)
    sample_df = pd.read_csv(SAMPLE_CLAIMS_CSV)
    
    print(f"  Claims to process: {len(claims_df)}")
    print(f"  User history entries: {len(user_history_df)}")
    print(f"  Evidence requirements: {len(evidence_req_df)}")
    
    # Initialize processor
    print("\n[2/4] Initializing VLM processor...")
    processor = ClaimProcessor(
        user_history_df=user_history_df,
        evidence_req_df=evidence_req_df,
    )
    print("  Gemini client initialized.")
    
    # Process each claim
    print(f"\n[3/4] Processing {len(claims_df)} claims...")
    log_entry("START_PROCESSING", f"Processing {len(claims_df)} claims from {claims_csv_path}")
    
    results = []
    for idx, row in claims_df.iterrows():
        claim_num = idx + 1
        print(f"\n--- Claim {claim_num}/{len(claims_df)} ---")
        print(f"  User: {row['user_id']}, Object: {row['claim_object']}")
        
        # Process the claim
        output = processor.process_claim(row.to_dict())
        
        # Build output row (input columns + output columns)
        output_row = {
            'user_id': row['user_id'],
            'image_paths': row['image_paths'],
            'user_claim': row['user_claim'],
            'claim_object': row['claim_object'],
        }
        output_row.update(output)
        results.append(output_row)
        
        print(f"  Status: {output['claim_status']} | "
              f"Issue: {output['issue_type']} | "
              f"Part: {output['object_part']} | "
              f"Severity: {output['severity']}")
        
        log_entry(
            f"CLAIM_PROCESSED",
            f"claim={claim_num}, user={row['user_id']}, object={row['claim_object']}, "
            f"status={output['claim_status']}, issue={output['issue_type']}, "
            f"part={output['object_part']}"
        )
    
    # Write output CSV
    print(f"\n[4/4] Writing output to {output_csv_path}...")
    output_df = pd.DataFrame(results)
    # Ensure column order matches expected schema
    output_df = output_df[OUTPUT_COLUMNS]
    output_df.to_csv(output_csv_path, index=False, quoting=1)  # quoting=1 = QUOTE_ALL
    
    print(f"\n{'=' * 60}")
    print(f"DONE! Output written to: {output_csv_path}")
    print(f"Total claims processed: {len(results)}")
    print(f"{'=' * 60}")
    
    log_entry("PROCESSING_COMPLETE", f"Wrote {len(results)} results to {output_csv_path}")
    
    return output_df


if __name__ == "__main__":
    setup_logging()
    log_entry("SESSION_START", "Main processing run started")
    
    # Check for API key
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set!")
        print("Get a free key from: https://aistudio.google.com/apikey")
        print("Then run: $env:GEMINI_API_KEY = 'your-key-here'")
        sys.exit(1)
    
    # Determine paths — use dataset dir relative to project root or from claims data
    claims_path = CLAIMS_CSV
    output_path = OUTPUT_CSV
    
    # Check if claims.csv exists at expected location
    if not claims_path.exists():
        # Try relative to current directory
        alt_claims = Path("dataset/claims.csv")
        if alt_claims.exists():
            claims_path = alt_claims
            output_path = Path("output.csv")
        else:
            # Try the extracted claims directory
            alt_claims2 = Path("claims/claims/claims.csv")
            if alt_claims2.exists():
                claims_path = alt_claims2
                output_path = Path("output.csv")
            else:
                print(f"ERROR: Cannot find claims.csv at {claims_path}")
                print("Searched also: dataset/claims.csv, claims/claims/claims.csv")
                sys.exit(1)
    
    print(f"Using claims from: {claims_path}")
    print(f"Output will be written to: {output_path}")
    
    run_claims(claims_path, output_path)
