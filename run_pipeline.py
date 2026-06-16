import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, required=True, help="Path to input candidates.jsonl")
    parser.add_argument("--out", type=str, required=True, help="Path to output submission CSV")
    args = parser.parse_args()
    
    print(f"Running pipeline on {args.candidates} to produce {args.out}...")
    
    scripts = [
        ["python", "scripts/resumescore.py", "--input", args.candidates],
        ["python", "scripts/bm25_score.py"],
        ["python", "scripts/feature_engineering.py"],
        ["python", "scripts/ranking.py", "--output", args.out]
    ]
    
    for cmd in scripts:
        print(f"\n>>> Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Error executing {' '.join(cmd)}")
            sys.exit(1)
            
    print(f"\nPipeline complete! Output saved to {args.out}")

if __name__ == "__main__":
    main()
