#!/usr/bin/env python3
"""Verify the prepared splits and reproduce their stratification membership."""
import argparse
import json
from pathlib import Path
from verify_splits import audit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data/scierc")
    args = parser.parse_args()
    print(json.dumps(audit(args.data_dir), indent=2))
