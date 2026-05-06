from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Exit immediately with a non-zero code.")
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--wait-timeout-sec", type=float, default=120.0)
    parser.add_argument("--poll-interval-sec", type=float, default=0.01)
    parser.parse_args()
    return 17


if __name__ == "__main__":
    raise SystemExit(main())
