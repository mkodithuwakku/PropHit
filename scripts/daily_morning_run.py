#!/usr/bin/env python
"""Convenience entrypoint for the Prophit daily morning pipeline.

Equivalent to ``python -m src.publication.daily_refresh`` — pulls the
zero-auth consensus feeds (PrizePicks -> Underdog fallback), generates
plays, and rebuilds the static dashboard bundle.

Usage:
    python scripts/daily_morning_run.py [--out site] [--live] [--publish]
                                        [--dry-run] [--date YYYY-MM-DD]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.publication.daily_refresh import main

if __name__ == "__main__":
    main()
