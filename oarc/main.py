#!/usr/bin/env python3
"""Main functionality for the OARC package."""

import sys
from oarc.cli.router import handle

def main(**kwargs):
    """Main CLI entry point."""
    return handle(**kwargs)

if __name__ == "__main__":
    sys.exit(main())
