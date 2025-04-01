"""
Main functionality for the OARC package.
"""

import sys
from oarc.utils.log import log
from oarc.cli.cli import cli


def hello():
    """Return a greeting from OARC."""
    log.info("Hello function called")
    return "Hello from OARC main module!"


def main(**kwargs):
    """Main CLI entry point."""
    log.info("OARC main entry point called")
    return cli(**kwargs)


if __name__ == "__main__":
    sys.exit(main())
