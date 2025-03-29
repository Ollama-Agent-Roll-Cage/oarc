"""Main functionality for the OARC package."""
import sys
from oarc.cli import cli


def hello():
    """Return a greeting from OARC."""
    return "Hello from OARC main module!"


def main(**kwargs):
    """Main CLI entry point."""
    return cli(**kwargs)


if __name__ == "__main__":
    sys.exit(main())
