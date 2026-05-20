"""``python -m ale ...`` dispatches to :func:`ale.cli.main`."""
from .cli import main
import sys

if __name__ == "__main__":
    sys.exit(main())
