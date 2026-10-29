"""Entry point for ``python -m bidswarm``."""

import sys

from .cli import main

if __name__ == "__main__":  # guard required for spawn-based process pools
    sys.exit(main())
