"""Stand-in for the trainer's ``library.utils``.

The vendored modules only consume ``setup_logging`` from here, so this is a
minimal shim that wires Python's ``logging`` to stderr the first time it's
called and is a no-op afterwards.
"""

import logging
import os
import sys


def setup_logging(args=None, log_level=None, reset=False):
    if logging.root.handlers and not reset:
        return
    if reset:
        for h in logging.root.handlers[:]:
            logging.root.removeHandler(h)
    level_name = log_level or os.environ.get("LOG_LEVEL") or "INFO"
    level = getattr(logging, level_name.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                                           datefmt="%Y-%m-%d %H:%M:%S"))
    logging.root.setLevel(level)
    logging.root.addHandler(handler)


setup_logging()
