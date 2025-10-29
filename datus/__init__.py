# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Datus - AI-powered SQL command-line interface"""

import os
import sys

# Work around macOS Accelerate crashes when importing NumPy on Python 3.11.
# NumPy performs a sanity check that may segfault on some macOS setups.
# Import NumPy once with sys.platform temporarily set to "linux" so the
# check is skipped. Subsequent imports reuse the loaded module safely.
if sys.platform == "darwin" and "numpy" not in sys.modules:
    original_platform = sys.platform
    try:  # noqa: SIM105
        sys.platform = "linux"
        try:
            import numpy  # noqa: F401
        except Exception:
            pass
    finally:
        sys.platform = original_platform

__version__ = "0.2.1"
