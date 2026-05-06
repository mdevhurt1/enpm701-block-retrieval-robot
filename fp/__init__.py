# Final Project - fp Package Init
# Configures package-wide logging at import time so all modules share
# a consistent timestamp format without each module calling basicConfig.
#
# Requirements: logging, sys
import logging
import sys

# Configure once at package import. Mission code is import-heavy at startup
# (vision.py loads camera calibration at module level), so basicConfig has to
# happen here rather than in cli.main() to catch the early diagnostics.
# stream=sys.stdout matches the prior `print()` destination so existing log
# capture / redirection setups keep working unchanged.
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s.%(msecs)03d] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
