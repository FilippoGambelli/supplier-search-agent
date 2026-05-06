import logging
from pathlib import Path

LOG_FILE = Path(__file__).parent / "app.log"

logger = logging.getLogger("ai-search")
logger.setLevel(logging.INFO)

# Avoid duplicate handlers (important with reload)
if not logger.handlers:

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File only handler
    file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)