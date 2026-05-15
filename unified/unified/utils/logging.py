from __future__ import annotations
import logging
import sys
from pathlib import Path


def setup_logging(log_dir: str | Path | None = None, level: int = logging.INFO):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(log_dir) / "run.log"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
