import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_file: Optional[Path | str] = None,
    level: int = logging.INFO
) -> None:
    """Configures root logging for console and optional file output.

    Args:
        log_file: Path to the log file. If provided, parent directories will be created.
        level: Logging level (e.g. logging.INFO, logging.DEBUG).
    """
    handlers = []

    # Console Handler (Stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)

    # File Handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True  # Reset any prior configuration
    )


def get_logger(name: str) -> logging.Logger:
    """Gets a logger instance with a specific name.

    Args:
        name: Name of the logger module.

    Returns:
        logging.Logger: The logger instance.
    """
    return logging.getLogger(name)
