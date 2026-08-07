#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------#
"""
Shared logging utilities for the wavy package.

Each module that wants structured logging should do::

    from wavy.logmod import get_logger
    logger = get_logger(__name__)

A ``configure_logging`` convenience function can then be defined in the
module as a thin wrapper (so users can target module-specific loggers by
name)::

    def configure_logging(level="DEBUG"):
        logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))
        logger.info("Logging for '%s' set to %s", __name__, level.upper())
"""

import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger, adding a :class:`StreamHandler` if needed.

    If the logger already has handlers (configured by a previous call or
    by the caller), no new handler is added, so this function is safe to
    call multiple times with the same name.

    Args:
        name:  Logger name — pass ``__name__`` from the calling module.
        level: Initial logging level (default :data:`logging.INFO`).

    Returns:
        Configured :class:`logging.Logger`.

    Example::

        from wavy.logmod import get_logger
        logger = get_logger(__name__)
        logger.info("Module loaded")
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
