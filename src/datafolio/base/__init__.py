"""Base classes and utilities for datafolio handler system.

This module provides the foundation for the handler-based architecture:
- BaseHandler: Abstract base class for all data type handlers
- HandlerRegistry: Registry for managing handlers
- Convenience functions: register_handler, get_handler, detect_handler
"""

from datafolio.base.handler import BaseHandler
from datafolio.base.registry import (
    HandlerRegistry,
    detect_handler,
    get_handler,
    get_registry,
    register_handler,
)

__all__ = [
    "BaseHandler",
    "HandlerRegistry",
    "register_handler",
    "get_handler",
    "detect_handler",
    "get_registry",
]
