"""Handler registry for managing data type handlers.

This module provides a global registry for registering and retrieving handlers.
The registry supports:
1. Registering handlers by item_type
2. Retrieving handlers by item_type
3. Auto-detecting handlers based on data type
"""

from typing import Any, Dict, List, Optional

from datafolio.base.handler import BaseHandler


class HandlerRegistry:
    """Global registry of data type handlers.

    The registry maintains a mapping of item_type -> handler and provides
    auto-detection capabilities for the add_data() generic API.

    Examples:
        Create and use a registry:
        >>> registry = HandlerRegistry()
        >>> registry.register(PandasHandler())
        >>> handler = registry.get('included_table')
        >>> handler = registry.detect(pd.DataFrame())
    """

    def __init__(self):
        """Initialize an empty registry."""
        self._handlers: Dict[str, BaseHandler] = {}

    def register(self, handler: BaseHandler) -> None:
        """Register a handler.

        Args:
            handler: Handler instance to register

        Raises:
            ValueError: If item_type already registered

        Examples:
            >>> registry = HandlerRegistry()
            >>> registry.register(PandasHandler())
            >>> registry.register(NumpyHandler())
        """
        item_type = handler.item_type
        if item_type in self._handlers:
            raise ValueError(
                f"Handler already registered for item_type: {item_type}. "
                f"Cannot register {handler.__class__.__name__}."
            )
        self._handlers[item_type] = handler

    def get(self, item_type: str) -> BaseHandler:
        """Get handler by item_type.

        Args:
            item_type: Item type string (e.g., 'included_table')

        Returns:
            Handler instance for the given item_type

        Raises:
            KeyError: If no handler registered for item_type

        Examples:
            >>> registry = HandlerRegistry()
            >>> registry.register(PandasHandler())
            >>> handler = registry.get('included_table')
            >>> isinstance(handler, PandasHandler)
            True
        """
        if item_type not in self._handlers:
            available = ", ".join(sorted(self._handlers.keys()))
            raise KeyError(
                f"No handler registered for item_type: '{item_type}'. "
                f"Available types: {available}"
            )
        return self._handlers[item_type]

    def detect(self, data: Any) -> Optional[BaseHandler]:
        """Auto-detect handler for data.

        Tries each registered handler's can_handle() method.
        Returns first handler that can handle the data.

        Args:
            data: Data object to detect type for

        Returns:
            Handler that can handle data, or None if no match

        Examples:
            >>> registry = HandlerRegistry()
            >>> registry.register(PandasHandler())
            >>> registry.register(NumpyHandler())
            >>> handler = registry.detect(pd.DataFrame())
            >>> isinstance(handler, PandasHandler)
            True
            >>> handler = registry.detect(np.array([1, 2, 3]))
            >>> isinstance(handler, NumpyHandler)
            True
            >>> handler = registry.detect("unknown type")
            >>> handler is None
            True
        """
        for handler in self._handlers.values():
            if handler.can_handle(data):
                return handler
        return None

    def list_types(self) -> List[str]:
        """List all registered item types.

        Returns:
            List of item_type strings

        Examples:
            >>> registry = HandlerRegistry()
            >>> registry.register(PandasHandler())
            >>> registry.register(NumpyHandler())
            >>> sorted(registry.list_types())
            ['included_table', 'numpy_array']
        """
        return list(self._handlers.keys())

    def is_registered(self, item_type: str) -> bool:
        """Check if a handler is registered for the given item_type.

        Args:
            item_type: Item type string to check

        Returns:
            True if handler is registered, False otherwise

        Examples:
            >>> registry = HandlerRegistry()
            >>> registry.register(PandasHandler())
            >>> registry.is_registered('included_table')
            True
            >>> registry.is_registered('unknown_type')
            False
        """
        return item_type in self._handlers


# Global singleton registry
_registry = HandlerRegistry()


def get_registry() -> HandlerRegistry:
    """Get the global handler registry.

    Returns:
        The global HandlerRegistry instance

    Examples:
        >>> registry = get_registry()
        >>> registry.list_types()
        ['included_table', 'numpy_array', ...]
    """
    return _registry


def register_handler(handler: BaseHandler) -> None:
    """Register a handler in the global registry.

    Convenience function for registering handlers without accessing
    the registry directly.

    Args:
        handler: Handler instance to register

    Raises:
        ValueError: If item_type already registered

    Examples:
        >>> register_handler(PandasHandler())
        >>> register_handler(NumpyHandler())
    """
    _registry.register(handler)


def get_handler(item_type: str) -> BaseHandler:
    """Get handler by item_type from global registry.

    Convenience function for getting handlers without accessing
    the registry directly.

    Args:
        item_type: Item type string (e.g., 'included_table')

    Returns:
        Handler instance for the given item_type

    Raises:
        KeyError: If no handler registered for item_type

    Examples:
        >>> handler = get_handler('included_table')
        >>> isinstance(handler, PandasHandler)
        True
    """
    return _registry.get(item_type)


def detect_handler(data: Any) -> Optional[BaseHandler]:
    """Auto-detect handler for data using global registry.

    Convenience function for detecting handlers without accessing
    the registry directly.

    Args:
        data: Data object to detect type for

    Returns:
        Handler that can handle data, or None if no match

    Examples:
        >>> handler = detect_handler(pd.DataFrame())
        >>> isinstance(handler, PandasHandler)
        True
        >>> handler = detect_handler("unknown")
        >>> handler is None
        True
    """
    return _registry.detect(data)
