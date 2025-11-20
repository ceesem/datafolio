"""Tests for BaseHandler and HandlerRegistry."""

from typing import Any, Dict, Optional

import pytest

from datafolio.base import BaseHandler, HandlerRegistry, get_registry, register_handler


class MockHandler(BaseHandler):
    """Mock handler for testing."""

    def __init__(self, item_type_value: str = "mock_type"):
        self._item_type = item_type_value

    @property
    def item_type(self) -> str:
        return self._item_type

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, str) and data.startswith("mock:")

    def add(
        self,
        folio: Any,
        name: str,
        data: Any,
        description: Optional[str] = None,
        inputs: Optional[list[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        return {
            "name": name,
            "item_type": self.item_type,
            "data": data,
            "description": description,
            "inputs": inputs,
        }

    def get(self, folio: Any, name: str, **kwargs) -> Any:
        return f"retrieved:{name}"

    def get_storage_subdir(self) -> str:
        return "mock"


class AnotherMockHandler(BaseHandler):
    """Another mock handler for testing."""

    @property
    def item_type(self) -> str:
        return "another_type"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, int)

    def add(
        self,
        folio: Any,
        name: str,
        data: Any,
        description: Optional[str] = None,
        inputs: Optional[list[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        return {"name": name, "item_type": self.item_type, "value": data}

    def get(self, folio: Any, name: str, **kwargs) -> Any:
        return 42

    def get_storage_subdir(self) -> str:
        return "another"


class TestBaseHandler:
    """Tests for BaseHandler abstract class."""

    def test_cannot_instantiate_base_handler(self):
        """Test that BaseHandler cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseHandler()

    def test_mock_handler_implements_interface(self):
        """Test that mock handler implements all required methods."""
        handler = MockHandler()

        assert handler.item_type == "mock_type"
        assert handler.can_handle("mock:data") is True
        assert handler.can_handle("other") is False
        assert handler.get_storage_subdir() == "mock"

        # Test add
        metadata = handler.add(
            None, "test", "mock:value", description="Test", inputs=["input1"]
        )
        assert metadata["name"] == "test"
        assert metadata["item_type"] == "mock_type"
        assert metadata["data"] == "mock:value"
        assert metadata["description"] == "Test"
        assert metadata["inputs"] == ["input1"]

        # Test get
        result = handler.get(None, "test")
        assert result == "retrieved:test"


class TestHandlerRegistry:
    """Tests for HandlerRegistry class."""

    def test_empty_registry(self):
        """Test that new registry is empty."""
        registry = HandlerRegistry()
        assert registry.list_types() == []

    def test_register_handler(self):
        """Test registering a handler."""
        registry = HandlerRegistry()
        handler = MockHandler()

        registry.register(handler)

        assert "mock_type" in registry.list_types()
        assert registry.is_registered("mock_type") is True
        assert registry.is_registered("unknown_type") is False

    def test_register_duplicate_raises_error(self):
        """Test that registering duplicate item_type raises error."""
        registry = HandlerRegistry()
        handler1 = MockHandler()
        handler2 = MockHandler()  # Same item_type

        registry.register(handler1)

        with pytest.raises(ValueError, match="Handler already registered"):
            registry.register(handler2)

    def test_get_handler(self):
        """Test retrieving a handler by item_type."""
        registry = HandlerRegistry()
        handler = MockHandler()

        registry.register(handler)

        retrieved = registry.get("mock_type")
        assert retrieved is handler
        assert retrieved.item_type == "mock_type"

    def test_get_unregistered_handler_raises_error(self):
        """Test that getting unregistered handler raises error."""
        registry = HandlerRegistry()

        with pytest.raises(KeyError, match="No handler registered"):
            registry.get("unknown_type")

    def test_detect_handler(self):
        """Test auto-detecting handler by data type."""
        registry = HandlerRegistry()
        mock_handler = MockHandler()
        another_handler = AnotherMockHandler()

        registry.register(mock_handler)
        registry.register(another_handler)

        # Test detection
        detected = registry.detect("mock:data")
        assert detected is mock_handler

        detected = registry.detect(42)
        assert detected is another_handler

        detected = registry.detect([1, 2, 3])
        assert detected is None

    def test_list_types(self):
        """Test listing all registered types."""
        registry = HandlerRegistry()

        assert registry.list_types() == []

        registry.register(MockHandler())
        assert registry.list_types() == ["mock_type"]

        registry.register(AnotherMockHandler())
        assert sorted(registry.list_types()) == ["another_type", "mock_type"]

    def test_multiple_handlers_detection_order(self):
        """Test that first matching handler is returned."""
        registry = HandlerRegistry()

        # Both handlers can handle strings starting with "mock:"
        handler1 = MockHandler("type1")
        handler2 = MockHandler("type2")

        registry.register(handler1)
        registry.register(handler2)

        # First registered handler should be detected
        detected = registry.detect("mock:data")
        assert detected.item_type == "type1"


class TestGlobalRegistry:
    """Tests for global registry convenience functions."""

    def setup_method(self):
        """Clear global registry before each test and re-register built-in handlers."""
        # Get fresh registry for each test
        registry = get_registry()
        registry._handlers.clear()

        # Re-register all built-in handlers after clearing
        from datafolio.handlers import (
            ArtifactHandler,
            JsonHandler,
            NumpyHandler,
            PandasHandler,
            PyTorchHandler,
            ReferenceTableHandler,
            SklearnHandler,
            TimestampHandler,
        )

        register_handler(PandasHandler())
        register_handler(ReferenceTableHandler())
        register_handler(NumpyHandler())
        register_handler(JsonHandler())
        register_handler(TimestampHandler())
        register_handler(ArtifactHandler())
        register_handler(SklearnHandler())
        register_handler(PyTorchHandler())

    def test_register_handler_function(self):
        """Test register_handler convenience function."""
        handler = MockHandler()
        register_handler(handler)

        registry = get_registry()
        assert registry.is_registered("mock_type")

    def test_get_handler_function(self):
        """Test get_handler convenience function."""
        from datafolio.base import get_handler

        handler = MockHandler()
        register_handler(handler)

        retrieved = get_handler("mock_type")
        assert retrieved is handler

    def test_detect_handler_function(self):
        """Test detect_handler convenience function."""
        from datafolio.base import detect_handler

        handler = MockHandler()
        register_handler(handler)

        detected = detect_handler("mock:data")
        assert detected is handler

        detected = detect_handler("other")
        assert detected is None
