"""Simple registry for methods, datasets, and backbones."""

from typing import Dict, Type


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type] = {}

    def register(self, key: str):
        def decorator(cls):
            if key in self._registry:
                raise ValueError(f"{self.name} '{key}' already registered.")
            self._registry[key] = cls
            return cls
        return decorator

    def get(self, key: str) -> Type:
        if key not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(f"{self.name} '{key}' not found. Available: {available}")
        return self._registry[key]

    def list(self) -> list:
        return sorted(self._registry.keys())


METHOD_REGISTRY = Registry("Method")
DATASET_REGISTRY = Registry("Dataset")
BACKBONE_REGISTRY = Registry("Backbone")
