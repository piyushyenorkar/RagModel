"""
Chunking strategies interface and registry.
All 4 strategies implement the same interface so retrieval code stays identical.
"""

from abc import ABC, abstractmethod
from backend.models import Chunk


class ChunkStrategy(ABC):
    """Base class for all chunking strategies."""
    
    strategy_name: str = "base"
    
    @abstractmethod
    def chunk(self, text: str, passage_id: str, **kwargs) -> list[Chunk]:
        """Split a passage into chunks."""
        ...


# Registry for easy lookup
_STRATEGIES: dict[str, type[ChunkStrategy]] = {}


def register_strategy(cls: type[ChunkStrategy]) -> type[ChunkStrategy]:
    """Decorator to register a chunking strategy."""
    _STRATEGIES[cls.strategy_name] = cls
    return cls


def get_strategy(name: str) -> ChunkStrategy:
    """Get an instantiated strategy by name."""
    if name not in _STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(_STRATEGIES.keys())}")
    return _STRATEGIES[name]()


def get_all_strategies() -> list[ChunkStrategy]:
    """Get all registered strategies."""
    return [cls() for cls in _STRATEGIES.values()]
