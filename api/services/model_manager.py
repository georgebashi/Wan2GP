"""
Model Manager - Singleton for managing loaded model state.

This wraps the global model state from wgp.py and provides a clean interface
for the HTTP API to load and query models.
"""

import threading
from typing import Optional, Dict, Any, List


class ModelManager:
    """Singleton that manages model loading and caching."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.model_lock = threading.Lock()
        self._wgp_imported = False

    def _ensure_wgp_imported(self):
        """Lazy import of wgp module to avoid circular imports."""
        if not self._wgp_imported:
            import wgp
            self._wgp = wgp
            self._wgp_imported = True

    def load_model(self, model_type: str, profile: int = -1) -> bool:
        """
        Load a model, replacing any currently loaded model.

        Args:
            model_type: The model type identifier (e.g., "t2v", "i2v")
            profile: Memory profile (0-5), -1 for default

        Returns:
            True if model loaded successfully
        """
        with self.model_lock:
            self._ensure_wgp_imported()

            current = self.get_current_model_type()
            if current == model_type:
                return True  # Already loaded

            # Unload current model if any
            if self._wgp.wan_model is not None:
                self._wgp.release_model()

            # Load new model
            self._wgp.wan_model, self._wgp.offloadobj = self._wgp.load_models(
                model_type,
                override_profile=profile
            )
            self._wgp.reload_needed = False
            return True

    def get_current_model_type(self) -> Optional[str]:
        """Get the currently loaded model type."""
        self._ensure_wgp_imported()
        if self._wgp.wan_model is None:
            return None
        return self._wgp.transformer_type

    def get_model(self):
        """Get the currently loaded model and offload object."""
        self._ensure_wgp_imported()
        return self._wgp.wan_model, self._wgp.offloadobj

    def is_loaded(self, model_type: str = None) -> bool:
        """Check if a model is loaded (optionally check specific type)."""
        self._ensure_wgp_imported()
        if self._wgp.wan_model is None:
            return False
        if model_type:
            return self._wgp.transformer_type == model_type
        return True

    def get_available_models(self) -> Dict[str, List[str]]:
        """Get all available model types grouped by family."""
        self._ensure_wgp_imported()

        families = {}
        for base_type, handler in self._wgp.model_types_handlers.items():
            family = handler.query_model_family()
            if family not in families:
                families[family] = []
            supported = handler.query_supported_types()
            if isinstance(supported, list):
                families[family].extend(supported)
            else:
                families[family].append(base_type)

        return families

    def get_all_model_types(self) -> List[str]:
        """Get flat list of all available model types."""
        families = self.get_available_models()
        all_types = []
        for types in families.values():
            all_types.extend(types)
        return sorted(set(all_types))


# Global singleton instance
model_manager = ModelManager()
