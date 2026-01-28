"""
Model Manager - Singleton for managing loaded model state.

This wraps the global model state from wgp.py and provides a clean interface
for the HTTP API to load and query models.
"""

import threading
import time
from typing import Optional, Dict, List
from api.profiling import ProfileManager

# Import wgp at module level to ensure compilation happens at startup
print(f"[{time.time():.3f}] model_manager: Starting wgp import...", flush=True)
_t = time.time()
import wgp
print(f"[{time.time():.3f}] model_manager: wgp imported in {time.time() - _t:.2f}s", flush=True)


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

    def load_model(self, model_type: str, profile: int = -1, job_id: str = None) -> bool:
        """
        Load a model, replacing any currently loaded model.

        Args:
            model_type: The model type identifier (e.g., "t2v", "i2v")
            profile: Memory profile (0-5), -1 for default

        Returns:
            True if model loaded successfully
        """
        with self.model_lock:
            current = self.get_current_model_type()
            if current == model_type:
                return True  # Already loaded

            # Unload current model if any
            if wgp.wan_model is not None:
                wgp.release_model()

            # Start model-load profile if job_id provided
            profile_mgr = ProfileManager.get_instance()
            if job_id:
                profile_mgr.start("model-load", job_id)

            # Load new model
            wgp.wan_model, wgp.offloadobj = wgp.load_models(
                model_type,
                override_profile=profile
            )
            wgp.reload_needed = False

            # Stop model-load profile
            if job_id:
                profile_mgr.stop()

            return True

    def get_current_model_type(self) -> Optional[str]:
        """Get the currently loaded model type."""
        if wgp.wan_model is None:
            return None
        return wgp.transformer_type

    def get_model(self):
        """Get the currently loaded model and offload object."""
        return wgp.wan_model, wgp.offloadobj

    def is_loaded(self, model_type: str = None) -> bool:
        """Check if a model is loaded (optionally check specific type)."""
        if wgp.wan_model is None:
            return False
        if model_type:
            return wgp.transformer_type == model_type
        return True

    def get_available_models(self) -> Dict[str, List[str]]:
        """Get all available model types grouped by family."""
        families = {}
        for base_type, handler in wgp.model_types_handlers.items():
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
