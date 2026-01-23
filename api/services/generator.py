"""
Generator Service - Wraps the existing generate_video function for API use.
"""

import uuid
import inspect
import base64
import tempfile
import os
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass, field
from PIL import Image
import io


@dataclass
class GenerationResult:
    """Result of a generation request."""
    success: bool
    output_path: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class GeneratorService:
    """Wraps the existing generate_video function for API use."""

    def __init__(self, model_manager):
        self.model_manager = model_manager
        self._wgp_imported = False

    def _ensure_wgp_imported(self):
        """Lazy import of wgp module."""
        if not self._wgp_imported:
            import wgp
            self._wgp = wgp
            self._wgp_imported = True

    def _decode_base64_image(self, data: str) -> Optional[str]:
        """
        Decode a base64 image string and save to a temp file.
        Returns the temp file path.
        """
        if data is None:
            return None

        # Handle data URI format
        if data.startswith("data:"):
            # Extract the base64 part after the comma
            try:
                header, encoded = data.split(",", 1)
            except ValueError:
                return None
        else:
            encoded = data

        try:
            img_bytes = base64.b64decode(encoded)
            img = Image.open(io.BytesIO(img_bytes))

            # Save to temp file
            suffix = ".png"
            if "jpeg" in data.lower() or "jpg" in data.lower():
                suffix = ".jpg"

            fd, path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            img.save(path)
            return path
        except Exception as e:
            print(f"Error decoding base64 image: {e}")
            return None

    def generate(
        self,
        params: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> GenerationResult:
        """
        Execute a generation request.

        Args:
            params: Generation parameters (prompt, resolution, etc.)
            progress_callback: Optional callback for progress updates

        Returns:
            GenerationResult with output path or error
        """
        self._ensure_wgp_imported()

        # Ensure model is loaded
        model_type = params.get("model_type")
        if model_type:
            if not self.model_manager.is_loaded(model_type):
                try:
                    self.model_manager.load_model(model_type)
                except Exception as e:
                    return GenerationResult(
                        success=False,
                        error=f"Failed to load model: {e}",
                        error_type="model_load_error"
                    )
        elif not self.model_manager.is_loaded():
            return GenerationResult(
                success=False,
                error="No model loaded. Please load a model first or specify model_type.",
                error_type="no_model_error"
            )
        else:
            # Use currently loaded model
            model_type = self.model_manager.get_current_model_type()

        # Create task structure
        task_id = f"api_{uuid.uuid4().hex[:8]}"
        task = {
            "id": task_id,
            "prompt": params.get("prompt", ""),
            "params": params
        }

        # Create state
        state = self._create_state()

        # Start with primary settings as defaults
        inputs = self._wgp.primary_settings.copy()

        # Handle base64 images - decode to temp files
        temp_files = []
        for img_field in ["image_start", "image_end"]:
            if img_field in params and params[img_field]:
                img_data = params[img_field]
                if isinstance(img_data, str) and (img_data.startswith("data:") or len(img_data) > 500):
                    # Looks like base64 data
                    temp_path = self._decode_base64_image(img_data)
                    if temp_path:
                        temp_files.append(temp_path)
                        params[img_field] = [temp_path]  # Wrap in list as expected

        # Override with provided params
        inputs.update(params)
        inputs["prompt"] = task["prompt"]
        inputs.setdefault("mode", "")

        # Set model_type
        inputs["model_type"] = model_type

        # Setup result container
        result_container = {"output_path": None, "error": None}

        def send_cmd(cmd_type, data=None):
            """Callback to capture progress and results."""
            if cmd_type == "output":
                gen = self._wgp.get_gen_info(state)
                if gen.get("file_list"):
                    result_container["output_path"] = gen["file_list"][-1]
            elif cmd_type == "error":
                result_container["error"] = str(data)
            elif progress_callback:
                progress_callback(cmd_type, data)

        # Get the valid parameter names for generate_video
        sig = inspect.signature(self._wgp.generate_video)
        expected_args = set(sig.parameters.keys())

        # Filter inputs to only valid parameters
        filtered_params = {}
        for k, v in inputs.items():
            if k in expected_args:
                filtered_params[k] = v

        # Add required parameters
        filtered_params["task"] = task
        filtered_params["send_cmd"] = send_cmd
        filtered_params["state"] = state

        # Ensure we have image_mode
        if "image_mode" not in filtered_params:
            filtered_params["image_mode"] = 0

        try:
            # Run generation
            self._wgp.generate_video(**filtered_params)

            # Cleanup temp files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass

            if result_container["error"]:
                return GenerationResult(
                    success=False,
                    error=result_container["error"],
                    error_type="generation_error"
                )

            # Get output path from state
            gen = self._wgp.get_gen_info(state)
            output_path = None
            if gen.get("file_list"):
                output_path = gen["file_list"][-1]

            return GenerationResult(
                success=True,
                output_path=output_path,
                metadata={
                    "seed": inputs.get("seed"),
                    "model_type": model_type,
                    "resolution": inputs.get("resolution"),
                    "video_length": inputs.get("video_length"),
                    "num_inference_steps": inputs.get("num_inference_steps")
                }
            )

        except Exception as e:
            # Cleanup temp files on error
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass

            import traceback
            traceback.print_exc()
            return GenerationResult(
                success=False,
                error=str(e),
                error_type="exception"
            )

    def _create_state(self) -> Dict:
        """Create a minimal state dict for generation."""
        return {
            "gen": {
                "queue": [],
                "in_progress": False,
                "file_list": [],
                "file_settings_list": [],
                "audio_file_list": [],
                "audio_file_settings_list": [],
                "selected": 0,
                "audio_selected": 0,
                "prompt_no": 1,
                "prompts_max": 1,
                "repeat_no": 1,
                "total_generation": 1,
                "window_no": 1,
                "total_windows": 1,
                "progress_status": "",
                "process_status": "process:main",
                "abort": False,
            },
            "loras": [],
            "model_type": self.model_manager.get_current_model_type(),
        }
