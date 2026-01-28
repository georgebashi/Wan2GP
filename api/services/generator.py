"""
Generator Service - Wraps the existing generate_video function for API use.
"""

import uuid
import inspect
import base64
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass, field
from api.profiling import ProfileManager
from PIL import Image
import io

import wgp


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

    def _decode_base64_image(self, data: str) -> Optional[Image.Image]:
        """
        Decode a base64 image string to a PIL Image.
        Returns the PIL Image object.
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
            # Convert to RGB if necessary (e.g., RGBA or palette images)
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            return img
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
        # Handle dynamic model definition
        model_def = params.pop("model_def", None)
        job_id = params.pop("job_id", None)
        model_type = params.get("model_type")

        force_reload = False
        if model_def:
            # Register the model definition dynamically
            if not model_type:
                return GenerationResult(
                    success=False,
                    error="model_type is required when using model_def",
                    error_type="validation_error"
                )
            try:
                # Convert pydantic model to dict if needed
                if hasattr(model_def, "model_dump"):
                    model_def_dict = model_def.model_dump(exclude_none=True)
                else:
                    model_def_dict = dict(model_def)
                _, force_reload = wgp.register_model_def(model_type, model_def_dict)
            except Exception as e:
                return GenerationResult(
                    success=False,
                    error=f"Failed to register model definition: {e}",
                    error_type="model_registration_error"
                )

        # Ensure model is loaded
        if model_type:
            if force_reload or not self.model_manager.is_loaded(model_type):
                try:
                    self.model_manager.load_model(model_type, job_id=job_id)
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
        inputs = wgp.primary_settings.copy()

        # Handle base64 images - decode to PIL Image objects
        for img_field in ["image_start", "image_end"]:
            if img_field in params and params[img_field]:
                img_data = params[img_field]
                if isinstance(img_data, str) and (img_data.startswith("data:") or len(img_data) > 500):
                    # Looks like base64 data
                    pil_image = self._decode_base64_image(img_data)
                    if pil_image:
                        params[img_field] = pil_image

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
                gen = wgp.get_gen_info(state)
                if gen.get("file_list"):
                    result_container["output_path"] = gen["file_list"][-1]
            elif cmd_type == "error":
                result_container["error"] = str(data)
            elif progress_callback:
                progress_callback(cmd_type, data)

        # Get the valid parameter names for generate_video
        sig = inspect.signature(wgp.generate_video)
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

        # Handle video_quality parameter by temporarily overriding server_config
        video_quality = params.get("video_quality")
        original_codec = None
        if video_quality:
            original_codec = wgp.server_config.get("video_output_codec")
            wgp.server_config["video_output_codec"] = video_quality

        # Start inference profile
        profile_mgr = ProfileManager.get_instance()
        if job_id:
            profile_mgr.start("inference", job_id)

        try:
            # Run generation
            wgp.generate_video(**filtered_params)

            # Stop inference profile
            if job_id:
                profile_mgr.stop()

            if result_container["error"]:
                return GenerationResult(
                    success=False,
                    error=result_container["error"],
                    error_type="generation_error"
                )

            # Get output path from state
            gen = wgp.get_gen_info(state)
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
            import traceback
            traceback.print_exc()
            # Stop inference profile on error
            if job_id:
                try:
                    profile_mgr.stop()
                except:
                    pass  # Don't mask original error
            return GenerationResult(
                success=False,
                error=str(e),
                error_type="exception"
            )
        finally:
            # Restore original video codec setting
            if original_codec is not None:
                wgp.server_config["video_output_codec"] = original_codec

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
